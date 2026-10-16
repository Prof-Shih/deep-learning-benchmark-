# debug_efficientnetv2s_hef_acc.py

import argparse
import time
import threading
import subprocess
import re
from pathlib import Path
from collections import Counter

import numpy as np
import psutil
from PIL import Image

from hailo_platform import VDevice, FormatType


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jpeg"}


# =========================================================
# Dataset
# =========================================================
def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path))
    ]


def build_imagenet_dataset(val_dir, limit=0):
    val_dir = Path(val_dir)

    if not val_dir.exists():
        raise RuntimeError(f"Dataset folder not found: {val_dir}")

    class_dirs = sorted(
        [p for p in val_dir.iterdir() if p.is_dir()],
        key=natural_sort_key
    )

    if len(class_dirs) == 0:
        raise RuntimeError(f"No class folders found under: {val_dir}")

    samples = []

    for cls_idx, cls_dir in enumerate(class_dirs):
        imgs = []

        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                imgs.append(p)

        imgs = sorted(imgs, key=natural_sort_key)

        for img_path in imgs:
            samples.append((str(img_path), cls_idx, cls_dir.name))

    if len(samples) == 0:
        raise RuntimeError(f"No images found under: {val_dir}")

    if limit > 0:
        samples = samples[:limit]

    return samples, class_dirs


# =========================================================
# Monitor
# =========================================================
def read_pmic_power_w():
    try:
        out = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"],
            text=True,
            timeout=1.0
        )
    except Exception:
        return None

    currents = {}
    volts = {}

    for line in out.splitlines():
        m_i = re.match(r"(.+?)_A\s+current\(\d+\)=([0-9.]+)A", line.strip())
        if m_i:
            currents[m_i.group(1).strip()] = float(m_i.group(2))
            continue

        m_v = re.match(r"(.+?)_V\s+volt\(\d+\)=([0-9.]+)V", line.strip())
        if m_v:
            volts[m_v.group(1).strip()] = float(m_v.group(2))
            continue

    total_w = 0.0

    for rail, current in currents.items():
        if rail in volts:
            total_w += current * volts[rail]

    return total_w


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


class Monitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.stop_event = threading.Event()
        self.proc = psutil.Process()
        self.ram = []
        self.power = []
        self.temp = []
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.ram.append(self.proc.memory_info().rss / 1024 / 1024)
            except Exception:
                pass

            p = read_pmic_power_w()
            if p is not None:
                self.power.append(p)

            t = read_temp_c()
            if t is not None:
                self.temp.append(t)

            time.sleep(self.interval)


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


# =========================================================
# Preprocess
# =========================================================
def get_bilinear():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def resize_direct(img, input_h, input_w):
    return img.resize((input_w, input_h), get_bilinear())


def resize_keep_ratio_center_crop(img, input_h, input_w):
    w, h = img.size
    bilinear = get_bilinear()

    resize_size = max(input_h, input_w)

    if w < h:
        new_w = resize_size
        new_h = int(round(h * resize_size / w))
    else:
        new_h = resize_size
        new_w = int(round(w * resize_size / h))

    img = img.resize((new_w, new_h), bilinear)

    left = max(0, (new_w - input_w) // 2)
    top = max(0, (new_h - input_h) // 2)

    img = img.crop((left, top, left + input_w, top + input_h))
    return img


def preprocess_image(img_path, input_h, input_w, mode, input_format):
    """
    mode:
      rgb_resize
      rgb_center_crop
      bgr_resize
      bgr_center_crop
      float_norm_resize
      float_norm_center_crop
      float_raw01_resize
      float_raw255_resize

    input_format:
      uint8
      float32
    """
    img = Image.open(img_path).convert("RGB")

    use_center_crop = "center_crop" in mode

    if use_center_crop:
        img = resize_keep_ratio_center_crop(img, input_h, input_w)
    else:
        img = resize_direct(img, input_h, input_w)

    arr = np.array(img, dtype=np.uint8, copy=True)

    # BGR 測試
    if mode.startswith("bgr"):
        arr = arr[:, :, ::-1].copy()

    # UINT8 input 測試
    if input_format == "uint8":
        arr = arr.astype(np.uint8, copy=True)

    # FLOAT32 input 測試
    elif input_format == "float32":
        if mode.startswith("float_norm"):
            arr = arr.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std

        elif mode.startswith("float_raw01"):
            arr = arr.astype(np.float32) / 255.0

        elif mode.startswith("float_raw255"):
            arr = arr.astype(np.float32)

        else:
            # 若 input_format=float32 但 mode 是 rgb_resize，就當 raw01
            arr = arr.astype(np.float32) / 255.0

    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    arr = np.ascontiguousarray(arr)
    arr.setflags(write=True)
    return arr


# =========================================================
# Accuracy / Stats
# =========================================================
def topk_indices(x, k=5):
    x = x.reshape(-1)
    k = min(k, x.shape[0])
    return np.argsort(x)[-k:][::-1]


def get_hw_from_shape(shape):
    shape = list(shape)

    # parse-hef 顯示 NHWC(384x384x3)
    # hailo API 通常會是 [384,384,3] 或 [1,384,384,3]
    if len(shape) == 4:
        h, w = shape[1], shape[2]
    elif len(shape) == 3:
        h, w = shape[0], shape[1]
    else:
        raise ValueError(f"Unsupported input shape: {shape}")

    return int(h), int(w)


def set_format(infer_model, input_format, output_format):
    if input_format == "uint8":
        infer_model.input().set_format_type(FormatType.UINT8)
    elif input_format == "float32":
        infer_model.input().set_format_type(FormatType.FLOAT32)
    else:
        raise ValueError(input_format)

    if output_format == "uint8":
        infer_model.output().set_format_type(FormatType.UINT8)
    elif output_format == "float32":
        infer_model.output().set_format_type(FormatType.FLOAT32)
    else:
        raise ValueError(output_format)


def run_one_test(args, samples, class_dirs, test_name, input_format, output_format, preprocess_mode):
    print("\n" + "=" * 80)
    print(f"[TEST] {test_name}")
    print(f"input_format  : {input_format}")
    print(f"output_format : {output_format}")
    print(f"preprocess    : {preprocess_mode}")
    print("=" * 80)

    top1_correct = 0
    top5_correct = 0
    processed = 0
    inference_time = 0.0

    pred_counter = Counter()
    output_mins = []
    output_maxs = []
    output_means = []
    output_stds = []

    first_outputs = []

    try:
        with VDevice() as vdevice:
            infer_model = vdevice.create_infer_model(args.hef)
            infer_model.set_batch_size(1)

            set_format(infer_model, input_format, output_format)

            input_info = infer_model.input()
            output_info = infer_model.output()

            input_name = input_info.name
            output_name = output_info.name

            input_shape = tuple(input_info.shape)
            output_shape = tuple(output_info.shape)

            input_h, input_w = get_hw_from_shape(input_shape)

            if output_format == "float32":
                output_dtype = np.float32
            else:
                output_dtype = np.uint8

            print("========== HEF INFO ==========")
            print(f"Input name   : {input_name}")
            print(f"Output name  : {output_name}")
            print(f"Input shape  : {input_shape}")
            print(f"Output shape : {output_shape}")
            print(f"Input HxW    : {input_h}x{input_w}")
            print(f"Output dtype : {output_dtype}")
            print("==============================")

            output_buffer = np.empty(output_shape, dtype=output_dtype)
            output_buffer = np.ascontiguousarray(output_buffer)
            output_buffer.setflags(write=True)

            with infer_model.configure() as configured_model:

                # Warmup
                for i in range(args.warmup):
                    img_path, _, _ = samples[i % len(samples)]

                    input_buffer = preprocess_image(
                        img_path,
                        input_h,
                        input_w,
                        mode=preprocess_mode,
                        input_format=input_format
                    )

                    bindings = configured_model.create_bindings()

                    try:
                        bindings.input(input_name).set_buffer(input_buffer)
                        bindings.output(output_name).set_buffer(output_buffer)
                    except TypeError:
                        bindings.input().set_buffer(input_buffer)
                        bindings.output().set_buffer(output_buffer)

                    configured_model.run([bindings], args.timeout_ms)

                # Evaluation
                total_start = time.perf_counter()

                for idx, (img_path, gt_label, class_name) in enumerate(samples, start=1):
                    input_buffer = preprocess_image(
                        img_path,
                        input_h,
                        input_w,
                        mode=preprocess_mode,
                        input_format=input_format
                    )

                    bindings = configured_model.create_bindings()

                    try:
                        bindings.input(input_name).set_buffer(input_buffer)
                        bindings.output(output_name).set_buffer(output_buffer)
                    except TypeError:
                        bindings.input().set_buffer(input_buffer)
                        bindings.output().set_buffer(output_buffer)

                    t0 = time.perf_counter()
                    configured_model.run([bindings], args.timeout_ms)
                    t1 = time.perf_counter()

                    inference_time += t1 - t0

                    logits = np.array(output_buffer, copy=True).reshape(-1)

                    if logits.shape[0] == 1001 and len(class_dirs) == 1000:
                        logits = logits[1:]

                    if logits.shape[0] != 1000:
                        print(f"[WARN] Output classes = {logits.shape[0]}, expected 1000")

                    top5 = topk_indices(logits, k=5)
                    pred1 = int(top5[0])

                    pred_counter[pred1] += 1

                    output_mins.append(float(np.min(logits)))
                    output_maxs.append(float(np.max(logits)))
                    output_means.append(float(np.mean(logits)))
                    output_stds.append(float(np.std(logits)))

                    if len(first_outputs) < 5:
                        first_outputs.append(logits.astype(np.float32).copy())

                    if pred1 == gt_label:
                        top1_correct += 1

                    if gt_label in top5:
                        top5_correct += 1

                    processed += 1

                    if args.print_samples and idx <= args.sample_print_count:
                        print(
                            f"[sample {idx}] "
                            f"gt={gt_label}({class_name}) "
                            f"pred1={pred1} "
                            f"top5={top5.tolist()} "
                            f"max={float(np.max(logits)):.4f} "
                            f"min={float(np.min(logits)):.4f} "
                            f"std={float(np.std(logits)):.4f}"
                        )

                total_elapsed = time.perf_counter() - total_start

    except Exception as e:
        print(f"[FAILED] {test_name}")
        print(f"Reason: {repr(e)}")
        return {
            "test": test_name,
            "ok": False,
            "reason": repr(e),
            "top1": 0.0,
            "top5": 0.0,
        }

    top1_acc = top1_correct / processed * 100.0 if processed else 0.0
    top5_acc = top5_correct / processed * 100.0 if processed else 0.0
    infer_fps = processed / inference_time if inference_time > 0 else 0.0
    overall_fps = processed / total_elapsed if total_elapsed > 0 else 0.0

    most_common = pred_counter.most_common(10)
    unique_preds = len(pred_counter)

    # output difference check
    pairwise_l1 = []
    if len(first_outputs) >= 2:
        for i in range(len(first_outputs) - 1):
            d = np.mean(np.abs(first_outputs[i] - first_outputs[i + 1]))
            pairwise_l1.append(float(d))

    print("\n========== TEST RESULT ==========")
    print(f"Test                      : {test_name}")
    print(f"Processed                 : {processed}")
    print(f"Top-1 Correct             : {top1_correct}/{processed}")
    print(f"Top-5 Correct             : {top5_correct}/{processed}")
    print(f"Top-1 Accuracy            : {top1_acc:.2f}%")
    print(f"Top-5 Accuracy            : {top5_acc:.2f}%")
    print(f"Inference Time            : {inference_time:.4f} s")
    print(f"Inference FPS             : {infer_fps:.2f}")
    print(f"Overall FPS               : {overall_fps:.2f}")
    print(f"Output min avg            : {np.mean(output_mins):.4f}")
    print(f"Output max avg            : {np.mean(output_maxs):.4f}")
    print(f"Output mean avg           : {np.mean(output_means):.4f}")
    print(f"Output std avg            : {np.mean(output_stds):.4f}")
    print(f"Unique pred1 count        : {unique_preds}")
    print(f"Most common pred1         : {most_common}")
    if pairwise_l1:
        print(f"First outputs mean L1 diff: {np.mean(pairwise_l1):.6f}")
    print("=================================")

    return {
        "test": test_name,
        "ok": True,
        "input_format": input_format,
        "output_format": output_format,
        "preprocess": preprocess_mode,
        "top1": top1_acc,
        "top5": top5_acc,
        "unique_preds": unique_preds,
        "most_common": most_common,
        "output_std_avg": float(np.mean(output_stds)),
        "output_l1_diff": float(np.mean(pairwise_l1)) if pairwise_l1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hef", type=str, default="efficientnet_s.hef")
    parser.add_argument("--image-dir", "--dir", "--val-dir", dest="image_dir", type=str, default="../val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--sample-print-count", type=int, default=10)

    args = parser.parse_args()

    samples, class_dirs = build_imagenet_dataset(args.image_dir, limit=args.limit)

    print("========== DEBUG DATASET INFO ==========")
    print(f"HEF        : {args.hef}")
    print(f"Dataset    : {args.image_dir}")
    print(f"Classes    : {len(class_dirs)}")
    print(f"Images     : {len(samples)}")
    print(f"Limit      : {args.limit}")
    print("Label mode : sorted folder order")
    print("========================================")

    # 這裡是重點：自動測多種可能
    tests = [
        # 完全照 parse-hef：UINT8 input / UINT8 output
        ("uint8_rgb_resize__uint8_out",        "uint8",   "uint8",   "rgb_resize"),
        ("uint8_rgb_centercrop__uint8_out",    "uint8",   "uint8",   "rgb_center_crop"),

        # 測 channel 是否反了
        ("uint8_bgr_resize__uint8_out",        "uint8",   "uint8",   "bgr_resize"),
        ("uint8_bgr_centercrop__uint8_out",    "uint8",   "uint8",   "bgr_center_crop"),

        # 測 output 是否能用 float32 反量化
        ("uint8_rgb_resize__float32_out",      "uint8",   "float32", "rgb_resize"),
        ("uint8_rgb_centercrop__float32_out",  "uint8",   "float32", "rgb_center_crop"),

        # 測 input 是否能用 float32 normalized NHWC
        ("float32_norm_resize__uint8_out",     "float32", "uint8",   "float_norm_resize"),
        ("float32_norm_crop__uint8_out",       "float32", "uint8",   "float_norm_center_crop"),

        # 測 input float32 + output float32
        ("float32_norm_resize__float32_out",   "float32", "float32", "float_norm_resize"),
        ("float32_norm_crop__float32_out",     "float32", "float32", "float_norm_center_crop"),

        # 測 input float32 但不 normalize，只 /255
        ("float32_raw01_resize__uint8_out",    "float32", "uint8",   "float_raw01_resize"),
        ("float32_raw01_resize__float32_out",  "float32", "float32", "float_raw01_resize"),

        # 測 input float32 0~255
        ("float32_raw255_resize__uint8_out",   "float32", "uint8",   "float_raw255_resize"),
    ]

    results = []

    total_monitor = Monitor(interval=args.monitor_interval)
    total_monitor.start()

    try:
        for test_name, input_format, output_format, preprocess_mode in tests:
            result = run_one_test(
                args,
                samples,
                class_dirs,
                test_name,
                input_format,
                output_format,
                preprocess_mode
            )
            results.append(result)

    finally:
        total_monitor.stop()

    print("\n\n" + "#" * 90)
    print("FINAL SUMMARY")
    print("#" * 90)

    ok_results = [r for r in results if r.get("ok")]

    ok_results_sorted = sorted(
        ok_results,
        key=lambda x: (x.get("top1", 0), x.get("top5", 0)),
        reverse=True
    )

    print("\n========== RANKING BY TOP-1 ==========")
    for r in ok_results_sorted:
        print(
            f"{r['test']:<40} "
            f"Top1={r['top1']:.2f}% "
            f"Top5={r['top5']:.2f}% "
            f"UniquePred={r['unique_preds']} "
            f"OutStd={r['output_std_avg']:.4f} "
            f"L1Diff={r['output_l1_diff']:.6f}"
        )

    failed = [r for r in results if not r.get("ok")]
    if failed:
        print("\n========== FAILED TESTS ==========")
        for r in failed:
            print(f"{r['test']:<40} {r['reason']}")

    print("\n========== SYSTEM MONITOR ==========")
    if total_monitor.ram:
        print(f"Average RAM   : {avg(total_monitor.ram):.2f} MB")
    else:
        print("Average RAM   : N/A")

    if total_monitor.power:
        print(f"Average Power : {avg(total_monitor.power):.4f} W")
    else:
        print("Average Power : N/A")

    if total_monitor.temp:
        print(f"Temp Max      : {max(total_monitor.temp):.2f} °C")
        print(f"Temp Min      : {min(total_monitor.temp):.2f} °C")
        print(f"Temp Avg      : {avg(total_monitor.temp):.2f} °C")
    else:
        print("Temperature   : N/A")

    print("\n========== HOW TO READ ==========")
    print("1. 如果所有 Top1/Top5 都接近 0，而且 UniquePred 很少，代表 input quantization / preprocess 很可能錯。")
    print("2. 如果 float32 input 測試有成功且準確率變高，代表不能直接餵 raw uint8。")
    print("3. 如果 BGR 比 RGB 高很多，代表 channel 順序錯。")
    print("4. 如果 float32 output 比 uint8 output 高很多，代表 raw uint8 output 不適合直接 argmax。")
    print("5. 如果所有組合都失敗或都 0，通常要回頭查 HAR 的 input quantization / normalize 設定。")
    print("#" * 90)


if __name__ == "__main__":
    main()