# infer_regnetx_hef_acc.py

import argparse
import time
import threading
import subprocess
import re
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

from hailo_platform import VDevice


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# Dataset
# =========================================================
def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path))
    ]


def build_imagenet_dataset(val_dir, limit=0):
    """
    資料夾格式：

    ../val/
      n01440764/
        xxx.JPEG
      n01443537/
        xxx.JPEG
      ...

    label 會依照資料夾名稱排序後決定。
    """
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
            samples.append((str(img_path), cls_idx))

    if len(samples) == 0:
        raise RuntimeError(f"No images found under: {val_dir}")

    if limit > 0:
        samples = samples[:limit]

    return samples, class_dirs


# =========================================================
# Raspberry Pi 5 Power / Temperature
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


# =========================================================
# Monitor
# =========================================================
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
# Hailo dtype / shape helpers
# =========================================================
def dtype_from_format(fmt):
    s = str(fmt).upper()

    if "UINT8" in s:
        return np.uint8
    if "UINT16" in s:
        return np.uint16
    if "INT8" in s:
        return np.int8
    if "INT16" in s:
        return np.int16
    if "FLOAT32" in s or "FLOAT" in s:
        return np.float32

    return np.uint8


def get_format_type(info):
    try:
        return info.format.type
    except Exception:
        try:
            return info.format_type
        except Exception:
            return "UINT8"


def get_hw_from_shape(shape):
    shape = list(shape)

    # 常見 Hailo HEF input:
    # [H, W, C]
    # [1, H, W, C]
    if len(shape) == 4:
        h, w = shape[1], shape[2]
    elif len(shape) == 3:
        h, w = shape[0], shape[1]
    else:
        raise ValueError(f"Unsupported input shape: {shape}")

    return int(h), int(w)


# =========================================================
# Preprocess
# =========================================================
def get_bilinear():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def preprocess_regnetx_hef(img_path, input_h, input_w, dtype):
    img = Image.open(img_path).convert("RGB")
    bilinear = get_bilinear()

    # RegNetX-800MF ImageNet 分類通常是 224x224。
    # 這裡照你 HEF benchmark 腳本邏輯：
    # Resize 256 -> CenterCrop HEF input size。
    # 如果 HEF input 是 UINT8，就不 normalize。
    resize_size = 256

    img = img.resize((resize_size, resize_size), bilinear)

    left = (resize_size - input_w) // 2
    top = (resize_size - input_h) // 2

    img = img.crop((left, top, left + input_w, top + input_h))

    arr = np.array(img, dtype=np.uint8, copy=True)

    if dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        arr = (arr - mean) / std
    else:
        arr = arr.astype(dtype, copy=True)

    arr = np.ascontiguousarray(arr)
    arr.setflags(write=True)

    return arr


# =========================================================
# Accuracy
# =========================================================
def topk_indices(logits, k=5):
    logits = logits.reshape(-1)
    k = min(k, logits.shape[0])
    return np.argsort(logits)[-k:][::-1]


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hef",
        type=str,
        default="regnetx_800mf.hef",
        help="Path to RegNetX-800MF HEF"
    )

    parser.add_argument(
        "--image-dir",
        "--dir",
        "--val-dir",
        dest="image_dir",
        type=str,
        default="../val",
        help="ImageNet val folder. Default: ../val"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit images. 0 means all images."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup runs"
    )

    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=50
    )

    args = parser.parse_args()

    # =====================================================
    # Dataset
    # =====================================================
    samples, class_dirs = build_imagenet_dataset(
        args.image_dir,
        limit=args.limit
    )

    total_images = len(samples)

    print("========== INFO ==========")
    print(f"HEF: {args.hef}")
    print(f"Dataset: {args.image_dir}")
    print(f"Classes: {len(class_dirs)}")
    print(f"Images: {total_images}")
    print("Label mapping: sorted folder order")
    print("API: InferModel")
    print("==========================")

    top1_correct = 0
    top5_correct = 0
    processed_images = 0

    total_infer_time = 0.0

    # =====================================================
    # Hailo Inference
    # =====================================================
    with VDevice() as vdevice:
        infer_model = vdevice.create_infer_model(args.hef)
        infer_model.set_batch_size(1)

        input_info = infer_model.input()
        output_info = infer_model.output()

        input_name = input_info.name
        output_name = output_info.name

        input_shape = tuple(input_info.shape)
        output_shape = tuple(output_info.shape)

        input_dtype = dtype_from_format(get_format_type(input_info))
        output_dtype = dtype_from_format(get_format_type(output_info))

        input_h, input_w = get_hw_from_shape(input_shape)

        print("========== HEF INFO ==========")
        print(f"Input name : {input_name}")
        print(f"Output name: {output_name}")
        print(f"Input shape : {input_shape}")
        print(f"Output shape: {output_shape}")
        print(f"Input dtype : {input_dtype}")
        print(f"Output dtype: {output_dtype}")
        print(f"Input HxW   : {input_h}x{input_w}")
        print("==============================")

        output_buffer = np.empty(output_shape, dtype=output_dtype)
        output_buffer = np.ascontiguousarray(output_buffer)
        output_buffer.setflags(write=True)

        with infer_model.configure() as configured_model:

            # =================================================
            # Warmup
            # =================================================
            if args.warmup > 0:
                print("\n[Warmup]")

                for i in range(args.warmup):
                    img_path, _ = samples[i % total_images]

                    input_buffer = preprocess_regnetx_hef(
                        img_path,
                        input_h,
                        input_w,
                        input_dtype
                    )

                    bindings = configured_model.create_bindings()

                    try:
                        bindings.input(input_name).set_buffer(input_buffer)
                        bindings.output(output_name).set_buffer(output_buffer)
                    except TypeError:
                        bindings.input().set_buffer(input_buffer)
                        bindings.output().set_buffer(output_buffer)

                    configured_model.run([bindings], args.timeout_ms)

                    if (i + 1) % 10 == 0 or (i + 1) == args.warmup:
                        print(f"Warmup {i + 1}/{args.warmup}")

            # =================================================
            # Evaluation
            # =================================================
            print("\n[Start Evaluation]")

            monitor = Monitor(interval=args.monitor_interval)
            monitor.start()

            total_start = time.perf_counter()

            for idx, (img_path, gt_label) in enumerate(samples, start=1):
                input_buffer = preprocess_regnetx_hef(
                    img_path,
                    input_h,
                    input_w,
                    input_dtype
                )

                bindings = configured_model.create_bindings()

                try:
                    bindings.input(input_name).set_buffer(input_buffer)
                    bindings.output(output_name).set_buffer(output_buffer)
                except TypeError:
                    bindings.input().set_buffer(input_buffer)
                    bindings.output().set_buffer(output_buffer)

                infer_start = time.perf_counter()
                configured_model.run([bindings], args.timeout_ms)
                infer_end = time.perf_counter()

                total_infer_time += infer_end - infer_start

                logits = np.array(output_buffer).reshape(-1)

                # 有些 ImageNet 模型可能輸出 1001 類，index 0 是 background。
                # 如果遇到 1001，這裡自動丟掉 background。
                if logits.shape[0] == 1001 and len(class_dirs) == 1000:
                    logits = logits[1:]

                if gt_label >= logits.shape[0]:
                    raise RuntimeError(
                        f"Ground truth label {gt_label} >= output classes {logits.shape[0]}. "
                        "請確認 ../val 資料夾排序是否和模型 label 對應。"
                    )

                top5 = topk_indices(logits, k=5)
                pred1 = int(top5[0])

                if pred1 == gt_label:
                    top1_correct += 1

                if gt_label in top5:
                    top5_correct += 1

                processed_images += 1

                if (
                    args.print_every > 0
                    and (processed_images % args.print_every == 0 or processed_images == total_images)
                ):
                    cur_top1 = top1_correct / processed_images * 100.0
                    cur_top5 = top5_correct / processed_images * 100.0

                    print(
                        f"[{processed_images:04d}/{total_images:04d}] "
                        f"Top1={cur_top1:.2f}% "
                        f"Top5={cur_top5:.2f}%"
                    )

            total_end = time.perf_counter()
            monitor.stop()

    # =====================================================
    # Result
    # =====================================================
    total_elapsed = total_end - total_start

    top1_acc = top1_correct / processed_images * 100.0 if processed_images > 0 else 0.0
    top5_acc = top5_correct / processed_images * 100.0 if processed_images > 0 else 0.0

    avg_infer_ms = total_infer_time / processed_images * 1000.0 if processed_images > 0 else 0.0
    infer_fps = processed_images / total_infer_time if total_infer_time > 0 else 0.0
    overall_fps = processed_images / total_elapsed if total_elapsed > 0 else 0.0

    print("\n========== RegNetX-800MF HEF Accuracy Result ==========")
    print(f"HEF                       : {args.hef}")
    print(f"Dataset                   : {args.image_dir}")
    print(f"Total images              : {processed_images}")
    print(f"Classes                   : {len(class_dirs)}")
    print(f"Top-1 Correct             : {top1_correct}/{processed_images}")
    print(f"Top-5 Correct             : {top5_correct}/{processed_images}")
    print(f"Top-1 Accuracy            : {top1_acc:.2f}%")
    print(f"Top-5 Accuracy            : {top5_acc:.2f}%")
    print(f"Total pure inference time : {total_infer_time:.4f} s")
    print(f"Average inference time    : {avg_infer_ms:.4f} ms/image")
    print(f"Inference FPS Only        : {infer_fps:.2f}")
    print(f"Total elapsed time        : {total_elapsed:.4f} s")
    print(f"Overall FPS               : {overall_fps:.2f}")

    if monitor.ram:
        print(f"Average RAM               : {avg(monitor.ram):.2f} MB")
    else:
        print("Average RAM               : N/A")

    if monitor.power:
        print(f"Average Power             : {avg(monitor.power):.4f} W")
    else:
        print("Average Power             : N/A")

    if monitor.temp:
        print(f"Temperature Max           : {max(monitor.temp):.2f} °C")
        print(f"Temperature Min           : {min(monitor.temp):.2f} °C")
        print(f"Temperature Avg           : {avg(monitor.temp):.2f} °C")
    else:
        print("Temperature Max           : N/A")
        print("Temperature Min           : N/A")
        print("Temperature Avg           : N/A")

    print("=======================================================")


if __name__ == "__main__":
    main()