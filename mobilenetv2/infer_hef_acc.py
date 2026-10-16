import time
import re
import json
import argparse
import threading
import subprocess
from pathlib import Path

import psutil
import numpy as np
from PIL import Image

from hailo_platform import (
    VDevice,
    FormatType,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# Raspberry Pi 5 PMIC Power
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
# Monitor RAM / Power / Temperature
# =========================================================
class Monitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.ram_mb = []
        self.power_w = []
        self.temp_c = []
        self.thread = None

    def _worker(self):
        proc = psutil.Process()

        while self.running:
            try:
                self.ram_mb.append(proc.memory_info().rss / 1024 / 1024)
            except Exception:
                pass

            p = read_pmic_power_w()
            if p is not None:
                self.power_w.append(p)

            t = read_temp_c()
            if t is not None:
                self.temp_c.append(t)

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()


# =========================================================
# ImageNet-mini label loading
# =========================================================
def load_imagenet_class_index_json(json_path):
    """
    Optional:
    imagenet_class_index.json format:
    {
        "0": ["n01440764", "tench"],
        "1": ["n01443537", "goldfish"],
        ...
    }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"class index json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wnid_to_idx = {}
    name_to_idx = {}

    for k, v in data.items():
        idx = int(k)

        if isinstance(v, (list, tuple)) and len(v) >= 2:
            wnid = str(v[0])
            name = str(v[1])
            wnid_to_idx[wnid] = idx
            name_to_idx[name] = idx
        elif isinstance(v, dict):
            if "wnid" in v:
                wnid_to_idx[str(v["wnid"])] = idx
            if "name" in v:
                name_to_idx[str(v["name"])] = idx

    return wnid_to_idx, name_to_idx


def build_imagenet_samples(image_dir, class_index_json=None):
    """
    支援幾種常見 ImageNet-mini val 結構：

    1. ../imagenet-mini/val/n01440764/xxx.JPEG
       用資料夾排序對應 ImageNet 1000 類 index

    2. ../imagenet-mini/val/0/xxx.JPEG
       直接把資料夾名稱當 label index

    3. 搭配 --class-index-json imagenet_class_index.json
       用 json 的 wnid 對應 label index
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise RuntimeError(f"IMAGE_DIR not found: {image_dir}")

    class_dirs = sorted([p for p in image_dir.iterdir() if p.is_dir()])

    if len(class_dirs) == 0:
        raise RuntimeError(
            f"No class folders found in {image_dir}. "
            "Expected format: val/class_name/image.jpg"
        )

    class_to_idx = {}
    mapping_mode = ""

    # Case 1: user provided imagenet_class_index.json
    if class_index_json is not None:
        wnid_to_idx, name_to_idx = load_imagenet_class_index_json(class_index_json)

        ok = True
        for d in class_dirs:
            key = d.name.split()[0]
            if key in wnid_to_idx:
                class_to_idx[d.name] = wnid_to_idx[key]
            elif d.name in name_to_idx:
                class_to_idx[d.name] = name_to_idx[d.name]
            else:
                ok = False
                break

        if ok:
            mapping_mode = "imagenet_class_index_json"

    # Case 2: folder name is numeric label
    if not class_to_idx:
        numeric_ok = all(d.name.isdigit() for d in class_dirs)
        if numeric_ok:
            class_to_idx = {d.name: int(d.name) for d in class_dirs}
            mapping_mode = "numeric_folder_name"

    # Case 3: default ImageNet WNID sorted order
    if not class_to_idx:
        class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}
        mapping_mode = "sorted_folder_order"

    samples = []

    for d in class_dirs:
        label = class_to_idx[d.name]

        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), label))

    if len(samples) == 0:
        raise RuntimeError(f"No images found under {image_dir}")

    return samples, class_dirs, mapping_mode


# =========================================================
# Preprocess
# =========================================================
def get_bilinear():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def preprocess_mobilenetv2_uint8_nhwc(img_path, input_size=224, mode="center_crop"):
    img = Image.open(img_path).convert("RGB")
    bilinear = get_bilinear()

    if mode == "center_crop":
        # ImageNet 常見驗證前處理：short side resize 256 -> center crop 224
        resize_size = 256

        w, h = img.size
        if w < h:
            new_w = resize_size
            new_h = int(h * resize_size / w)
        else:
            new_h = resize_size
            new_w = int(w * resize_size / h)

        img = img.resize((new_w, new_h), bilinear)

        left = (new_w - input_size) // 2
        top = (new_h - input_size) // 2
        img = img.crop((left, top, left + input_size, top + input_size))

    elif mode == "resize":
        # 完全照你原本速度測試那種：直接 resize 224x224
        img = img.resize((input_size, input_size), bilinear)

    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    img = np.array(img, dtype=np.uint8, copy=True)

    # Hailo HEF input: UINT8 NHWC
    img = np.ascontiguousarray(img)
    img.setflags(write=True)

    return img


# =========================================================
# Accuracy utils
# =========================================================
def topk_indices(scores, k=5):
    scores = scores.reshape(-1)
    k = min(k, scores.shape[0])
    return np.argsort(scores)[-k:][::-1]


def configure_infer_model(vdevice, hef_path):
    """
    優先用 FLOAT32 output，因為算 accuracy 比較適合。
    如果你的 Hailo runtime / HEF 不支援，就自動退回 UINT8 output。
    """
    try:
        infer_model = vdevice.create_infer_model(hef_path)
        infer_model.set_batch_size(1)

        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        configured_model = infer_model.configure()
        output_dtype = np.float32
        output_format_name = "FLOAT32"

        return infer_model, configured_model, output_dtype, output_format_name

    except Exception as e:
        print("[WARN] FLOAT32 output configure failed.")
        print(f"[WARN] Reason: {e}")
        print("[WARN] Retry with UINT8 output...")

        infer_model = vdevice.create_infer_model(hef_path)
        infer_model.set_batch_size(1)

        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.UINT8)

        configured_model = infer_model.configure()
        output_dtype = np.uint8
        output_format_name = "UINT8"

        return infer_model, configured_model, output_dtype, output_format_name


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=str, default="mobilenet_v2_1.0.hef")
    parser.add_argument("--image-dir", type=str, default="../imagenet-mini/val")
    parser.add_argument("--class-index-json", type=str, default=None)
    parser.add_argument(
        "--preprocess",
        type=str,
        default="center_crop",
        choices=["center_crop", "resize"],
        help="center_crop: ImageNet eval style. resize: same as old speed script."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args()

    samples, class_dirs, mapping_mode = build_imagenet_samples(
        args.image_dir,
        class_index_json=args.class_index_json
    )

    if args.limit is not None:
        samples = samples[:args.limit]

    num_images = len(samples)
    max_label = max(label for _, label in samples)

    print("========== INFO ==========")
    print(f"HEF: {args.hef}")
    print(f"Image dir: {args.image_dir}")
    print(f"Images: {num_images}")
    print(f"Classes found: {len(class_dirs)}")
    print(f"Label mapping: {mapping_mode}")
    print(f"Preprocess: {args.preprocess}")
    print("Input format: UINT8 NHWC 224x224x3")
    print("API: InferModel")

    if mapping_mode == "sorted_folder_order" and len(class_dirs) != 1000:
        print(
            "[WARN] You are using sorted folder order, but class folder count is not 1000."
        )
        print(
            "[WARN] If accuracy looks wrong, provide --class-index-json imagenet_class_index.json"
        )

    monitor = Monitor(interval=0.5)

    total_start = time.perf_counter()
    inference_time = 0.0

    correct1 = 0
    correct5 = 0

    monitor.start()

    try:
        with VDevice() as vdevice:
            infer_model, configured_model, output_dtype, output_format_name = configure_infer_model(
                vdevice,
                args.hef
            )

            input_name = infer_model.input().name
            output_name = infer_model.output().name
            input_shape = infer_model.input().shape
            output_shape = infer_model.output().shape

            print(f"Input name: {input_name}")
            print(f"Output name: {output_name}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Output format: {output_format_name}")

            output_buffer = np.zeros(output_shape, dtype=output_dtype)
            output_buffer = np.ascontiguousarray(output_buffer)
            output_buffer.setflags(write=True)

            for idx, (img_path, label) in enumerate(samples, start=1):
                input_buffer = preprocess_mobilenetv2_uint8_nhwc(
                    img_path,
                    input_size=224,
                    mode=args.preprocess
                )

                bindings = configured_model.create_bindings()
                bindings.input().set_buffer(input_buffer)
                bindings.output().set_buffer(output_buffer)

                infer_start = time.perf_counter()
                configured_model.run([bindings], 10000)
                inference_time += time.perf_counter() - infer_start

                scores = np.array(output_buffer).reshape(-1)

                # 有些 ImageNet 模型會輸出 1001 類，index 0 是 background。
                # 如果遇到 1001 類，這裡自動去掉 background。
                if scores.shape[0] == 1001 and max_label <= 999:
                    scores = scores[1:]

                if label >= scores.shape[0]:
                    raise RuntimeError(
                        f"Label index {label} >= output classes {scores.shape[0]}. "
                        "Your label mapping may be wrong. Try --class-index-json."
                    )

                top5 = topk_indices(scores, k=5)
                pred1 = int(top5[0])

                if pred1 == label:
                    correct1 += 1

                if label in top5:
                    correct5 += 1

                if args.progress_interval > 0 and idx % args.progress_interval == 0:
                    top1_now = correct1 / idx * 100.0
                    top5_now = correct5 / idx * 100.0
                    print(
                        f"Processed {idx}/{num_images} | "
                        f"Top-1: {top1_now:.2f}% | Top-5: {top5_now:.2f}%"
                    )

    finally:
        monitor.stop()

    total_time = time.perf_counter() - total_start

    top1_acc = correct1 / num_images * 100.0
    top5_acc = correct5 / num_images * 100.0

    print("\n========== RESULT ==========")
    print(f"Total images: {num_images}")
    print(f"Top-1 Correct: {correct1}/{num_images}")
    print(f"Top-5 Correct: {correct5}/{num_images}")
    print(f"Top-1 Accuracy (%): {top1_acc:.2f}")
    print(f"Top-5 Accuracy (%): {top5_acc:.2f}")

    print(f"Total Time (s): {total_time:.4f}")
    print(f"Inference Time Only (s): {inference_time:.4f}")

    if total_time > 0:
        print(f"Overall FPS: {num_images / total_time:.2f}")
    else:
        print("Overall FPS: N/A")

    if inference_time > 0:
        print(f"Inference FPS Only: {num_images / inference_time:.2f}")
    else:
        print("Inference FPS Only: N/A")

    if monitor.ram_mb:
        print(f"Average RAM Usage (MB): {np.mean(monitor.ram_mb):.2f}")
    else:
        print("Average RAM Usage (MB): N/A")

    if monitor.power_w:
        print(f"Average Power (W): {np.mean(monitor.power_w):.3f}")
    else:
        print("Average Power (W): N/A")

    if monitor.temp_c:
        print(f"Temperature Max (C): {np.max(monitor.temp_c):.2f}")
        print(f"Temperature Min (C): {np.min(monitor.temp_c):.2f}")
        print(f"Temperature Avg (C): {np.mean(monitor.temp_c):.2f}")
    else:
        print("Temperature: N/A")


if __name__ == "__main__":
    main()