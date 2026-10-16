# infer_yolov8n_onnx_rpi5_batch.py

import os
import re
import time
import argparse
import threading
import subprocess
from pathlib import Path

import cv2
import numpy as np
import psutil
import onnxruntime as ort


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# Power / Temp / RAM Monitor
# =========================

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
        m_i = re.match(
            r"(.+?)_A\s+current\(\d+\)=([0-9.]+)A",
            line.strip()
        )
        if m_i:
            currents[m_i.group(1).strip()] = float(m_i.group(2))
            continue

        m_v = re.match(
            r"(.+?)_V\s+volt\(\d+\)=([0-9.]+)V",
            line.strip()
        )
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


class SystemMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.running = False
        self.thread = None

        self.ram_samples = []
        self.power_samples = []
        self.temp_samples = []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _loop(self):
        proc = psutil.Process(os.getpid())

        while self.running:
            try:
                ram_mb = proc.memory_info().rss / 1024 / 1024
                self.ram_samples.append(ram_mb)
            except Exception:
                pass

            power_w = read_pmic_power_w()
            if power_w is not None:
                self.power_samples.append(power_w)

            temp_c = read_temp_c()
            if temp_c is not None:
                self.temp_samples.append(temp_c)

            time.sleep(self.interval)

    def summary(self):
        def avg(x):
            return sum(x) / len(x) if x else 0.0

        return {
            "avg_ram_mb": avg(self.ram_samples),
            "avg_power_w": avg(self.power_samples),
            "min_temp_c": min(self.temp_samples) if self.temp_samples else 0.0,
            "max_temp_c": max(self.temp_samples) if self.temp_samples else 0.0,
            "avg_temp_c": avg(self.temp_samples),
        }


# =========================
# Image / YOLOv8 Preprocess
# =========================

def get_images(image_dir):
    image_dir = Path(image_dir)
    imgs = []

    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(str(p))

    return sorted(imgs)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)

    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded = cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img_padded


def preprocess_yolov8(img_path, img_size=640):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = letterbox(img, new_shape=(img_size, img_size))

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HWC -> CHW
    img = img.transpose(2, 0, 1)

    # uint8 -> float32, 0~1
    img = img.astype(np.float32) / 255.0

    return img


def make_batch(img_paths, img_size):
    batch = [preprocess_yolov8(p, img_size) for p in img_paths]
    return np.stack(batch, axis=0)


# =========================
# Main Benchmark
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="yolov8n_dynamic.onnx",
                        help="Path to YOLOv8n ONNX model")
    parser.add_argument("--image_dir", type=str, default="../Test",
                        help="Path to image folder")
    parser.add_argument("--img_size", type=int, default=640,
                        help="YOLOv8 input size, default 640")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Dynamic batch size")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of images, 0 means all")
    parser.add_argument("--monitor_interval", type=float, default=0.2,
                        help="Monitor sampling interval in seconds")

    args = parser.parse_args()

    images = get_images(args.image_dir)

    if args.limit > 0:
        images = images[:args.limit]

    if len(images) == 0:
        raise RuntimeError("No images found.")

    print("========== Config ==========")
    print(f"Model       : {args.model}")
    print(f"Image dir   : {args.image_dir}")
    print(f"Images      : {len(images)}")
    print(f"Input size  : {args.img_size}x{args.img_size}")
    print(f"Batch size  : {args.batch_size}")
    print("============================")

    providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        args.model,
        providers=providers
    )

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"Input name  : {input_name}")
    print(f"Output names: {output_names}")

    # Warmup
    print("\nWarming up...")
    warmup_imgs = images[:args.batch_size]
    warmup_batch = make_batch(warmup_imgs, args.img_size)

    for _ in range(args.warmup):
        session.run(output_names, {input_name: warmup_batch})

    monitor = SystemMonitor(interval=args.monitor_interval)

    total_images = 0
    total_infer_time = 0.0

    print("\nStart benchmark...")

    monitor.start()
    total_start = time.perf_counter()

    for i in range(0, len(images), args.batch_size):
        batch_paths = images[i:i + args.batch_size]
        batch = make_batch(batch_paths, args.img_size)

        infer_start = time.perf_counter()
        outputs = session.run(output_names, {input_name: batch})
        infer_end = time.perf_counter()

        infer_time = infer_end - infer_start

        total_infer_time += infer_time
        total_images += len(batch_paths)

        print(
            f"[{total_images:5d}/{len(images):5d}] "
            f"batch={len(batch_paths)} "
            f"infer_time={infer_time:.4f}s "
            f"infer_fps={len(batch_paths) / infer_time:.2f}"
        )

    total_end = time.perf_counter()
    monitor.stop()

    total_time = total_end - total_start

    overall_fps = total_images / total_time if total_time > 0 else 0.0
    infer_fps = total_images / total_infer_time if total_infer_time > 0 else 0.0

    sys_info = monitor.summary()

    print("\n========== Result ==========")
    print(f"Total images             : {total_images}")
    print(f"Total test time          : {total_time:.4f} s")
    print(f"Pure inference time      : {total_infer_time:.4f} s")
    print(f"Overall FPS              : {overall_fps:.4f}")
    print(f"Pure inference FPS       : {infer_fps:.4f}")
    print(f"Average RAM usage        : {sys_info['avg_ram_mb']:.2f} MB")
    print(f"Average Power            : {sys_info['avg_power_w']:.4f} W")
    print(f"Temperature min          : {sys_info['min_temp_c']:.2f} °C")
    print(f"Temperature max          : {sys_info['max_temp_c']:.2f} °C")
    print(f"Temperature avg          : {sys_info['avg_temp_c']:.2f} °C")
    print("============================")


if __name__ == "__main__":
    main()