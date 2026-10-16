import os
import re
import time
import argparse
import threading
import subprocess
from pathlib import Path

import psutil
import numpy as np
from PIL import Image
import onnxruntime as ort


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_images(image_dir):
    image_dir = Path(image_dir)
    imgs = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(str(p))
    return sorted(imgs)


def preprocess_effnetv2_s(img_path, input_size=384):
    img = Image.open(img_path).convert("RGB")

    # Torchvision EfficientNet V2 S 官方常用推論 preprocessing:
    # Resize -> CenterCrop -> Normalize
    resize_size = 384
    img = img.resize((resize_size, resize_size), Image.BILINEAR)

    left = (resize_size - input_size) // 2
    top = (resize_size - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW

    return arr


def make_batch(img_paths, input_size):
    batch = []
    for p in img_paths:
        try:
            batch.append(preprocess_effnetv2_s(p, input_size))
        except Exception as e:
            print(f"[WARN] skip image: {p}, error: {e}")

    if len(batch) == 0:
        return None

    return np.stack(batch, axis=0).astype(np.float32)


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_pmic_power_w():
    """
    使用 Raspberry Pi 5:
        vcgencmd pmic_read_adc

    自動找出：
        XXX_A × XXX_V

    並加總成 total board power。
    """

    try:
        result = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True,
            text=True,
            timeout=1.0
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    currents = {}
    voltages = {}

    for line in result.stdout.splitlines():
        line = line.strip()

        # 例：
        # 3V7_WL_SW_A current(0)=0.123456A
        # 3V7_WL_SW_V volt(0)=3.700000V

        m_current = re.search(r"^(.+)_A\s+.*?=\s*([0-9.]+)A", line)
        m_voltage = re.search(r"^(.+)_V\s+.*?=\s*([0-9.]+)V", line)

        if m_current:
            name = m_current.group(1)
            value = float(m_current.group(2))
            currents[name] = value

        if m_voltage:
            name = m_voltage.group(1)
            value = float(m_voltage.group(2))
            voltages[name] = value

    total_power = 0.0
    matched = 0

    for name, current_a in currents.items():
        if name in voltages:
            voltage_v = voltages[name]
            total_power += current_a * voltage_v
            matched += 1

    if matched == 0:
        return None

    return total_power


class SystemMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.running = False

        self.ram_samples = []
        self.power_samples = []
        self.temp_samples = []

        self.process = psutil.Process(os.getpid())
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _run(self):
        while self.running:
            try:
                ram_mb = self.process.memory_info().rss / 1024 / 1024
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

    def get_result(self):
        def avg(x):
            return float(np.mean(x)) if len(x) > 0 else 0.0

        def minv(x):
            return float(np.min(x)) if len(x) > 0 else 0.0

        def maxv(x):
            return float(np.max(x)) if len(x) > 0 else 0.0

        return {
            "avg_ram_mb": avg(self.ram_samples),

            "avg_power_w": avg(self.power_samples),
            "min_power_w": minv(self.power_samples),
            "max_power_w": maxv(self.power_samples),

            "avg_temp_c": avg(self.temp_samples),
            "min_temp_c": minv(self.temp_samples),
            "max_temp_c": maxv(self.temp_samples),

            "power_samples": len(self.power_samples),
            "temp_samples": len(self.temp_samples),
            "ram_samples": len(self.ram_samples),
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="efficientnet_v2_s_opset11_dynamic.onnx")
    parser.add_argument("--image_dir", type=str, default="../Test")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--monitor_interval", type=float, default=0.2)
    parser.add_argument("--num_threads", type=int, default=4)

    args = parser.parse_args()

    image_paths = get_images(args.image_dir)

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    if len(image_paths) == 0:
        raise RuntimeError("No images found.")

    print("========== Config ==========")
    print(f"Model path        : {args.model}")
    print(f"Image dir         : {args.image_dir}")
    print(f"Total images      : {len(image_paths)}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Input size        : {args.input_size}")
    print(f"Warmup            : {args.warmup}")
    print(f"Monitor interval  : {args.monitor_interval}")
    print("============================")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        args.model,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # warmup
    dummy = np.random.randn(
        args.batch_size, 3, args.input_size, args.input_size
    ).astype(np.float32)

    for _ in range(args.warmup):
        session.run(None, {input_name: dummy})

    monitor = SystemMonitor(interval=args.monitor_interval)
    monitor.start()

    total_start = time.perf_counter()

    pure_infer_time = 0.0
    processed_images = 0
    processed_batches = 0

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch = make_batch(batch_paths, args.input_size)

        if batch is None:
            continue

        real_batch_size = batch.shape[0]

        infer_start = time.perf_counter()
        outputs = session.run(None, {input_name: batch})
        infer_end = time.perf_counter()

        pure_infer_time += infer_end - infer_start
        processed_images += real_batch_size
        processed_batches += 1

        if processed_batches % 10 == 0:
            print(
                f"[Progress] batches={processed_batches}, "
                f"images={processed_images}/{len(image_paths)}"
            )

    total_end = time.perf_counter()
    monitor.stop()

    total_time = total_end - total_start

    overall_fps = processed_images / total_time if total_time > 0 else 0.0
    pure_infer_fps = processed_images / pure_infer_time if pure_infer_time > 0 else 0.0

    stat = monitor.get_result()

    print("\n========== Result ==========")
    print(f"Processed images          : {processed_images}")
    print(f"Processed batches         : {processed_batches}")
    print(f"Batch size                : {args.batch_size}")

    print(f"Total test time (s)       : {total_time:.6f}")
    print(f"Pure inference time (s)   : {pure_infer_time:.6f}")

    print(f"Overall FPS               : {overall_fps:.4f}")
    print(f"Pure inference FPS        : {pure_infer_fps:.4f}")

    print(f"Average RAM (MB)          : {stat['avg_ram_mb']:.2f}")

    print(f"Average Power (W)         : {stat['avg_power_w']:.4f}")
    print(f"Min Power (W)             : {stat['min_power_w']:.4f}")
    print(f"Max Power (W)             : {stat['max_power_w']:.4f}")
    print(f"Power samples             : {stat['power_samples']}")

    print(f"Temperature Min (C)       : {stat['min_temp_c']:.2f}")
    print(f"Temperature Max (C)       : {stat['max_temp_c']:.2f}")
    print(f"Temperature Avg (C)       : {stat['avg_temp_c']:.2f}")
    print(f"Temperature samples       : {stat['temp_samples']}")

    print("============================")


if __name__ == "__main__":
    main()