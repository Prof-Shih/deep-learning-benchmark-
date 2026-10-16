# infer_deeplabv3_mobilenetv2_onnx_rpi5_batch.py

import os
import re
import time
import argparse
import threading
import subprocess
from pathlib import Path

import numpy as np
import psutil
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

        self.process = psutil.Process(os.getpid())

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _loop(self):
        while self.running:
            try:
                ram_mb = self.process.memory_info().rss / 1024 / 1024
                self.ram_samples.append(ram_mb)
            except Exception:
                pass

            p = read_pmic_power_w()
            if p is not None:
                self.power_samples.append(p)

            t = read_temp_c()
            if t is not None:
                self.temp_samples.append(t)

            time.sleep(self.interval)

    def avg_ram(self):
        if not self.ram_samples:
            return None
        return float(np.mean(self.ram_samples))

    def avg_power(self):
        if not self.power_samples:
            return None
        return float(np.mean(self.power_samples))

    def temp_min(self):
        if not self.temp_samples:
            return None
        return float(np.min(self.temp_samples))

    def temp_max(self):
        if not self.temp_samples:
            return None
        return float(np.max(self.temp_samples))

    def temp_avg(self):
        if not self.temp_samples:
            return None
        return float(np.mean(self.temp_samples))


def parse_input_shape(session, default_h=513, default_w=513):
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape

    if len(shape) != 4:
        raise RuntimeError(f"Only 4D input is supported, got shape: {shape}")

    # 你的模型：
    # input_1:0 ['batch', 513, 513, 3]
    # 格式是 NHWC
    n, h, w, c = shape

    if isinstance(h, int):
        input_h = h
    else:
        input_h = default_h

    if isinstance(w, int):
        input_w = w
    else:
        input_w = default_w

    if isinstance(c, int):
        input_c = c
    else:
        input_c = 3

    return input_meta.name, shape, input_c, input_h, input_w


def preprocess_deeplab_nhwc(img_path, input_h=513, input_w=513):
    img = Image.open(img_path).convert("RGB")

    # DeepLabV3 MobileNetV2 這個 ONNX input 是 NHWC 513x513x3
    img = img.resize((input_w, input_h), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0

    # ImageNet normalize
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # 保持 NHWC，不做 transpose
    return arr.astype(np.float32)


def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="deeplab_v3_mobilenet_v2_dynamic.onnx",
                        help="Path to DeepLabV3 MobileNetV2 ONNX model")
    parser.add_argument("--image_dir", type=str, default="../Test",
                        help="Image folder path")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of images, 0 means use all")
    parser.add_argument("--threads", type=int, default=0,
                        help="ONNXRuntime intra_op threads, 0 means default")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations")
    parser.add_argument("--monitor_interval", type=float, default=0.2,
                        help="RAM/Power/Temp sampling interval seconds")
    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError("--batch must be >= 1")

    images = get_images(args.image_dir)
    if args.limit > 0:
        images = images[:args.limit]

    if len(images) == 0:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    print("========== INFO ==========")
    print(f"ONNX: {args.onnx}")
    print(f"Images: {len(images)}")
    print(f"Image dir: {args.image_dir}")
    print(f"Batch size: {args.batch}")

    sess_options = ort.SessionOptions()
    if args.threads > 0:
        sess_options.intra_op_num_threads = args.threads
        sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        args.onnx,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name, input_shape, input_c, input_h, input_w = parse_input_shape(session)

    output_names = [o.name for o in session.get_outputs()]
    output_shapes = [o.shape for o in session.get_outputs()]

    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Parsed input size: H={input_h}, W={input_w}, C={input_c}")
    print(f"Output names: {output_names}")
    print(f"Output shapes: {output_shapes}")
    print("==========================")

    if input_c != 3:
        raise RuntimeError(f"Expected 3-channel NHWC input, got C={input_c}")

    # Warmup
    if args.warmup > 0:
        dummy = np.zeros((args.batch, input_h, input_w, 3), dtype=np.float32)
        for _ in range(args.warmup):
            _ = session.run(output_names, {input_name: dummy})

    monitor = SystemMonitor(interval=args.monitor_interval)
    monitor.start()

    total_start = time.perf_counter()

    inference_time_only = 0.0
    processed = 0

    try:
        for batch_idx, batch_paths in enumerate(make_batches(images, args.batch), start=1):
            batch_data = []

            for img_path in batch_paths:
                arr = preprocess_deeplab_nhwc(img_path, input_h, input_w)
                batch_data.append(arr)

            x = np.stack(batch_data, axis=0).astype(np.float32)

            infer_start = time.perf_counter()
            _ = session.run(output_names, {input_name: x})
            infer_end = time.perf_counter()

            inference_time_only += infer_end - infer_start
            processed += len(batch_paths)

            if processed % 10 == 0 or processed == len(images):
                print(f"Processed {processed}/{len(images)}")

    finally:
        total_end = time.perf_counter()
        monitor.stop()

    total_time = total_end - total_start

    overall_fps = processed / total_time if total_time > 0 else 0.0
    inference_fps = processed / inference_time_only if inference_time_only > 0 else 0.0

    avg_ram = monitor.avg_ram()
    avg_power = monitor.avg_power()
    temp_min = monitor.temp_min()
    temp_max = monitor.temp_max()
    temp_avg = monitor.temp_avg()

    print()
    print("========== RESULTS ==========")
    print(f"Processed Images       : {processed}")
    print(f"Batch Size             : {args.batch}")
    print(f"Total Time             : {total_time:.4f} s")
    print(f"Inference Time Only    : {inference_time_only:.4f} s")
    print(f"Overall FPS            : {overall_fps:.2f}")
    print(f"Inference FPS Only     : {inference_fps:.2f}")

    if avg_ram is not None:
        print(f"Average RAM            : {avg_ram:.2f} MB")
    else:
        print("Average RAM            : N/A")

    if avg_power is not None:
        print(f"Average Power          : {avg_power:.3f} W")
    else:
        print("Average Power          : N/A")

    if temp_min is not None and temp_max is not None and temp_avg is not None:
        print(f"Temperature Min        : {temp_min:.2f} °C")
        print(f"Temperature Max        : {temp_max:.2f} °C")
        print(f"Temperature Avg        : {temp_avg:.2f} °C")
    else:
        print("Temperature Min/Max/Avg: N/A")

    print("=============================")


if __name__ == "__main__":
    main()