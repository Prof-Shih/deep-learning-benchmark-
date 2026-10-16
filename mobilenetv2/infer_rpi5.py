import os
import time
import argparse
import threading
import subprocess
import re
from pathlib import Path

import numpy as np
import psutil
from PIL import Image
import onnxruntime as ort


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(image_dir):
    image_dir = Path(image_dir)
    imgs = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(str(p))
    return sorted(imgs)


def preprocess_mobilenetv2(img_path, input_size=224):
    img = Image.open(img_path).convert("RGB")

    # Torchvision MobileNetV2:
    # Resize 256 -> CenterCrop 224 -> Normalize
    resize_size = 256
    img = img.resize((resize_size, resize_size), Image.BILINEAR)

    left = (resize_size - input_size) // 2
    top = (resize_size - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))

    arr = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr.astype(np.float32)


def make_batches(images, batch_size):
    for i in range(0, len(images), batch_size):
        yield images[i:i + batch_size]


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_pmic_power_w():
    """
    Raspberry Pi 5:
    使用 vcgencmd pmic_read_adc 估算 PMIC power。
    如果讀不到，就回傳 None。
    """
    try:
        out = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0
        )
    except Exception:
        return None

    currents = {}
    volts = {}

    for line in out.splitlines():
        line = line.strip()

        m_i = re.match(r"(.+?)_A\s+current\(\d+\)=([0-9.]+)A", line)
        if m_i:
            currents[m_i.group(1).strip()] = float(m_i.group(2))
            continue

        m_v = re.match(r"(.+?)_V\s+volt\(\d+\)=([0-9.]+)V", line)
        if m_v:
            volts[m_v.group(1).strip()] = float(m_v.group(2))
            continue

    total_w = 0.0
    count = 0
    for name, cur in currents.items():
        if name in volts:
            total_w += cur * volts[name]
            count += 1

    return total_w if count > 0 else None


class Monitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.running = False
        self.thread = None
        self.proc = psutil.Process(os.getpid())

        self.ram_mb = []
        self.temp_c = []
        self.power_w = []

    def _loop(self):
        while self.running:
            try:
                rss_mb = self.proc.memory_info().rss / (1024 * 1024)
                self.ram_mb.append(rss_mb)
            except Exception:
                pass

            t = read_temp_c()
            if t is not None:
                self.temp_c.append(t)

            p = read_pmic_power_w()
            if p is not None:
                self.power_w.append(p)

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    @staticmethod
    def avg(x):
        return float(np.mean(x)) if len(x) else None

    @staticmethod
    def min(x):
        return float(np.min(x)) if len(x) else None

    @staticmethod
    def max(x):
        return float(np.max(x)) if len(x) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mobilenet_v2_opset11_dynamic.onnx")
    parser.add_argument("--image_dir", type=str, default="../Test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--monitor_interval", type=float, default=0.2)
    args = parser.parse_args()

    images = get_images(args.image_dir)
    if args.limit > 0:
        images = images[:args.limit]

    if len(images) == 0:
        raise RuntimeError("No images found.")

    print(f"[INFO] Model       : {args.model}")
    print(f"[INFO] Images      : {len(images)}")
    print(f"[INFO] Batch size  : {args.batch_size}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        args.model,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"[INFO] Input name  : {input_name}")
    print(f"[INFO] Output name : {output_name}")

    # Warmup
    warm_paths = images[:args.batch_size]
    warm_batch = np.stack(
        [preprocess_mobilenetv2(p, args.input_size) for p in warm_paths],
        axis=0
    )

    for _ in range(args.warmup):
        _ = session.run([output_name], {input_name: warm_batch})

    monitor = Monitor(interval=args.monitor_interval)

    total_infer_time = 0.0
    total_batches = 0
    processed_images = 0

    total_start = time.perf_counter()
    monitor.start()

    for batch_idx, batch_paths in enumerate(make_batches(images, args.batch_size), 1):
        batch_input = np.stack(
            [preprocess_mobilenetv2(p, args.input_size) for p in batch_paths],
            axis=0
        )

        infer_start = time.perf_counter()
        _ = session.run([output_name], {input_name: batch_input})
        infer_end = time.perf_counter()

        total_infer_time += infer_end - infer_start
        total_batches += 1
        processed_images += len(batch_paths)

        if processed_images % 100 == 0 or processed_images == len(images):
            print(f"[INFO] {processed_images}/{len(images)} images done")

    monitor.stop()
    total_end = time.perf_counter()

    total_time = total_end - total_start

    # FPS 都用「圖片數量」算，不是 batch 數量
    total_fps = processed_images / total_time
    pure_infer_fps = processed_images / total_infer_time

    avg_ram = monitor.avg(monitor.ram_mb)
    avg_power = monitor.avg(monitor.power_w)

    temp_min = monitor.min(monitor.temp_c)
    temp_max = monitor.max(monitor.temp_c)
    temp_avg = monitor.avg(monitor.temp_c)

    print("\n========== RESULT ==========")
    print(f"Model                  : {args.model}")
    print(f"Images                 : {processed_images}")
    print(f"Batch size             : {args.batch_size}")
    print(f"Total batches          : {total_batches}")
    print(f"Total time             : {total_time:.4f} s")
    print(f"Pure inference time    : {total_infer_time:.4f} s")
    print(f"Total FPS              : {total_fps:.4f}")
    print(f"Pure inference FPS     : {pure_infer_fps:.4f}")

    if avg_ram is not None:
        print(f"Average RAM            : {avg_ram:.2f} MB")
    else:
        print("Average RAM            : N/A")

    if avg_power is not None:
        print(f"Average Power          : {avg_power:.4f} W")
    else:
        print("Average Power          : N/A")

    if temp_min is not None:
        print(f"Temperature min        : {temp_min:.2f} °C")
        print(f"Temperature max        : {temp_max:.2f} °C")
        print(f"Temperature avg        : {temp_avg:.2f} °C")
    else:
        print("Temperature            : N/A")


if __name__ == "__main__":
    main()