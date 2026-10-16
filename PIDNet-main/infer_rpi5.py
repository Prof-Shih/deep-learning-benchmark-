# infer_pidnet_onnx_rpi5_batch.py

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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_images(image_dir):
    image_dir = Path(image_dir)
    imgs = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(str(p))
    return sorted(imgs)


def make_batches(images, batch_size):
    for i in range(0, len(images), batch_size):
        yield images[i:i + batch_size]


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_power_w():
    """
    Raspberry Pi 5:
    使用 vcgencmd pmic_read_adc

    對應格式：
    xxx_A current(0)=0.123A
    xxx_V volt(0)=3.300V

    會把能配對的 xxx_A × xxx_V 全部加總。
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

            p = read_power_w()
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


def preprocess_pidnet(img_path, input_h, input_w):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    arr = img.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW

    return arr.astype(np.float32)


def postprocess_seg_output(outputs):
    """
    benchmark 用，不存圖。
    PIDNet 可能輸出 [N,C,H,W] logits。
    如果多輸出，先取第一個。
    """
    out = outputs[0] if isinstance(outputs, list) else outputs

    if out.ndim == 4:
        pred = np.argmax(out, axis=1)
    elif out.ndim == 3:
        pred = np.argmax(out, axis=0)
    else:
        pred = out

    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PIDNet_S_Cityscapes_val_dynamic.onnx", help="PIDNet ONNX path")
    parser.add_argument("--image_dir", type=str, default="../Test", help="image folder")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_h", type=int, default=1024)
    parser.add_argument("--input_w", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--monitor_interval", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    images = get_images(args.image_dir)
    if args.limit > 0:
        images = images[:args.limit]

    if len(images) == 0:
        raise RuntimeError("No images found.")

    print(f"[INFO] Model       : {args.model}")
    print(f"[INFO] Images      : {len(images)}")
    print(f"[INFO] Batch size  : {args.batch_size}")
    print(f"[INFO] Input size  : {args.input_h}x{args.input_w}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        args.model,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"[INFO] Input name  : {input_name}")
    print(f"[INFO] Output names: {output_names}")

    # Warmup
    warm_batch = np.zeros(
        (args.batch_size, 3, args.input_h, args.input_w),
        dtype=np.float32
    )

    for _ in range(args.warmup):
        _ = session.run(output_names, {input_name: warm_batch})

    monitor = Monitor(interval=args.monitor_interval)

    total_infer_time = 0.0
    total_batches = 0
    processed_images = 0

    total_start = time.perf_counter()
    monitor.start()

    for batch_idx, batch_paths in enumerate(make_batches(images, args.batch_size), 1):
        batch_input = np.stack(
            [preprocess_pidnet(p, args.input_h, args.input_w) for p in batch_paths],
            axis=0
        )

        infer_start = time.perf_counter()
        outputs = session.run(output_names, {input_name: batch_input})
        infer_end = time.perf_counter()

        total_infer_time += infer_end - infer_start
        total_batches += 1
        processed_images += len(batch_paths)

        _ = postprocess_seg_output(outputs)

        print(
            f"[INFO] {processed_images}/{len(images)} images done",
            end="\r"
        )

    monitor.stop()
    total_end = time.perf_counter()

    total_time = total_end - total_start

    total_fps = processed_images / total_time if total_time > 0 else 0.0
    pure_infer_fps = processed_images / total_infer_time if total_infer_time > 0 else 0.0

    avg_ram = monitor.avg(monitor.ram_mb)
    avg_power = monitor.avg(monitor.power_w)

    temp_min = monitor.min(monitor.temp_c)
    temp_max = monitor.max(monitor.temp_c)
    temp_avg = monitor.avg(monitor.temp_c)

    print("\n\n========== RESULT ==========")
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

    print("============================")


if __name__ == "__main__":
    main()