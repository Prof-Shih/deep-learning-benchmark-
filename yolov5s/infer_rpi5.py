# infer_yolov5n_onnx_rpi5_batch.py

import os
import re
import cv2
import time
import argparse
import threading
import subprocess
from pathlib import Path

import numpy as np
import psutil
import onnxruntime as ort


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(image_dir):
    image_dir = Path(image_dir)
    imgs = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(str(p))
    return sorted(imgs)


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


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


class Monitor:
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
            "ram_samples": len(self.ram_samples),
            "power_samples": len(self.power_samples),
            "temp_samples": len(self.temp_samples),
        }


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]

    r = min(new_shape / h, new_shape / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    dw = new_shape - new_unpad_w
    dh = new_shape - new_unpad_h
    dw /= 2
    dh /= 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img


def preprocess_yolov5(img_path, imgsz=640):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = letterbox(img, new_shape=imgsz)

    # BGR -> RGB, HWC -> CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)

    # 0~255 -> 0~1
    img /= 255.0

    return img


def make_batches(image_paths, batch_size, imgsz):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = []

        for p in batch_paths:
            batch_imgs.append(preprocess_yolov5(p, imgsz))

        batch = np.stack(batch_imgs, axis=0)
        yield batch, len(batch_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5n_dynamic.onnx", help="Path to yolov5n.onnx")
    parser.add_argument("--image-dir", type=str, default="../Test", help="Image folder")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 = use all images")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    args = parser.parse_args()

    image_paths = get_images(args.image_dir)

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    if len(image_paths) == 0:
        raise RuntimeError("No images found.")

    print("========== YOLOv5n ONNX Raspberry Pi 5 Test ==========")
    print(f"Model       : {args.model}")
    print(f"Image dir   : {args.image_dir}")
    print(f"Images      : {len(image_paths)}")
    print(f"Input size  : {args.imgsz}x{args.imgsz}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Threads     : {args.threads}")
    print("======================================================")

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

    print(f"Input name  : {input_name}")
    print(f"Output names: {output_names}")

    dummy = np.zeros(
        (args.batch_size, 3, args.imgsz, args.imgsz),
        dtype=np.float32
    )

    print("\nWarming up...")
    for _ in range(args.warmup):
        _ = session.run(output_names, {input_name: dummy})

    monitor = Monitor(interval=args.monitor_interval)

    total_images = 0
    infer_time = 0.0

    print("\nStart testing...")

    monitor.start()
    total_start = time.perf_counter()

    for batch, real_bs in make_batches(image_paths, args.batch_size, args.imgsz):
        t0 = time.perf_counter()
        _ = session.run(output_names, {input_name: batch})
        t1 = time.perf_counter()

        infer_time += (t1 - t0)
        total_images += real_bs

    total_end = time.perf_counter()
    monitor.stop()

    total_time = total_end - total_start

    overall_fps = total_images / total_time if total_time > 0 else 0.0
    infer_fps = total_images / infer_time if infer_time > 0 else 0.0

    mon = monitor.summary()

    print("\n==================== Result ====================")
    print(f"Total images              : {total_images}")
    print(f"Batch size                : {args.batch_size}")
    print(f"Total test time           : {total_time:.4f} s")
    print(f"Pure inference time       : {infer_time:.4f} s")
    print(f"Overall FPS               : {overall_fps:.4f}")
    print(f"Pure inference FPS        : {infer_fps:.4f}")

    print(f"Average RAM usage         : {mon['avg_ram_mb']:.2f} MB")
    print(f"Average Power             : {mon['avg_power_w']:.4f} W")

    print(f"Temperature min           : {mon['min_temp_c']:.2f} °C")
    print(f"Temperature max           : {mon['max_temp_c']:.2f} °C")
    print(f"Temperature avg           : {mon['avg_temp_c']:.2f} °C")

    print(f"RAM samples               : {mon['ram_samples']}")
    print(f"Power samples             : {mon['power_samples']}")
    print(f"Temp samples              : {mon['temp_samples']}")
    print("================================================")


if __name__ == "__main__":
    main()