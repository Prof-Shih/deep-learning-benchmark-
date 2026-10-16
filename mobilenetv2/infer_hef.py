import time
import re
import gc
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
        self.thread = None

        self.ram_mb = []
        self.power_w = []
        self.temp_c = []

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
# Image
# =========================================================
def get_images(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise RuntimeError(f"Image directory not found: {image_dir}")

    imgs = []
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(str(p))

    return sorted(imgs)


def parse_input_hw(input_shape):
    """
    Hailo InferModel 常見 input shape:
    - [H, W, C]
    - [N, H, W, C]
    你的 mobilenet_v2_1.0.hef 之前是 [224, 224, 3]
    """
    shape = list(input_shape)

    if len(shape) == 3:
        h, w, c = shape
        return int(h), int(w), int(c)

    if len(shape) == 4:
        _, h, w, c = shape
        return int(h), int(w), int(c)

    raise RuntimeError(f"Unsupported input shape: {input_shape}")


def preprocess_uint8_nhwc(img_path, width, height):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)

    arr = np.array(img, dtype=np.uint8, copy=True)
    arr = np.ascontiguousarray(arr)
    arr.setflags(write=True)

    return arr


def make_output_buffer(output_shape, dtype):
    out = np.zeros(output_shape, dtype=dtype)
    out = np.ascontiguousarray(out)
    out.setflags(write=True)
    return out


def get_output_dtype(output_format):
    if output_format == "uint8":
        return np.uint8
    elif output_format == "float32":
        return np.float32
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def set_hailo_format(infer_model, input_format, output_format):
    if input_format == "uint8":
        infer_model.input().set_format_type(FormatType.UINT8)
    elif input_format == "float32":
        infer_model.input().set_format_type(FormatType.FLOAT32)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    if output_format == "uint8":
        infer_model.output().set_format_type(FormatType.UINT8)
    elif output_format == "float32":
        infer_model.output().set_format_type(FormatType.FLOAT32)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hef", type=str, default="mobilenet_v2_1.0.hef")
    parser.add_argument("--image-dir", "--dir", dest="image_dir", type=str, default="../Test")

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--monitor-interval", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=50)

    parser.add_argument(
        "--input-format",
        type=str,
        default="uint8",
        choices=["uint8", "float32"]
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="uint8",
        choices=["uint8", "float32"]
    )

    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError("--batch must be >= 1")

    image_paths = get_images(args.image_dir)

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    if len(image_paths) == 0:
        raise RuntimeError("No images found. Please check --image-dir.")

    print("========== INFO ==========")
    print(f"HEF: {args.hef}")
    print(f"Image dir: {args.image_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Batch size: {args.batch}")
    print(f"Input format: {args.input_format}")
    print(f"Output format: {args.output_format}")
    print(f"Timeout ms: {args.timeout_ms}")
    print("API: InferModel")
    print("==========================")

    monitor = Monitor(interval=args.monitor_interval)

    total_start = time.perf_counter()
    inference_time = 0.0

    valid_images = 0
    submitted_frames = 0
    padded_frames = 0

    output_dtype = get_output_dtype(args.output_format)

    monitor.start()

    try:
        with VDevice() as vdevice:
            infer_model = vdevice.create_infer_model(args.hef)

            # =====================================================
            # Important:
            # set_batch_size 一定要在 configure() 前面
            # =====================================================
            infer_model.set_batch_size(args.batch)

            set_hailo_format(
                infer_model,
                input_format=args.input_format,
                output_format=args.output_format
            )

            # =====================================================
            # Important:
            # 一定用 with，避免 VDevice 太早釋放造成 CIM 還在用
            # =====================================================
            with infer_model.configure() as configured_model:
                input_name = infer_model.input().name
                output_name = infer_model.output().name

                input_shape = infer_model.input().shape
                output_shape = infer_model.output().shape

                input_h, input_w, input_c = parse_input_hw(input_shape)

                print(f"Input name: {input_name}")
                print(f"Output name: {output_name}")
                print(f"Input shape: {input_shape}")
                print(f"Output shape: {output_shape}")
                print(f"Parsed input: H={input_h}, W={input_w}, C={input_c}")
                print("==========================")

                for start_idx in range(0, len(image_paths), args.batch):
                    batch_paths = image_paths[start_idx:start_idx + args.batch]
                    real_batch_size = len(batch_paths)

                    # 最後一批不足 batch 時，用最後一張補滿
                    # 這樣可以避免 Hailo batch bindings 數量不足
                    if real_batch_size < args.batch:
                        pad_count = args.batch - real_batch_size
                        batch_paths = batch_paths + [batch_paths[-1]] * pad_count
                    else:
                        pad_count = 0

                    bindings_list = []
                    output_buffers = []

                    for img_path in batch_paths:
                        input_buffer = preprocess_uint8_nhwc(
                            img_path,
                            width=input_w,
                            height=input_h
                        )

                        output_buffer = make_output_buffer(
                            output_shape,
                            dtype=output_dtype
                        )

                        bindings = configured_model.create_bindings()
                        bindings.input().set_buffer(input_buffer)
                        bindings.output().set_buffer(output_buffer)

                        bindings_list.append(bindings)
                        output_buffers.append(output_buffer)

                    infer_start = time.perf_counter()

                    # =====================================================
                    # Hailo 正確 batch 跑法：
                    # batch=8，就是放 8 個 bindings
                    # 不是把 input 做成 [8, H, W, C]
                    # =====================================================
                    configured_model.run(bindings_list, args.timeout_ms)

                    inference_time += time.perf_counter() - infer_start

                    valid_images += real_batch_size
                    submitted_frames += len(bindings_list)
                    padded_frames += pad_count

                    if args.print_every > 0:
                        if valid_images % args.print_every < real_batch_size:
                            print(
                                f"Processed valid images: {valid_images}/{len(image_paths)} "
                                f"(submitted frames incl. padding: {submitted_frames})"
                            )

            # 讓 configured_model 先完整釋放
            time.sleep(0.5)

    finally:
        monitor.stop()

        # 幫助 Python 釋放 Hailo object
        gc.collect()
        time.sleep(0.5)

    total_time = time.perf_counter() - total_start

    print("\n========== RESULT ==========")
    print(f"Valid images: {valid_images}")
    print(f"Submitted frames incl. padding: {submitted_frames}")
    print(f"Padded frames: {padded_frames}")
    print(f"Batch size: {args.batch}")
    print(f"Total Time (s): {total_time:.4f}")
    print(f"Inference Time Only (s): {inference_time:.4f}")

    if total_time > 0:
        print(f"Overall FPS valid images: {valid_images / total_time:.2f}")
        print(f"Overall FPS submitted frames: {submitted_frames / total_time:.2f}")
    else:
        print("Overall FPS: N/A")

    if inference_time > 0:
        print(f"Inference FPS valid images: {valid_images / inference_time:.2f}")
        print(f"Inference FPS submitted frames: {submitted_frames / inference_time:.2f}")
    else:
        print("Inference FPS: N/A")

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