# benchmark_bytetrack_yolox_nano_hef_images_infermodel.py

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


def get_images(image_dir):
    image_dir = Path(image_dir)
    return sorted([
        str(p) for p in image_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])


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

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _loop(self):
        while not self.stop_event.is_set():
            self.ram.append(self.proc.memory_info().rss / 1024 / 1024)

            p = read_pmic_power_w()
            if p is not None:
                self.power.append(p)

            t = read_temp_c()
            if t is not None:
                self.temp.append(t)

            time.sleep(self.interval)


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


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

    if len(shape) == 4:
        h, w = shape[1], shape[2]
    elif len(shape) == 3:
        h, w = shape[0], shape[1]
    else:
        raise ValueError(f"Unsupported input shape: {shape}")

    return int(h), int(w)


def letterbox(img, input_h, input_w, color=(114, 114, 114)):
    orig_w, orig_h = img.size

    scale = min(input_w / orig_w, input_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (input_w, input_h), color)

    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2

    canvas.paste(resized, (pad_x, pad_y))
    return canvas


def preprocess_bytetrack_yolox(img_path, input_h, input_w, dtype):
    img = Image.open(img_path).convert("RGB")

    # YOLOX / ByteTrack detector input: letterbox 到 HEF input size
    img = letterbox(img, input_h, input_w)

    arr = np.array(img, dtype=np.uint8, copy=True)

    if dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    else:
        # UINT8 NHWC：不要 normalize
        arr = arr.astype(dtype, copy=True)

    arr = np.ascontiguousarray(arr)
    arr.setflags(write=True)
    return arr


def get_all_outputs(infer_model):
    outputs = infer_model.outputs

    if callable(outputs):
        outputs = outputs()

    if not isinstance(outputs, list):
        outputs = list(outputs)

    return outputs


def set_input_buffer(bindings, input_name, input_buffer):
    try:
        bindings.input(input_name).set_buffer(input_buffer)
    except TypeError:
        bindings.input().set_buffer(input_buffer)


def set_output_buffers(bindings, output_infos, output_buffers):
    for out_info in output_infos:
        bindings.output(out_info.name).set_buffer(output_buffers[out_info.name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default="bytetrack_yolox_nano.hef")
    parser.add_argument("--image-dir", default="../Test")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    args = parser.parse_args()

    images = get_images(args.image_dir)
    if not images:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    with VDevice() as vdevice:
        infer_model = vdevice.create_infer_model(args.hef)
        infer_model.set_batch_size(args.batch_size)

        input_info = infer_model.input()
        output_infos = get_all_outputs(infer_model)

        input_name = input_info.name
        input_shape = tuple(input_info.shape)
        input_dtype = dtype_from_format(get_format_type(input_info))
        input_h, input_w = get_hw_from_shape(input_shape)

        output_buffers = {}

        print("========== INFO ==========")
        print(f"HEF: {args.hef}")
        print(f"Images folder: {args.image_dir}")
        print(f"Images found: {len(images)}")
        print("Model content: YOLOX nano detector raw heads for ByteTrack")
        print("Decode/NMS/Tracking: NOT included")
        print("API: InferModel")
        print(f"Input name: {input_name}")
        print(f"Input shape: {input_shape}")
        print(f"Input dtype: {input_dtype}")
        print(f"Input HxW: {input_h}x{input_w}")
        print(f"Batch size: {args.batch_size}")

        print("\nOutputs:")
        for out_info in output_infos:
            out_name = out_info.name
            out_shape = tuple(out_info.shape)
            out_dtype = dtype_from_format(get_format_type(out_info))
            output_buffers[out_name] = np.empty(out_shape, dtype=out_dtype)

            print(f"  {out_name}: shape={out_shape}, dtype={out_dtype}")

        print("==========================")

        with infer_model.configure() as configured_model:

            # warmup
            for i in range(args.warmup):
                img_path = images[i % len(images)]
                input_buffer = preprocess_bytetrack_yolox(
                    img_path, input_h, input_w, input_dtype
                )

                bindings = configured_model.create_bindings()
                set_input_buffer(bindings, input_name, input_buffer)
                set_output_buffers(bindings, output_infos, output_buffers)

                configured_model.run([bindings], args.timeout_ms)

            monitor = Monitor(interval=args.monitor_interval)
            monitor.start()

            total_start = time.perf_counter()
            inference_time = 0.0

            for i, img_path in enumerate(images):
                input_buffer = preprocess_bytetrack_yolox(
                    img_path, input_h, input_w, input_dtype
                )

                bindings = configured_model.create_bindings()
                set_input_buffer(bindings, input_name, input_buffer)
                set_output_buffers(bindings, output_infos, output_buffers)

                infer_start = time.perf_counter()
                configured_model.run([bindings], args.timeout_ms)
                infer_end = time.perf_counter()

                inference_time += infer_end - infer_start

                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(images)}")

            total_end = time.perf_counter()
            monitor.stop()

    total_time = total_end - total_start
    total_images = len(images) * args.batch_size

    overall_fps = total_images / total_time if total_time > 0 else 0.0
    inference_fps = total_images / inference_time if inference_time > 0 else 0.0

    print("\n========== RESULT ==========")
    print(f"Total Images: {total_images}")
    print(f"Total Time: {total_time:.4f} s")
    print(f"Inference Time Only: {inference_time:.4f} s")
    print(f"Overall FPS: {overall_fps:.4f}")
    print(f"Inference FPS Only: {inference_fps:.4f}")
    print(f"Average RAM: {avg(monitor.ram):.2f} MB")
    print(f"Average Power: {avg(monitor.power):.4f} W")

    if monitor.temp:
        print(f"Temperature Max: {max(monitor.temp):.2f} °C")
        print(f"Temperature Min: {min(monitor.temp):.2f} °C")
        print(f"Temperature Avg: {avg(monitor.temp):.2f} °C")
    else:
        print("Temperature Max: 0.00 °C")
        print("Temperature Min: 0.00 °C")
        print("Temperature Avg: 0.00 °C")

    print("============================")


if __name__ == "__main__":
    main()