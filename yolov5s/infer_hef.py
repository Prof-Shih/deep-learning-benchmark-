# benchmark_yolov5s_hef_images_infermodel_batch.py

import argparse
import time
import threading
import subprocess
import re
import gc
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

from hailo_platform import VDevice


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# Image Loader
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


# =========================================================
# Raspberry Pi 5 Power / Temp
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
        if self.thread is not None and self.thread.is_alive():
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
# Hailo format helpers
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

    # Common Hailo HEF input:
    # [H, W, C] or [1, H, W, C]
    if len(shape) == 4:
        h, w = shape[1], shape[2]
    elif len(shape) == 3:
        h, w = shape[0], shape[1]
    else:
        raise ValueError(f"Unsupported input shape: {shape}")

    return int(h), int(w)


def get_all_outputs(infer_model):
    outputs = infer_model.outputs

    if callable(outputs):
        outputs = outputs()

    if not isinstance(outputs, list):
        outputs = list(outputs)

    return outputs


# =========================================================
# YOLOv5 Preprocess
# =========================================================
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


def preprocess_yolov5(img_path, input_h, input_w, dtype):
    img = Image.open(img_path).convert("RGB")
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


# =========================================================
# Bindings helpers
# =========================================================
def make_output_buffers(output_infos):
    output_buffers = {}

    for out_info in output_infos:
        out_name = out_info.name
        out_shape = tuple(out_info.shape)
        out_dtype = dtype_from_format(get_format_type(out_info))

        out = np.empty(out_shape, dtype=out_dtype)
        out = np.ascontiguousarray(out)
        out.setflags(write=True)

        output_buffers[out_name] = out

    return output_buffers


def set_input_buffer(bindings, input_name, input_buffer):
    try:
        bindings.input(input_name).set_buffer(input_buffer)
    except TypeError:
        bindings.input().set_buffer(input_buffer)


def set_output_buffers(bindings, output_infos, output_buffers):
    for out_info in output_infos:
        try:
            bindings.output(out_info.name).set_buffer(output_buffers[out_info.name])
        except TypeError:
            # 通常多 output 需要 name；這裡只是保險
            bindings.output().set_buffer(output_buffers[out_info.name])


def create_batch_bindings(
    configured_model,
    batch_paths,
    input_h,
    input_w,
    input_dtype,
    input_name,
    output_infos
):
    bindings_list = []
    input_buffers = []
    output_buffers_list = []

    for img_path in batch_paths:
        input_buffer = preprocess_yolov5(
            img_path,
            input_h=input_h,
            input_w=input_w,
            dtype=input_dtype
        )

        output_buffers = make_output_buffers(output_infos)

        bindings = configured_model.create_bindings()
        set_input_buffer(bindings, input_name, input_buffer)
        set_output_buffers(bindings, output_infos, output_buffers)

        bindings_list.append(bindings)
        input_buffers.append(input_buffer)
        output_buffers_list.append(output_buffers)

    return bindings_list, input_buffers, output_buffers_list


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hef", default="yolov5s.hef", help="Path to YOLOv5s HEF")
    parser.add_argument("--image-dir", "--dir", dest="image_dir", default="../Test", help="Folder of images")

    # 支援兩種名稱，讓 infer_all.py 通用
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=1)

    # 支援兩種名稱，讓 infer_all.py 通用
    parser.add_argument("--limit", "--frames", dest="frames", type=int, default=None)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    parser.add_argument("--print-every", type=int, default=50)

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch must be >= 1")

    images = get_images(args.image_dir)

    if not images:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    if args.frames is None:
        args.frames = len(images)

    if args.frames <= 0:
        raise ValueError("--frames / --limit must be > 0")

    valid_frames_to_run = args.frames

    total_start = None
    total_end = None
    inference_time = 0.0

    valid_images = 0
    submitted_frames = 0
    padded_frames = 0

    monitor = None

    try:
        with VDevice() as vdevice:
            infer_model = vdevice.create_infer_model(args.hef)

            # 重要：Hailo batch size 一定要在 configure 前設定
            infer_model.set_batch_size(args.batch_size)

            input_info = infer_model.input()
            output_infos = get_all_outputs(infer_model)

            input_name = input_info.name
            input_shape = tuple(input_info.shape)
            input_dtype = dtype_from_format(get_format_type(input_info))
            input_h, input_w = get_hw_from_shape(input_shape)

            print("========== INFO ==========")
            print(f"HEF: {args.hef}")
            print(f"Image dir: {args.image_dir}")
            print(f"Images found: {len(images)}")
            print(f"Frames to run: {valid_frames_to_run}")
            print("Model: YOLOv5s with Hailo NMS postprocess")
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
                print(f"  {out_name}: shape={out_shape}, dtype={out_dtype}")

            print("==========================")

            # 重要：用 with，避免 VDevice 太早釋放
            with infer_model.configure() as configured_model:

                # ==============================
                # Warmup
                # ==============================
                warmup_done = 0

                while warmup_done < args.warmup:
                    real_batch_size = min(args.batch_size, args.warmup - warmup_done)

                    batch_paths = []
                    for j in range(real_batch_size):
                        img_idx = (warmup_done + j) % len(images)
                        batch_paths.append(images[img_idx])

                    if real_batch_size < args.batch_size:
                        pad_count = args.batch_size - real_batch_size
                        batch_paths = batch_paths + [batch_paths[-1]] * pad_count

                    bindings_list, input_buffers, output_buffers_list = create_batch_bindings(
                        configured_model=configured_model,
                        batch_paths=batch_paths,
                        input_h=input_h,
                        input_w=input_w,
                        input_dtype=input_dtype,
                        input_name=input_name,
                        output_infos=output_infos
                    )

                    configured_model.run(bindings_list, args.timeout_ms)

                    warmup_done += real_batch_size

                # ==============================
                # Real benchmark
                # ==============================
                monitor = Monitor(interval=args.monitor_interval)
                monitor.start()

                total_start = time.perf_counter()

                frame_index = 0

                while valid_images < valid_frames_to_run:
                    real_batch_size = min(args.batch_size, valid_frames_to_run - valid_images)

                    batch_paths = []
                    for j in range(real_batch_size):
                        img_idx = (frame_index + j) % len(images)
                        batch_paths.append(images[img_idx])

                    if real_batch_size < args.batch_size:
                        pad_count = args.batch_size - real_batch_size
                        batch_paths = batch_paths + [batch_paths[-1]] * pad_count
                    else:
                        pad_count = 0

                    bindings_list, input_buffers, output_buffers_list = create_batch_bindings(
                        configured_model=configured_model,
                        batch_paths=batch_paths,
                        input_h=input_h,
                        input_w=input_w,
                        input_dtype=input_dtype,
                        input_name=input_name,
                        output_infos=output_infos
                    )

                    infer_start = time.perf_counter()

                    # Hailo 正確 batch：
                    # batch=8 就是 8 個 bindings，每個 binding 一張圖
                    configured_model.run(bindings_list, args.timeout_ms)

                    infer_end = time.perf_counter()

                    inference_time += infer_end - infer_start

                    valid_images += real_batch_size
                    submitted_frames += len(bindings_list)
                    padded_frames += pad_count
                    frame_index += real_batch_size

                    if args.print_every > 0:
                        if valid_images % args.print_every < real_batch_size:
                            print(
                                f"Processed valid images: {valid_images}/{valid_frames_to_run} "
                                f"(submitted frames incl. padding: {submitted_frames})"
                            )

                total_end = time.perf_counter()

                monitor.stop()

            time.sleep(0.5)

    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass

        gc.collect()
        time.sleep(0.5)

    if total_start is None or total_end is None:
        raise RuntimeError("Benchmark did not run.")

    total_time = total_end - total_start

    overall_fps_valid = valid_images / total_time if total_time > 0 else 0.0
    inference_fps_valid = valid_images / inference_time if inference_time > 0 else 0.0

    overall_fps_submitted = submitted_frames / total_time if total_time > 0 else 0.0
    inference_fps_submitted = submitted_frames / inference_time if inference_time > 0 else 0.0

    print("\n========== RESULT ==========")
    print(f"Valid Images: {valid_images}")
    print(f"Submitted Frames incl. padding: {submitted_frames}")
    print(f"Padded Frames: {padded_frames}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total Time: {total_time:.4f} s")
    print(f"Inference Time Only: {inference_time:.4f} s")
    print(f"Overall FPS valid images: {overall_fps_valid:.4f}")
    print(f"Inference FPS valid images: {inference_fps_valid:.4f}")
    print(f"Overall FPS submitted frames: {overall_fps_submitted:.4f}")
    print(f"Inference FPS submitted frames: {inference_fps_submitted:.4f}")

    if monitor is not None:
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
    else:
        print("Average RAM: 0.00 MB")
        print("Average Power: 0.0000 W")
        print("Temperature Max: 0.00 °C")
        print("Temperature Min: 0.00 °C")
        print("Temperature Avg: 0.00 °C")

    print("============================")


if __name__ == "__main__":
    main()