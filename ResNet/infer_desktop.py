import os
import re
import csv
import glob
import time
import atexit
import threading
import subprocess
from pathlib import Path

import cv2
import numpy as np
import psutil

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =========================
# Default Config
# =========================
ENGINE_PATH = "onnx_outputs/resnet18_224_int8.engine"
IMAGE_DIR = "../Test"
OUTPUT_DIR = "trt_windows_results"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# =========================
# Helpers
# =========================
def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def softmax_batch(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def mean_ignore_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_images(folder):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths), key=natural_sort_key)


def fmt(v, nd=4, suffix=""):
    if v is None:
        return "N/A"
    return f"{v:.{nd}f}{suffix}"


def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# =========================
# nvidia-smi Sampler
# =========================
class NvidiaSMISampler:
    def __init__(self, interval_sec=0.1, gpu_index=0):
        self.interval_sec = max(0.05, float(interval_sec))
        self.gpu_index = gpu_index
        self.samples = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _worker(self):
        while not self.stop_event.is_set():
            sample = self._read_once()
            if sample is not None:
                with self.lock:
                    self.samples.append(sample)
            time.sleep(self.interval_sec)

    def _read_once(self):
        t = time.time()
        sys_ram = psutil.virtual_memory()

        sample = {
            "ts": t,
            "sys_ram_used_mb": sys_ram.used / (1024 * 1024),
            "sys_ram_total_mb": sys_ram.total / (1024 * 1024),
            "gpu_power_w": None,
            "gpu_temp_c": None,
            "gpu_mem_used_mb": None,
            "gpu_util": None,
        }

        try:
            cmd = [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=power.draw,temperature.gpu,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                timeout=2,
            )

            if result.returncode != 0:
                return sample

            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]

            if len(parts) >= 4:
                sample["gpu_power_w"] = self._to_float(parts[0])
                sample["gpu_temp_c"] = self._to_float(parts[1])
                sample["gpu_mem_used_mb"] = self._to_float(parts[2])
                sample["gpu_util"] = self._to_float(parts[3])

        except Exception:
            pass

        return sample

    @staticmethod
    def _to_float(x):
        try:
            if x in ("N/A", "", "[N/A]"):
                return None
            return float(x)
        except Exception:
            return None

    def get_samples_between(self, t0, t1):
        with self.lock:
            return [s for s in self.samples if t0 <= s["ts"] <= t1]

    def get_latest(self):
        with self.lock:
            return self.samples[-1] if self.samples else None


# =========================
# TensorRT Classifier with Batch Support
# =========================
class TRTClassifier:
    def __init__(self, engine_path, max_batch=1):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.max_batch = int(max_batch)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        self.num_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io)]

        self.input_name = None
        self.output_name = None

        for name in self.tensor_names:
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Failed to find input/output tensor names.")

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        raw_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

        if len(raw_input_shape) != 4:
            raise ValueError(f"Expected NCHW input, got: {raw_input_shape}")

        n, c, h, w = raw_input_shape

        if n == -1:
            self.dynamic_batch = True
            self.c, self.h, self.w = c, h, w
            self.alloc_batch = self.max_batch
            self.context.set_input_shape(self.input_name, (self.alloc_batch, self.c, self.h, self.w))
        else:
            self.dynamic_batch = False
            self.c, self.h, self.w = c, h, w
            self.alloc_batch = n

            if self.max_batch > n:
                raise ValueError(
                    f"你的 engine 是固定 batch={n}，不能用 --batch {self.max_batch}。"
                    f"請重新轉成 dynamic batch 或 max batch 較大的 engine。"
                )

            self.context.set_input_shape(self.input_name, (n, self.c, self.h, self.w))

        self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        self.stream = cuda.Stream()

        self.max_input_shape = (self.alloc_batch, self.c, self.h, self.w)
        input_nbytes = int(np.prod(self.max_input_shape)) * np.dtype(self.input_dtype).itemsize

        max_output_shape = list(self.output_shape)
        if len(max_output_shape) >= 1:
            max_output_shape[0] = self.alloc_batch
        self.max_output_shape = tuple(max_output_shape)

        output_nbytes = int(np.prod(self.max_output_shape)) * np.dtype(self.output_dtype).itemsize

        self.host_input = cuda.pagelocked_empty(int(np.prod(self.max_input_shape)), self.input_dtype)
        self.host_output = cuda.pagelocked_empty(int(np.prod(self.max_output_shape)), self.output_dtype)

        self.device_input = cuda.mem_alloc(input_nbytes)
        self.device_output = cuda.mem_alloc(output_nbytes)

        self.context.set_tensor_address(self.input_name, int(self.device_input))
        self.context.set_tensor_address(self.output_name, int(self.device_output))

    def preprocess_one(self, image_bgr):
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(img, dtype=np.float32)

    def preprocess_batch(self, image_paths):
        imgs = []
        valid_paths = []

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue

            inp = self.preprocess_one(img)
            imgs.append(inp)
            valid_paths.append(img_path)

        if not imgs:
            return None, []

        batch = np.stack(imgs, axis=0)
        return np.ascontiguousarray(batch, dtype=np.float32), valid_paths

    def infer(self, input_tensor):
        current_batch = input_tensor.shape[0]

        if current_batch > self.alloc_batch:
            raise ValueError(f"Current batch {current_batch} > allocated batch {self.alloc_batch}")

        if self.dynamic_batch:
            self.context.set_input_shape(
                self.input_name,
                (current_batch, self.c, self.h, self.w)
            )

        current_input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        current_output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        input_size = int(np.prod(current_input_shape))
        output_size = int(np.prod(current_output_shape))

        np.copyto(self.host_input[:input_size], input_tensor.ravel())

        start_evt = cuda.Event()
        end_evt = cuda.Event()

        t0 = time.time()
        start_evt.record(self.stream)

        cuda.memcpy_htod_async(
            self.device_input,
            self.host_input[:input_size],
            self.stream
        )

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        cuda.memcpy_dtoh_async(
            self.host_output[:output_size],
            self.device_output,
            self.stream
        )

        end_evt.record(self.stream)
        self.stream.synchronize()
        t1 = time.time()

        wall_s = t1 - t0
        gpu_s = start_evt.time_till(end_evt) / 1000.0

        output = np.array(self.host_output[:output_size]).reshape(current_output_shape)
        return output, wall_s, gpu_s


# =========================
# Main
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT batch inference on Windows NVIDIA GPU")
    parser.add_argument("--engine", type=str, default=ENGINE_PATH, help="Path to TensorRT engine")
    parser.add_argument("--input_dir", type=str, default=IMAGE_DIR, help="Folder containing images")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output folder")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for nvidia-smi")
    parser.add_argument("--sample_interval", type=float, default=0.1, help="nvidia-smi polling interval")
    parser.add_argument("--save_csv", action="store_true", help="Save per-image CSV")
    args = parser.parse_args()

    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    ensure_dir(args.output_dir)

    image_paths = find_images(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in folder: {args.input_dir}")

    print(f"[INFO] Engine: {args.engine}")
    print(f"[INFO] Input dir: {args.input_dir}")
    print(f"[INFO] Total images found: {len(image_paths)}")
    print(f"[INFO] Batch size: {args.batch}")

    model = TRTClassifier(args.engine, max_batch=args.batch)

    print(f"[INFO] Engine input shape now: {model.input_shape}")
    print(f"[INFO] Engine output shape now: {model.output_shape}")
    print(f"[INFO] Dynamic batch: {model.dynamic_batch}")

    sampler = NvidiaSMISampler(interval_sec=args.sample_interval, gpu_index=args.gpu)
    sampler.start()
    atexit.register(sampler.stop)

    time.sleep(0.3)

    per_image_results = []
    per_batch_results = []

    total_wall_start = time.time()

    processed = 0
    total_images_all = len(image_paths)

    for batch_idx, batch_paths in enumerate(make_batches(image_paths, args.batch), start=1):
        input_tensor, valid_paths = model.preprocess_batch(batch_paths)

        if input_tensor is None or len(valid_paths) == 0:
            continue

        current_batch = input_tensor.shape[0]

        infer_t0 = time.time()
        output, infer_wall_s, infer_gpu_s = model.infer(input_tensor)
        infer_t1 = time.time()

        samples = sampler.get_samples_between(infer_t0, infer_t1)
        latest = sampler.get_latest()

        sys_ram_used_mb = mean_ignore_none([s["sys_ram_used_mb"] for s in samples])
        sys_ram_total_mb = mean_ignore_none([s["sys_ram_total_mb"] for s in samples])
        gpu_power_w = mean_ignore_none([s["gpu_power_w"] for s in samples])
        gpu_temp_c = mean_ignore_none([s["gpu_temp_c"] for s in samples])
        gpu_mem_used_mb = mean_ignore_none([s["gpu_mem_used_mb"] for s in samples])

        if latest is not None:
            if sys_ram_used_mb is None:
                sys_ram_used_mb = latest["sys_ram_used_mb"]
            if sys_ram_total_mb is None:
                sys_ram_total_mb = latest["sys_ram_total_mb"]
            if gpu_power_w is None:
                gpu_power_w = latest["gpu_power_w"]
            if gpu_temp_c is None:
                gpu_temp_c = latest["gpu_temp_c"]
            if gpu_mem_used_mb is None:
                gpu_mem_used_mb = latest["gpu_mem_used_mb"]

        batch_energy_j = gpu_power_w * infer_wall_s if gpu_power_w is not None else None
        energy_per_image_j = batch_energy_j / current_batch if batch_energy_j is not None else None

        batch_fps = current_batch / infer_wall_s if infer_wall_s > 0 else None
        infer_time_per_image_s = infer_wall_s / current_batch if current_batch > 0 else None
        gpu_time_per_image_s = infer_gpu_s / current_batch if current_batch > 0 else None

        logits = output.reshape(current_batch, -1)
        probs = softmax_batch(logits)
        top1_idx_list = np.argmax(probs, axis=1)
        top1_score_list = np.max(probs, axis=1)

        per_batch_results.append({
            "batch_idx": batch_idx,
            "batch_size": current_batch,
            "infer_time_s": infer_wall_s,
            "gpu_kernel_time_s": infer_gpu_s,
            "batch_fps": batch_fps,
            "sys_ram_used_mb": sys_ram_used_mb,
            "sys_ram_total_mb": sys_ram_total_mb,
            "gpu_mem_used_mb": gpu_mem_used_mb,
            "gpu_power_w": gpu_power_w,
            "batch_energy_j": batch_energy_j,
            "gpu_temp_c": gpu_temp_c,
        })

        for i, img_path in enumerate(valid_paths):
            processed += 1

            row = {
                "index": processed,
                "batch_idx": batch_idx,
                "image": os.path.basename(img_path),
                "top1_idx": int(top1_idx_list[i]),
                "top1_score": float(top1_score_list[i]),
                "infer_time_per_image_s": infer_time_per_image_s,
                "gpu_kernel_time_per_image_s": gpu_time_per_image_s,
                "batch_infer_time_s": infer_wall_s,
                "batch_fps": batch_fps,
                "sys_ram_used_mb": sys_ram_used_mb,
                "sys_ram_total_mb": sys_ram_total_mb,
                "gpu_mem_used_mb": gpu_mem_used_mb,
                "gpu_power_w": gpu_power_w,
                "energy_j": energy_per_image_j,
                "gpu_temp_c": gpu_temp_c,
            }
            per_image_results.append(row)

        if processed % 10 == 0 or processed == total_images_all:
            print(
                f"[{processed:04d}/{total_images_all:04d}] "
                f"Batch={current_batch} | "
                f"Batch Infer={infer_wall_s * 1000:.2f} ms | "
                f"Avg/Image={infer_time_per_image_s * 1000:.2f} ms | "
                f"Batch FPS={fmt(batch_fps, 2)} | "
                f"RAM={fmt(sys_ram_used_mb, 1)} MB | "
                f"GPU_MEM={fmt(gpu_mem_used_mb, 1)} MB | "
                f"Power={fmt(gpu_power_w, 3)} W | "
                f"Energy/Image={fmt(energy_per_image_j, 4)} J | "
                f"GPU Temp={fmt(gpu_temp_c, 2)} C"
            )

    total_wall_end = time.time()
    sampler.stop()

    valid_images = per_image_results
    valid_batches = per_batch_results

    if not valid_images or not valid_batches:
        raise RuntimeError("No valid inference results.")

    total_images = len(valid_images)
    total_batches = len(valid_batches)

    total_wall_time_s = total_wall_end - total_wall_start

    total_pure_infer_time_s = sum(r["infer_time_s"] for r in valid_batches)
    total_gpu_kernel_time_s = sum(r["gpu_kernel_time_s"] for r in valid_batches)

    avg_infer_time_per_image_s = total_pure_infer_time_s / total_images
    avg_gpu_kernel_time_per_image_s = total_gpu_kernel_time_s / total_images

    pure_infer_fps = total_images / total_pure_infer_time_s if total_pure_infer_time_s > 0 else None
    overall_fps = total_images / total_wall_time_s if total_wall_time_s > 0 else None

    avg_sys_ram_mb = mean_ignore_none([r["sys_ram_used_mb"] for r in valid_batches])
    peak_sys_ram_mb = max([r["sys_ram_used_mb"] for r in valid_batches if r["sys_ram_used_mb"] is not None], default=None)

    avg_gpu_mem_mb = mean_ignore_none([r["gpu_mem_used_mb"] for r in valid_batches])
    peak_gpu_mem_mb = max([r["gpu_mem_used_mb"] for r in valid_batches if r["gpu_mem_used_mb"] is not None], default=None)

    avg_power_w = mean_ignore_none([r["gpu_power_w"] for r in valid_batches])
    peak_power_w = max([r["gpu_power_w"] for r in valid_batches if r["gpu_power_w"] is not None], default=None)

    total_energy_j = (
        sum([r["batch_energy_j"] for r in valid_batches if r["batch_energy_j"] is not None])
        if any(r["batch_energy_j"] is not None for r in valid_batches)
        else None
    )
    energy_per_image_j = total_energy_j / total_images if total_energy_j is not None else None

    avg_gpu_temp_c = mean_ignore_none([r["gpu_temp_c"] for r in valid_batches])
    peak_gpu_temp_c = max([r["gpu_temp_c"] for r in valid_batches if r["gpu_temp_c"] is not None], default=None)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Total batches: {total_batches}")
    print(f"Batch size setting: {args.batch}")
    print("")
    print(f"Total wall time: {total_wall_time_s:.6f} s")
    print(f"Total pure inference time: {total_pure_infer_time_s:.6f} s")
    print(f"Total GPU kernel time: {total_gpu_kernel_time_s:.6f} s")
    print(f"Average inference time / image: {avg_infer_time_per_image_s * 1000:.3f} ms")
    print(f"Average GPU kernel time / image: {avg_gpu_kernel_time_per_image_s * 1000:.3f} ms")
    print("")
    print(f"Overall FPS: {fmt(overall_fps, 4)}")
    print(f"Pure inference FPS: {fmt(pure_infer_fps, 4)}")
    print("")
    print(f"Average RAM: {fmt(avg_sys_ram_mb, 3)} MB")
    print(f"Peak RAM: {fmt(peak_sys_ram_mb, 3)} MB")
    print(f"Average GPU Memory: {fmt(avg_gpu_mem_mb, 3)} MB")
    print(f"Peak GPU Memory: {fmt(peak_gpu_mem_mb, 3)} MB")
    print("")
    print(f"Average Power: {fmt(avg_power_w, 6)} W")
    print(f"Peak Power: {fmt(peak_power_w, 6)} W")
    print(f"Total Energy: {fmt(total_energy_j, 6)} J")
    print(f"Energy / image: {fmt(energy_per_image_j, 6)} J")
    print("")
    print(f"Average GPU Temp: {fmt(avg_gpu_temp_c, 3)} C")
    print(f"Peak GPU Temp: {fmt(peak_gpu_temp_c, 3)} C")

    report_txt = os.path.join(args.output_dir, "report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("FINAL SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Engine: {args.engine}\n")
        f.write(f"Input dir: {args.input_dir}\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Total batches: {total_batches}\n")
        f.write(f"Batch size setting: {args.batch}\n\n")

        f.write(f"Total wall time: {total_wall_time_s:.6f} s\n")
        f.write(f"Total pure inference time: {total_pure_infer_time_s:.6f} s\n")
        f.write(f"Total GPU kernel time: {total_gpu_kernel_time_s:.6f} s\n")
        f.write(f"Average inference time / image: {avg_infer_time_per_image_s * 1000:.3f} ms\n")
        f.write(f"Average GPU kernel time / image: {avg_gpu_kernel_time_per_image_s * 1000:.3f} ms\n\n")

        f.write(f"Overall FPS: {fmt(overall_fps, 4)}\n")
        f.write(f"Pure inference FPS: {fmt(pure_infer_fps, 4)}\n\n")

        f.write(f"Average RAM: {fmt(avg_sys_ram_mb, 3)} MB\n")
        f.write(f"Peak RAM: {fmt(peak_sys_ram_mb, 3)} MB\n")
        f.write(f"Average GPU Memory: {fmt(avg_gpu_mem_mb, 3)} MB\n")
        f.write(f"Peak GPU Memory: {fmt(peak_gpu_mem_mb, 3)} MB\n\n")

        f.write(f"Average Power: {fmt(avg_power_w, 6)} W\n")
        f.write(f"Peak Power: {fmt(peak_power_w, 6)} W\n")
        f.write(f"Total Energy: {fmt(total_energy_j, 6)} J\n")
        f.write(f"Energy / image: {fmt(energy_per_image_j, 6)} J\n\n")

        f.write(f"Average GPU Temp: {fmt(avg_gpu_temp_c, 3)} C\n")
        f.write(f"Peak GPU Temp: {fmt(peak_gpu_temp_c, 3)} C\n")

    if args.save_csv:
        csv_path = os.path.join(args.output_dir, "per_image.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "batch_idx",
                    "image",
                    "top1_idx",
                    "top1_score",
                    "infer_time_per_image_s",
                    "gpu_kernel_time_per_image_s",
                    "batch_infer_time_s",
                    "batch_fps",
                    "sys_ram_used_mb",
                    "sys_ram_total_mb",
                    "gpu_mem_used_mb",
                    "gpu_power_w",
                    "energy_j",
                    "gpu_temp_c",
                ],
            )
            writer.writeheader()
            for row in valid_images:
                writer.writerow(row)
        print(f"Saved CSV: {csv_path}")

    print(f"Saved TXT: {report_txt}")


if __name__ == "__main__":
    main()