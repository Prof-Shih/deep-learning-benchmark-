#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import time
import psutil
import threading
import subprocess
from pathlib import Path

import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_all_images(folder):
    folder = Path(folder)
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(str(p))
    files.sort()
    return files


def preprocess_pidnet(img_bgr, input_h, input_w):
    resized = cv2.resize(img_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32)[:, :, ::-1]
    img /= 255.0
    img -= MEAN
    img /= STD
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img


def format_num(v, digits=2, na="N/A"):
    if v is None:
        return na
    return f"{v:.{digits}f}"


class TegraStatsMonitor:
    def __init__(self, interval_ms=500):
        self.interval_ms = int(interval_ms)
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()

        self.power_w = []
        self.cpu_temp_c = []
        self.gpu_temp_c = []
        self.tj_temp_c = []
        self.all_temp_c = []

    def _find_tegrastats(self):
        for p in ["/usr/bin/tegrastats", "/bin/tegrastats"]:
            if os.path.exists(p):
                return p
        return None

    def _parse_power(self, line):
        import re
        m = re.search(r"VDD_IN\s+([0-9]+(?:\.[0-9]+)?)mW", line)
        if m:
            return float(m.group(1)) / 1000.0
        return None

    def _parse_temp_by_key(self, line, key):
        import re
        m = re.search(rf"{key}@([0-9]+(?:\.[0-9]+)?)C", line)
        if m:
            return float(m.group(1))
        return None

    def _parse_all_temps(self, line):
        import re
        vals = re.findall(r"([A-Za-z0-9_]+)@([0-9]+(?:\.[0-9]+)?)C", line)
        out = []
        for _, v in vals:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    def _reader(self):
        while not self.stop_event.is_set():
            if self.proc is None or self.proc.stdout is None:
                break

            line = self.proc.stdout.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            power = self._parse_power(line)
            if power is not None:
                self.power_w.append(power)

            cpu_t = self._parse_temp_by_key(line, "cpu")
            gpu_t = self._parse_temp_by_key(line, "gpu")
            tj_t = self._parse_temp_by_key(line, "tj")

            if cpu_t is not None:
                self.cpu_temp_c.append(cpu_t)
            if gpu_t is not None:
                self.gpu_temp_c.append(gpu_t)
            if tj_t is not None:
                self.tj_temp_c.append(tj_t)

            temps = self._parse_all_temps(line)
            if temps:
                self.all_temp_c.extend(temps)

    def start(self):
        tegrastats_bin = self._find_tegrastats()
        if tegrastats_bin is None:
            print("[WARN] tegrastats not found, power/temp will be N/A")
            return False

        cmd = [tegrastats_bin, "--interval", str(self.interval_ms)]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"[WARN] failed to start tegrastats: {e}")
            self.proc = None
            return False

    def stop(self):
        self.stop_event.set()
        try:
            if self.proc is not None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass

        if self.thread is not None:
            self.thread.join(timeout=2)

    @staticmethod
    def _avg(arr):
        return float(np.mean(arr)) if len(arr) > 0 else None

    @staticmethod
    def _max(arr):
        return float(np.max(arr)) if len(arr) > 0 else None

    def summary(self):
        return {
            "avg_power_w": self._avg(self.power_w),
            "peak_power_w": self._max(self.power_w),
            "avg_cpu_temp_c": self._avg(self.cpu_temp_c),
            "avg_gpu_temp_c": self._avg(self.gpu_temp_c),
            "avg_tj_temp_c": self._avg(self.tj_temp_c),
            "peak_temp_c": self._max(self.all_temp_c),
        }


class TRTInfer:
    def __init__(self, engine_path, batch_size):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected exactly 1 input tensor, got {len(self.input_names)}: {self.input_names}")

        self.input_name = self.input_names[0]
        engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

        if len(engine_input_shape) != 4:
            raise RuntimeError(f"Expected 4D input, got: {engine_input_shape}")

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise RuntimeError("--batch must be > 0")

        _, self.input_c, self.input_h, self.input_w = engine_input_shape

        if any(d < 0 for d in engine_input_shape):
            self.input_shape = (self.batch_size, self.input_c, self.input_h, self.input_w)
            self.context.set_input_shape(self.input_name, self.input_shape)
        else:
            fixed_batch = engine_input_shape[0]
            if fixed_batch != self.batch_size:
                raise RuntimeError(
                    f"Engine is fixed batch={fixed_batch}, but --batch={self.batch_size}. "
                    f"Please rebuild engine with dynamic batch or use --batch {fixed_batch}."
                )
            self.input_shape = engine_input_shape

        self.device_allocs = {}
        self.host_outputs = {}
        self.bindings = {}

        self._alloc_tensor(self.input_name)
        for name in self.output_names:
            self._alloc_tensor(name)

    def _alloc_tensor(self, name):
        shape = tuple(self.context.get_tensor_shape(name))
        dtype = trt.nptype(self.engine.get_tensor_dtype(name))
        size = int(np.prod(shape))

        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem = cuda.mem_alloc(host_mem.nbytes)

        self.device_allocs[name] = dev_mem
        self.bindings[name] = int(dev_mem)
        self.context.set_tensor_address(name, int(dev_mem))

        if name in self.output_names:
            self.host_outputs[name] = host_mem

    def infer(self, input_tensor):
        if input_tensor.shape != self.input_shape:
            raise ValueError(f"Input shape mismatch: got {input_tensor.shape}, expected {self.input_shape}")

        t0 = time.perf_counter()

        cuda.memcpy_htod_async(self.device_allocs[self.input_name], input_tensor, self.stream)

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_outputs[name], self.device_allocs[name], self.stream)

        self.stream.synchronize()

        t1 = time.perf_counter()
        return t1 - t0


def make_batch(batch_paths, batch_size, input_h, input_w):
    batch_imgs = []
    valid_count = 0

    for img_path in batch_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable: {os.path.basename(img_path)}")
            continue

        inp = preprocess_pidnet(img, input_h, input_w)
        batch_imgs.append(inp)
        valid_count += 1

    if valid_count == 0:
        return None, 0

    while len(batch_imgs) < batch_size:
        batch_imgs.append(np.zeros_like(batch_imgs[0], dtype=np.float32))

    batch_tensor = np.stack(batch_imgs, axis=0)
    batch_tensor = np.ascontiguousarray(batch_tensor, dtype=np.float32)
    return batch_tensor, valid_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PIDNet-S Cityscapes TensorRT benchmark on Jetson Orin Nano")
    parser.add_argument("--engine", default="PIDNet_S_Cityscapes_val_dynamic.engine", help="Path to TensorRT .engine")
    parser.add_argument("--dir", default="../Test", help="Image directory")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--max-images", type=int, default=0, help="0 means run all images")
    parser.add_argument("--tegrastats-interval", type=int, default=500, help="tegrastats polling interval in ms")
    args = parser.parse_args()

    image_paths = get_all_images(args.dir)
    if len(image_paths) == 0:
        print(f"[ERROR] No images found in: {args.dir}")
        sys.exit(1)

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print("[INFO] Loading TensorRT engine...")
    runner = TRTInfer(args.engine, args.batch)

    print(f"[INFO] Engine input shape : {runner.input_shape}")
    print(f"[INFO] Input tensor name  : {runner.input_name}")
    print(f"[INFO] Output tensor names: {runner.output_names}")
    print(f"[INFO] Batch size        : {args.batch}")
    print(f"[INFO] Total target images: {len(image_paths)}")

    warmup_img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if warmup_img is None:
        print(f"[ERROR] Failed to read warmup image: {image_paths[0]}")
        sys.exit(1)

    one = preprocess_pidnet(warmup_img, runner.input_h, runner.input_w)
    warmup_input = np.stack([one for _ in range(args.batch)], axis=0)
    warmup_input = np.ascontiguousarray(warmup_input, dtype=np.float32)

    print(f"[INFO] Warmup x {args.warmup} ...")
    for _ in range(args.warmup):
        _ = runner.infer(warmup_input)

    proc = psutil.Process(os.getpid())
    tegra = TegraStatsMonitor(interval_ms=args.tegrastats_interval)
    tegra.start()

    pure_infer_times = []
    ram_samples_mb = []
    ok_count = 0

    total_start = time.perf_counter()

    total_images = len(image_paths)

    for start_idx in range(0, total_images, args.batch):
        batch_paths = image_paths[start_idx:start_idx + args.batch]
        inp, valid_count = make_batch(batch_paths, args.batch, runner.input_h, runner.input_w)

        if inp is None:
            continue

        infer_t = runner.infer(inp)

        pure_infer_times.append(infer_t)
        ok_count += valid_count

        rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
        ram_samples_mb.append(rss_mb)

        now_idx = min(start_idx + args.batch, total_images)
        print(f"[{now_idx}/{total_images}] batch={valid_count}")

    total_end = time.perf_counter()
    tegra.stop()

    if ok_count == 0:
        print("[ERROR] No valid images processed.")
        sys.exit(1)

    total_elapsed = total_end - total_start
    total_pure_infer = float(np.sum(pure_infer_times))

    avg_infer_s_per_image = total_pure_infer / ok_count
    avg_infer_ms_per_image = avg_infer_s_per_image * 1000.0

    avg_infer_s_per_batch = total_pure_infer / len(pure_infer_times)
    avg_infer_ms_per_batch = avg_infer_s_per_batch * 1000.0

    avg_fps = ok_count / total_pure_infer if total_pure_infer > 0 else 0.0
    overall_fps = ok_count / total_elapsed if total_elapsed > 0 else 0.0

    avg_ram = float(np.mean(ram_samples_mb)) if ram_samples_mb else None
    peak_ram = float(np.max(ram_samples_mb)) if ram_samples_mb else None

    ts = tegra.summary()
    avg_power = ts["avg_power_w"]
    peak_power = ts["peak_power_w"]
    avg_cpu_temp = ts["avg_cpu_temp_c"]
    avg_gpu_temp = ts["avg_gpu_temp_c"]
    avg_tj_temp = ts["avg_tj_temp_c"]
    peak_temp = ts["peak_temp_c"]

    total_energy = (avg_power * total_pure_infer) if avg_power is not None else None
    energy_per_image = (total_energy / ok_count) if total_energy is not None else None

    print()
    print(f"Total images: {ok_count}")
    print(f"Batch size: {args.batch}")
    print()
    print(f"Total elapsed time: {total_elapsed:.3f} s")
    print(f"Total pure inference time: {total_pure_infer:.3f} s")
    print(f"Average inference time / image: {avg_infer_ms_per_image:.2f} ms")
    print(f"Average inference time / batch: {avg_infer_ms_per_batch:.2f} ms")
    print()
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Overall FPS: {overall_fps:.1f}")
    print()
    print(f"Average RAM: {format_num(avg_ram, 0)} MB")
    print(f"Peak RAM: {format_num(peak_ram, 0)} MB")
    print()
    print(f"Average Power: {format_num(avg_power, 2)} W")
    print(f"Peak Power: {format_num(peak_power, 2)} W")
    print()
    print(f"Total Energy: {format_num(total_energy, 2)} J")
    print(f"Energy / image: {format_num(energy_per_image, 4)} J")
    print()
    print(f"Average CPU Temp: {format_num(avg_cpu_temp, 1)} C")
    print(f"Average GPU Temp: {format_num(avg_gpu_temp, 1)} C")
    print(f"Average TJ Temp: {format_num(avg_tj_temp, 1)} C")
    print(f"Peak Temp: {format_num(peak_temp, 1)} C")


if __name__ == "__main__":
    main()