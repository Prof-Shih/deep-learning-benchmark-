#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cv2
import time
import math
import psutil
import signal
import argparse
import threading
import subprocess
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TegraStatsMonitor:
    def __init__(self, tegrastats_cmd="tegrastats", interval_ms=200):
        self.tegrastats_cmd = tegrastats_cmd
        self.interval_ms = int(interval_ms)
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.samples = {
            "power_w": [],
            "cpu_temp_c": [],
            "gpu_temp_c": [],
            "tj_temp_c": [],
        }

    def _parse_line(self, line):
        cpu_match = re.search(r"cpu@([0-9.]+)C", line)
        gpu_match = re.search(r"gpu@([0-9.]+)C", line)
        tj_match = re.search(r"tj@([0-9.]+)C", line)

        # 這段保留你原本的功耗寫法：讀 tegrastats 裡的 VDD_IN
        vdd_match = re.search(r"VDD_IN\s+([0-9.]+)mW(?:/([0-9.]+)mW)?", line)

        with self.lock:
            if cpu_match:
                self.samples["cpu_temp_c"].append(float(cpu_match.group(1)))
            if gpu_match:
                self.samples["gpu_temp_c"].append(float(gpu_match.group(1)))
            if tj_match:
                self.samples["tj_temp_c"].append(float(tj_match.group(1)))
            if vdd_match:
                pw_mw = vdd_match.group(2) if vdd_match.group(2) else vdd_match.group(1)
                self.samples["power_w"].append(float(pw_mw) / 1000.0)

    def _reader(self):
        while not self.stop_event.is_set():
            if self.proc is None or self.proc.stdout is None:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            self._parse_line(line.strip())

    def start(self):
        self.proc = subprocess.Popen(
            [self.tegrastats_cmd, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def summary(self):
        def avg(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        def peak(xs):
            return max(xs) if xs else float("nan")

        with self.lock:
            power = list(self.samples["power_w"])
            cpu_t = list(self.samples["cpu_temp_c"])
            gpu_t = list(self.samples["gpu_temp_c"])
            tj_t = list(self.samples["tj_temp_c"])

        all_temp = cpu_t + gpu_t + tj_t

        return {
            "avg_power_w": avg(power),
            "peak_power_w": peak(power),
            "avg_cpu_temp_c": avg(cpu_t),
            "avg_gpu_temp_c": avg(gpu_t),
            "avg_tj_temp_c": avg(tj_t),
            "peak_temp_c": peak(all_temp),
        }


class TRTInfer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_names = []

        print("========== ENGINE INFO ==========")

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)

            try:
                shape = self.engine.get_tensor_shape(name)
            except Exception:
                shape = "unknown"

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                print(f"Input : {name} {shape} {dtype}")
            else:
                self.output_names.append(name)
                print(f"Output: {name} {shape} {dtype}")

        print("=================================")
        print()

        if self.input_name is None:
            raise RuntimeError("No input tensor found")

        self.d_input = None
        self.input_bytes = 0
        self.d_outputs = {}
        self.output_bytes = {}

    def infer(self, x):
        x = np.ascontiguousarray(x.astype(np.float32))

        # dynamic batch
        self.context.set_input_shape(self.input_name, x.shape)

        if self.d_input is None or x.nbytes > self.input_bytes:
            self.d_input = cuda.mem_alloc(x.nbytes)
            self.input_bytes = x.nbytes

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        cuda.memcpy_htod_async(self.d_input, x, self.stream)

        output_host = {}

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            host = np.empty(size, dtype=dtype)
            nbytes = host.nbytes

            if name not in self.d_outputs or nbytes > self.output_bytes.get(name, 0):
                self.d_outputs[name] = cuda.mem_alloc(nbytes)
                self.output_bytes[name] = nbytes

            self.context.set_tensor_address(name, int(self.d_outputs[name]))
            output_host[name] = (host, shape)

        t0 = time.perf_counter()
        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        t1 = time.perf_counter()

        if not ok:
            raise RuntimeError("TensorRT execution failed")

        outputs = []
        for name in self.output_names:
            host, shape = output_host[name]
            cuda.memcpy_dtoh_async(host, self.d_outputs[name], self.stream)
            self.stream.synchronize()
            outputs.append(host.reshape(shape))

        return outputs, t1 - t0


def list_images(input_dir):
    files = []
    for name in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in IMG_EXTS:
            files.append(p)
    return files


def preprocess_deeplab_nhwc(img_bgr, input_h, input_w, mean, std):
    # OpenCV 讀進來是 BGR，轉成 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # resize to 513x513
    img_resized = cv2.resize(img_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    # ImageNet normalize
    x = img_resized.astype(np.float32) / 255.0
    x = (x - mean) / std

    # 你的 engine input 是 NHWC: [batch, 513, 513, 3]
    return np.ascontiguousarray(x, dtype=np.float32)


def get_segmentation_map(output):
    """
    DeepLab output:
    [B, 513, 513, 21] -> argmax last dim -> [B, 513, 513]
    """
    out = output

    if out.ndim == 4:
        # NHWC
        if out.shape[-1] <= 256:
            return np.argmax(out, axis=-1).astype(np.uint8)

        # NCHW fallback
        if out.shape[1] <= 256:
            return np.argmax(out, axis=1).astype(np.uint8)

    if out.ndim == 3:
        return out.astype(np.uint8)

    if out.ndim == 2:
        return np.expand_dims(out.astype(np.uint8), axis=0)

    raise RuntimeError(f"Unsupported output shape: {out.shape}")


def fmt(v, digits=2):
    if v is None or math.isnan(v) or math.isinf(v):
        return "N/A"
    return f"{v:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="deeplab_v3_mobilenet_v2_dynamic.engine")
    parser.add_argument("--input_dir", type=str, default="../Test")
    parser.add_argument("--input_h", type=int, default=513)
    parser.add_argument("--input_w", type=int, default=513)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tegrastats", type=str, default="auto")
    parser.add_argument("--tegrastats_interval", type=int, default=200)
    args = parser.parse_args()

    if not os.path.isfile(args.engine):
        raise FileNotFoundError(args.engine)

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(args.input_dir)

    image_paths = list_images(args.input_dir)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    total_images = len(image_paths)

    if total_images == 0:
        raise RuntimeError(f"No images found in {args.input_dir}")

    print(f"Total images: {total_images}")
    print(f"Batch size: {args.batch}")
    print(f"Input size: {args.input_h}x{args.input_w}")
    print()

    # RGB ImageNet mean/std，NHWC 用
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    model = TRTInfer(args.engine)

    # warmup
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")

    warm_one = preprocess_deeplab_nhwc(first_img, args.input_h, args.input_w, mean, std)
    warm_x = np.repeat(np.expand_dims(warm_one, axis=0), args.batch, axis=0)

    print("Warmup...")
    for _ in range(args.warmup):
        outputs, _ = model.infer(warm_x)

    print("First output shape after warmup:")
    for idx, out in enumerate(outputs):
        print(f"output[{idx}] shape: {out.shape}")
    print()

    tegra = None
    if args.tegrastats != "off":
        tegra_cmd = "tegrastats" if args.tegrastats == "auto" else args.tegrastats
        tegra = TegraStatsMonitor(tegra_cmd, args.tegrastats_interval)
        tegra.start()

    process = psutil.Process(os.getpid())

    pure_times = []
    rss_samples_mb = []
    processed = 0

    t_total0 = time.perf_counter()

    for i in range(0, total_images, args.batch):
        batch_paths = image_paths[i:i + args.batch]
        batch_inputs = []

        for p in batch_paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[Warning] Failed to read image: {p}")
                continue

            x = preprocess_deeplab_nhwc(img, args.input_h, args.input_w, mean, std)
            batch_inputs.append(x)

        if len(batch_inputs) == 0:
            continue

        # NHWC batch: [B, 513, 513, 3]
        batch_x = np.stack(batch_inputs, axis=0).astype(np.float32)

        outputs, pure_t = model.infer(batch_x)

        # 不存圖，只確認 segmentation map 可以正常產生
        seg_map = get_segmentation_map(outputs[0])

        pure_times.append(pure_t)
        processed += batch_x.shape[0]

        rss_samples_mb.append(process.memory_info().rss / 1024.0 / 1024.0)

        print(
            f"[{processed}/{total_images}] "
            f"input={batch_x.shape}, output={outputs[0].shape}, seg={seg_map.shape}",
            flush=True,
        )

    t_total1 = time.perf_counter()

    if tegra is not None:
        tegra.stop()
        tegra_sum = tegra.summary()
    else:
        tegra_sum = {
            "avg_power_w": float("nan"),
            "peak_power_w": float("nan"),
            "avg_cpu_temp_c": float("nan"),
            "avg_gpu_temp_c": float("nan"),
            "avg_tj_temp_c": float("nan"),
            "peak_temp_c": float("nan"),
        }

    total_pure = sum(pure_times)
    total_elapsed = t_total1 - t_total0

    avg_infer_ms_per_batch = total_pure / len(pure_times) * 1000.0 if pure_times else float("nan")
    avg_infer_ms_per_image = total_pure / processed * 1000.0 if processed > 0 else float("nan")

    infer_fps = processed / total_pure if total_pure > 0 else float("nan")
    overall_fps = processed / total_elapsed if total_elapsed > 0 else float("nan")

    avg_ram = sum(rss_samples_mb) / len(rss_samples_mb) if rss_samples_mb else float("nan")
    peak_ram = max(rss_samples_mb) if rss_samples_mb else float("nan")

    total_energy = (
        tegra_sum["avg_power_w"] * total_pure
        if not math.isnan(tegra_sum["avg_power_w"])
        else float("nan")
    )
    energy_per_image = (
        total_energy / processed
        if processed > 0 and not math.isnan(total_energy)
        else float("nan")
    )

    print()
    print("========== RESULT ==========")
    print(f"Total images: {processed}")
    print(f"Batch size: {args.batch}")
    print()
    print(f"Total elapsed time: {fmt(total_elapsed, 3)} s")
    print(f"Total pure inference time: {fmt(total_pure, 3)} s")
    print(f"Average inference time / batch: {fmt(avg_infer_ms_per_batch, 2)} ms")
    print(f"Average inference time / image: {fmt(avg_infer_ms_per_image, 2)} ms")
    print()
    print(f"Inference FPS Only: {fmt(infer_fps, 2)}")
    print(f"Overall FPS: {fmt(overall_fps, 2)}")
    print()
    print(f"Average RAM: {fmt(avg_ram, 2)} MB")
    print(f"Peak RAM: {fmt(peak_ram, 2)} MB")
    print()
    print(f"Average Power: {fmt(tegra_sum['avg_power_w'], 2)} W")
    print(f"Peak Power: {fmt(tegra_sum['peak_power_w'], 2)} W")
    print()
    print(f"Total Energy: {fmt(total_energy, 2)} J")
    print(f"Energy / image: {fmt(energy_per_image, 4)} J")
    print()
    print(f"Average CPU Temp: {fmt(tegra_sum['avg_cpu_temp_c'], 1)} C")
    print(f"Average GPU Temp: {fmt(tegra_sum['avg_gpu_temp_c'], 1)} C")
    print(f"Average TJ Temp: {fmt(tegra_sum['avg_tj_temp_c'], 1)} C")
    print(f"Peak Temp: {fmt(tegra_sum['peak_temp_c'], 1)} C")


if __name__ == "__main__":
    main()