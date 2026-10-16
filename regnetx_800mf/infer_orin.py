import os
import glob
import time
import threading
import subprocess
import argparse

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_dynamic_engine_path(engine_path):
    root, ext = os.path.splitext(engine_path)
    if root.endswith("_dynamic"):
        return engine_path
    return root + "_dynamic" + ext


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
    return engine


class TegraMonitor:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.running = False
        self.stats = []
        self.proc = None
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _worker(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
        except FileNotFoundError:
            self.proc = None
            return

        while self.running and self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                continue
            parsed = self.parse_line(line.strip())
            if parsed:
                self.stats.append(parsed)

    @staticmethod
    def _extract_temp(line, key):
        pos = line.find(key)
        if pos == -1:
            return None
        start = pos + len(key)
        end = line.find("C", start)
        if end == -1:
            return None
        try:
            return float(line[start:end])
        except Exception:
            return None

    @staticmethod
    def parse_line(line):
        data = {}

        ram_pos = line.find("RAM ")
        if ram_pos != -1:
            try:
                sub = line[ram_pos + 4:]
                used = sub.split("/")[0].strip()
                data["ram_mb"] = float(used)
            except Exception:
                pass

        p_pos = line.find("VDD_IN ")
        if p_pos != -1:
            try:
                sub = line[p_pos + 7:]
                first = sub.split("mW")[0].strip()
                data["power_w"] = float(first) / 1000.0
            except Exception:
                pass

        cpu_t = TegraMonitor._extract_temp(line, "cpu@")
        gpu_t = TegraMonitor._extract_temp(line, "gpu@")
        tj_t = TegraMonitor._extract_temp(line, "tj@")

        if cpu_t is not None:
            data["cpu_temp_c"] = cpu_t
        if gpu_t is not None:
            data["gpu_temp_c"] = gpu_t
        if tj_t is not None:
            data["tj_temp_c"] = tj_t

        return data


def get_io_tensor_names(engine):
    input_names = []
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(f"Expected 1 input tensor, got {len(input_names)}: {input_names}")
    if len(output_names) < 1:
        raise RuntimeError("No output tensor found in engine.")

    return input_names[0], output_names


def volume(shape):
    v = 1
    for x in shape:
        v *= int(x)
    return int(v)


def preprocess_regnetx_800mf(img_path, h, w):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img, dtype=np.float32)


def main(engine_path, img_dir, batch_size=1, warmup=5, print_every=10):
    engine_path = to_dynamic_engine_path(engine_path)

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    input_name, output_names = get_io_tensor_names(engine)

    raw_shape = tuple(engine.get_tensor_shape(input_name))
    if len(raw_shape) != 4:
        raise RuntimeError(f"Expected 4D input shape, got {raw_shape}")

    _, c, h, w = raw_shape
    if int(c) != 3:
        raise RuntimeError(f"Expected 3-channel input, got {raw_shape}")

    input_shape = (batch_size, int(c), int(h), int(w))
    context.set_input_shape(input_name, input_shape)

    print(f"Input tensor name: {input_name}")
    print(f"Engine path: {engine_path}")
    print(f"Engine input shape: {input_shape}")

    input_nbytes = int(volume(input_shape) * np.dtype(np.float32).itemsize)
    d_input = cuda.mem_alloc(input_nbytes)
    context.set_tensor_address(input_name, int(d_input))

    output_buffers = []
    d_outputs = []

    for out_name in output_names:
        out_shape = list(context.get_tensor_shape(out_name))
        if any(x < 0 for x in out_shape):
            raise RuntimeError(f"Dynamic output shape unresolved for tensor {out_name}: {out_shape}")

        out_shape[0] = batch_size
        out_shape = tuple(out_shape)

        out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))
        host_out = np.empty(out_shape, dtype=out_dtype)
        d_out = cuda.mem_alloc(int(host_out.nbytes))

        context.set_tensor_address(out_name, int(d_out))
        output_buffers.append((out_name, host_out))
        d_outputs.append(d_out)

        print(f"Output tensor: {out_name}, shape={out_shape}, dtype={out_dtype}")

    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths = sorted(img_paths)

    total_images = len(img_paths)
    if total_images == 0:
        raise RuntimeError(f"No images found in: {img_dir}")

    print(f"Total images found: {total_images}")
    print(f"Batch size: {batch_size}")

    warmup_imgs = []
    for i in range(min(batch_size, total_images)):
        warmup_imgs.append(preprocess_regnetx_800mf(img_paths[i], int(h), int(w)))

    while len(warmup_imgs) < batch_size:
        warmup_imgs.append(warmup_imgs[-1])

    warmup_batch = np.stack(warmup_imgs, axis=0)

    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, warmup_batch, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for (_, host_out), d_out in zip(output_buffers, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, stream)
        stream.synchronize()

    monitor = TegraMonitor(interval_ms=100)
    monitor.start()

    infer_times = []
    wall_start = time.time()

    for start_idx in range(0, total_images, batch_size):
        batch_paths = img_paths[start_idx:start_idx + batch_size]
        real_batch = len(batch_paths)

        batch_imgs = [preprocess_regnetx_800mf(p, int(h), int(w)) for p in batch_paths]

        while len(batch_imgs) < batch_size:
            batch_imgs.append(batch_imgs[-1])

        inp = np.stack(batch_imgs, axis=0)

        t0 = time.time()

        cuda.memcpy_htod_async(d_input, inp, stream)
        context.execute_async_v3(stream_handle=stream.handle)

        for (_, host_out), d_out in zip(output_buffers, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, stream)

        stream.synchronize()
        t1 = time.time()

        infer_times.append(t1 - t0)

        first_out = output_buffers[0][1]
        _ = np.argmax(first_out[:real_batch].reshape(real_batch, -1), axis=1)

        current = min(start_idx + batch_size, total_images)
        if current % print_every == 0 or current == total_images:
            print(f"{current:04d}/{total_images:04d}")

    wall_end = time.time()
    monitor.stop()

    total_all_time = wall_end - wall_start
    total_pure_infer = float(np.sum(infer_times))

    avg_infer = total_pure_infer / total_images
    avg_fps = total_images / total_pure_infer if total_pure_infer > 0 else 0.0
    overall_fps = total_images / total_all_time if total_all_time > 0 else 0.0

    stats = monitor.stats

    ram_vals = [x["ram_mb"] for x in stats if "ram_mb" in x]
    power_vals = [x["power_w"] for x in stats if "power_w" in x]
    cpu_temps = [x["cpu_temp_c"] for x in stats if "cpu_temp_c" in x]
    gpu_temps = [x["gpu_temp_c"] for x in stats if "gpu_temp_c" in x]
    tj_temps = [x["tj_temp_c"] for x in stats if "tj_temp_c" in x]

    avg_ram = float(np.mean(ram_vals)) if ram_vals else 0.0
    peak_ram = float(np.max(ram_vals)) if ram_vals else 0.0

    avg_power = float(np.mean(power_vals)) if power_vals else 0.0
    peak_power = float(np.max(power_vals)) if power_vals else 0.0

    total_energy = avg_power * total_pure_infer
    energy_per_image = total_energy / total_images if total_images > 0 else 0.0

    avg_cpu_temp = float(np.mean(cpu_temps)) if cpu_temps else 0.0
    avg_gpu_temp = float(np.mean(gpu_temps)) if gpu_temps else 0.0
    avg_tj_temp = float(np.mean(tj_temps)) if tj_temps else 0.0

    all_temps = cpu_temps + gpu_temps + tj_temps
    peak_temp = float(np.max(all_temps)) if all_temps else 0.0
    min_temp = float(np.min(all_temps)) if all_temps else 0.0

    print()
    print(f"Total images: {total_images}")
    print(f"Batch size: {batch_size}")
    print()
    print(f"Total time: {total_all_time:.3f} s")
    print(f"Total pure inference time: {total_pure_infer:.3f} s")
    print(f"Average inference time: {avg_infer * 1000:.2f} ms")
    print()
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Overall FPS: {overall_fps:.1f}")
    print()
    print(f"Average RAM: {avg_ram:.0f} MB")
    print(f"Peak RAM: {peak_ram:.0f} MB")
    print()
    print(f"Average Power: {avg_power:.2f} W")
    print(f"Peak Power: {peak_power:.2f} W")
    print()
    print(f"Total Energy: {total_energy:.2f} J")
    print(f"Energy / image: {energy_per_image:.4f} J")
    print()
    print(f"Average CPU Temp: {avg_cpu_temp:.1f} C")
    print(f"Average GPU Temp: {avg_gpu_temp:.1f} C")
    print(f"Average TJ Temp: {avg_tj_temp:.1f} C")
    print(f"Peak Temp: {peak_temp:.1f} C")
    print(f"Min Temp: {min_temp:.1f} C")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="RegNetX-800MF_dds_8gpu_dynamic.engine", help="Path to RegNetX-800MF TensorRT engine file")
    parser.add_argument("--dir", default="../Test", help="Path to image folder")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs before measurement")
    parser.add_argument("--print_every", type=int, default=10, help="Print progress every N images")
    args = parser.parse_args()

    main(
        engine_path=args.engine,
        img_dir=args.dir,
        batch_size=args.batch,
        warmup=args.warmup,
        print_every=args.print_every
    )
