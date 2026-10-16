import os
import cv2
import time
import math
import psutil
import queue
import atexit
import signal
import argparse
import threading
import subprocess
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import re


# =========================================================
# tegrastats monitor
# =========================================================
class TegraStatsMonitor:
    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.running = False

        self.power_w = []
        self.cpu_temp_c = []
        self.gpu_temp_c = []
        self.tj_temp_c = []

    def start(self):
        if self.running:
            return
        self.running = True
        cmd = ["tegrastats", "--interval", str(self.interval_ms)]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass

    def _reader(self):
        while self.running and self.proc is not None and self.proc.stdout is not None:
            line = self.proc.stdout.readline()
            if not line:
                continue

            # 功耗：VDD_IN 3846mW/3846mW 或類似格式
            m = re.search(r"VDD_IN\s+(\d+)mW(?:/\d+mW)?", line)
            if m:
                self.power_w.append(float(m.group(1)) / 1000.0)

            # 溫度：cpu@48.906C gpu@48.875C tj@48.906C
            m = re.search(r"cpu@([0-9.]+)C", line)
            if m:
                self.cpu_temp_c.append(float(m.group(1)))

            m = re.search(r"gpu@([0-9.]+)C", line)
            if m:
                self.gpu_temp_c.append(float(m.group(1)))

            m = re.search(r"tj@([0-9.]+)C", line)
            if m:
                self.tj_temp_c.append(float(m.group(1)))


# =========================================================
# TensorRT helpers
# =========================================================
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def load_engine(engine_path: str) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load engine: {engine_path}")
    return engine


def trt_nptype(dtype):
    return np.dtype(trt.nptype(dtype))


def volume(shape):
    v = 1
    for s in shape:
        v *= int(s)
    return int(v)


class TRTInference:
    def __init__(self, engine_path: str, input_h: int = 640, input_w: int = 640):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()
        self.input_h = input_h
        self.input_w = input_w

        self.tensor_mode_api = hasattr(self.engine, "num_io_tensors")
        self._setup_bindings()

    def _setup_bindings(self):
        self.bindings = []
        self.allocations = {}
        self.host_buffers = {}
        self.device_buffers = {}
        self.input_names = []
        self.output_names = []

        if self.tensor_mode_api:
            # TRT 10 style
            self.num_tensors = self.engine.num_io_tensors
            self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_tensors)]

            # 先設定動態 shape
            for name in self.tensor_names:
                mode = self.engine.get_tensor_mode(name)
                shape = list(self.engine.get_tensor_shape(name))
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                    if -1 in shape:
                        shape = [1, 3, self.input_h, self.input_w]
                        self.context.set_input_shape(name, tuple(shape))
                else:
                    self.output_names.append(name)

            for name in self.tensor_names:
                mode = self.engine.get_tensor_mode(name)
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt_nptype(self.engine.get_tensor_dtype(name))
                size = volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                self.host_buffers[name] = host_mem
                self.device_buffers[name] = device_mem
                self.allocations[name] = int(device_mem)

                if mode == trt.TensorIOMode.INPUT:
                    pass
                else:
                    pass

            # execute_async_v3 需 set_tensor_address
            for name in self.tensor_names:
                self.context.set_tensor_address(name, self.allocations[name])

        else:
            # TRT 8 style
            self.num_bindings = self.engine.num_bindings
            for i in range(self.num_bindings):
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                dtype = trt_nptype(self.engine.get_binding_dtype(i))
                shape = list(self.engine.get_binding_shape(i))

                if is_input:
                    self.input_names.append(name)
                    if -1 in shape:
                        shape = [1, 3, self.input_h, self.input_w]
                        self.context.set_binding_shape(i, tuple(shape))
                else:
                    self.output_names.append(name)

            for i in range(self.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = trt_nptype(self.engine.get_binding_dtype(i))
                shape = tuple(self.context.get_binding_shape(i))
                size = volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                self.host_buffers[name] = host_mem
                self.device_buffers[name] = device_mem
                self.bindings.append(int(device_mem))

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected 1 input, got {len(self.input_names)}: {self.input_names}")

        self.input_name = self.input_names[0]

    def infer(self, input_tensor: np.ndarray) -> float:
        """
        回傳 pure inference time（ms）
        只量 execute，不含讀圖/resize/normalize
        """
        inp = input_tensor.ravel()
        np.copyto(self.host_buffers[self.input_name], inp)

        cuda.memcpy_htod_async(self.device_buffers[self.input_name], self.host_buffers[self.input_name], self.stream)

        t0 = time.perf_counter()

        if self.tensor_mode_api:
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        if not ok:
            raise RuntimeError("TensorRT execute failed")

        # 這裡雖然不需要輸出內容，仍然做 D2H，確保執行完整
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name], self.stream)

        self.stream.synchronize()
        t1 = time.perf_counter()

        return (t1 - t0) * 1000.0


# =========================================================
# image preprocess
# =========================================================
def preprocess_bgr(img_bgr: np.ndarray, input_h: int, input_w: int, input_dtype: np.dtype) -> np.ndarray:
    img = cv2.resize(img_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # -> NCHW

    if input_dtype == np.float16:
        img = img.astype(np.float16)
    else:
        img = img.astype(np.float32)

    return np.ascontiguousarray(img)


def list_images(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
            files.append(path)
    files.sort()
    return files


def mean_or_zero(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def max_or_zero(x):
    return float(np.max(x)) if len(x) > 0 else 0.0


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True, help="Path to TensorRT engine")
    parser.add_argument("--dir", type=str, required=True, help="Image folder")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (square)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--tegrastats-interval", type=int, default=200, help="tegrastats interval ms")
    args = parser.parse_args()

    if not os.path.isfile(args.engine):
        raise FileNotFoundError(f"Engine not found: {args.engine}")
    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f"Image folder not found: {args.dir}")

    image_paths = list_images(args.dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {args.dir}")

    runner = TRTInference(args.engine, input_h=args.imgsz, input_w=args.imgsz)

    # 推斷 input dtype
    if runner.tensor_mode_api:
        input_dtype = trt_nptype(runner.engine.get_tensor_dtype(runner.input_name))
    else:
        input_idx = [i for i in range(runner.engine.num_bindings) if runner.engine.binding_is_input(i)][0]
        input_dtype = trt_nptype(runner.engine.get_binding_dtype(input_idx))

    # warmup
    warm_img = cv2.imread(image_paths[0])
    if warm_img is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")
    warm_tensor = preprocess_bgr(warm_img, args.imgsz, args.imgsz, input_dtype)

    for _ in range(max(args.warmup, 0)):
        _ = runner.infer(warm_tensor)

    monitor = TegraStatsMonitor(interval_ms=args.tegrastats_interval)
    monitor.start()

    process = psutil.Process(os.getpid())
    ram_mb_samples = []
    infer_ms_list = []

    t_total_start = time.perf_counter()

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        inp = preprocess_bgr(img, args.imgsz, args.imgsz, input_dtype)

        # RAM sample
        ram_mb_samples.append(process.memory_info().rss / 1024.0 / 1024.0)

        infer_ms = runner.infer(inp)
        infer_ms_list.append(infer_ms)

    t_total_end = time.perf_counter()

    monitor.stop()

    total_images = len(infer_ms_list)
    if total_images == 0:
        raise RuntimeError("No valid images were inferred.")

    total_pure_infer_s = sum(infer_ms_list) / 1000.0
    avg_infer_ms = sum(infer_ms_list) / total_images

    avg_fps = 1000.0 / avg_infer_ms
    overall_elapsed_s = t_total_end - t_total_start
    overall_fps = total_images / overall_elapsed_s if overall_elapsed_s > 0 else 0.0

    avg_ram_mb = mean_or_zero(ram_mb_samples)
    peak_ram_mb = max_or_zero(ram_mb_samples)

    avg_power_w = mean_or_zero(monitor.power_w)
    peak_power_w = max_or_zero(monitor.power_w)

    total_energy_j = avg_power_w * total_pure_infer_s
    energy_per_image_j = total_energy_j / total_images

    avg_cpu_temp = mean_or_zero(monitor.cpu_temp_c)
    avg_gpu_temp = mean_or_zero(monitor.gpu_temp_c)
    avg_tj_temp = mean_or_zero(monitor.tj_temp_c)
    peak_temp = max(
        max_or_zero(monitor.cpu_temp_c),
        max_or_zero(monitor.gpu_temp_c),
        max_or_zero(monitor.tj_temp_c),
    )

    print(f"Total images: {total_images}\n")

    print(f"Total pure inference time: {total_pure_infer_s:.3f} s")
    print(f"Average inference time: {avg_infer_ms:.2f} ms\n")

    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Overall FPS: {overall_fps:.1f}\n")

    print(f"Average RAM: {avg_ram_mb:.0f} MB")
    print(f"Peak RAM: {peak_ram_mb:.0f} MB\n")

    print(f"Average Power: {avg_power_w:.2f} W")
    print(f"Peak Power: {peak_power_w:.2f} W\n")

    print(f"Total Energy: {total_energy_j:.2f} J")
    print(f"Energy / image: {energy_per_image_j:.4f} J\n")

    print(f"Average CPU Temp: {avg_cpu_temp:.1f} C")
    print(f"Average GPU Temp: {avg_gpu_temp:.1f} C")
    print(f"Average TJ Temp: {avg_tj_temp:.1f} C")
    print(f"Peak Temp: {peak_temp:.1f} C")


if __name__ == "__main__":
    main()