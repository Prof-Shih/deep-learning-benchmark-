import os
import re
import cv2
import time
import math
import queue
import signal
import argparse
import threading
import subprocess
from typing import List, Tuple, Optional

import numpy as np
import psutil

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(img_dir: str) -> List[str]:
    files = []
    for name in os.listdir(img_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(os.path.join(img_dir, name))
    files.sort()
    return files


def find_tegrastats() -> Optional[str]:
    candidates = [
        "tegrastats",
        "/usr/bin/tegrastats",
        "/bin/tegrastats",
        "/usr/sbin/tegrastats",
    ]
    for c in candidates:
        if os.path.isabs(c):
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        else:
            from shutil import which
            p = which(c)
            if p:
                return p
    return None


class TelemetryMonitor:
    """
    讀 tegrastats:
      RAM 2787/7620MB
      cpu@48.906C gpu@48.875C tj@48.906C
      VDD_IN 3846mW/3846mW
    """

    def __init__(self, tegrastats: str = "auto", interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.tegrastats = find_tegrastats() if tegrastats == "auto" else tegrastats
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.samples = []
        self.lock = threading.Lock()

    def start(self):
        self.stop_event.clear()

        if self.tegrastats is None:
            # fallback: 只抓 RAM
            self.thread = threading.Thread(target=self._fallback_loop, daemon=True)
            self.thread.start()
            return

        cmd = [self.tegrastats, "--interval", str(self.interval_ms)]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid
        )
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass

        if self.thread is not None:
            self.thread.join(timeout=2)

    def _fallback_loop(self):
        p = psutil.Process(os.getpid())
        while not self.stop_event.is_set():
            ram_mb = p.memory_info().rss / (1024 * 1024)
            sample = {
                "ram_mb": ram_mb,
                "power_w": None,
                "cpu_temp": None,
                "gpu_temp": None,
                "tj_temp": None,
                "peak_temp": None,
            }
            with self.lock:
                self.samples.append(sample)
            time.sleep(self.interval_ms / 1000.0)

    def _read_loop(self):
        while not self.stop_event.is_set() and self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            sample = self._parse_tegrastats_line(line.strip())
            if sample is not None:
                with self.lock:
                    self.samples.append(sample)

    @staticmethod
    def _parse_tegrastats_line(line: str):
        # RAM
        ram_mb = None
        m_ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if m_ram:
            ram_mb = float(m_ram.group(1))

        # Power: VDD_IN 3846mW/3846mW
        power_w = None
        m_pwr = re.search(r"VDD_IN\s+(\d+)mW(?:/(\d+)mW)?", line)
        if m_pwr:
            power_w = float(m_pwr.group(1)) / 1000.0

        # Temps
        cpu_temp = None
        gpu_temp = None
        tj_temp = None

        m_cpu = re.search(r"cpu@([0-9.]+)C", line)
        if m_cpu:
            cpu_temp = float(m_cpu.group(1))

        m_gpu = re.search(r"gpu@([0-9.]+)C", line)
        if m_gpu:
            gpu_temp = float(m_gpu.group(1))

        m_tj = re.search(r"tj@([0-9.]+)C", line)
        if m_tj:
            tj_temp = float(m_tj.group(1))

        temps = [t for t in [cpu_temp, gpu_temp, tj_temp] if t is not None]
        peak_temp = max(temps) if temps else None

        return {
            "ram_mb": ram_mb,
            "power_w": power_w,
            "cpu_temp": cpu_temp,
            "gpu_temp": gpu_temp,
            "tj_temp": tj_temp,
            "peak_temp": peak_temp,
        }

    def summary(self):
        with self.lock:
            samples = list(self.samples)

        if not samples:
            return {
                "avg_ram_mb": None,
                "peak_ram_mb": None,
                "avg_power_w": None,
                "peak_power_w": None,
                "avg_cpu_temp": None,
                "avg_gpu_temp": None,
                "avg_tj_temp": None,
                "peak_temp": None,
            }

        def avg_of(key):
            vals = [s[key] for s in samples if s.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        def max_of(key):
            vals = [s[key] for s in samples if s.get(key) is not None]
            return max(vals) if vals else None

        return {
            "avg_ram_mb": avg_of("ram_mb"),
            "peak_ram_mb": max_of("ram_mb"),
            "avg_power_w": avg_of("power_w"),
            "peak_power_w": max_of("power_w"),
            "avg_cpu_temp": avg_of("cpu_temp"),
            "avg_gpu_temp": avg_of("gpu_temp"),
            "avg_tj_temp": avg_of("tj_temp"),
            "peak_temp": max_of("peak_temp"),
        }


def fmt_num(v, digits=2, suffix=""):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}{suffix}"


class TRTInfer:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()

        self.is_trt10 = hasattr(self.engine, "num_io_tensors")
        self.input_name = None
        self.output_names = []

        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}
        self.bindings = []

        self.input_h = None
        self.input_w = None
        self.input_c = None
        self.batch_size = 1

        self._setup_bindings()

    def _volume(self, shape):
        v = 1
        for s in shape:
            v *= int(s)
        return int(v)

    def _setup_bindings(self):
        if self.is_trt10:
            tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.bindings = [None] * self.engine.num_io_tensors

            for i, name in enumerate(tensor_names):
                mode = self.engine.get_tensor_mode(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = list(self.engine.get_tensor_shape(name))

                if -1 in shape:
                    if mode == trt.TensorIOMode.INPUT:
                        # 預設動態 shape 給 1x3x640x640
                        guess = [1, 3, 640, 640]
                        self.context.set_input_shape(name, guess)
                        shape = list(self.context.get_tensor_shape(name))
                    else:
                        shape = list(self.context.get_tensor_shape(name))

                size = self._volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[i] = int(device_mem)

                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    self.host_inputs[name] = host_mem
                    self.device_inputs[name] = device_mem
                    if len(shape) == 4:
                        self.batch_size, self.input_c, self.input_h, self.input_w = map(int, shape)
                else:
                    self.output_names.append(name)
                    self.host_outputs[name] = host_mem
                    self.device_outputs[name] = device_mem

            for i, name in enumerate(tensor_names):
                self.context.set_tensor_address(name, self.bindings[i])

        else:
            nb = self.engine.num_bindings
            self.bindings = [None] * nb

            for i in range(nb):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = list(self.engine.get_binding_shape(i))

                if self.engine.binding_is_input(i):
                    self.input_name = name
                    if -1 in shape:
                        guess = [1, 3, 640, 640]
                        self.context.set_binding_shape(i, guess)
                        shape = list(self.context.get_binding_shape(i))
                    if len(shape) == 4:
                        self.batch_size, self.input_c, self.input_h, self.input_w = map(int, shape)

                else:
                    if -1 in shape:
                        shape = list(self.context.get_binding_shape(i))

                size = self._volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[i] = int(device_mem)

                if self.engine.binding_is_input(i):
                    self.host_inputs[name] = host_mem
                    self.device_inputs[name] = device_mem
                else:
                    self.output_names.append(name)
                    self.host_outputs[name] = host_mem
                    self.device_outputs[name] = device_mem

        if self.input_h is None or self.input_w is None:
            raise RuntimeError("Could not determine engine input shape")

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        純 benchmark，用直接 resize，不做 letterbox。
        輸出 NCHW float32，範圍 0~1
        """
        img = cv2.resize(img_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # -> NCHW
        return np.ascontiguousarray(img)

    def infer(self, input_tensor: np.ndarray) -> float:
        """
        回傳 pure inference time (seconds)
        """
        inp = self.host_inputs[self.input_name]
        np.copyto(inp, input_tensor.ravel())

        cuda.memcpy_htod_async(self.device_inputs[self.input_name], inp, self.stream)

        t0 = time.perf_counter()

        if self.is_trt10:
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        self.stream.synchronize()
        t1 = time.perf_counter()

        if not ok:
            raise RuntimeError("TensorRT inference failed")

        # 為了流程完整，把 output 拉回來；但不 decode
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.host_outputs[name], self.device_outputs[name], self.stream)
        self.stream.synchronize()

        return t1 - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TensorRT engine path")
    ap.add_argument("--dir", required=True, help="Image directory")
    ap.add_argument("--max-images", type=int, default=100, help="How many images to benchmark")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    ap.add_argument("--tegrastats", default="auto", help="'auto' or absolute path to tegrastats")
    ap.add_argument("--interval-ms", type=int, default=200, help="Telemetry polling interval")
    args = ap.parse_args()

    img_paths = list_images(args.dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {args.dir}")

    img_paths = img_paths[:args.max_images]

    trt_infer = TRTInfer(args.engine)

    # warmup
    dummy = np.zeros((1, trt_infer.input_c, trt_infer.input_h, trt_infer.input_w), dtype=np.float32)
    for _ in range(args.warmup):
        _ = trt_infer.infer(dummy)

    monitor = TelemetryMonitor(tegrastats=args.tegrastats, interval_ms=args.interval_ms)
    monitor.start()

    pure_times = []
    total_start = time.perf_counter()

    total_images = 0
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        inp = trt_infer.preprocess(img)
        pure_t = trt_infer.infer(inp)
        pure_times.append(pure_t)
        total_images += 1

    total_end = time.perf_counter()
    monitor.stop()

    total_elapsed = total_end - total_start
    total_pure = sum(pure_times)
    avg_pure = (total_pure / total_images) if total_images > 0 else 0.0

    avg_fps = (1.0 / avg_pure) if avg_pure > 0 else 0.0
    overall_fps = (total_images / total_elapsed) if total_elapsed > 0 else 0.0

    telem = monitor.summary()

    avg_power = telem["avg_power_w"]
    total_energy = (avg_power * total_pure) if avg_power is not None else None
    energy_per_image = (total_energy / total_images) if (total_energy is not None and total_images > 0) else None

    print(f"Total images: {total_images}")
    print()
    print(f"Total pure inference time: {fmt_num(total_pure, 3, ' s')}")
    print(f"Average inference time: {fmt_num(avg_pure * 1000.0, 2, ' ms')}")
    print()
    print(f"Average FPS: {fmt_num(avg_fps, 1)}")
    print(f"Overall FPS: {fmt_num(overall_fps, 1)}")
    print()
    print(f"Average RAM: {fmt_num(telem['avg_ram_mb'], 0, ' MB')}")
    print(f"Peak RAM: {fmt_num(telem['peak_ram_mb'], 0, ' MB')}")
    print()
    print(f"Average Power: {fmt_num(avg_power, 2, ' W')}")
    print(f"Peak Power: {fmt_num(telem['peak_power_w'], 2, ' W')}")
    print()
    print(f"Total Energy: {fmt_num(total_energy, 2, ' J')}")
    print(f"Energy / image: {fmt_num(energy_per_image, 4, ' J')}")
    print()
    print(f"Average CPU Temp: {fmt_num(telem['avg_cpu_temp'], 1, ' C')}")
    print(f"Average GPU Temp: {fmt_num(telem['avg_gpu_temp'], 1, ' C')}")
    print(f"Average TJ Temp: {fmt_num(telem['avg_tj_temp'], 1, ' C')}")
    print(f"Peak Temp: {fmt_num(telem['peak_temp'], 1, ' C')}")


if __name__ == "__main__":
    main()