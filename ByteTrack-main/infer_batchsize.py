#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import time
import math
import atexit
import signal
import shutil
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import tensorrt as trt
except Exception as e:
    print(f"[ERROR] Failed to import tensorrt: {e}")
    sys.exit(1)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except Exception as e:
    print(f"[ERROR] Failed to import pycuda: {e}")
    sys.exit(1)


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


@dataclass
class TelemetrySample:
    ram_mb: Optional[float] = None
    power_w: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    tj_temp_c: Optional[float] = None
    peak_temp_c: Optional[float] = None
    ts: float = field(default_factory=time.time)


class TegrastatsMonitor:
    def __init__(self, interval_ms: int = 200):
        self.interval_ms = max(50, int(interval_ms))
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.samples: List[TelemetrySample] = []
        self._lock = threading.Lock()

    def start(self):
        tegrastats = shutil.which("tegrastats")
        if tegrastats is None:
            raise RuntimeError("tegrastats not found in PATH")
        self.proc = subprocess.Popen(
            [tegrastats, "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                    else:
                        self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1)

    def _reader_loop(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if self.stop_event.is_set():
                break
            sample = self._parse_line(line.strip())
            if sample is not None:
                with self._lock:
                    self.samples.append(sample)

    @staticmethod
    def _find_temp(line: str, key: str) -> Optional[float]:
        m = re.search(rf"{re.escape(key)}@([0-9]+(?:\.[0-9]+)?)C", line)
        return float(m.group(1)) if m else None

    def _parse_line(self, line: str) -> Optional[TelemetrySample]:
        if not line:
            return None

        ram_mb = None
        m_ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if m_ram:
            ram_mb = float(m_ram.group(1))

        power_w = None
        m_pwr = re.search(r"VDD_IN\s+(\d+)mW(?:/(\d+)mW)?", line)
        if m_pwr:
            power_w = float(m_pwr.group(1)) / 1000.0

        cpu_temp = self._find_temp(line, "cpu")
        gpu_temp = self._find_temp(line, "gpu")
        tj_temp = self._find_temp(line, "tj")

        temps = [t for t in [cpu_temp, gpu_temp, tj_temp] if t is not None]
        peak_temp = max(temps) if temps else None

        return TelemetrySample(
            ram_mb=ram_mb,
            power_w=power_w,
            cpu_temp_c=cpu_temp,
            gpu_temp_c=gpu_temp,
            tj_temp_c=tj_temp,
            peak_temp_c=peak_temp,
        )

    def get_samples(self):
        with self._lock:
            return list(self.samples)


def avg(vals):
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def peak(vals):
    vals = [v for v in vals if v is not None]
    return float(max(vals)) if vals else None


def format_num(v, digits=2, default="N/A"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return default
    return f"{v:.{digits}f}"


def parse_hw(s: str) -> Tuple[int, int]:
    h, w = map(int, s.split(","))
    return h, w


def open_input(input_arg: str):
    if input_arg.isdigit():
        return cv2.VideoCapture(int(input_arg))
    return cv2.VideoCapture(input_arg)


def load_engine(engine_path: str):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        return engine


class TRTInfer:
    def __init__(self, engine_path: str, input_hw: Tuple[int, int], batch_size: int, warmup: int = 10):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()
        self.input_h, self.input_w = input_hw
        self.batch_size = int(batch_size)

        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]

        self.input_names = []
        self.output_names = []
        for name in self.tensor_names:
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected exactly 1 input tensor, got {len(self.input_names)}")

        self.input_name = self.input_names[0]
        self.bindings: Dict[str, Dict[str, object]] = {}

        self._set_shape_and_allocate(self.batch_size)

        dummy = np.zeros((self.batch_size, 3, self.input_h, self.input_w), dtype=np.float32)
        for _ in range(max(0, warmup)):
            self.infer(dummy)

    def _set_shape_and_allocate(self, batch_size: int):
        in_shape = list(self.engine.get_tensor_shape(self.input_name))

        if -1 in in_shape:
            self.context.set_input_shape(
                self.input_name,
                (batch_size, 3, self.input_h, self.input_w)
            )
        else:
            fixed_shape = tuple(in_shape)
            expected = (batch_size, 3, self.input_h, self.input_w)
            if fixed_shape != expected:
                raise RuntimeError(
                    f"Engine is fixed shape {fixed_shape}, but got batch/input {expected}. "
                    f"請確認 engine 是用 dynamic batch 建的。"
                )

        self.bindings.clear()

        for name in self.tensor_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Unresolved tensor shape for {name}: {shape}")

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))

            self.bindings[name] = {
                "shape": shape,
                "dtype": dtype,
                "host": host,
                "device": device,
                "nbytes": host.nbytes,
            }

    def preprocess_one(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def preprocess_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        batch = [self.preprocess_one(f) for f in frames]
        arr = np.stack(batch, axis=0)
        return np.ascontiguousarray(arr)

    def infer(self, input_tensor: np.ndarray):
        current_batch = input_tensor.shape[0]

        if tuple(input_tensor.shape) != tuple(self.bindings[self.input_name]["shape"]):
            self._set_shape_and_allocate(current_batch)

        inp = self.bindings[self.input_name]

        if input_tensor.dtype != inp["dtype"]:
            input_tensor = input_tensor.astype(inp["dtype"], copy=False)

        cuda.memcpy_htod_async(inp["device"], input_tensor, self.stream)

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        for name in self.output_names:
            out = self.bindings[name]
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        outputs = []
        for name in self.output_names:
            outputs.append(np.array(self.bindings[name]["host"], copy=True))
        return outputs


def build_argparser():
    ap = argparse.ArgumentParser("TensorRT video benchmark for Jetson Orin Nano")

    ap.add_argument("--engine", default="bytetrack_nano_mot17.engine", type=str)
    ap.add_argument("--input", default="video.mp4", type=str)
    ap.add_argument("--input-size", default="608,1088", type=parse_hw)

    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--telemetry-interval-ms", type=int, default=200)
    ap.add_argument("--progress-every", type=int, default=10)

    return ap


def main():
    args = build_argparser().parse_args()

    if args.batch <= 0:
        print("[ERROR] --batch must be >= 1")
        sys.exit(1)

    if not os.path.exists(args.engine):
        print(f"[ERROR] Engine not found: {args.engine}")
        sys.exit(1)

    cap = open_input(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input: {args.input}")
        sys.exit(1)

    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    has_total_frames = total_frames_meta > 0
    target_total = min(args.limit, total_frames_meta) if (args.limit > 0 and has_total_frames) else args.limit

    infer = TRTInfer(args.engine, args.input_size, batch_size=args.batch, warmup=args.warmup)

    mon = TegrastatsMonitor(interval_ms=args.telemetry_interval_ms)
    atexit.register(mon.stop)
    mon.start()

    frame_idx = 0
    batch_count = 0
    pure_infer_total_s = 0.0
    infer_times_ms: List[float] = []

    t_all_start = time.time()

    try:
        while True:
            frames = []

            for _ in range(args.batch):
                if args.limit > 0 and frame_idx + len(frames) >= args.limit:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)

            if len(frames) == 0:
                break

            input_tensor = infer.preprocess_batch(frames)

            t0 = time.time()
            _ = infer.infer(input_tensor)
            dt = time.time() - t0

            batch_count += 1
            frame_idx += len(frames)

            pure_infer_total_s += dt
            infer_times_ms.append(dt * 1000.0)

            should_print = (
                frame_idx == len(frames)
                or (args.progress_every > 0 and frame_idx % args.progress_every < len(frames))
                or (args.limit > 0 and frame_idx >= args.limit)
            )

            if should_print:
                if has_total_frames and args.limit > 0:
                    print(f"Processing frame: {frame_idx}/{target_total}")
                elif has_total_frames and args.limit <= 0:
                    print(f"Processing frame: {frame_idx}/{total_frames_meta}")
                elif args.limit > 0:
                    print(f"Processing frame: {frame_idx}/{args.limit}")
                else:
                    print(f"Processing frame: {frame_idx}")

    finally:
        total_elapsed_s = time.time() - t_all_start
        cap.release()
        mon.stop()

    samples = mon.get_samples()

    ram_vals = [s.ram_mb for s in samples if s.ram_mb is not None]
    power_vals = [s.power_w for s in samples if s.power_w is not None]
    cpu_temp_vals = [s.cpu_temp_c for s in samples if s.cpu_temp_c is not None]
    gpu_temp_vals = [s.gpu_temp_c for s in samples if s.gpu_temp_c is not None]
    tj_temp_vals = [s.tj_temp_c for s in samples if s.tj_temp_c is not None]
    peak_temp_vals = [s.peak_temp_c for s in samples if s.peak_temp_c is not None]

    avg_batch_infer_ms = avg(infer_times_ms) if infer_times_ms else 0.0
    avg_infer_ms_per_frame = (pure_infer_total_s / frame_idx * 1000.0) if frame_idx > 0 else 0.0

    avg_fps = (frame_idx / pure_infer_total_s) if pure_infer_total_s > 0 else 0.0
    overall_fps = (frame_idx / total_elapsed_s) if total_elapsed_s > 0 else 0.0

    avg_ram = avg(ram_vals)
    peak_ram = peak(ram_vals)

    avg_power = avg(power_vals)
    peak_power = peak(power_vals)

    total_energy_j = (avg_power * pure_infer_total_s) if avg_power is not None else None
    energy_per_frame_j = (total_energy_j / frame_idx) if (total_energy_j is not None and frame_idx > 0) else None

    avg_cpu_temp = avg(cpu_temp_vals)
    avg_gpu_temp = avg(gpu_temp_vals)
    avg_tj_temp = avg(tj_temp_vals)
    peak_temp = peak(peak_temp_vals)

    print("=" * 56)
    print(f"Batch size: {args.batch}")
    print(f"Total batches: {batch_count}")
    print(f"Total frames: {frame_idx}")
    print("")
    print(f"Total elapsed time: {format_num(total_elapsed_s, 3)} s")
    print(f"Total pure inference time: {format_num(pure_infer_total_s, 3)} s")
    print(f"Average batch inference time: {format_num(avg_batch_infer_ms, 2)} ms")
    print(f"Average inference time / frame: {format_num(avg_infer_ms_per_frame, 2)} ms")
    print("")
    print(f"Average FPS: {format_num(avg_fps, 1)}")
    print(f"Overall FPS: {format_num(overall_fps, 1)}")
    print("")
    print(f"Average RAM: {format_num(avg_ram, 0)} MB")
    print(f"Peak RAM: {format_num(peak_ram, 0)} MB")
    print("")
    print(f"Average Power: {format_num(avg_power, 2)} W")
    print(f"Peak Power: {format_num(peak_power, 2)} W")
    print("")
    print(f"Total Energy: {format_num(total_energy_j, 2)} J")
    print(f"Energy / frame: {format_num(energy_per_frame_j, 4)} J")
    print("")
    print(f"Average CPU Temp: {format_num(avg_cpu_temp, 1)} C")
    print(f"Average GPU Temp: {format_num(avg_gpu_temp, 1)} C")
    print(f"Average TJ Temp: {format_num(avg_tj_temp, 1)} C")
    print(f"Peak Temp: {format_num(peak_temp, 1)} C")
    print("=" * 56)


if __name__ == "__main__":
    main()