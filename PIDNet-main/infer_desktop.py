import os
import sys
import time
import ctypes
import ctypes.util
import threading
import subprocess
import argparse
from pathlib import Path

import cv2
import psutil
import numpy as np
import tensorrt as trt


# =========================
# Fixed paths
# =========================
ENGINE_PATH = "PIDNet_S_Cityscapes_val_dynamic.engine"
IMAGE_DIR = "../Test"
PROGRESS_EVERY = 10

# PIDNet / Cityscapes 常用輸入尺寸
DEFAULT_INPUT_H = 1024
DEFAULT_INPUT_W = 2048

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# =========================
# CUDA Runtime Helper (Linux Docker)
# =========================
class CudaRT:
    def __init__(self):
        self.lib = None

        linux_candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so.11.0",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
            "/usr/lib/x86_64-linux-gnu/libcudart.so.11.0",
        ]

        for libname in linux_candidates:
            if not libname:
                continue
            try:
                self.lib = ctypes.CDLL(libname)
                print(f"[CUDA] Loaded runtime: {libname}")
                break
            except Exception:
                continue

        if self.lib is None:
            raise RuntimeError(
                "Cannot load CUDA runtime library (libcudart.so). "
                "Please check CUDA runtime inside container."
            )

        self.cudaMemcpyHostToDevice = 1
        self.cudaMemcpyDeviceToHost = 2

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int

        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int

        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int

        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int

        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        err = self.lib.cudaMalloc(ctypes.byref(ptr), nbytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc failed with error code {err}")
        return ptr.value

    def free(self, ptr: int):
        if ptr:
            self.lib.cudaFree(ctypes.c_void_p(ptr))

    def memcpy_htod(self, dst_ptr: int, src_np: np.ndarray):
        src_ptr = src_np.ctypes.data_as(ctypes.c_void_p)
        err = self.lib.cudaMemcpy(
            ctypes.c_void_p(dst_ptr),
            src_ptr,
            src_np.nbytes,
            self.cudaMemcpyHostToDevice
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed with error code {err}")

    def memcpy_dtoh(self, dst_np: np.ndarray, src_ptr: int):
        dst_ptr = dst_np.ctypes.data_as(ctypes.c_void_p)
        err = self.lib.cudaMemcpy(
            dst_ptr,
            ctypes.c_void_p(src_ptr),
            dst_np.nbytes,
            self.cudaMemcpyDeviceToHost
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy D2H failed with error code {err}")

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        err = self.lib.cudaStreamCreate(ctypes.byref(stream))
        if err != 0:
            raise RuntimeError(f"cudaStreamCreate failed with error code {err}")
        return stream.value

    def destroy_stream(self, stream: int):
        if stream:
            self.lib.cudaStreamDestroy(ctypes.c_void_p(stream))

    def synchronize_stream(self, stream: int):
        err = self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream))
        if err != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed with error code {err}")


cuda_rt = CudaRT()


# =========================
# Monitor
# =========================
class MonitorThread(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = False
        self.samples_ram_mb = []
        self.samples_temp_c = []
        self.samples_power_w = []

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        proc = psutil.Process(os.getpid())

        while self.running:
            try:
                ram_mb = proc.memory_info().rss / (1024 * 1024)
                self.samples_ram_mb.append(ram_mb)
            except Exception:
                pass

            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    encoding="utf-8",
                ).strip()

                if out:
                    first_line = out.splitlines()[0].strip()
                    parts = [x.strip() for x in first_line.split(",")]
                    if len(parts) >= 2:
                        temp_c = float(parts[0])
                        power_w = float(parts[1])
                        self.samples_temp_c.append(temp_c)
                        self.samples_power_w.append(power_w)
            except Exception:
                pass

            time.sleep(self.interval)


# =========================
# Helpers
# =========================
def trt_nptype(dtype):
    return np.dtype(trt.nptype(dtype))


def volume(shape):
    v = 1
    for x in shape:
        v *= int(x)
    return int(v)


def list_images(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
    paths.sort()
    return paths


def replace_dynamic_shape(shape, batch_size=1):
    fixed = []
    for i, d in enumerate(shape):
        if d != -1:
            fixed.append(int(d))
        else:
            if i == 0:
                fixed.append(batch_size)
            elif i == 1:
                fixed.append(3)
            elif i == 2:
                fixed.append(DEFAULT_INPUT_H)
            elif i == 3:
                fixed.append(DEFAULT_INPUT_W)
            else:
                fixed.append(1)
    return tuple(fixed)


def preprocess_pidnet(image_bgr, input_shape):
    if len(input_shape) != 4:
        raise RuntimeError(f"Unexpected input shape: {input_shape}")

    _, c, h, w = input_shape

    if c != 3:
        raise RuntimeError(f"Expected 3 channels, got: {input_shape}")

    if h == -1:
        h = DEFAULT_INPUT_H
    if w == -1:
        w = DEFAULT_INPUT_W

    img = cv2.resize(image_bgr, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)


def make_batch(image_paths, input_shape):
    batch_imgs = []

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        img = preprocess_pidnet(image, input_shape)
        batch_imgs.append(img)

    if not batch_imgs:
        return None

    return np.ascontiguousarray(np.stack(batch_imgs, axis=0))


# =========================
# TensorRT Inference
# =========================
class TRTInfer:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda_rt.create_stream()

        self.d_input = None
        self.input_nbytes = 0
        self.output_device_ptrs = {}
        self.host_outputs = {}

        self.is_trt10 = hasattr(self.engine, "num_io_tensors")

        if self.is_trt10:
            self._setup_trt10()
        else:
            self._setup_legacy()

    def _setup_trt10(self):
        self.tensor_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
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
            raise RuntimeError(f"Expected 1 input tensor, got {len(self.input_names)}")

        self.input_name = self.input_names[0]
        self.input_dtype = trt_nptype(self.engine.get_tensor_dtype(self.input_name))
        self.engine_input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

        if -1 in self.engine_input_shape:
            init_shape = replace_dynamic_shape(self.engine_input_shape, batch_size=1)
            self.context.set_input_shape(self.input_name, init_shape)

        self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))

        self._allocate_input()
        self._allocate_outputs_trt10()

    def _setup_legacy(self):
        self.num_bindings = self.engine.num_bindings
        self.binding_names = [
            self.engine.get_binding_name(i)
            for i in range(self.num_bindings)
        ]

        input_indices = []
        output_indices = []

        for i in range(self.num_bindings):
            if self.engine.binding_is_input(i):
                input_indices.append(i)
            else:
                output_indices.append(i)

        if len(input_indices) != 1:
            raise RuntimeError(f"Expected 1 input binding, got {len(input_indices)}")

        self.input_idx = input_indices[0]
        self.output_idxs = output_indices
        self.bindings = [0] * self.num_bindings

        self.input_name = self.engine.get_binding_name(self.input_idx)
        self.input_dtype = trt_nptype(self.engine.get_binding_dtype(self.input_idx))
        self.engine_input_shape = tuple(self.context.get_binding_shape(self.input_idx))

        if -1 in self.engine_input_shape:
            init_shape = replace_dynamic_shape(self.engine_input_shape, batch_size=1)
            self.context.set_binding_shape(self.input_idx, init_shape)

        self.input_shape = tuple(self.context.get_binding_shape(self.input_idx))

        self._allocate_input()
        self.bindings[self.input_idx] = int(self.d_input)

        self._allocate_outputs_legacy()

    def _allocate_input(self):
        if self.d_input:
            cuda_rt.free(self.d_input)

        self.input_nbytes = volume(self.input_shape) * np.dtype(self.input_dtype).itemsize
        self.d_input = cuda_rt.malloc(self.input_nbytes)

        if self.is_trt10:
            self.context.set_tensor_address(self.input_name, self.d_input)

    def _free_outputs(self):
        for _, ptr in self.output_device_ptrs.items():
            cuda_rt.free(ptr)

        self.output_device_ptrs = {}
        self.host_outputs = {}

    def _allocate_outputs_trt10(self):
        self._free_outputs()

        for out_name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(out_name))
            out_dtype = trt_nptype(self.engine.get_tensor_dtype(out_name))

            if -1 in out_shape:
                raise RuntimeError(
                    f"Output shape still dynamic after setting input shape: {out_name} {out_shape}"
                )

            out_nbytes = volume(out_shape) * np.dtype(out_dtype).itemsize
            d_out = cuda_rt.malloc(out_nbytes)

            self.output_device_ptrs[out_name] = d_out
            self.context.set_tensor_address(out_name, d_out)
            self.host_outputs[out_name] = np.empty(out_shape, dtype=out_dtype)

    def _allocate_outputs_legacy(self):
        self._free_outputs()

        for out_idx in self.output_idxs:
            out_name = self.engine.get_binding_name(out_idx)
            out_shape = tuple(self.context.get_binding_shape(out_idx))
            out_dtype = trt_nptype(self.engine.get_binding_dtype(out_idx))

            if -1 in out_shape:
                raise RuntimeError(
                    f"Output shape still dynamic after setting input shape: {out_name} {out_shape}"
                )

            out_nbytes = volume(out_shape) * np.dtype(out_dtype).itemsize
            d_out = cuda_rt.malloc(out_nbytes)

            self.output_device_ptrs[out_name] = d_out
            self.bindings[out_idx] = int(d_out)
            self.host_outputs[out_name] = np.empty(out_shape, dtype=out_dtype)

    def set_input_shape_if_needed(self, input_tensor):
        new_shape = tuple(input_tensor.shape)

        if new_shape == tuple(self.input_shape):
            return

        # static engine 不能換 batch
        if -1 not in self.engine_input_shape:
            raise RuntimeError(
                f"Engine is static shape {self.engine_input_shape}, "
                f"but got input {new_shape}. "
                f"請用 --batch 1，或重新 build dynamic batch engine。"
            )

        if self.is_trt10:
            self.context.set_input_shape(self.input_name, new_shape)
            self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))

            self._allocate_input()
            self._allocate_outputs_trt10()

        else:
            self.context.set_binding_shape(self.input_idx, new_shape)
            self.input_shape = tuple(self.context.get_binding_shape(self.input_idx))

            self._allocate_input()
            self.bindings[self.input_idx] = int(self.d_input)

            self._allocate_outputs_legacy()

    def infer(self, input_tensor: np.ndarray):
        if input_tensor.dtype != self.input_dtype:
            input_tensor = input_tensor.astype(self.input_dtype, copy=False)

        input_tensor = np.ascontiguousarray(input_tensor)

        self.set_input_shape_if_needed(input_tensor)

        if input_tensor.nbytes > self.input_nbytes:
            raise RuntimeError(
                f"Input tensor too large. Input nbytes={input_tensor.nbytes}, "
                f"allocated={self.input_nbytes}"
            )

        cuda_rt.memcpy_htod(self.d_input, input_tensor)

        if self.is_trt10:
            ok = self.context.execute_async_v3(self.stream)
            if not ok:
                raise RuntimeError("execute_async_v3 failed")
        else:
            if hasattr(self.context, "execute_async_v2"):
                ok = self.context.execute_async_v2(
                    bindings=self.bindings,
                    stream_handle=self.stream
                )
                if not ok:
                    raise RuntimeError("execute_async_v2 failed")
            else:
                ok = self.context.execute_v2(bindings=self.bindings)
                if not ok:
                    raise RuntimeError("execute_v2 failed")

        cuda_rt.synchronize_stream(self.stream)

        outputs = {}
        for out_name, h_out in self.host_outputs.items():
            cuda_rt.memcpy_dtoh(h_out, self.output_device_ptrs[out_name])
            outputs[out_name] = h_out.copy()

        return outputs

    def release(self):
        try:
            if self.d_input:
                cuda_rt.free(self.d_input)
                self.d_input = None

            self._free_outputs()

        except Exception:
            pass

        try:
            cuda_rt.destroy_stream(self.stream)
        except Exception:
            pass


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if args.batch < 1:
        print("batch must be >= 1")
        sys.exit(1)

    if not os.path.exists(ENGINE_PATH):
        print(f"Engine not found: {ENGINE_PATH}")
        sys.exit(1)

    if not os.path.isdir(IMAGE_DIR):
        print(f"Image dir not found: {IMAGE_DIR}")
        sys.exit(1)

    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        print(f"No images found in: {IMAGE_DIR}")
        sys.exit(1)

    trt_model = TRTInfer(ENGINE_PATH)

    print(f"Engine: {ENGINE_PATH}")
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Engine input shape: {trt_model.engine_input_shape}")
    print(f"Runtime input shape: {trt_model.input_shape}")
    print(f"Batch size: {args.batch}")
    print(f"Total images: {len(image_paths)}")
    print("Start inference...\n")

    pure_infer_times = []
    processed_count = 0
    monitor = MonitorThread(interval=0.2)

    try:
        monitor.start()
        total_start = time.perf_counter()

        for start in range(0, len(image_paths), args.batch):
            batch_paths = image_paths[start:start + args.batch]

            input_tensor = make_batch(batch_paths, trt_model.input_shape)
            if input_tensor is None:
                continue

            current_batch = input_tensor.shape[0]

            t0 = time.perf_counter()
            _ = trt_model.infer(input_tensor)
            t1 = time.perf_counter()

            pure_infer_times.append(t1 - t0)
            processed_count += current_batch

            if processed_count % PROGRESS_EVERY == 0 or processed_count == len(image_paths):
                print(f"{processed_count}/{len(image_paths)}")

        total_end = time.perf_counter()

    finally:
        monitor.stop()
        monitor.join(timeout=1.0)
        trt_model.release()

    if processed_count == 0:
        print("No valid images processed.")
        sys.exit(1)

    total_time = total_end - total_start
    pure_total = float(sum(pure_infer_times))
    avg_infer = pure_total / processed_count if processed_count else 0.0

    overall_fps = processed_count / total_time if total_time > 0 else 0.0
    pure_fps = processed_count / pure_total if pure_total > 0 else 0.0

    ram_avg = float(np.mean(monitor.samples_ram_mb)) if monitor.samples_ram_mb else 0.0
    ram_peak = float(np.max(monitor.samples_ram_mb)) if monitor.samples_ram_mb else 0.0

    temp_avg = float(np.mean(monitor.samples_temp_c)) if monitor.samples_temp_c else 0.0
    temp_max = float(np.max(monitor.samples_temp_c)) if monitor.samples_temp_c else 0.0
    temp_min = float(np.min(monitor.samples_temp_c)) if monitor.samples_temp_c else 0.0

    power_avg = float(np.mean(monitor.samples_power_w)) if monitor.samples_power_w else 0.0
    power_peak = float(np.max(monitor.samples_power_w)) if monitor.samples_power_w else 0.0

    total_energy = power_avg * total_time if power_avg > 0 else 0.0
    energy_per_image = total_energy / processed_count if processed_count > 0 else 0.0

    print("\n========== Final Results ==========")
    print(f"Total images: {processed_count}")
    print()
    print(f"Total inference time: {total_time:.3f} s")
    print(f"Pure inference time:  {pure_total:.3f} s")
    print(f"Average inference:    {avg_infer * 1000:.2f} ms")
    print()
    print(f"Overall FPS:          {overall_fps:.2f}")
    print(f"Pure inference FPS:   {pure_fps:.2f}")
    print()
    print(f"Temperature max:      {temp_max:.2f} C")
    print(f"Temperature min:      {temp_min:.2f} C")
    print(f"Temperature avg:      {temp_avg:.2f} C")
    print()
    print(f"Average RAM:          {ram_avg:.2f} MB")
    print(f"Peak RAM:             {ram_peak:.2f} MB")
    print()
    print(f"Average Power:        {power_avg:.2f} W")
    print(f"Peak Power:           {power_peak:.2f} W")
    print(f"Total Energy:         {total_energy:.2f} J")
    print(f"Energy / image:       {energy_per_image:.4f} J/image")


if __name__ == "__main__":
    main()