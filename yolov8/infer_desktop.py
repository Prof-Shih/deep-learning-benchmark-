import os
import time
import threading
import subprocess
import argparse
from typing import List, Optional, Tuple

import cv2
import numpy as np
import psutil
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


ENGINE_PATH = "yolov8n_dynamic.engine"
IMAGE_DIR = "../Test"
PROGRESS_EVERY = 10
NVIDIA_SMI_INTERVAL_SEC = 0.2
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class NvidiaSMIMonitor:
    def __init__(self, interval_sec: float = 0.2):
        self.interval_sec = interval_sec
        self.running = False
        self.thread = None
        self.samples: List[Tuple[float, float]] = []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)

    def _loop(self):
        while self.running:
            try:
                cmd = (
                    "nvidia-smi "
                    "--query-gpu=power.draw,temperature.gpu "
                    "--format=csv,noheader,nounits"
                )
                result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
                text = result.decode("utf-8", errors="ignore").strip()
                if text:
                    first_line = text.splitlines()[0]
                    parts = [x.strip() for x in first_line.split(",")]
                    if len(parts) >= 2:
                        power_w = float(parts[0])
                        temp_c = float(parts[1])
                        self.samples.append((power_w, temp_c))
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def summary(self):
        if not self.samples:
            return None, None, None, None
        powers = [x[0] for x in self.samples]
        temps = [x[1] for x in self.samples]
        return (
            float(np.mean(powers)),
            float(np.max(powers)),
            float(np.mean(temps)),
            float(np.max(temps)),
        )


class TRTInfer:
    def __init__(self, engine_path: str, batch_size: int):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.batch_size = batch_size

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()
        self.is_trt10 = hasattr(self.engine, "num_io_tensors")

        self.input_name: Optional[str] = None
        self.output_names: List[str] = []

        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}
        self.bindings = []

        self.input_shape = None
        self.input_h = None
        self.input_w = None
        self.input_c = None

        self._setup()

    @staticmethod
    def _volume(shape) -> int:
        v = 1
        for s in shape:
            v *= int(s)
        return int(v)

    def _setup(self):
        if self.is_trt10:
            num_tensors = self.engine.num_io_tensors
            self.bindings = [None] * num_tensors

            for i in range(num_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = list(self.engine.get_tensor_shape(name))

                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name

                    if len(shape) != 4:
                        raise RuntimeError(f"Expected 4D input, got {shape}")

                    shape[0] = self.batch_size
                    if shape[1] == -1:
                        shape[1] = 3
                    if shape[2] == -1:
                        shape[2] = 640
                    if shape[3] == -1:
                        shape[3] = 640

                    self.context.set_input_shape(name, tuple(shape))
                    shape = list(self.context.get_tensor_shape(name))

                    self.input_shape = tuple(int(x) for x in shape)
                    _, self.input_c, self.input_h, self.input_w = self.input_shape

                else:
                    shape = list(self.context.get_tensor_shape(name))
                    if -1 in shape:
                        raise RuntimeError(f"Output shape still dynamic: {name}, {shape}")

                size = self._volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(int(host_mem.nbytes))
                self.bindings[i] = int(device_mem)

                if mode == trt.TensorIOMode.INPUT:
                    self.host_inputs[name] = host_mem
                    self.device_inputs[name] = device_mem
                else:
                    self.output_names.append(name)
                    self.host_outputs[name] = host_mem
                    self.device_outputs[name] = device_mem

            for i in range(num_tensors):
                name = self.engine.get_tensor_name(i)
                self.context.set_tensor_address(name, self.bindings[i])

        else:
            num_bindings = self.engine.num_bindings
            self.bindings = [None] * num_bindings

            for i in range(num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = list(self.engine.get_binding_shape(i))

                if self.engine.binding_is_input(i):
                    self.input_name = name

                    if len(shape) != 4:
                        raise RuntimeError(f"Expected 4D input, got {shape}")

                    shape[0] = self.batch_size
                    if shape[1] == -1:
                        shape[1] = 3
                    if shape[2] == -1:
                        shape[2] = 640
                    if shape[3] == -1:
                        shape[3] = 640

                    self.context.set_binding_shape(i, tuple(shape))
                    shape = list(self.context.get_binding_shape(i))

                    self.input_shape = tuple(int(x) for x in shape)
                    _, self.input_c, self.input_h, self.input_w = self.input_shape

                else:
                    shape = list(self.context.get_binding_shape(i))
                    if -1 in shape:
                        raise RuntimeError(f"Output shape still dynamic: {name}, {shape}")

                size = self._volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(int(host_mem.nbytes))
                self.bindings[i] = int(device_mem)

                if self.engine.binding_is_input(i):
                    self.host_inputs[name] = host_mem
                    self.device_inputs[name] = device_mem
                else:
                    self.output_names.append(name)
                    self.host_outputs[name] = host_mem
                    self.device_outputs[name] = device_mem

        if self.input_name is None:
            raise RuntimeError("No input tensor found")

        print(f"Input tensor name: {self.input_name}")
        print(f"Engine input shape: {self.input_shape}")
        print(f"Batch size: {self.batch_size}")
        print(f"Output tensors: {self.output_names}")

    def preprocess_one(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def preprocess_batch(self, imgs_bgr: List[np.ndarray]) -> np.ndarray:
        batch = np.zeros(self.input_shape, dtype=np.float32)

        for i, img in enumerate(imgs_bgr):
            batch[i] = self.preprocess_one(img)

        return np.ascontiguousarray(batch)

    def infer(self, input_tensor: np.ndarray) -> float:
        np.copyto(self.host_inputs[self.input_name], input_tensor.ravel())

        cuda.memcpy_htod_async(
            self.device_inputs[self.input_name],
            self.host_inputs[self.input_name],
            self.stream,
        )

        t0 = time.perf_counter()

        if self.is_trt10:
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            ok = self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle,
            )

        self.stream.synchronize()
        t1 = time.perf_counter()

        if not ok:
            raise RuntimeError("TensorRT inference failed")

        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.host_outputs[name],
                self.device_outputs[name],
                self.stream,
            )

        self.stream.synchronize()

        return t1 - t0


def list_images(image_dir: str) -> List[str]:
    files = []
    for name in os.listdir(image_dir):
        if name.lower().endswith(IMAGE_EXTS):
            files.append(os.path.join(image_dir, name))
    files.sort()
    return files


def fmt(v, digits=2, suffix=""):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}{suffix}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch size for TensorRT inference")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    engine_path = ENGINE_PATH
    image_dir = IMAGE_DIR
    batch_size = args.batch

    if not os.path.isfile(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    trt_model = TRTInfer(engine_path, batch_size=batch_size)
    process = psutil.Process(os.getpid())
    gpu_monitor = NvidiaSMIMonitor(interval_sec=NVIDIA_SMI_INTERVAL_SEC)

    dummy = np.zeros(trt_model.input_shape, dtype=np.float32)
    for _ in range(10):
        _ = trt_model.infer(dummy)

    total_images = 0
    total_batches = 0
    pure_times = []
    ram_samples = []

    gpu_monitor.start()
    total_start = time.perf_counter()

    for start_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start_idx:start_idx + batch_size]
        imgs = []

        for path in batch_paths:
            img = cv2.imread(path)
            if img is not None:
                imgs.append(img)

        if not imgs:
            continue

        real_batch = len(imgs)

        inp = trt_model.preprocess_batch(imgs)
        infer_t = trt_model.infer(inp)

        pure_times.append(infer_t)
        total_batches += 1
        total_images += real_batch

        ram_mb = process.memory_info().rss / (1024 * 1024)
        ram_samples.append(ram_mb)

        if total_images % PROGRESS_EVERY == 0 or total_images == len(image_paths):
            print(f"Processed {total_images}/{len(image_paths)}")

    total_end = time.perf_counter()
    gpu_monitor.stop()

    if total_images == 0:
        raise RuntimeError("No valid images were processed")

    total_elapsed = total_end - total_start
    total_pure = float(sum(pure_times))

    avg_infer_per_batch = total_pure / total_batches
    avg_infer_per_image = total_pure / total_images

    average_fps = total_images / total_pure if total_pure > 0 else 0.0
    overall_fps = total_images / total_elapsed if total_elapsed > 0 else 0.0

    avg_ram = float(np.mean(ram_samples)) if ram_samples else None
    peak_ram = float(np.max(ram_samples)) if ram_samples else None

    avg_power, peak_power, avg_gpu_temp, peak_gpu_temp = gpu_monitor.summary()

    total_energy = (avg_power * total_pure) if avg_power is not None else None
    energy_per_image = (total_energy / total_images) if total_energy is not None else None

    print("\n===== RESULT =====")
    print(f"Total images: {total_images}")
    print(f"Total batches: {total_batches}")
    print(f"Batch size: {batch_size}\n")

    print(f"Total elapsed time: {fmt(total_elapsed, 3, ' s')}")
    print(f"Total pure inference time: {fmt(total_pure, 3, ' s')}")
    print(f"Average inference time / batch: {fmt(avg_infer_per_batch * 1000.0, 2, ' ms')}")
    print(f"Average inference time / image: {fmt(avg_infer_per_image * 1000.0, 2, ' ms')}\n")

    print(f"Average FPS: {fmt(average_fps, 1)}")
    print(f"Overall FPS: {fmt(overall_fps, 1)}\n")

    print(f"Average RAM: {fmt(avg_ram, 0, ' MB')}")
    print(f"Peak RAM: {fmt(peak_ram, 0, ' MB')}\n")

    print(f"Average Power: {fmt(avg_power, 2, ' W')}")
    print(f"Peak Power: {fmt(peak_power, 2, ' W')}\n")

    print(f"Total Energy: {fmt(total_energy, 2, ' J')}")
    print(f"Energy / image: {fmt(energy_per_image, 4, ' J')}\n")

    print(f"Average GPU Temp: {fmt(avg_gpu_temp, 1, ' C')}")
    print(f"Peak GPU Temp: {fmt(peak_gpu_temp, 1, ' C')}")


if __name__ == "__main__":
    main()