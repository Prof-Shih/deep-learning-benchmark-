# infer_regnetx_800mf_trt.py

import os
import time
import glob
import argparse
import threading
import subprocess
import numpy as np
import cv2
import psutil
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =========================================================
# NVIDIA-SMI 監測執行緒
# =========================================================
class NvidiaSMIMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.running = False
        self.thread = None
        self.power_list = []
        self.temp_list = []

    def _query_nvidia_smi(self):
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            line = result.decode().strip().splitlines()[0]
            parts = [x.strip() for x in line.split(",")]
            return float(parts[0]), float(parts[1])
        except Exception:
            return None, None

    def _worker(self):
        while self.running:
            power, temp = self._query_nvidia_smi()
            if power is not None:
                self.power_list.append(power)
            if temp is not None:
                self.temp_list.append(temp)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def get_stats(self):
        avg_power = float(np.mean(self.power_list)) if len(self.power_list) > 0 else 0.0
        peak_power = float(np.max(self.power_list)) if len(self.power_list) > 0 else 0.0
        avg_temp = float(np.mean(self.temp_list)) if len(self.temp_list) > 0 else 0.0
        peak_temp = float(np.max(self.temp_list)) if len(self.temp_list) > 0 else 0.0
        return avg_power, peak_power, avg_temp, peak_temp


# =========================================================
# TensorRT Inference
# =========================================================
class TRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        self.input_name = None
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise RuntimeError("No input tensor found in engine.")

        self.stream = cuda.Stream()
        self.input_device = None
        self.output_devices = {}

    def infer(self, input_data):
        input_data = np.ascontiguousarray(input_data.astype(np.float32))

        self.context.set_input_shape(self.input_name, input_data.shape)

        actual_input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        if tuple(input_data.shape) != actual_input_shape:
            raise RuntimeError(
                f"Input shape mismatch: got {tuple(input_data.shape)}, expected {actual_input_shape}"
            )

        if self.input_device is not None:
            self.input_device.free()
        self.input_device = cuda.mem_alloc(input_data.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.input_device))

        host_outputs = []
        for out_name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(out_name))
            out_dtype = trt.nptype(self.engine.get_tensor_dtype(out_name))

            if any(d < 0 for d in out_shape):
                raise RuntimeError(f"Invalid output shape for {out_name}: {out_shape}")

            host_out = np.empty(out_shape, dtype=out_dtype)

            if out_name in self.output_devices:
                self.output_devices[out_name].free()
            dev_out = cuda.mem_alloc(host_out.nbytes)
            self.output_devices[out_name] = dev_out

            self.context.set_tensor_address(out_name, int(dev_out))
            host_outputs.append((out_name, host_out, dev_out))

        cuda.memcpy_htod_async(self.input_device, input_data, self.stream)

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        outputs = []
        for out_name, host_out, dev_out in host_outputs:
            cuda.memcpy_dtoh_async(host_out, dev_out, self.stream)
            outputs.append(host_out)

        self.stream.synchronize()
        return outputs

    def get_input_spec(self):
        dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        return self.input_name, shape, dtype


# =========================================================
# Preprocess - RegNetX-800MF
# input.1: [N, 3, 224, 224]
# output: 502 [N, 1000]
# =========================================================
def preprocess_regnetx_800mf(img, h, w):
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img


# =========================================================
# Main
# =========================================================
def main(engine_path, image_dir, batch_size=1, monitor_interval=0.2):
    if batch_size < 1:
        raise ValueError("--batch must be >= 1")

    trt_model = TRTInfer(engine_path)

    input_name, engine_input_shape, engine_input_dtype = trt_model.get_input_spec()
    print(f"Input tensor name: {input_name}")
    print(f"Engine input shape: {engine_input_shape}")
    print(f"Engine input dtype: {engine_input_dtype}")
    print(f"Batch size: {batch_size}")

    if len(engine_input_shape) != 4:
        raise RuntimeError(f"Unsupported input shape: {engine_input_shape}")

    h = engine_input_shape[2] if engine_input_shape[2] > 0 else 224
    w = engine_input_shape[3] if engine_input_shape[3] > 0 else 224

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(image_paths)

    total_images = len(image_paths)
    if total_images == 0:
        print("No images found.")
        return

    process = psutil.Process(os.getpid())

    ram_list = []
    infer_time_total = 0.0
    processed = 0

    monitor = NvidiaSMIMonitor(interval=monitor_interval)

    total_start = time.time()
    monitor.start()

    try:
        for start_idx in range(0, total_images, batch_size):
            batch_paths = image_paths[start_idx:start_idx + batch_size]

            batch_imgs = []
            valid_count = 0

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skip unreadable image: {img_path}")
                    continue

                img_tensor = preprocess_regnetx_800mf(img, h, w)
                batch_imgs.append(img_tensor)
                valid_count += 1

            if valid_count == 0:
                continue

            input_tensor = np.stack(batch_imgs, axis=0)  # NCHW

            start_infer = time.time()
            _ = trt_model.infer(input_tensor)
            infer_time = time.time() - start_infer

            infer_time_total += infer_time

            ram_mb = process.memory_info().rss / (1024 ** 2)
            ram_list.append(ram_mb)

            processed += valid_count

            if processed % 100 == 0 or processed >= total_images:
                print(f"{processed}/{total_images}")

    finally:
        monitor.stop()

    total_time = time.time() - total_start

    if processed == 0:
        print("No valid images processed.")
        return

    avg_infer_time_per_image = infer_time_total / processed
    overall_fps = processed / total_time
    infer_fps = processed / infer_time_total if infer_time_total > 0 else 0.0

    avg_ram = float(np.mean(ram_list)) if len(ram_list) > 0 else 0.0
    peak_ram = float(np.max(ram_list)) if len(ram_list) > 0 else 0.0

    avg_power, peak_power, avg_temp, peak_temp = monitor.get_stats()

    total_energy = avg_power * total_time
    energy_per_image = total_energy / processed if processed > 0 else 0.0

    print("\n========== RESULT ==========")
    print(f"Total images: {processed}")
    print(f"Batch size: {batch_size}")

    print(f"\nTotal time: {total_time:.4f} s")
    print(f"Total pure inference time: {infer_time_total:.4f} s")
    print(f"Average inference time: {avg_infer_time_per_image * 1000:.4f} ms / image")

    print(f"\nAverage FPS: {infer_fps:.2f}")
    print(f"Overall FPS: {overall_fps:.2f}")

    print(f"\nAverage RAM: {avg_ram:.2f} MB")
    print(f"Peak RAM: {peak_ram:.2f} MB")

    print(f"\nAverage Power: {avg_power:.2f} W")
    print(f"Peak Power: {peak_power:.2f} W")

    print(f"\nTotal Energy: {total_energy:.4f} J")
    print(f"Energy / image: {energy_per_image:.6f} J")

    print(f"\nAverage GPU Temp: {avg_temp:.2f} °C")
    print(f"Peak GPU Temp: {peak_temp:.2f} °C")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="RegNetX-800MF_fp16.engine")
    parser.add_argument("--dir", type=str, default="../Test")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=0.2,
        help="nvidia-smi sampling interval in seconds"
    )
    args = parser.parse_args()

    main(args.engine, args.dir, args.batch, args.monitor_interval)