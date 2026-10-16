import cv2
import time
import argparse
import numpy as np
import subprocess
import psutil
import threading

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# ======================
# GPU Monitor
# ======================
class GPUMonitor:
    def __init__(self):
        self.temps = []
        self.powers = []
        self.running = True

    def query(self):
        try:
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits",
                shell=True
            ).decode().strip()
            temp, power = result.split(", ")
            return float(temp), float(power)
        except:
            return 0.0, 0.0

    def run(self):
        while self.running:
            t, p = self.query()
            self.temps.append(t)
            self.powers.append(p)
            time.sleep(0.5)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()


# ======================
# TensorRT v3 Inference
# ======================
class TRTInfer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.stream = cuda.Stream()

        # 取得 input / output
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = int(np.prod(shape))
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
                self.input_dtype = dtype
                self.input_mem = device_mem
            else:
                self.output_name = name
                self.output_shape = shape
                self.output_dtype = dtype
                self.output_mem = device_mem

            # ⭐ TRT v3 必要
            self.context.set_tensor_address(name, int(device_mem))

        print("Input shape:", self.input_shape)

    def infer(self, img):
        cuda.memcpy_htod_async(self.input_mem, img, self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        output = np.empty(self.output_shape, dtype=self.output_dtype)
        cuda.memcpy_dtoh_async(output, self.output_mem, self.stream)

        self.stream.synchronize()
        return output


# ======================
# Preprocess
# ======================
def preprocess(frame, input_shape):
    _, c, h, w = input_shape

    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return np.ascontiguousarray(img)


# ======================
# Main
# ======================
def main(engine_path, video_path, limit):

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video open failed"

    trt_model = TRTInfer(engine_path)

    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    frame_count = 0
    infer_time_total = 0
    ram_usage = []

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if limit > 0 and frame_count >= limit:
            break

        input_tensor = preprocess(frame, trt_model.input_shape)

        t0 = time.time()
        _ = trt_model.infer(input_tensor)
        t1 = time.time()

        infer_time_total += (t1 - t0)

        ram_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"{frame_count} frames processed")

    total_time = time.time() - start_time

    gpu_monitor.stop()

    # ======================
    # Stats
    # ======================
    avg_inf_time = infer_time_total / frame_count
    inf_fps = 1.0 / avg_inf_time
    overall_fps = frame_count / total_time

    avg_ram = np.mean(ram_usage)
    peak_ram = np.max(ram_usage)

    temps = gpu_monitor.temps
    powers = gpu_monitor.powers

    avg_temp = np.mean(temps)
    max_temp = np.max(temps)
    min_temp = np.min(temps)

    avg_power = np.mean(powers)
    total_energy = avg_power * total_time

    print("\n===== RESULT =====")
    print(f"Total frames: {frame_count}")

    print(f"\nTotal inference time: {infer_time_total:.3f} s")
    print(f"Average inference time: {avg_inf_time*1000:.3f} ms")

    print(f"\nInference FPS: {inf_fps:.2f}")
    print(f"Overall FPS: {overall_fps:.2f}")

    print(f"\nAverage RAM: {avg_ram:.1f} MB")
    print(f"Peak RAM: {peak_ram:.1f} MB")

    print(f"\nTemperature (C): avg={avg_temp:.1f}, max={max_temp:.1f}, min={min_temp:.1f}")

    print(f"\nAverage Power: {avg_power:.2f} W")
    print(f"Total Energy: {total_energy:.2f} J")


# ======================
# 固定路徑 + limit
# ======================
if __name__ == "__main__":

    ENGINE_PATH = "nano_mot17_608,1088/bytetrack_nano_mot17.engine"
    VIDEO_PATH = "test.mp4"

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    print("Engine:", ENGINE_PATH)
    print("Video:", VIDEO_PATH)

    main(ENGINE_PATH, VIDEO_PATH, args.limit)