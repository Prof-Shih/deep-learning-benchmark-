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
# FairMOT Config
# ======================
FAIRMOT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
FAIRMOT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 你前面 FairMOT ONNX 是 1x3x608x1088
FAIRMOT_INPUT_H = 608
FAIRMOT_INPUT_W = 1088


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

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.stream = cuda.Stream()

        self.input_name = None
        self.input_shape = None
        self.input_dtype = None
        self.input_mem = None

        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes = {}
        self.output_mems = {}

        # ======================
        # 先找 input
        # ======================
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                shape = tuple(self.engine.get_tensor_shape(name))

                # dynamic shape 情況
                if any(dim < 0 for dim in shape):
                    self.context.set_input_shape(
                        name,
                        (1, 3, FAIRMOT_INPUT_H, FAIRMOT_INPUT_W)
                    )
                    shape = tuple(self.context.get_tensor_shape(name))

                self.input_shape = shape
                self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(name))

        if self.input_name is None:
            raise RuntimeError("No input tensor found")

        # ======================
        # 配置所有 input / output memory
        # FairMOT 有 4 個 output:
        # hm, wh, id, reg
        # ======================
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = int(np.prod(shape))

                device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

                self.input_name = name
                self.input_shape = shape
                self.input_dtype = dtype
                self.input_mem = device_mem

                self.context.set_tensor_address(name, int(device_mem))

            else:
                shape = tuple(self.context.get_tensor_shape(name))

                if any(dim < 0 for dim in shape):
                    shape = tuple(self.engine.get_tensor_shape(name))

                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = int(np.prod(shape))

                device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

                self.output_names.append(name)
                self.output_shapes[name] = shape
                self.output_dtypes[name] = dtype
                self.output_mems[name] = device_mem

                self.context.set_tensor_address(name, int(device_mem))

        print("Input name:", self.input_name)
        print("Input shape:", self.input_shape)
        print("Output tensors:")

        for name in self.output_names:
            print(f"  {name}: shape={self.output_shapes[name]}")

    def infer(self, img):
        cuda.memcpy_htod_async(self.input_mem, img, self.stream)

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        outputs = {}

        for name in self.output_names:
            output = np.empty(
                self.output_shapes[name],
                dtype=self.output_dtypes[name]
            )

            cuda.memcpy_dtoh_async(
                output,
                self.output_mems[name],
                self.stream
            )

            outputs[name] = output

        self.stream.synchronize()

        return outputs


# ======================
# Preprocess
# ======================
def preprocess(frame, input_shape):
    _, c, h, w = input_shape

    img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = (img - FAIRMOT_MEAN) / FAIRMOT_STD

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return np.ascontiguousarray(img, dtype=np.float32)


# ======================
# Optional: FairMOT output check
# ======================
def print_output_shapes_once(outputs):
    print("\n[FairMOT Output Shapes]")
    for name, out in outputs.items():
        print(f"{name}: {out.shape}, dtype={out.dtype}")


# ======================
# Main
# ======================
def main(engine_path, video_path, limit, print_shapes):

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Video open failed"

    trt_model = TRTInfer(engine_path)

    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    frame_count = 0
    infer_time_total = 0.0
    ram_usage = []

    printed_shapes = False

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if limit > 0 and frame_count >= limit:
            break

        input_tensor = preprocess(frame, trt_model.input_shape)

        t0 = time.time()
        outputs = trt_model.infer(input_tensor)
        t1 = time.time()

        infer_time_total += (t1 - t0)

        if print_shapes and not printed_shapes:
            print_output_shapes_once(outputs)
            printed_shapes = True

        ram_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"{frame_count} frames processed")

    total_time = time.time() - start_time

    cap.release()
    gpu_monitor.stop()

    # ======================
    # Stats
    # ======================
    if frame_count == 0:
        print("No frames processed")
        return

    avg_inf_time = infer_time_total / frame_count
    inf_fps = 1.0 / avg_inf_time if avg_inf_time > 0 else 0.0
    overall_fps = frame_count / total_time if total_time > 0 else 0.0

    avg_ram = np.mean(ram_usage)
    peak_ram = np.max(ram_usage)

    temps = gpu_monitor.temps
    powers = gpu_monitor.powers

    avg_temp = np.mean(temps) if len(temps) > 0 else 0.0
    max_temp = np.max(temps) if len(temps) > 0 else 0.0
    min_temp = np.min(temps) if len(temps) > 0 else 0.0

    avg_power = np.mean(powers) if len(powers) > 0 else 0.0

    # 保留你原本的算法：avg_power * total_time
    total_energy = avg_power * total_time

    print("\n===== RESULT =====")
    print(f"Total frames: {frame_count}")

    print(f"\nTotal inference time: {infer_time_total:.3f} s")
    print(f"Average inference time: {avg_inf_time * 1000:.3f} ms")

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

    ENGINE_PATH = "fairmot_dla34_no_dcn_opset11.engine"
    VIDEO_PATH = "test.mp4"

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-shapes", action="store_true")

    args = parser.parse_args()

    print("Engine:", ENGINE_PATH)
    print("Video:", VIDEO_PATH)

    main(
        engine_path=ENGINE_PATH,
        video_path=VIDEO_PATH,
        limit=args.limit,
        print_shapes=args.print_shapes
    )