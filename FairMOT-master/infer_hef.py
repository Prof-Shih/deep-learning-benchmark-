# benchmark_fairmot_hef_video_infermodel.py

import argparse
import time
import threading
import subprocess
import re

import cv2
import numpy as np
import psutil

from hailo_platform import VDevice


def read_pmic_power_w():
    try:
        out = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"],
            text=True,
            timeout=1.0
        )
    except Exception:
        return None

    currents = {}
    volts = {}

    for line in out.splitlines():
        m_i = re.match(r"(.+?)_A\s+current\(\d+\)=([0-9.]+)A", line.strip())
        if m_i:
            currents[m_i.group(1).strip()] = float(m_i.group(2))
            continue

        m_v = re.match(r"(.+?)_V\s+volt\(\d+\)=([0-9.]+)V", line.strip())
        if m_v:
            volts[m_v.group(1).strip()] = float(m_v.group(2))
            continue

    total_w = 0.0
    for rail, current in currents.items():
        if rail in volts:
            total_w += current * volts[rail]

    return total_w


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


class Monitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.stop_event = threading.Event()
        self.proc = psutil.Process()
        self.ram = []
        self.power = []
        self.temp = []

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _loop(self):
        while not self.stop_event.is_set():
            self.ram.append(self.proc.memory_info().rss / 1024 / 1024)

            p = read_pmic_power_w()
            if p is not None:
                self.power.append(p)

            t = read_temp_c()
            if t is not None:
                self.temp.append(t)

            time.sleep(self.interval)


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def dtype_from_format(fmt):
    s = str(fmt).upper()

    if "UINT8" in s:
        return np.uint8
    if "UINT16" in s:
        return np.uint16
    if "INT8" in s:
        return np.int8
    if "INT16" in s:
        return np.int16
    if "FLOAT32" in s or "FLOAT" in s:
        return np.float32

    return np.uint8


def get_format_type(info):
    try:
        return info.format.type
    except Exception:
        try:
            return info.format_type
        except Exception:
            return "UINT8"


def get_input_layout_and_hw(shape):
    shape = list(shape)

    # FairMOT 原始 ONNX: [1, 3, 608, 1088]
    # 但 Hailo HEF 也可能變成 [608, 1088, 3]
    if len(shape) == 4:
        if shape[1] == 3:
            return "NCHW", int(shape[2]), int(shape[3])
        elif shape[3] == 3:
            return "NHWC", int(shape[1]), int(shape[2])
    elif len(shape) == 3:
        if shape[0] == 3:
            return "CHW", int(shape[1]), int(shape[2])
        elif shape[2] == 3:
            return "HWC", int(shape[0]), int(shape[1])

    raise ValueError(f"Unsupported input shape: {shape}")


def preprocess_fairmot_frame(frame_bgr, input_h, input_w, layout, dtype):
    # FairMOT 用影像 resize 到 608x1088
    frame = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if dtype == np.float32:
        arr = frame.astype(np.float32) / 255.0

        # FairMOT 常見 normalize，只有 HEF input 是 float 才做
        mean = np.array([0.408, 0.447, 0.470], dtype=np.float32)
        std = np.array([0.289, 0.274, 0.278], dtype=np.float32)
        arr = (arr - mean) / std
    else:
        # UINT8 input：不要 normalize
        arr = np.array(frame, dtype=dtype, copy=True)

    if layout in ("NCHW", "CHW"):
        arr = np.transpose(arr, (2, 0, 1))

    if layout == "NCHW":
        arr = np.expand_dims(arr, axis=0)
    elif layout == "NHWC":
        arr = np.expand_dims(arr, axis=0)

    arr = np.array(arr, dtype=dtype, copy=True)
    arr = np.ascontiguousarray(arr)
    arr.setflags(write=True)
    return arr


def get_all_outputs(infer_model):
    outputs = infer_model.outputs

    if callable(outputs):
        outputs = outputs()

    if not isinstance(outputs, list):
        outputs = list(outputs)

    return outputs


def set_input_buffer(bindings, input_name, input_buffer):
    try:
        bindings.input(input_name).set_buffer(input_buffer)
    except TypeError:
        bindings.input().set_buffer(input_buffer)


def set_output_buffers(bindings, output_infos, output_buffers):
    for out_info in output_infos:
        bindings.output(out_info.name).set_buffer(output_buffers[out_info.name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default="fairmot_dla34_no_dcn_opset11.hef")
    parser.add_argument("--video", default="../test.mp4")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    with VDevice() as vdevice:
        infer_model = vdevice.create_infer_model(args.hef)
        infer_model.set_batch_size(args.batch_size)

        input_info = infer_model.input()
        output_infos = get_all_outputs(infer_model)

        input_name = input_info.name
        input_shape = tuple(input_info.shape)
        input_dtype = dtype_from_format(get_format_type(input_info))

        layout, input_h, input_w = get_input_layout_and_hw(input_shape)

        output_buffers = {}

        print("========== INFO ==========")
        print(f"HEF: {args.hef}")
        print(f"Video: {args.video}")
        print("Model: FairMOT raw heads")
        print("Benchmark only: no sigmoid / decode / tracking")
        print("API: InferModel")
        print(f"Input name: {input_name}")
        print(f"Input shape: {input_shape}")
        print(f"Input layout: {layout}")
        print(f"Input dtype: {input_dtype}")
        print(f"Input HxW: {input_h}x{input_w}")
        print(f"Batch size: {args.batch_size}")

        print("\nOutputs:")
        for out_info in output_infos:
            out_name = out_info.name
            out_shape = tuple(out_info.shape)
            out_dtype = dtype_from_format(get_format_type(out_info))
            output_buffers[out_name] = np.empty(out_shape, dtype=out_dtype)
            print(f"  {out_name}: shape={out_shape}, dtype={out_dtype}")

        print("==========================")

        with infer_model.configure() as configured_model:

            # warmup
            warmup_done = 0
            while warmup_done < args.warmup:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                input_buffer = preprocess_fairmot_frame(
                    frame, input_h, input_w, layout, input_dtype
                )

                bindings = configured_model.create_bindings()
                set_input_buffer(bindings, input_name, input_buffer)
                set_output_buffers(bindings, output_infos, output_buffers)

                configured_model.run([bindings], args.timeout_ms)
                warmup_done += 1

            # 正式測試從影片開頭開始
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            monitor = Monitor(interval=args.monitor_interval)
            monitor.start()

            total_start = time.perf_counter()
            inference_time = 0.0
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                input_buffer = preprocess_fairmot_frame(
                    frame, input_h, input_w, layout, input_dtype
                )

                bindings = configured_model.create_bindings()
                set_input_buffer(bindings, input_name, input_buffer)
                set_output_buffers(bindings, output_infos, output_buffers)

                infer_start = time.perf_counter()
                configured_model.run([bindings], args.timeout_ms)
                infer_end = time.perf_counter()

                inference_time += infer_end - infer_start
                frame_count += 1

                if frame_count % 50 == 0:
                    print(f"Processed {frame_count} frames")

            total_end = time.perf_counter()
            monitor.stop()

    cap.release()

    total_time = total_end - total_start
    total_frames = frame_count * args.batch_size

    overall_fps = total_frames / total_time if total_time > 0 else 0.0
    inference_fps = total_frames / inference_time if inference_time > 0 else 0.0

    print("\n========== RESULT ==========")
    print(f"Total Frames: {total_frames}")
    print(f"Total Time: {total_time:.4f} s")
    print(f"Inference Time Only: {inference_time:.4f} s")
    print(f"Overall FPS: {overall_fps:.4f}")
    print(f"Inference FPS Only: {inference_fps:.4f}")
    print(f"Average RAM: {avg(monitor.ram):.2f} MB")
    print(f"Average Power: {avg(monitor.power):.4f} W")

    if monitor.temp:
        print(f"Temperature Max: {max(monitor.temp):.2f} °C")
        print(f"Temperature Min: {min(monitor.temp):.2f} °C")
        print(f"Temperature Avg: {avg(monitor.temp):.2f} °C")
    else:
        print("Temperature Max: 0.00 °C")
        print("Temperature Min: 0.00 °C")
        print("Temperature Avg: 0.00 °C")

    print("============================")


if __name__ == "__main__":
    main()