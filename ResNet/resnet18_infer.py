import os
import re
import csv
import time
import glob
import math
import queue
import atexit
import signal
import threading
import subprocess
from pathlib import Path

import cv2
import numpy as np
import psutil

# TensorRT / CUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =========================
# User Config Defaults
# =========================
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================
# Utility
# =========================
def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def load_labels(label_path=None, num_classes=1000):
    if label_path and os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) < num_classes:
            labels += [f"class_{i}" for i in range(len(labels), num_classes)]
        return labels
    return [f"class_{i}" for i in range(num_classes)]


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# Tegrastats / Telemetry
# =========================
class TelemetrySampler:
    """
    Try tegrastats first.
    If unavailable, fallback to limited psutil/sysfs readings.
    """

    def __init__(self, interval_ms=100):
        self.interval_ms = max(20, int(interval_ms))
        self.samples = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.proc = None
        self.mode = None  # "tegrastats" or "fallback"

    def start(self):
        tegrastats_path = self._find_tegrastats()
        if tegrastats_path is not None:
            self.mode = "tegrastats"
            cmd = [tegrastats_path, "--interval", str(self.interval_ms)]
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            self.thread = threading.Thread(target=self._reader_tegrastats, daemon=True)
            self.thread.start()
        else:
            self.mode = "fallback"
            self.thread = threading.Thread(target=self._reader_fallback, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            try:
                self.proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass

    def _find_tegrastats(self):
        candidates = [
            "/usr/bin/tegrastats",
            "/bin/tegrastats",
            "/usr/sbin/tegrastats",
            "/usr/local/bin/tegrastats",
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        return None

    def _reader_tegrastats(self):
        # Examples:
        # RAM 2787/7620MB (lfb 109x4MB) ... cpu@48.906C gpu@48.875C tj@48.906C ... VDD_IN 3846mW/3846mW
        for line in self.proc.stdout:
            if self.stop_event.is_set():
                break
            sample = self._parse_tegrastats_line(line.strip())
            if sample:
                with self.lock:
                    self.samples.append(sample)

    def _reader_fallback(self):
        while not self.stop_event.is_set():
            t = time.time()
            ram = psutil.virtual_memory()
            sample = {
                "ts": t,
                "ram_used_mb": ram.used / (1024 * 1024),
                "ram_total_mb": ram.total / (1024 * 1024),
                "power_inst_w": None,
                "power_avg_w": None,
                "temps": self._read_sysfs_temps(),
                "raw": "fallback",
            }
            # Try INA3221 VDD_IN sysfs for Orin Nano/NX
            p_inst = self._read_vdd_in_power_sysfs()
            if p_inst is not None:
                sample["power_inst_w"] = p_inst
            with self.lock:
                self.samples.append(sample)
            time.sleep(self.interval_ms / 1000.0)

    def _parse_tegrastats_line(self, line):
        try:
            t = time.time()
            sample = {
                "ts": t,
                "ram_used_mb": None,
                "ram_total_mb": None,
                "power_inst_w": None,
                "power_avg_w": None,
                "temps": {},
                "raw": line,
            }

            # RAM 2787/7620MB
            m_ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            if m_ram:
                sample["ram_used_mb"] = float(m_ram.group(1))
                sample["ram_total_mb"] = float(m_ram.group(2))

            # VDD_IN 3846mW/3846mW
            m_pwr = re.search(r"VDD_IN\s+(\d+)mW/(\d+)mW", line)
            if m_pwr:
                sample["power_inst_w"] = float(m_pwr.group(1)) / 1000.0
                sample["power_avg_w"] = float(m_pwr.group(2)) / 1000.0

            # temperature tokens like cpu@48.906C gpu@48.875C tj@48.906C
            temp_pairs = re.findall(r"([A-Za-z0-9_]+)@(-?\d+(?:\.\d+)?)C", line)
            for name, value in temp_pairs:
                sample["temps"][name.lower()] = float(value)

            return sample
        except Exception:
            return None

    def _read_sysfs_temps(self):
        temps = {}
        thermal_root = "/sys/class/thermal"
        if not os.path.isdir(thermal_root):
            return temps

        for zone in glob.glob(os.path.join(thermal_root, "thermal_zone*")):
            try:
                type_path = os.path.join(zone, "type")
                temp_path = os.path.join(zone, "temp")
                if os.path.isfile(type_path) and os.path.isfile(temp_path):
                    with open(type_path, "r") as f:
                        name = f.read().strip().lower()
                    with open(temp_path, "r") as f:
                        raw = f.read().strip()
                    val = float(raw)
                    if val > 1000:
                        val = val / 1000.0
                    temps[name] = val
            except Exception:
                continue
        return temps

    def _read_vdd_in_power_sysfs(self):
        # Jetson Orin Nano / NX docs mention INA3221 under:
        # /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon<x>
        base = "/sys/bus/i2c/drivers/ina3221/1-0040"
        if not os.path.isdir(base):
            return None

        try:
            hwmons = glob.glob(os.path.join(base, "hwmon", "hwmon*"))
            for hw in hwmons:
                # Try to find channel whose label/name contains VDD_IN
                labels = glob.glob(os.path.join(hw, "in*_label")) + glob.glob(os.path.join(hw, "curr*_label"))
                for lab in labels:
                    try:
                        with open(lab, "r") as f:
                            txt = f.read().strip()
                        if "VDD_IN" in txt.upper():
                            idx = re.findall(r"(\d+)", os.path.basename(lab))
                            if not idx:
                                continue
                            idx = idx[0]

                            # Common hwmon nodes
                            v_path = os.path.join(hw, f"in{idx}_input")
                            c_path = os.path.join(hw, f"curr{idx}_input")
                            p_path = os.path.join(hw, f"power{idx}_input")

                            if os.path.isfile(p_path):
                                with open(p_path, "r") as f:
                                    p = float(f.read().strip())
                                # Often microwatts in hwmon; convert conservatively
                                if p > 100000:
                                    return p / 1e6
                                return p / 1000.0

                            if os.path.isfile(v_path) and os.path.isfile(c_path):
                                with open(v_path, "r") as f:
                                    v = float(f.read().strip())
                                with open(c_path, "r") as f:
                                    c = float(f.read().strip())
                                # hwmon often uses mV and mA
                                return (v * c) / 1e6
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def get_samples_between(self, t0, t1):
        with self.lock:
            return [s for s in self.samples if t0 <= s["ts"] <= t1]

    def get_latest(self):
        with self.lock:
            return self.samples[-1] if self.samples else None


# =========================
# TensorRT Wrapper
# =========================
class TRTClassifier:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.num_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io)]

        self.input_name = None
        self.output_name = None
        for name in self.tensor_names:
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Could not identify input/output tensors.")

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # Dynamic shape support
        self.has_dynamic = any(dim == -1 for dim in self.input_shape)
        if self.has_dynamic:
            # default set batch=1 for dynamic engines
            inferred_shape = tuple(1 if d == -1 else d for d in self.input_shape)
            self.context.set_input_shape(self.input_name, inferred_shape)
            self.input_shape = tuple(self.context.get_tensor_shape(self.input_name))
            self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        if len(self.input_shape) != 4:
            raise ValueError(f"Expected NCHW 4D input, got: {self.input_shape}")

        self.batch_size, self.c, self.h, self.w = self.input_shape

        # Allocate buffers
        self.stream = cuda.Stream()
        self.bindings = {}

        input_nbytes = int(np.prod(self.input_shape)) * np.dtype(self.input_dtype).itemsize
        output_nbytes = int(np.prod(self.output_shape)) * np.dtype(self.output_dtype).itemsize

        self.host_input = cuda.pagelocked_empty(int(np.prod(self.input_shape)), self.input_dtype)
        self.host_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)), self.output_dtype)

        self.device_input = cuda.mem_alloc(input_nbytes)
        self.device_output = cuda.mem_alloc(output_nbytes)

        self.context.set_tensor_address(self.input_name, int(self.device_input))
        self.context.set_tensor_address(self.output_name, int(self.device_output))

    def preprocess(self, image_bgr, mean, std):
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # -> NCHW
        return np.ascontiguousarray(img, dtype=np.float32)

    def infer(self, input_tensor):
        np.copyto(self.host_input, input_tensor.ravel())

        start_event = cuda.Event()
        end_event = cuda.Event()

        start_wall = time.time()
        start_event.record(self.stream)

        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)

        end_event.record(self.stream)
        self.stream.synchronize()
        end_wall = time.time()

        gpu_ms = start_event.time_till(end_event)
        wall_s = end_wall - start_wall

        output = np.array(self.host_output).reshape(self.output_shape)
        return output, wall_s, gpu_ms / 1000.0


# =========================
# Reporting Helpers
# =========================
def summarize_temperature(temp_dict):
    if not temp_dict:
        return None, None, None, None

    cpu = temp_dict.get("cpu")
    gpu = temp_dict.get("gpu")
    tj = temp_dict.get("tj")

    max_name = None
    max_temp = None
    for k, v in temp_dict.items():
        if max_temp is None or v > max_temp:
            max_temp = v
            max_name = k
    return cpu, gpu, tj, (max_name, max_temp)


def mean_ignore_none(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def fmt(v, nd=4, suffix=""):
    if v is None:
        return "N/A"
    return f"{v:.{nd}f}{suffix}"


# =========================
# Main
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser("TensorRT folder inference on Jetson Orin Nano")
    parser.add_argument("--engine", type=str, required=True, help="TensorRT engine path")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder of input images")
    parser.add_argument("--output_dir", type=str, default="trt_orin_results", help="Output folder")
    parser.add_argument("--labels", type=str, default=None, help="Optional labels txt file")
    parser.add_argument("--telemetry_interval_ms", type=int, default=100, help="tegrastats interval")
    parser.add_argument("--mean", type=float, nargs=3, default=DEFAULT_MEAN, help="Normalization mean")
    parser.add_argument("--std", type=float, nargs=3, default=DEFAULT_STD, help="Normalization std")
    parser.add_argument("--save_images", action="store_true", help="Save overlay result images")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    report_txt = os.path.join(args.output_dir, "report.txt")
    report_csv = os.path.join(args.output_dir, "per_image_results.csv")

    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext.upper()}")))
    image_paths = sorted(set(image_paths), key=natural_sort_key)

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    print(f"[INFO] Found {len(image_paths)} images.")
    print(f"[INFO] Loading TensorRT engine: {args.engine}")

    model = TRTClassifier(args.engine)
    labels = load_labels(args.labels, num_classes=model.output_shape[-1] if len(model.output_shape) > 0 else 1000)

    print(f"[INFO] Input shape: {model.input_shape}")
    print(f"[INFO] Output shape: {model.output_shape}")

    telemetry = TelemetrySampler(interval_ms=args.telemetry_interval_ms)
    telemetry.start()
    atexit.register(telemetry.stop)

    time.sleep(0.3)  # let telemetry gather a few samples

    per_image_rows = []

    total_start = time.time()

    for idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Skip unreadable image: {img_path}")
            continue

        input_tensor = model.preprocess(
            img,
            mean=np.array(args.mean, dtype=np.float32).reshape(1, 1, 3),
            std=np.array(args.std, dtype=np.float32).reshape(1, 1, 3),
        )

        infer_t0 = time.time()
        output, wall_infer_s, gpu_infer_s = model.infer(input_tensor)
        infer_t1 = time.time()

        logits = output.reshape(-1)
        probs = softmax(logits)
        top1_idx = int(np.argmax(probs))
        top1_score = float(probs[top1_idx])
        top1_name = labels[top1_idx] if top1_idx < len(labels) else f"class_{top1_idx}"

        telemetry_window = telemetry.get_samples_between(infer_t0, infer_t1)
        latest = telemetry.get_latest()

        # Power
        p_inst_avg = mean_ignore_none([s.get("power_inst_w") for s in telemetry_window])
        p_avg_avg = mean_ignore_none([s.get("power_avg_w") for s in telemetry_window])

        if p_inst_avg is None and latest is not None:
            p_inst_avg = latest.get("power_inst_w")
        if p_avg_avg is None and latest is not None:
            p_avg_avg = latest.get("power_avg_w")

        # Prefer instantaneous average over window; fallback to average-power field
        chosen_power_w = p_inst_avg if p_inst_avg is not None else p_avg_avg
        energy_j = chosen_power_w * wall_infer_s if chosen_power_w is not None else None

        # RAM
        ram_used_mb = mean_ignore_none([s.get("ram_used_mb") for s in telemetry_window])
        ram_total_mb = mean_ignore_none([s.get("ram_total_mb") for s in telemetry_window])
        if ram_used_mb is None and latest is not None:
            ram_used_mb = latest.get("ram_used_mb")
            ram_total_mb = latest.get("ram_total_mb")

        # Temps
        merged_temps = {}
        for s in telemetry_window:
            for k, v in s.get("temps", {}).items():
                merged_temps[k] = v
        if not merged_temps and latest is not None:
            merged_temps = latest.get("temps", {})

        cpu_temp, gpu_temp, tj_temp, max_temp = summarize_temperature(merged_temps)

        fps_single = 1.0 / wall_infer_s if wall_infer_s > 0 else None

        row = {
            "index": idx,
            "image": os.path.basename(img_path),
            "top1_id": top1_idx,
            "top1_name": top1_name,
            "top1_score": top1_score,
            "infer_time_s": wall_infer_s,
            "gpu_kernel_time_s": gpu_infer_s,
            "fps": fps_single,
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "power_w": chosen_power_w,
            "energy_j": energy_j,
            "cpu_temp_c": cpu_temp,
            "gpu_temp_c": gpu_temp,
            "tj_temp_c": tj_temp,
            "max_temp_name": max_temp[0] if max_temp else None,
            "max_temp_c": max_temp[1] if max_temp else None,
        }
        per_image_rows.append(row)

        print(
            f"[{idx:04d}/{len(image_paths):04d}] {os.path.basename(img_path)} | "
            f"Top1={top1_name} ({top1_score:.4f}) | "
            f"Infer={wall_infer_s*1000:.2f} ms | FPS={fps_single:.2f} | "
            f"RAM={fmt(ram_used_mb, 1)} / {fmt(ram_total_mb, 1)} MB | "
            f"Power={fmt(chosen_power_w, 3)} W | Energy={fmt(energy_j, 4)} J | "
            f"CPU={fmt(cpu_temp, 2)} C GPU={fmt(gpu_temp, 2)} C TJ={fmt(tj_temp, 2)} C"
        )

        if args.save_images:
            vis = img.copy()
            text1 = f"Top1: {top1_name}"
            text2 = f"Score: {top1_score:.4f}"
            text3 = f"Infer: {wall_infer_s*1000:.2f} ms | FPS: {fps_single:.2f}"
            text4 = f"P: {fmt(chosen_power_w, 3)}W | Tj: {fmt(tj_temp, 2)}C"
            y0 = 30
            for i, txt in enumerate([text1, text2, text3, text4]):
                cv2.putText(
                    vis,
                    txt,
                    (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            out_img = os.path.join(args.output_dir, os.path.basename(img_path))
            cv2.imwrite(out_img, vis)

    total_end = time.time()
    telemetry.stop()

    valid_rows = [r for r in per_image_rows if r is not None]
    if not valid_rows:
        raise RuntimeError("No valid inference results.")

    total_images = len(valid_rows)
    total_wall_s = total_end - total_start
    total_infer_s = sum(r["infer_time_s"] for r in valid_rows)
    avg_infer_s = total_infer_s / total_images
    avg_fps = total_images / total_infer_s if total_infer_s > 0 else None
    overall_fps_wall = total_images / total_wall_s if total_wall_s > 0 else None

    avg_ram_used_mb = mean_ignore_none([r["ram_used_mb"] for r in valid_rows])
    max_ram_used_mb = max([r["ram_used_mb"] for r in valid_rows if r["ram_used_mb"] is not None], default=None)

    avg_power_w = mean_ignore_none([r["power_w"] for r in valid_rows])
    max_power_w = max([r["power_w"] for r in valid_rows if r["power_w"] is not None], default=None)

    total_energy_j = sum(r["energy_j"] for r in valid_rows if r["energy_j"] is not None)
    avg_energy_j = mean_ignore_none([r["energy_j"] for r in valid_rows])

    avg_cpu_temp = mean_ignore_none([r["cpu_temp_c"] for r in valid_rows])
    avg_gpu_temp = mean_ignore_none([r["gpu_temp_c"] for r in valid_rows])
    avg_tj_temp = mean_ignore_none([r["tj_temp_c"] for r in valid_rows])
    max_temp_c = max([r["max_temp_c"] for r in valid_rows if r["max_temp_c"] is not None], default=None)

    # Save CSV
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index", "image", "top1_id", "top1_name", "top1_score",
                "infer_time_s", "gpu_kernel_time_s", "fps",
                "ram_used_mb", "ram_total_mb",
                "power_w", "energy_j",
                "cpu_temp_c", "gpu_temp_c", "tj_temp_c",
                "max_temp_name", "max_temp_c"
            ],
        )
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(row)

    # Save TXT report
    lines = []
    lines.append("=== TensorRT Folder Inference Report (Jetson Orin Nano) ===")
    lines.append(f"Engine: {args.engine}")
    lines.append(f"Input dir: {args.input_dir}")
    lines.append(f"Output dir: {args.output_dir}")
    lines.append(f"Telemetry mode: {telemetry.mode}")
    lines.append(f"Total images: {total_images}")
    lines.append("")
    lines.append("=== Overall ===")
    lines.append(f"Total wall time (s): {total_wall_s:.6f}")
    lines.append(f"Total pure inference time (s): {total_infer_s:.6f}")
    lines.append(f"Average single-image inference time (s): {avg_infer_s:.6f}")
    lines.append(f"Average FPS (using pure inference time): {avg_fps:.6f}" if avg_fps is not None else "Average FPS: N/A")
    lines.append(f"Overall FPS (using total wall time): {overall_fps_wall:.6f}" if overall_fps_wall is not None else "Overall FPS: N/A")
    lines.append("")
    lines.append("=== Memory ===")
    lines.append(f"Average RAM used (MB): {avg_ram_used_mb:.3f}" if avg_ram_used_mb is not None else "Average RAM used (MB): N/A")
    lines.append(f"Peak RAM used (MB): {max_ram_used_mb:.3f}" if max_ram_used_mb is not None else "Peak RAM used (MB): N/A")
    lines.append("")
    lines.append("=== Power / Energy ===")
    lines.append(f"Average power (W): {avg_power_w:.6f}" if avg_power_w is not None else "Average power (W): N/A")
    lines.append(f"Peak power (W): {max_power_w:.6f}" if max_power_w is not None else "Peak power (W): N/A")
    lines.append(f"Total energy (J): {total_energy_j:.6f}" if total_energy_j is not None else "Total energy (J): N/A")
    lines.append(f"Average energy per image (J): {avg_energy_j:.6f}" if avg_energy_j is not None else "Average energy per image (J): N/A")
    lines.append("")
    lines.append("=== Temperature ===")
    lines.append(f"Average CPU temp (C): {avg_cpu_temp:.3f}" if avg_cpu_temp is not None else "Average CPU temp (C): N/A")
    lines.append(f"Average GPU temp (C): {avg_gpu_temp:.3f}" if avg_gpu_temp is not None else "Average GPU temp (C): N/A")
    lines.append(f"Average TJ temp (C): {avg_tj_temp:.3f}" if avg_tj_temp is not None else "Average TJ temp (C): N/A")
    lines.append(f"Peak observed temp (C): {max_temp_c:.3f}" if max_temp_c is not None else "Peak observed temp (C): N/A")
    lines.append("")
    lines.append("=== Per Image ===")
    for r in valid_rows:
        lines.append(
            f"{r['index']:04d} | {r['image']} | "
            f"Top1={r['top1_name']} ({r['top1_score']:.4f}) | "
            f"Infer={r['infer_time_s']*1000:.2f} ms | "
            f"FPS={r['fps']:.2f} | "
            f"RAM={fmt(r['ram_used_mb'], 1)} MB | "
            f"Power={fmt(r['power_w'], 3)} W | "
            f"Energy={fmt(r['energy_j'], 4)} J | "
            f"CPU={fmt(r['cpu_temp_c'], 2)} C | "
            f"GPU={fmt(r['gpu_temp_c'], 2)} C | "
            f"TJ={fmt(r['tj_temp_c'], 2)} C"
        )

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total images: {total_images}")
    print(f"Total wall time: {total_wall_s:.6f} s")
    print(f"Total pure inference time: {total_infer_s:.6f} s")
    print(f"Average single-image inference time: {avg_infer_s:.6f} s")
    print(f"Average FPS (pure inference): {avg_fps:.6f}" if avg_fps is not None else "Average FPS: N/A")
    print(f"Overall FPS (wall time): {overall_fps_wall:.6f}" if overall_fps_wall is not None else "Overall FPS: N/A")
    print(f"Average RAM used: {fmt(avg_ram_used_mb, 3)} MB")
    print(f"Peak RAM used: {fmt(max_ram_used_mb, 3)} MB")
    print(f"Average power: {fmt(avg_power_w, 6)} W")
    print(f"Peak power: {fmt(max_power_w, 6)} W")
    print(f"Total energy: {fmt(total_energy_j, 6)} J")
    print(f"Average energy/image: {fmt(avg_energy_j, 6)} J")
    print(f"Average CPU temp: {fmt(avg_cpu_temp, 3)} C")
    print(f"Average GPU temp: {fmt(avg_gpu_temp, 3)} C")
    print(f"Average TJ temp: {fmt(avg_tj_temp, 3)} C")
    print(f"Peak observed temp: {fmt(max_temp_c, 3)} C")
    print(f"Saved CSV: {report_csv}")
    print(f"Saved TXT: {report_txt}")


if __name__ == "__main__":
    main()