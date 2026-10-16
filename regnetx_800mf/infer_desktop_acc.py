# infer_regnetx_800mf_acc_trt.py

import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
            raise RuntimeError("No input tensor found.")

        if len(self.output_names) == 0:
            raise RuntimeError("No output tensor found.")

        self.stream = cuda.Stream()
        self.input_device = None
        self.output_devices = {}

    def get_input_spec(self):
        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        return self.input_name, shape, dtype

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


# =========================================================
# Dataset
# =========================================================
def collect_imagenet_folder(val_dir, limit=0):
    val_dir = Path(val_dir)

    if not val_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {val_dir}")

    class_dirs = [p for p in val_dir.iterdir() if p.is_dir()]

    if len(class_dirs) == 0:
        raise RuntimeError(
            f"No class folders found in {val_dir}. "
            "ImageNet-mini val should be folder-based."
        )

    # 如果資料夾是 0,1,2...999，就直接用數字當 label
    if all(p.name.isdigit() for p in class_dirs):
        class_dirs = sorted(class_dirs, key=lambda p: int(p.name))
        class_to_idx = {p.name: int(p.name) for p in class_dirs}
    else:
        # ImageNet WNID 資料夾，例如 n01440764，通常用 sorted 順序對應 0~999
        class_dirs = sorted(class_dirs, key=lambda p: p.name)
        class_to_idx = {p.name: idx for idx, p in enumerate(class_dirs)}

    samples = []

    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]

        for img_path in sorted(class_dir.rglob("*")):
            if img_path.suffix.lower() in IMG_EXTS:
                samples.append((str(img_path), label))

    if len(samples) == 0:
        raise RuntimeError(f"No images found in {val_dir}")

    if limit > 0:
        samples = samples[:limit]

    return samples, class_to_idx


# =========================================================
# Preprocess - RegNetX-800MF ImageNet
# input.1: [N, 3, 224, 224]
# output: 502 [N, 1000]
# =========================================================
def resize_shorter_side(img, size=256):
    h, w = img.shape[:2]

    if h < w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


def center_crop(img, crop_h=224, crop_w=224):
    h, w = img.shape[:2]

    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)

    img = img[top:top + crop_h, left:left + crop_w]
    return img


def preprocess_regnetx_800mf(img_path, input_h=224, input_w=224):
    img = cv2.imread(img_path)

    if img is None:
        return None

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ImageNet classification preprocess:
    # Resize shorter side to 256, then center crop 224x224
    img = resize_shorter_side(img, size=256)
    img = center_crop(img, crop_h=input_h, crop_w=input_w)

    if img.shape[0] != input_h or img.shape[1] != input_w:
        img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32)

    return img


# =========================================================
# Accuracy
# =========================================================
def get_logits(outputs):
    """
    RegNetX-800MF output should be [N, 1000].
    如果 engine 有多個 output，這裡會自動找最後一維是 1000 的。
    """
    for out in outputs:
        arr = np.asarray(out)

        if arr.ndim == 2 and arr.shape[-1] == 1000:
            return arr

        if arr.ndim == 1 and arr.shape[0] == 1000:
            return arr.reshape(1, 1000)

    raise RuntimeError(
        f"Cannot find classification logits [N,1000]. Output shapes: {[o.shape for o in outputs]}"
    )


def topk_accuracy(logits, labels, k=5):
    labels = np.asarray(labels, dtype=np.int64)

    top1 = np.argmax(logits, axis=1)
    top1_correct = np.sum(top1 == labels)

    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    topk_correct = 0

    for i in range(len(labels)):
        if labels[i] in topk[i]:
            topk_correct += 1

    return int(top1_correct), int(topk_correct)


# =========================================================
# Main
# =========================================================
def main(engine_path, val_dir, batch_size=1, limit=0):
    if batch_size < 1:
        raise ValueError("--batch must be >= 1")

    trt_model = TRTInfer(engine_path)

    input_name, engine_input_shape, engine_input_dtype = trt_model.get_input_spec()

    print("========== INFO ==========")
    print(f"Engine: {engine_path}")
    print(f"Dataset: {val_dir}")
    print(f"Input tensor name: {input_name}")
    print(f"Engine input shape: {engine_input_shape}")
    print(f"Engine input dtype: {engine_input_dtype}")
    print(f"Batch size: {batch_size}")

    if len(engine_input_shape) != 4:
        raise RuntimeError(f"Unsupported input shape: {engine_input_shape}")

    input_h = engine_input_shape[2] if engine_input_shape[2] > 0 else 224
    input_w = engine_input_shape[3] if engine_input_shape[3] > 0 else 224

    samples, class_to_idx = collect_imagenet_folder(val_dir, limit=limit)

    print(f"Classes: {len(class_to_idx)}")
    print(f"Images: {len(samples)}")
    print("==========================\n")

    total_images = len(samples)
    processed = 0
    skipped = 0

    correct_top1 = 0
    correct_top5 = 0

    infer_time_total = 0.0

    total_start = time.time()

    for start_idx in range(0, total_images, batch_size):
        batch_samples = samples[start_idx:start_idx + batch_size]

        batch_imgs = []
        batch_labels = []

        for img_path, label in batch_samples:
            img_tensor = preprocess_regnetx_800mf(
                img_path,
                input_h=input_h,
                input_w=input_w
            )

            if img_tensor is None:
                print(f"Skip unreadable image: {img_path}")
                skipped += 1
                continue

            batch_imgs.append(img_tensor)
            batch_labels.append(label)

        if len(batch_imgs) == 0:
            continue

        input_tensor = np.stack(batch_imgs, axis=0)  # NCHW

        infer_start = time.time()
        outputs = trt_model.infer(input_tensor)
        infer_time = time.time() - infer_start

        infer_time_total += infer_time

        logits = get_logits(outputs)

        if logits.shape[0] != len(batch_labels):
            raise RuntimeError(
                f"Output batch mismatch: logits batch={logits.shape[0]}, labels={len(batch_labels)}"
            )

        top1_c, top5_c = topk_accuracy(logits, batch_labels, k=5)

        correct_top1 += top1_c
        correct_top5 += top5_c
        processed += len(batch_labels)

        if processed % 100 == 0 or processed >= total_images:
            top1_now = correct_top1 / processed * 100.0
            top5_now = correct_top5 / processed * 100.0
            print(
                f"Processed {processed}/{total_images} | "
                f"Top-1: {top1_now:.2f}% | Top-5: {top5_now:.2f}%"
            )

    total_time = time.time() - total_start

    if processed == 0:
        print("No valid images processed.")
        return

    top1_acc = correct_top1 / processed * 100.0
    top5_acc = correct_top5 / processed * 100.0

    overall_fps = processed / total_time if total_time > 0 else 0.0
    infer_fps = processed / infer_time_total if infer_time_total > 0 else 0.0
    avg_infer_ms = infer_time_total / processed * 1000.0

    print("\n========== RESULT ==========")
    print(f"Engine: {engine_path}")
    print(f"Dataset: {val_dir}")
    print(f"Total images: {processed}")
    print(f"Skipped images: {skipped}")
    print(f"Batch size: {batch_size}")

    print(f"\nTop-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    print(f"\nTotal time: {total_time:.4f} s")
    print(f"Total pure inference time: {infer_time_total:.4f} s")
    print(f"Average inference time: {avg_infer_ms:.4f} ms / image")

    print(f"\nOverall FPS: {overall_fps:.2f}")
    print(f"Inference FPS Only: {infer_fps:.2f}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        type=str,
        default="RegNetX-800MF.engine"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="/workspace/data/imagenet-mini/val"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only test first N images. 0 means test all images."
    )

    args = parser.parse_args()

    main(
        engine_path=args.engine,
        val_dir=args.dir,
        batch_size=args.batch,
        limit=args.limit
    )