import os
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =========================
# Fixed paths
# =========================
ENGINE_PATH = "onnx_outputs/resnet18_224_int8.engine"
VAL_DIR = "/workspace/data/imagenet-mini/val"


# =========================
# ImageNet preprocess
# =========================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"]


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def collect_imagefolder_samples(val_dir):
    """
    取代 torchvision.datasets.ImageFolder。

    資料夾格式：
    val/
      n01440764/
        xxx.JPEG
      n01443537/
        xxx.JPEG

    label 會依照資料夾名稱排序後給 index。
    ImageNet 資料夾若是 n01440764, n01443537 ... 這種 WordNet ID，
    sorted 後通常會對應 ImageNet class index。
    """
    val_dir = Path(val_dir)

    class_dirs = [
        p for p in val_dir.iterdir()
        if p.is_dir()
    ]

    class_dirs = sorted(class_dirs, key=lambda x: x.name)

    classes = [p.name for p in class_dirs]
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    samples = []

    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]

        for root, _, files in os.walk(class_dir):
            for f in files:
                if Path(f).suffix in IMAGE_EXTS:
                    img_path = os.path.join(root, f)
                    samples.append((img_path, label))

    samples = sorted(samples, key=lambda x: x[0])

    return samples, classes, class_to_idx


def preprocess_image(image_path, input_h=224, input_w=224):
    """
    ResNet / TorchVision ImageNet preprocessing:
    Resize 256 -> CenterCrop 224 -> RGB -> Normalize -> CHW
    """
    img = Image.open(image_path).convert("RGB")

    resize_size = int(round(max(input_h, input_w) / 0.875))

    w, h = img.size

    if h < w:
        new_h = resize_size
        new_w = int(w * resize_size / h)
    else:
        new_w = resize_size
        new_h = int(h * resize_size / w)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - input_w) // 2
    top = (new_h - input_h) // 2
    img = img.crop((left, top, left + input_w, top + input_h))

    img = np.asarray(img).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = np.transpose(img, (2, 0, 1))

    return img.astype(np.float32)


def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")

    print(f"[INFO] Loading TensorRT engine: {engine_path}")

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError(
            "Failed to deserialize TensorRT engine. "
            "可能原因：engine 是在不同 GPU / 不同 TensorRT 版本上建立的，需要重新轉 engine。"
        )

    return engine


def trt_dtype_to_np(dtype):
    return trt.nptype(dtype)


class TRTClassifier:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_name = None

        self.input_dtype = None
        self.output_dtype = None

        self.input_shape = None
        self.output_shape = None

        self.use_v3 = hasattr(self.context, "execute_async_v3")

        self._parse_io()

    def _parse_io(self):
        if hasattr(self.engine, "num_io_tensors"):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                dtype = self.engine.get_tensor_dtype(name)
                shape = tuple(self.engine.get_tensor_shape(name))

                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                    self.input_dtype = dtype
                    self.input_shape = shape
                else:
                    self.output_name = name
                    self.output_dtype = dtype
                    self.output_shape = shape
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = tuple(self.engine.get_binding_shape(i))

                if self.engine.binding_is_input(i):
                    self.input_name = name
                    self.input_dtype = dtype
                    self.input_shape = shape
                else:
                    self.output_name = name
                    self.output_dtype = dtype
                    self.output_shape = shape

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Cannot find TensorRT input/output tensors.")

        print("[INFO] TensorRT IO")
        print(f"       Input name : {self.input_name}")
        print(f"       Input shape: {self.input_shape}")
        print(f"       Output name: {self.output_name}")
        print(f"       Output shape: {self.output_shape}")

    def get_input_hw(self):
        shape = self.input_shape

        if len(shape) != 4:
            raise RuntimeError(f"Expected NCHW input, got shape: {shape}")

        h = shape[2]
        w = shape[3]

        if h <= 0:
            h = 224
        if w <= 0:
            w = 224

        return int(h), int(w)

    def infer(self, batch_np):
        batch_np = np.ascontiguousarray(batch_np.astype(np.float32))
        batch_size = batch_np.shape[0]

        input_shape = tuple(batch_np.shape)

        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, input_shape)
        elif hasattr(self.context, "set_binding_shape"):
            input_idx = self.engine.get_binding_index(self.input_name)
            self.context.set_binding_shape(input_idx, input_shape)

        if hasattr(self.context, "get_tensor_shape"):
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            output_idx = self.engine.get_binding_index(self.output_name)
            out_shape = tuple(self.context.get_binding_shape(output_idx))

        out_shape = tuple(batch_size if d < 0 else d for d in out_shape)

        input_nbytes = batch_np.nbytes
        output_np = np.empty(out_shape, dtype=trt_dtype_to_np(self.output_dtype))
        output_nbytes = output_np.nbytes

        d_input = cuda.mem_alloc(input_nbytes)
        d_output = cuda.mem_alloc(output_nbytes)

        cuda.memcpy_htod_async(d_input, batch_np, self.stream)

        if self.use_v3:
            self.context.set_tensor_address(self.input_name, int(d_input))
            self.context.set_tensor_address(self.output_name, int(d_output))
            ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            bindings = [None] * self.engine.num_bindings
            input_idx = self.engine.get_binding_index(self.input_name)
            output_idx = self.engine.get_binding_index(self.output_name)
            bindings[input_idx] = int(d_input)
            bindings[output_idx] = int(d_output)
            ok = self.context.execute_async_v2(
                bindings=bindings,
                stream_handle=self.stream.handle
            )

        if not ok:
            raise RuntimeError("TensorRT inference failed.")

        cuda.memcpy_dtoh_async(output_np, d_output, self.stream)
        self.stream.synchronize()

        return output_np


def topk_accuracy(logits, targets, topk=(1, 5)):
    maxk = max(topk)

    pred = np.argsort(logits, axis=1)[:, -maxk:][:, ::-1]

    results = []

    for k in topk:
        correct = 0
        for i in range(len(targets)):
            if targets[i] in pred[i, :k]:
                correct += 1

        results.append(correct)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT ResNet18 ImageNet Accuracy Test"
    )

    parser.add_argument(
        "--engine",
        type=str,
        default=ENGINE_PATH,
        help="TensorRT engine path"
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default=VAL_DIR,
        help="ImageNet validation folder path"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images. 0 means all images."
    )

    args = parser.parse_args()

    engine_path = args.engine
    val_dir = args.val_dir
    batch_size = args.batch

    print("========== TensorRT ResNet18 Accuracy Test ==========")
    print(f"Engine : {engine_path}")
    print(f"Val dir: {val_dir}")
    print(f"Batch  : {batch_size}")
    print(f"Limit  : {args.limit if args.limit > 0 else 'All'}")
    print("=====================================================")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    samples, classes, class_to_idx = collect_imagefolder_samples(val_dir)

    if args.limit > 0:
        samples = samples[:args.limit]

    image_paths = [s[0] for s in samples]
    targets = [s[1] for s in samples]

    total_images = len(image_paths)

    print(f"[INFO] Classes found : {len(classes)}")
    print(f"[INFO] Images to test: {total_images}")

    if total_images == 0:
        raise RuntimeError("No images found in val_dir.")

    if len(classes) != 1000:
        print()
        print("[WARNING] 你的資料夾類別數不是 1000。")
        print("          如果不是完整 ImageNet 1000 類，label 對應可能不準。")
        print()

    model = TRTClassifier(engine_path)
    input_h, input_w = model.get_input_hw()

    print(f"[INFO] Preprocess input size: {input_h}x{input_w}")

    total_top1 = 0
    total_top5 = 0

    total_pure_infer_time = 0.0
    total_start = time.time()

    processed = 0

    while processed < total_images:
        batch_samples = samples[processed:processed + batch_size]
        batch_paths = [s[0] for s in batch_samples]
        batch_targets = [s[1] for s in batch_samples]

        batch_imgs = []

        for p in batch_paths:
            img = preprocess_image(p, input_h=input_h, input_w=input_w)
            batch_imgs.append(img)

        batch_np = np.stack(batch_imgs, axis=0).astype(np.float32)

        start_infer = time.time()
        logits = model.infer(batch_np)
        end_infer = time.time()

        infer_time = end_infer - start_infer
        total_pure_infer_time += infer_time

        logits = logits.reshape((len(batch_paths), -1))

        top1, top5 = topk_accuracy(
            logits,
            np.array(batch_targets, dtype=np.int64),
            topk=(1, 5)
        )

        total_top1 += top1
        total_top5 += top5

        processed += len(batch_paths)

        if processed % (batch_size * 10) == 0 or processed == total_images:
            current_top1 = total_top1 / processed * 100.0
            current_top5 = total_top5 / processed * 100.0

            print(
                f"[{processed:5d}/{total_images:5d}] "
                f"Top-1: {current_top1:.2f}% | "
                f"Top-5: {current_top5:.2f}%"
            )

    total_end = time.time()

    total_elapsed = total_end - total_start

    final_top1 = total_top1 / total_images * 100.0
    final_top5 = total_top5 / total_images * 100.0

    avg_infer_time = total_pure_infer_time / total_images
    pure_fps = total_images / total_pure_infer_time
    overall_fps = total_images / total_elapsed

    print()
    print("========== Final Result ==========")
    print(f"Total images              : {total_images}")
    print(f"Top-1 Accuracy            : {final_top1:.2f}%")
    print(f"Top-5 Accuracy            : {final_top5:.2f}%")
    print(f"Total pure inference time : {total_pure_infer_time:.4f} s")
    print(f"Average inference time    : {avg_infer_time * 1000:.4f} ms/image")
    print(f"Pure inference FPS        : {pure_fps:.2f}")
    print(f"Total elapsed time        : {total_elapsed:.4f} s")
    print(f"Overall FPS               : {overall_fps:.2f}")
    print("==================================")


if __name__ == "__main__":
    main()