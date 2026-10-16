import os
import glob
import time
import argparse

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# ============================================
# Config
# ============================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

IMG_EXTS = (
    "*.jpg", "*.jpeg", "*.png",
    "*.bmp", "*.webp",
    "*.JPG", "*.JPEG", "*.PNG",
    "*.BMP", "*.WEBP"
)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

TOPK = 5


# ============================================
# TensorRT helpers
# ============================================
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

    return engine


def get_io_tensor_names(engine):
    input_names = []
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(
            f"Expected 1 input tensor, got {len(input_names)}: {input_names}"
        )

    if len(output_names) < 1:
        raise RuntimeError("No output tensor found in engine.")

    return input_names[0], output_names


def volume(shape):
    v = 1
    for x in shape:
        v *= int(x)
    return int(v)


def resolve_input_shape(engine, context, input_name, batch_size, input_h=None, input_w=None):
    """
    會自動從 engine 讀 input shape。
    - 固定尺寸 engine：直接使用 engine 內建 H/W。
    - dynamic shape engine：若 H/W 是 -1，優先用 --input_h / --input_w；沒給就預設 224。
    """
    raw_shape = tuple(engine.get_tensor_shape(input_name))

    if len(raw_shape) != 4:
        raise RuntimeError(f"Expected 4D input shape, got {raw_shape}")

    n, c, h, w = raw_shape

    if c == -1:
        c = 3
    else:
        c = int(c)

    if c != 3:
        raise RuntimeError(f"Expected 3-channel input, got {raw_shape}")

    if h == -1:
        h = int(input_h) if input_h is not None else 224
    else:
        h = int(h)

    if w == -1:
        w = int(input_w) if input_w is not None else 224
    else:
        w = int(w)

    if n == -1:
        n = int(batch_size)
    else:
        # static batch engine 以 engine 內建 batch 為準
        n = int(n)

    input_shape = (n, c, h, w)

    try:
        context.set_input_shape(input_name, input_shape)
    except Exception:
        # static engine 可能不能 set shape，改用 context 目前 shape
        input_shape = tuple(context.get_tensor_shape(input_name))

    return input_shape


# ============================================
# Dataset
# ============================================
def build_imagenet_dataset(val_dir, limit=0):
    """
    ImageNet-mini / ImageNet val 資料夾格式：
        val/
          n01440764/
            xxx.JPEG
          n01443537/
            xxx.JPEG

    label 會依照 class folder 排序後對應 index。
    """
    class_names = sorted([
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    ])

    if len(class_names) == 0:
        raise RuntimeError(f"No class folders found under: {val_dir}")

    samples = []

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(val_dir, cls_name)

        imgs = []
        for ext in IMG_EXTS:
            # 第一層
            imgs.extend(glob.glob(os.path.join(cls_dir, ext)))

            # 子資料夾也一起找，避免 images/xxx.JPEG 這種結構
            imgs.extend(
                glob.glob(
                    os.path.join(cls_dir, "**", ext),
                    recursive=True
                )
            )

        imgs = sorted(list(set(imgs)))

        for img_path in imgs:
            samples.append((img_path, cls_idx))

    if len(samples) == 0:
        raise RuntimeError(
            f"No images found in class folders under: {val_dir}"
        )

    if limit > 0:
        samples = samples[:limit]

    return samples, class_names


# ============================================
# Preprocess
# ============================================
def preprocess_image(img_path, h, w):
    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(
        img,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    img = img.astype(np.float32) / 255.0

    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = np.transpose(img, (2, 0, 1))

    return np.ascontiguousarray(img, dtype=np.float32)


# ============================================
# Main
# ============================================
def main(
    engine_path,
    val_dir,
    batch_size=1,
    warmup=5,
    print_every=10,
    limit=0,
    input_h=None,
    input_w=None
):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    input_name, output_names = get_io_tensor_names(engine)

    input_shape = resolve_input_shape(
        engine=engine,
        context=context,
        input_name=input_name,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w
    )

    batch_size, c, h, w = map(int, input_shape)

    print("========== EfficientNet TensorRT Accuracy Test ==========")
    print(f"Engine path       : {engine_path}")
    print(f"Dataset           : {val_dir}")
    print(f"Input tensor name : {input_name}")
    print(f"Input shape       : {input_shape}")
    print(f"Input size        : {h}x{w}")
    print(f"Batch size        : {batch_size}")
    print("=========================================================")

    # ========================================
    # Allocate input
    # ========================================
    input_nbytes = int(volume(input_shape) * np.dtype(np.float32).itemsize)
    d_input = cuda.mem_alloc(input_nbytes)
    context.set_tensor_address(input_name, int(d_input))

    # ========================================
    # Allocate outputs
    # ========================================
    output_buffers = []
    d_outputs = []

    for out_name in output_names:
        out_shape = list(context.get_tensor_shape(out_name))

        if any(x < 0 for x in out_shape):
            raise RuntimeError(
                f"Dynamic output shape unresolved for tensor {out_name}: {out_shape}"
            )

        if len(out_shape) >= 1:
            out_shape[0] = batch_size

        out_shape = tuple(out_shape)
        out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))

        host_out = np.empty(out_shape, dtype=out_dtype)
        d_out = cuda.mem_alloc(int(host_out.nbytes))

        context.set_tensor_address(out_name, int(d_out))

        output_buffers.append((out_name, host_out))
        d_outputs.append(d_out)

        print(f"Output tensor     : {out_name}, shape={out_shape}, dtype={out_dtype}")

    # ========================================
    # Dataset
    # ========================================
    samples, class_names = build_imagenet_dataset(
        val_dir,
        limit=limit
    )

    total_images = len(samples)
    print(f"Total classes     : {len(class_names)}")
    print(f"Total images      : {total_images}")

    # ========================================
    # Warmup
    # ========================================
    warmup_imgs = []

    for i in range(min(batch_size, total_images)):
        img_path, _ = samples[i]
        warmup_imgs.append(preprocess_image(img_path, h, w))

    while len(warmup_imgs) < batch_size:
        warmup_imgs.append(warmup_imgs[-1])

    warmup_batch = np.stack(warmup_imgs, axis=0)

    print("\n[Warmup]")

    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, warmup_batch, stream)
        context.execute_async_v3(stream_handle=stream.handle)

        for (_, host_out), d_out in zip(output_buffers, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, stream)

        stream.synchronize()

    # ========================================
    # Evaluation
    # ========================================
    top1_correct = 0
    top5_correct = 0

    pure_infer_time = 0.0

    total_start = time.time()

    print("\n[Start Evaluation]")

    for start_idx in range(0, total_images, batch_size):
        batch_samples = samples[start_idx:start_idx + batch_size]
        real_batch = len(batch_samples)

        batch_imgs = []
        gt_labels = []

        for img_path, gt_label in batch_samples:
            batch_imgs.append(preprocess_image(img_path, h, w))
            gt_labels.append(gt_label)

        while len(batch_imgs) < batch_size:
            batch_imgs.append(batch_imgs[-1])

        inp = np.stack(batch_imgs, axis=0)

        t0 = time.time()

        cuda.memcpy_htod_async(d_input, inp, stream)
        context.execute_async_v3(stream_handle=stream.handle)

        for (_, host_out), d_out in zip(output_buffers, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, stream)

        stream.synchronize()

        t1 = time.time()

        pure_infer_time += (t1 - t0)

        first_out = output_buffers[0][1]
        output = first_out[:real_batch].reshape(real_batch, -1)

        for i in range(real_batch):
            logits = output[i]

            pred_top1 = int(np.argmax(logits))
            if pred_top1 == gt_labels[i]:
                top1_correct += 1

            pred_top5 = np.argsort(logits)[-TOPK:]
            if gt_labels[i] in pred_top5:
                top5_correct += 1

        current = min(start_idx + batch_size, total_images)

        if current % print_every == 0 or current == total_images:
            cur_top1 = top1_correct / current * 100
            cur_top5 = top5_correct / current * 100

            print(
                f"[{current:04d}/{total_images:04d}] "
                f"Top1={cur_top1:.2f}% "
                f"Top5={cur_top5:.2f}%"
            )

    total_elapsed = time.time() - total_start

    # ========================================
    # Final metrics
    # ========================================
    top1_acc = top1_correct / total_images * 100
    top5_acc = top5_correct / total_images * 100

    avg_infer_ms = pure_infer_time / total_images * 1000
    pure_fps = total_images / pure_infer_time if pure_infer_time > 0 else 0.0
    overall_fps = total_images / total_elapsed if total_elapsed > 0 else 0.0

    print("\n========== EfficientNet Final Result ==========")
    print(f"Engine path               : {engine_path}")
    print(f"Dataset                   : {val_dir}")
    print(f"Input size                : {h}x{w}")
    print(f"Total images              : {total_images}")
    print(f"Batch size                : {batch_size}")

    print(f"Top-1 Accuracy            : {top1_acc:.2f}%")
    print(f"Top-5 Accuracy            : {top5_acc:.2f}%")

    print(f"Total pure inference time : {pure_infer_time:.4f} s")
    print(f"Average inference time    : {avg_infer_ms:.4f} ms/image")
    print(f"Pure inference FPS        : {pure_fps:.2f}")

    print(f"Total elapsed time        : {total_elapsed:.4f} s")
    print(f"Overall FPS               : {overall_fps:.2f}")
    print("===============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        default="efficientnet_v2_s.engine",
        help="Path to TensorRT engine file"
    )

    parser.add_argument(
        "--dir",
        default="../imagenet-mini/val",
        help="ImageNet-mini / ImageNet val folder"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for dynamic-batch engine. Static engine will use engine batch size."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup runs before measurement"
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print progress every N images"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images, 0 means all"
    )

    parser.add_argument(
        "--input_h",
        type=int,
        default=None,
        help="Only used when engine input height is dynamic -1"
    )

    parser.add_argument(
        "--input_w",
        type=int,
        default=None,
        help="Only used when engine input width is dynamic -1"
    )

    args = parser.parse_args()

    main(
        engine_path=args.engine,
        val_dir=args.dir,
        batch_size=args.batch,
        warmup=args.warmup,
        print_every=args.print_every,
        limit=args.limit,
        input_h=args.input_h,
        input_w=args.input_w
    )
