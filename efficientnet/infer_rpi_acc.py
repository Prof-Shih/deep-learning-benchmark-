import os
import re
import glob
import time
import argparse

import cv2
import numpy as np
import onnxruntime as ort


# ============================================
# Config
# ============================================
DEFAULT_MODEL = "efficientnet_v2_s_384_fp32.onnx"
DEFAULT_VAL_DIR = "../imagenet-mini/val"

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

TOPK = 5

IMAGE_EXT_PATTERNS = (
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP"
)


# ============================================
# Utils
# ============================================
def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def build_imagenet_dataset(val_dir, limit=0):
    """
    資料夾格式：

    val/
      n01440764/
        xxx.JPEG
      n01443537/
        xxx.JPEG
      ...

    label 會依照資料夾名稱排序後決定。
    """

    if not os.path.isdir(val_dir):
        raise RuntimeError(f"Dataset folder not found: {val_dir}")

    class_names = sorted([
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    ], key=natural_sort_key)

    if len(class_names) == 0:
        raise RuntimeError(f"No class folders found under: {val_dir}")

    samples = []

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(val_dir, cls_name)

        imgs = []
        for pattern in IMAGE_EXT_PATTERNS:
            imgs.extend(glob.glob(os.path.join(cls_dir, pattern)))
            imgs.extend(glob.glob(os.path.join(cls_dir, "**", pattern), recursive=True))

        imgs = sorted(list(set(imgs)), key=natural_sort_key)

        for img_path in imgs:
            samples.append((img_path, cls_idx))

    if len(samples) == 0:
        raise RuntimeError(f"No images found under: {val_dir}")

    if limit > 0:
        samples = samples[:limit]

    return samples, class_names


def get_input_hw(session, fallback_size=384):
    input_shape = session.get_inputs()[0].shape

    try:
        h = int(input_shape[2])
        w = int(input_shape[3])
        return h, w
    except Exception:
        return fallback_size, fallback_size


def preprocess(img_path, h, w, mean, std):
    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))

    return np.ascontiguousarray(img, dtype=np.float32)


def make_batch(batch_samples, h, w, mean, std):
    batch_imgs = []
    gt_labels = []

    for img_path, gt_label in batch_samples:
        img = preprocess(img_path, h, w, mean, std)
        batch_imgs.append(img)
        gt_labels.append(gt_label)

    batch = np.stack(batch_imgs, axis=0).astype(np.float32)
    return batch, gt_labels


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="ONNX model path"
    )

    parser.add_argument(
        "--dir",
        "--val_dir",
        "--input_dir",
        dest="val_dir",
        type=str,
        default=DEFAULT_VAL_DIR,
        help="ImageNet val folder"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size. 固定 batch=1 的 ONNX 請用 1。"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit images. 0 means all images."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup runs"
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print progress every N images"
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=384,
        help="Fallback input size if ONNX shape is dynamic"
    )

    args = parser.parse_args()

    batch_size = max(1, int(args.batch))

    print("========== EfficientNet ONNXRuntime Accuracy Test ==========")
    print(f"Model      : {args.model}")
    print(f"Dataset    : {args.val_dir}")
    print(f"Batch size : {batch_size}")
    print(f"Limit      : {args.limit}")
    print(f"Provider   : CPUExecutionProvider")
    print("============================================================")

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    h, w = get_input_hw(session, fallback_size=args.input_size)

    print(f"Input name : {input_name}")
    print(f"Output name: {output_name}")
    print(f"Input size : {h}x{w}")
    print("============================================================")

    samples, class_names = build_imagenet_dataset(
        args.val_dir,
        limit=args.limit
    )

    total_images = len(samples)

    print(f"Classes     : {len(class_names)}")
    print(f"Total images: {total_images}")
    print("============================================================")

    mean = np.array(DEFAULT_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(DEFAULT_STD, dtype=np.float32).reshape(1, 1, 3)

    # ========================================
    # Warmup
    # ========================================
    if total_images > 0 and args.warmup > 0:
        print("\n[Warmup]")

        warmup_samples = samples[:batch_size]
        warmup_input, _ = make_batch(warmup_samples, h, w, mean, std)

        for i in range(args.warmup):
            _ = session.run([output_name], {input_name: warmup_input})
            print(f"Warmup {i + 1}/{args.warmup}")

    # ========================================
    # Evaluation
    # ========================================
    print("\n[Start Evaluation]")

    top1_correct = 0
    top5_correct = 0
    processed_images = 0
    total_pure_infer = 0.0

    total_start = time.time()

    for start_idx in range(0, total_images, batch_size):
        batch_samples = samples[start_idx:start_idx + batch_size]

        input_data, gt_labels = make_batch(
            batch_samples,
            h,
            w,
            mean,
            std
        )

        real_batch = input_data.shape[0]

        t0 = time.time()
        output = session.run([output_name], {input_name: input_data})[0]
        t1 = time.time()

        total_pure_infer += (t1 - t0)

        output = output.reshape(real_batch, -1)

        for i in range(real_batch):
            logits = output[i]
            gt = int(gt_labels[i])

            pred_top1 = int(np.argmax(logits))
            if pred_top1 == gt:
                top1_correct += 1

            pred_top5 = np.argsort(logits)[-TOPK:]
            if gt in pred_top5:
                top5_correct += 1

        processed_images += real_batch

        if processed_images % args.print_every == 0 or processed_images == total_images:
            cur_top1 = top1_correct / processed_images * 100
            cur_top5 = top5_correct / processed_images * 100

            print(
                f"[{processed_images:04d}/{total_images:04d}] "
                f"Top1={cur_top1:.2f}% "
                f"Top5={cur_top5:.2f}%"
            )

    total_elapsed = time.time() - total_start

    top1_acc = top1_correct / processed_images * 100 if processed_images > 0 else 0.0
    top5_acc = top5_correct / processed_images * 100 if processed_images > 0 else 0.0

    avg_infer_ms = total_pure_infer / processed_images * 1000 if processed_images > 0 else 0.0
    pure_fps = processed_images / total_pure_infer if total_pure_infer > 0 else 0.0
    overall_fps = processed_images / total_elapsed if total_elapsed > 0 else 0.0

    print("\n========== EfficientNet ONNXRuntime Final Result ==========")
    print(f"Model                     : {args.model}")
    print(f"Dataset                   : {args.val_dir}")
    print(f"Total images              : {processed_images}")
    print(f"Batch size                : {batch_size}")
    print(f"Top-1 Accuracy            : {top1_acc:.2f}%")
    print(f"Top-5 Accuracy            : {top5_acc:.2f}%")
    print(f"Total pure inference time : {total_pure_infer:.4f} s")
    print(f"Average inference time    : {avg_infer_ms:.4f} ms/image")
    print(f"Pure inference FPS        : {pure_fps:.2f}")
    print(f"Total elapsed time        : {total_elapsed:.4f} s")
    print(f"Overall FPS               : {overall_fps:.2f}")
    print("===========================================================")


if __name__ == "__main__":
    main()