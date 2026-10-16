# acc_onnx_rpi5.py

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================================================
# Dataset
# ImageNet / ImageNet-mini val folder format:
# ../val/class_xxx/image.jpg
# =========================================================
def collect_imagenet_folder(val_dir, limit=0):
    val_dir = Path(val_dir)

    if not val_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {val_dir}")

    class_dirs = [p for p in val_dir.iterdir() if p.is_dir()]

    if len(class_dirs) == 0:
        raise RuntimeError(
            f"No class folders found in {val_dir}. "
            "Your val folder should be like: ../val/n01440764/xxx.jpg"
        )

    # 如果資料夾是 0,1,2...999，就直接用數字當 label
    if all(p.name.isdigit() for p in class_dirs):
        class_dirs = sorted(class_dirs, key=lambda p: int(p.name))
        class_to_idx = {p.name: int(p.name) for p in class_dirs}
    else:
        # ImageNet WNID folder，例如 n01440764
        # 通常用 sorted folder name 對應 0~999
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
# ONNX input shape
# =========================================================
def parse_input_shape(session, default_h=224, default_w=224):
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape

    if len(shape) != 4:
        raise RuntimeError(f"Only 4D input is supported, got shape: {shape}")

    n, c, h, w = shape

    input_c = c if isinstance(c, int) else 3
    input_h = h if isinstance(h, int) else default_h
    input_w = w if isinstance(w, int) else default_w

    return input_meta.name, shape, input_c, input_h, input_w


# =========================================================
# Preprocess
# 主要照 infer_onnx5.py：
# RGB、Resize、CenterCrop、ImageNet normalize、NCHW
# =========================================================
def preprocess_regnetx(img_path, input_h, input_w):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    # 對 224x224 來說就是 Resize 256 -> CenterCrop 224
    resize_h = int(round(input_h * 256 / 224))
    resize_w = int(round(input_w * 256 / 224))

    img = img.resize((resize_w, resize_h), Image.BILINEAR)

    left = max((resize_w - input_w) // 2, 0)
    top = max((resize_h - input_h) // 2, 0)

    img = img.crop((left, top, left + input_w, top + input_h))

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    return arr


# =========================================================
# Output logits
# =========================================================
def get_logits(outputs):
    """
    自動找分類輸出 [N, 1000]
    如果 output 是 [1000] 也會轉成 [1, 1000]
    """
    for out in outputs:
        arr = np.asarray(out)

        if arr.ndim == 2 and arr.shape[-1] == 1000:
            return arr

        if arr.ndim == 1 and arr.shape[0] == 1000:
            return arr.reshape(1, 1000)

    raise RuntimeError(
        f"Cannot find classification logits [N,1000]. "
        f"Output shapes: {[np.asarray(o).shape for o in outputs]}"
    )


def topk_accuracy(logits, labels, k=5):
    labels = np.asarray(labels, dtype=np.int64)

    top1 = np.argmax(logits, axis=1)
    top1_correct = int(np.sum(top1 == labels))

    topk = np.argpartition(logits, -k, axis=1)[:, -k:]

    topk_correct = 0
    for i in range(len(labels)):
        if labels[i] in topk[i]:
            topk_correct += 1

    return top1_correct, topk_correct


def make_batches(samples, batch_size):
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx",
        type=str,
        default="RegNetX-800MF_dds_8gpu_dynamic.onnx",
        help="Path to ONNX model"
    )

    parser.add_argument(
        "--dir",
        "--image-dir",
        "--val-dir",
        dest="val_dir",
        type=str,
        default="../val",
        help="ImageNet val folder path"
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
        help="Only test first N images. 0 means test all images."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNXRuntime intra_op threads. 0 means default."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations"
    )

    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError("--batch must be >= 1")

    sess_options = ort.SessionOptions()

    if args.threads > 0:
        sess_options.intra_op_num_threads = args.threads
        sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        args.onnx,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

    input_name, input_shape, input_c, input_h, input_w = parse_input_shape(session)
    output_names = [o.name for o in session.get_outputs()]

    if input_c != 3:
        raise RuntimeError(f"Expected 3-channel input, got C={input_c}")

    samples, class_to_idx = collect_imagenet_folder(args.val_dir, limit=args.limit)

    print("========== INFO ==========")
    print(f"ONNX          : {args.onnx}")
    print(f"Dataset       : {args.val_dir}")
    print(f"Classes       : {len(class_to_idx)}")
    print(f"Images        : {len(samples)}")
    print(f"Batch size    : {args.batch}")
    print(f"Input name    : {input_name}")
    print(f"Input shape   : {input_shape}")
    print(f"Parsed size   : C={input_c}, H={input_h}, W={input_w}")
    print(f"Output names  : {output_names}")
    print("==========================\n")

    # Warmup
    if args.warmup > 0:
        dummy = np.zeros((args.batch, 3, input_h, input_w), dtype=np.float32)
        for _ in range(args.warmup):
            _ = session.run(output_names, {input_name: dummy})

    total_images = len(samples)
    processed = 0
    skipped = 0

    correct_top1 = 0
    correct_top5 = 0

    for batch_samples in make_batches(samples, args.batch):
        batch_imgs = []
        batch_labels = []

        for img_path, label in batch_samples:
            img_tensor = preprocess_regnetx(img_path, input_h, input_w)

            if img_tensor is None:
                print(f"Skip unreadable image: {img_path}")
                skipped += 1
                continue

            batch_imgs.append(img_tensor)
            batch_labels.append(label)

        if len(batch_imgs) == 0:
            continue

        x = np.stack(batch_imgs, axis=0).astype(np.float32)

        outputs = session.run(output_names, {input_name: x})
        logits = get_logits(outputs)

        if logits.shape[0] != len(batch_labels):
            raise RuntimeError(
                f"Output batch mismatch: logits batch={logits.shape[0]}, "
                f"labels={len(batch_labels)}"
            )

        top1_c, top5_c = topk_accuracy(logits, batch_labels, k=5)

        correct_top1 += top1_c
        correct_top5 += top5_c
        processed += len(batch_labels)

        if processed % 100 == 0 or processed >= total_images:
            print(f"Processed {processed}/{total_images}")

    if processed == 0:
        print("No valid images processed.")
        return

    top1_acc = correct_top1 / processed * 100.0
    top5_acc = correct_top5 / processed * 100.0

    print("\n========== RESULT ==========")
    print(f"ONNX          : {args.onnx}")
    print(f"Dataset       : {args.val_dir}")
    print(f"Total images  : {processed}")
    print(f"Skipped images: {skipped}")
    print(f"Batch size    : {args.batch}")
    print()
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("============================")


if __name__ == "__main__":
    main()