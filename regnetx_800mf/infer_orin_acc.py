import os
import glob
import time
import argparse

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_engine_path(engine_path):
    # Prefer the path the user gave. If it does not exist, try xxx_dynamic.engine.
    if os.path.exists(engine_path):
        return engine_path

    root, ext = os.path.splitext(engine_path)
    dynamic_path = engine_path if root.endswith("_dynamic") else root + "_dynamic" + ext
    if os.path.exists(dynamic_path):
        return dynamic_path

    return engine_path


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
        raise RuntimeError(f"Expected 1 input tensor, got {len(input_names)}: {input_names}")
    if len(output_names) < 1:
        raise RuntimeError("No output tensor found in engine.")

    return input_names[0], output_names


def volume(shape):
    v = 1
    for x in shape:
        v *= int(x)
    return int(v)


def center_crop(img, crop_h, crop_w):
    h, w = img.shape[:2]
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return img[top:top + crop_h, left:left + crop_w]


def preprocess_regnetx_800mf(img_path, h, w, mode="center_crop"):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mode == "center_crop":
        # Torchvision ImageNet eval style: resize shorter side to 256, then center crop 224x224.
        # If engine input is not 224, scale the resize size proportionally.
        resize_short = int(round(256 * max(h, w) / 224.0))
        ih, iw = img.shape[:2]
        if ih < iw:
            new_h = resize_short
            new_w = int(round(iw * resize_short / ih))
        else:
            new_w = resize_short
            new_h = int(round(ih * resize_short / iw))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = center_crop(img, h, w)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    elif mode == "direct_resize":
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img, dtype=np.float32)


def get_imagenet_mini_samples(val_dir):
    val_dir = os.path.abspath(val_dir)
    if not os.path.isdir(val_dir):
        raise RuntimeError(f"Dataset folder not found: {val_dir}")

    class_dirs = [d for d in sorted(os.listdir(val_dir)) if os.path.isdir(os.path.join(val_dir, d))]
    if len(class_dirs) == 0:
        raise RuntimeError(
            f"No class folders found in {val_dir}. Expected format: imagenet-mini/val/<class_folder>/*.JPEG"
        )

    class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}

    samples = []
    for class_name in class_dirs:
        class_path = os.path.join(val_dir, class_name)
        img_paths = []
        for ext in IMG_EXTS:
            img_paths.extend(glob.glob(os.path.join(class_path, ext)))
        img_paths = sorted(img_paths)
        label = class_to_idx[class_name]
        for p in img_paths:
            samples.append((p, label))

    if len(samples) == 0:
        raise RuntimeError(f"No images found under class folders in: {val_dir}")

    return samples, class_to_idx


def topk_correct(logits, labels, k=5):
    logits = logits.reshape(logits.shape[0], -1)
    labels = np.asarray(labels, dtype=np.int64)

    top1 = np.argmax(logits, axis=1)
    top1_ok = int(np.sum(top1 == labels))

    k = min(k, logits.shape[1])
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    topk_ok = int(np.sum([labels[i] in topk[i] for i in range(labels.shape[0])]))

    return top1_ok, topk_ok


def main(engine_path, data_dir, batch_size=1, warmup=5, print_every=100, limit=0, preprocess_mode="center_crop"):
    engine_path = resolve_engine_path(engine_path)

    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    input_name, output_names = get_io_tensor_names(engine)

    raw_shape = tuple(engine.get_tensor_shape(input_name))
    if len(raw_shape) != 4:
        raise RuntimeError(f"Expected 4D input shape, got {raw_shape}")

    _, c, h, w = raw_shape
    if int(c) != 3:
        raise RuntimeError(f"Expected 3-channel input, got {raw_shape}")

    input_shape = (batch_size, int(c), int(h), int(w))
    context.set_input_shape(input_name, input_shape)

    print(f"Input tensor name: {input_name}")
    print(f"Engine path: {engine_path}")
    print(f"Engine input shape: {input_shape}")

    input_nbytes = int(volume(input_shape) * np.dtype(np.float32).itemsize)
    d_input = cuda.mem_alloc(input_nbytes)
    context.set_tensor_address(input_name, int(d_input))

    output_buffers = []
    d_outputs = []

    for out_name in output_names:
        out_shape = list(context.get_tensor_shape(out_name))
        if any(x < 0 for x in out_shape):
            raise RuntimeError(f"Dynamic output shape unresolved for tensor {out_name}: {out_shape}")

        out_shape[0] = batch_size
        out_shape = tuple(out_shape)

        out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))
        host_out = np.empty(out_shape, dtype=out_dtype)
        d_out = cuda.mem_alloc(int(host_out.nbytes))

        context.set_tensor_address(out_name, int(d_out))
        output_buffers.append((out_name, host_out))
        d_outputs.append(d_out)

        print(f"Output tensor: {out_name}, shape={out_shape}, dtype={out_dtype}")

    samples, class_to_idx = get_imagenet_mini_samples(data_dir)
    if limit and limit > 0:
        samples = samples[:limit]

    total_images = len(samples)
    print(f"Dataset: {data_dir}")
    print(f"Classes found: {len(class_to_idx)}")
    print(f"Total images: {total_images}")
    print(f"Batch size: {batch_size}")
    print(f"Preprocess: {preprocess_mode}")

    warmup_imgs = []
    for i in range(min(batch_size, total_images)):
        warmup_imgs.append(preprocess_regnetx_800mf(samples[i][0], int(h), int(w), preprocess_mode))

    while len(warmup_imgs) < batch_size:
        warmup_imgs.append(warmup_imgs[-1])

    warmup_batch = np.stack(warmup_imgs, axis=0)

    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, warmup_batch, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for (_, host_out), d_out in zip(output_buffers, d_outputs):
            cuda.memcpy_dtoh_async(host_out, d_out, stream)
        stream.synchronize()

    top1_total = 0
    top5_total = 0
    infer_times = []

    wall_start = time.time()

    for start_idx in range(0, total_images, batch_size):
        batch_samples = samples[start_idx:start_idx + batch_size]
        real_batch = len(batch_samples)

        batch_imgs = [preprocess_regnetx_800mf(p, int(h), int(w), preprocess_mode) for p, _ in batch_samples]
        batch_labels = [label for _, label in batch_samples]

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

        infer_times.append(t1 - t0)

        logits = output_buffers[0][1][:real_batch]
        ok1, ok5 = topk_correct(logits, batch_labels, k=5)
        top1_total += ok1
        top5_total += ok5

        current = min(start_idx + batch_size, total_images)
        if current % print_every == 0 or current == total_images:
            print(f"{current:05d}/{total_images:05d} | Top1: {top1_total / current * 100:.2f}% | Top5: {top5_total / current * 100:.2f}%")

    wall_end = time.time()

    total_all_time = wall_end - wall_start
    total_pure_infer = float(np.sum(infer_times))

    avg_infer_per_image = total_pure_infer / total_images
    infer_fps = total_images / total_pure_infer if total_pure_infer > 0 else 0.0
    overall_fps = total_images / total_all_time if total_all_time > 0 else 0.0

    top1_acc = top1_total / total_images * 100.0
    top5_acc = top5_total / total_images * 100.0

    print()
    print("========== Accuracy ==========")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_total}/{total_images})")
    print(f"Top-5 Accuracy: {top5_acc:.2f}% ({top5_total}/{total_images})")
    print()
    print("========== Time / FPS ==========")
    print(f"Total images: {total_images}")
    print(f"Batch size: {batch_size}")
    print(f"Total time: {total_all_time:.3f} s")
    print(f"Total pure inference time: {total_pure_infer:.3f} s")
    print(f"Average inference time: {avg_infer_per_image * 1000:.2f} ms/image")
    print(f"Inference FPS only: {infer_fps:.2f}")
    print(f"Overall FPS: {overall_fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="regnetx_800mf.engine", help="Path to RegNetX-800MF TensorRT engine file")
    parser.add_argument("--data", default="imagenet-mini/val", help="Path to ImageNet-mini val folder")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs before measurement")
    parser.add_argument("--print_every", type=int, default=100, help="Print progress every N images")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images, 0 means use all")
    parser.add_argument("--preprocess", default="center_crop", choices=["center_crop", "direct_resize"], help="Image preprocessing mode")
    args = parser.parse_args()

    main(
        engine_path=args.engine,
        data_dir=args.data,
        batch_size=args.batch_size,
        warmup=args.warmup,
        print_every=args.print_every,
        limit=args.limit,
        preprocess_mode=args.preprocess,
    )
