import os
import time
import argparse
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# =========================
# 預設路徑：你可以直接改這裡
# =========================
DEFAULT_ENGINE_PATH = "onnx_outputs/mobilenet_v2_224_fp16_2.engine"
DEFAULT_VAL_DIR = "/workspace/data/imagenet-mini/val"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def preprocess_image(image_path):
    """
    MobileNetV2 / ImageNet 標準前處理：
    Resize 256 -> CenterCrop 224 -> Normalize
    輸出 shape: (1, 3, 224, 224)
    """
    img = Image.open(image_path).convert("RGB")

    # Resize: 短邊變 256，保持比例
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(h * 256 / w)
    else:
        new_h = 256
        new_w = int(w * 256 / h)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    img = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # add batch dim
    img = np.expand_dims(img, axis=0)

    return np.ascontiguousarray(img, dtype=np.float32)


def get_one_image_per_class(val_dir):
    """
    從 ImageFolder 格式資料夾中，每個類別取一張圖片。
    class index 依照資料夾名稱排序。
    """
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")

    classes = sorted([
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    ])

    samples = []

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(val_dir, class_name)

        images = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(image_exts)
        ])

        if len(images) == 0:
            print(f"[WARN] 類別資料夾沒有圖片: {class_name}")
            continue

        image_path = os.path.join(class_dir, images[0])
        samples.append((image_path, label, class_name))

    return samples, classes


class TRTModel:
    def __init__(self, engine_path):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"找不到 engine: {engine_path}")

        print(f"[INFO] Loading TensorRT engine: {engine_path}")

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("TensorRT engine 載入失敗")

        self.context = self.engine.create_execution_context()

        self.stream = cuda.Stream()

        self.input_name = None
        self.output_name = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("找不到 TensorRT input/output tensor")

        print(f"[INFO] Input tensor : {self.input_name}")
        print(f"[INFO] Output tensor: {self.output_name}")

        # 設定 input shape
        # MobileNetV2 一般是 1x3x224x224
        self.input_shape = (1, 3, 224, 224)

        engine_input_shape = self.engine.get_tensor_shape(self.input_name)
        print(f"[INFO] Engine input shape: {engine_input_shape}")

        # 如果是 dynamic shape，例如 (-1, 3, 224, 224)，要設定實際 shape
        if -1 in engine_input_shape:
            self.context.set_input_shape(self.input_name, self.input_shape)
        else:
            self.input_shape = tuple(engine_input_shape)

        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        print(f"[INFO] Output shape: {output_shape}")

        self.input_nbytes = int(np.prod(self.input_shape) * np.dtype(np.float32).itemsize)
        self.output_nbytes = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

        self.d_input = cuda.mem_alloc(self.input_nbytes)
        self.d_output = cuda.mem_alloc(self.output_nbytes)

        self.output_shape = output_shape

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, input_array):
        input_array = np.ascontiguousarray(input_array, dtype=np.float32)

        output = np.empty(self.output_shape, dtype=np.float32)

        cuda.memcpy_htod_async(self.d_input, input_array, self.stream)

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        return output


def topk_correct(output, target, topk=(1, 5)):
    """
    output shape 通常是 (1, 1000)
    target 是 int label
    """
    output = output.reshape(-1)

    maxk = max(topk)
    pred = np.argsort(output)[::-1][:maxk]

    top1 = int(target in pred[:1])
    top5 = int(target in pred[:5])

    return top1, top5, pred[:5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default=DEFAULT_ENGINE_PATH)
    parser.add_argument("--val_dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--progress", type=int, default=10, help="每幾張顯示一次進度")
    args = parser.parse_args()

    print("========== TensorRT MobileNetV2 Accuracy Test ==========")
    print(f"Engine : {args.engine}")
    print(f"Val dir: {args.val_dir}")
    print("Mode   : one image per class")
    print("========================================================")

    if not os.path.isdir(args.val_dir):
        raise FileNotFoundError(f"找不到 val 資料夾: {args.val_dir}")

    samples, classes = get_one_image_per_class(args.val_dir)

    print(f"[INFO] Classes found : {len(classes)}")
    print(f"[INFO] Images to test: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("沒有找到任何圖片")

    model = TRTModel(args.engine)

    top1_correct = 0
    top5_correct = 0
    total = 0

    pure_infer_time = 0.0
    total_start = time.time()

    for idx, (image_path, label, class_name) in enumerate(samples, start=1):
        input_tensor = preprocess_image(image_path)

        infer_start = time.time()
        output = model.infer(input_tensor)
        infer_end = time.time()

        pure_infer_time += infer_end - infer_start

        c1, c5, pred5 = topk_correct(output, label)

        top1_correct += c1
        top5_correct += c5
        total += 1

        if idx % args.progress == 0 or idx == total or idx == len(samples):
            print(f"[{idx:04d}/{len(samples):04d}] "
                  f"Top1: {top1_correct / total * 100:.2f}% | "
                  f"Top5: {top5_correct / total * 100:.2f}%")

    total_end = time.time()
    total_time = total_end - total_start

    top1_acc = top1_correct / total * 100.0
    top5_acc = top5_correct / total * 100.0

    avg_infer_ms = pure_infer_time / total * 1000.0
    pure_fps = total / pure_infer_time
    overall_fps = total / total_time

    print("\n========== Result ==========")
    print(f"Total classes/images tested : {total}")
    print(f"Top-1 Accuracy              : {top1_acc:.2f}%")
    print(f"Top-5 Accuracy              : {top5_acc:.2f}%")
    print(f"Total pure inference time   : {pure_infer_time:.4f} sec")
    print(f"Average inference time      : {avg_infer_ms:.4f} ms/image")
    print(f"Pure inference FPS          : {pure_fps:.2f}")
    print(f"Total elapsed time          : {total_time:.4f} sec")
    print(f"Overall FPS                 : {overall_fps:.2f}")
    print("============================")


if __name__ == "__main__":
    main()