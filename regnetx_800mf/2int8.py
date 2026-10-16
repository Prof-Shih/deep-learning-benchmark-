import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_images(img_dir, limit=None):
    img_dir = Path(img_dir)
    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    imgs = sorted(imgs)
    if limit:
        imgs = imgs[:limit]
    return imgs


def preprocess_regnet(img_path, size=224):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ImageNet classification 常用：resize 256 -> center crop 224
    resize_size = 256
    img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    img = img[top:top + size, left:left + size]

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> NCHW
    img = img.transpose(2, 0, 1)

    # batch = 1
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


class RegNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, limit=100):
        self.input_name = input_name
        self.images = get_images(image_dir, limit)
        self.idx = 0

        if len(self.images) == 0:
            raise RuntimeError(f"No calibration images found in: {image_dir}")

        print(f"[INFO] Calibration images: {len(self.images)}")

    def get_next(self):
        if self.idx >= len(self.images):
            return None

        img_path = self.images[self.idx]
        self.idx += 1

        img = preprocess_regnet(img_path)
        return {self.input_name: img}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="RegNetX-800MF_dds_8gpu.onnx")
    parser.add_argument("--output", default="RegNetX-800MF_dds_8gpu_int8.onnx")
    parser.add_argument("--calib_dir", default="images")
    parser.add_argument("--input_name", default="input.1")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    data_reader = RegNetCalibrationDataReader(
        image_dir=args.calib_dir,
        input_name=args.input_name,
        limit=args.limit
    )

    quantize_static(
        model_input=args.onnx,
        model_output=args.output,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
    )

    print(f"[OK] Saved INT8 calibrated ONNX: {args.output}")


if __name__ == "__main__":
    main()