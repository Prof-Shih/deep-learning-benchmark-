import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import onnx
from torchvision import models

from onnxconverter_common import float16

from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)


# =========================================================
# ResNet18
# PTH -> FP32 ONNX -> FP16 ONNX + INT8 QDQ ONNX
#
# fixed shape: 1 x 3 x 224 x 224
# opset 11
# =========================================================


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_path, input_size=224):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((input_size, input_size), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, input_size=224, max_images=100):
        self.input_name = input_name
        self.input_size = input_size

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_paths = []

        for ext in exts:
            image_paths.extend(
                glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
            )

        image_paths = sorted(image_paths)

        if max_images is not None and max_images > 0:
            image_paths = image_paths[:max_images]

        if len(image_paths) == 0:
            raise RuntimeError(f"No calibration images found in: {image_dir}")

        self.image_paths = image_paths
        self.index = 0

        print(f"[Calibration] images: {len(self.image_paths)}")

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        self.index += 1

        input_data = preprocess_image(img_path, self.input_size)

        if self.index % 10 == 0 or self.index == len(self.image_paths):
            print(f"[Calibration] {self.index}/{len(self.image_paths)}")

        return {self.input_name: input_data}


def check_onnx(onnx_path):
    print(f"\nChecking ONNX: {onnx_path}")

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    for imp in model.opset_import:
        print(f"ONNX opset version: {imp.version}")

    print("\n[ONNX Inputs]")
    for i, inp in enumerate(model.graph.input):
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        print(f"  {i}: name={inp.name}, shape={dims}")

    print("\n[ONNX Outputs]")
    for i, out in enumerate(model.graph.output):
        dims = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        print(f"  {i}: name={out.name}, shape={dims}")

    print("ONNX model is valid ✅")


def load_resnet18_weights(model, weight_path, device="cpu"):
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError("Loaded checkpoint is not a dict/state_dict.")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print("Weights loaded.")

    if missing:
        print("[Warning] Missing keys:")
        for k in missing:
            print("  ", k)

    if unexpected:
        print("[Warning] Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    return model


def export_fp32_onnx(weight_path, fp32_path, input_size=224, num_classes=1000, opset=11):
    print("\n" + "=" * 80)
    print("Export ResNet18 FP32 ONNX")
    print(f"Weights    : {weight_path}")
    print(f"Input size : 1 x 3 x {input_size} x {input_size}")
    print(f"Output     : {fp32_path}")
    print("=" * 80)

    model = models.resnet18(num_classes=num_classes)
    model = load_resnet18_weights(model, weight_path)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],

        # 固定 shape，不開 dynamic_axes
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     "output": {0: "batch_size"},
        # }
    )

    check_onnx(fp32_path)
    return fp32_path


def convert_fp16(fp32_path):
    fp16_path = fp32_path.replace("_fp32.onnx", "_fp16.onnx")

    print("\n" + "=" * 80)
    print("Convert FP32 ONNX -> FP16 ONNX")
    print(f"Input : {fp32_path}")
    print(f"Output: {fp16_path}")
    print("=" * 80)

    model_fp32 = onnx.load(fp32_path)

    model_fp16 = float16.convert_float_to_float16(
        model_fp32,
        keep_io_types=True,
    )

    onnx.save(model_fp16, fp16_path)

    check_onnx(fp16_path)
    return fp16_path


def convert_int8_static(fp32_path, image_dir, input_size=224, max_calib_images=100):
    int8_path = fp32_path.replace("_fp32.onnx", "_int8_qdq.onnx")

    print("\n" + "=" * 80)
    print("Convert FP32 ONNX -> INT8 QDQ ONNX")
    print(f"Input ONNX : {fp32_path}")
    print(f"Calib dir  : {image_dir}")
    print(f"Output     : {int8_path}")
    print("=" * 80)

    model = onnx.load(fp32_path)
    input_name = model.graph.input[0].name
    print(f"Input name: {input_name}")

    data_reader = ImageCalibrationDataReader(
        image_dir=image_dir,
        input_name=input_name,
        input_size=input_size,
        max_images=max_calib_images,
    )

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,

        # opset 11 + QDQ 不要開 per_channel
        per_channel=False,
    )

    check_onnx(int8_path)
    return int8_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default="resnet18-f37072fd.pth",
        help="Path to ResNet18 .pth weights",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size, default 224",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of output classes, default 1000",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version, default 11",
    )

    parser.add_argument(
        "--calib-dir",
        type=str,
        default="../data/imagenet-mini/val",
        help="Calibration image folder",
    )

    parser.add_argument(
        "--max-calib-images",
        type=int,
        default=100,
        help="Max calibration images for INT8",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_outputs",
        help="Output folder",
    )

    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "fp32", "fp16", "int8"],
        help="all = fp32 + fp16 + int8",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fp32_path = os.path.join(
        args.output_dir,
        f"resnet18_{args.input_size}_fp32.onnx"
    )

    print("========== ResNet18 ONNX Export ==========")
    print(f"Weights          : {args.weights}")
    print(f"Input size       : {args.input_size}")
    print(f"Num classes      : {args.num_classes}")
    print(f"Opset            : {args.opset}")
    print(f"Calibration dir  : {args.calib_dir}")
    print(f"Max calib images : {args.max_calib_images}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Only             : {args.only}")
    print("==========================================")

    export_fp32_onnx(
        weight_path=args.weights,
        fp32_path=fp32_path,
        input_size=args.input_size,
        num_classes=args.num_classes,
        opset=args.opset,
    )

    if args.only in ["all", "fp16"]:
        convert_fp16(fp32_path)

    if args.only in ["all", "int8"]:
        convert_int8_static(
            fp32_path=fp32_path,
            image_dir=args.calib_dir,
            input_size=args.input_size,
            max_calib_images=args.max_calib_images,
        )

    print("\n全部完成 ✅")


if __name__ == "__main__":
    main()