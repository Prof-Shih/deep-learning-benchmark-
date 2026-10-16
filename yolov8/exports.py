import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 .pt to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic axes")
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX (only works in some environments)")
    parser.add_argument("--device", type=str, default="cpu", help="Export device: cpu or 0")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"找不到權重檔: {args.weights}")

    print("========================================")
    print("YOLOv8 -> ONNX Export")
    print(f"Weights   : {args.weights}")
    print(f"Image size: {args.imgsz}")
    print(f"Opset     : {args.opset}")
    print(f"Simplify  : {args.simplify}")
    print(f"Dynamic   : {args.dynamic}")
    print(f"Half      : {args.half}")
    print(f"Device    : {args.device}")
    print("========================================")

    model = YOLO(args.weights)

    result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        device=args.device
    )

    print("========================================")
    print("Export finished")
    print(f"Output: {result}")
    print("========================================")


if __name__ == "__main__":
    main()