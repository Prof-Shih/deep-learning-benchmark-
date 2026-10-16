import os
import argparse
import torch
from torchvision import models


def build_model(num_classes=1000):
    model = models.resnet18(num_classes=num_classes)
    model.eval()
    return model


def load_weights(model, weight_path, device="cpu"):
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    ckpt = torch.load(weight_path, map_location=device)

    # 官方 resnet18-f37072fd.pth 通常就是純 state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
        else:
            raise ValueError(
                "Unsupported checkpoint format. "
                "Expected a state_dict or a dict containing 'state_dict'/'model'."
            )
    else:
        raise ValueError("Loaded checkpoint is not a dict/state_dict.")

    # 去掉可能的 module. 前綴
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


def export_onnx(model, output_path, input_size=224, batch_size=1, opset=11):
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print(f"ONNX exported to: {output_path}")
    print(f"Opset version: {opset}")
    print(f"Dynamic input shape: (batch_size, 3, {input_size}, {input_size})")

def verify_onnx(onnx_path):
    try:
        import onnx

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model verification: PASSED")

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

    except ImportError:
        print("onnx package not installed, skip verification.")
    except Exception as e:
        print(f"ONNX model verification: FAILED\n{e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ResNet18 .pth to fixed-shape ONNX opset 11 for KL730"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="resnet18-f37072fd.pth",
        help="Path to the .pth weights file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resnet18_opset11_dynamic.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fixed batch size for export (default: 1)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of output classes (default: 1000 for ImageNet)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size != 1:
        print("[Warning] For KL730, batch_size=1 is strongly recommended.")

    print("Building ResNet18 model...")
    model = build_model(num_classes=args.num_classes)

    print("Loading weights...")
    model = load_weights(model, args.weights)

    print("Exporting fixed-shape ONNX...")
    export_onnx(
        model=model,
        output_path=args.output,
        input_size=args.input_size,
        batch_size=args.batch_size,
        opset=args.opset,
    )

    print("Verifying ONNX...")
    verify_onnx(args.output)

    print("Done.")


if __name__ == "__main__":
    main()