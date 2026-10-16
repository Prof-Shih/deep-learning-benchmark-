import torch
import onnx

from torchvision.models import (
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)

# =========================================================
# EfficientNetV2 S/M/L 多尺寸 ONNX Export
# sizes: 224, 384, 480, 640, 768
# fixed shape: 1 x 3 x H x W
# opset 11
# =========================================================

# ====== 權重路徑 ======
PTH_PATHS = {
    "s": "efficientnet_v2_s.pth",
    "m": "efficientnet_v2_m-dc08266a.pth",
    "l": "efficientnet_v2_l-59c71312.pth",
}

# ====== 模型建立函式 ======
MODEL_BUILDERS = {
    "s": efficientnet_v2_s,
    "m": efficientnet_v2_m,
    "l": efficientnet_v2_l,
}

# ====== 要輸出的尺寸 ======
INPUT_SIZES = [224, 384, 480, 640, 768]

# ====== 要輸出的模型 ======
MODEL_VARIANTS = ["s", "m", "l"]


def print_onnx_info(onnx_path):
    print(f"\nChecking ONNX model: {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    for imp in onnx_model.opset_import:
        print(f"ONNX opset version: {imp.version}")

    print("\n[ONNX Inputs]")
    for i, inp in enumerate(onnx_model.graph.input):
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
    for i, out in enumerate(onnx_model.graph.output):
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


def export_one_model(variant, input_size):
    model_name = f"efficientnet_v2_{variant}"
    pth_path = PTH_PATHS[variant]
    onnx_path = f"{model_name}_{input_size}.onnx"

    print("\n" + "=" * 70)
    print(f"Exporting: {model_name}")
    print(f"Input size: 1 x 3 x {input_size} x {input_size}")
    print(f"PTH path  : {pth_path}")
    print(f"ONNX path : {onnx_path}")
    print("=" * 70)

    # ====== 載入模型 ======
    model = MODEL_BUILDERS[variant]()
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # ====== dummy input ======
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # ====== export ONNX fixed shape ======
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,

        # 不要開 dynamic_axes，Kneo Pi / Kneron 建議固定 shape
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     "output": {0: "batch_size"},
        # }
    )

    print(f"Export finished: {onnx_path}")

    # ====== 驗證 ONNX ======
    print_onnx_info(onnx_path)


def main():
    for variant in MODEL_VARIANTS:
        for size in INPUT_SIZES:
            export_one_model(variant, size)

    print("\n全部 export 完成 ✅")


if __name__ == "__main__":
    main()