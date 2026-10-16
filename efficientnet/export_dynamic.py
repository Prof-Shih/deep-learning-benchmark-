import torch
import onnx
from torchvision.models import efficientnet_v2_m

# ====== 設定 ======
PTH_PATH = "efficientnet_v2_m-dc08266a.pth"
ONNX_PATH = "efficientnet_v2_m_dynamic.onnx"

# EfficientNetV2-L 建議 input size：480x480
INPUT_SIZE = (1, 3, 480, 480)

# ====== 載入模型 ======
print("Loading model...")

model = efficientnet_v2_m()

state_dict = torch.load(PTH_PATH, map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

# ====== 建立 dummy input ======
dummy_input = torch.randn(*INPUT_SIZE)

# ====== 轉 ONNX ======
print("Exporting EfficientNetV2-L to ONNX (opset 11, fixed shape)...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    dynamo=False,

    # ====== 之後如果要改成動態 batch，就把下面註解打開 ======
#    dynamic_axes={
#        "input": {0: "batch_size"},
#        "output": {0: "batch_size"}
#    }
    )

print("Export finished!")

# ====== 驗證 ONNX ======
print("Checking ONNX model...")

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

# ====== 顯示 opset ======
for imp in onnx_model.opset_import:
    print(f"ONNX opset version: {imp.version}")

# ====== 顯示 input / output shape ======
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

print("\nONNX model is valid ✅")
print(f"Saved to: {ONNX_PATH}")