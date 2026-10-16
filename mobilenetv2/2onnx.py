import torch
from torchvision import models
import onnx

# Load model
model = models.mobilenet_v2()
state_dict = torch.load("mobilenet_v2-b0353104.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mobilenet_v2_opset11_dynamic.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

# Check ONNX
model_onnx = onnx.load("mobilenet_v2_opset11_dynamic.onnx")
onnx.checker.check_model(model_onnx)

print("✅ ONNX export success!")