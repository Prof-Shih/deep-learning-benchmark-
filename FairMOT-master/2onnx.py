import os
import sys
import torch
import torch.nn as nn

# =========================
# Path
# =========================
FAIRMOT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_LIB = os.path.join(FAIRMOT_ROOT, "src", "lib")
sys.path.insert(0, SRC_LIB)

# =========================
# Create fake dcn_v2.py
# Replace DCNv2 with normal Conv2d
# =========================
fake_dcn_path = os.path.join(SRC_LIB, "dcn_v2.py")

fake_dcn_code = r'''
import torch
import torch.nn as nn

class DCN(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        deformable_groups=1,
        bias=True
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias
        )
'''

with open(fake_dcn_path, "w", encoding="utf-8") as f:
    f.write(fake_dcn_code)

print("[INFO] Fake dcn_v2.py created:", fake_dcn_path)

# =========================
# Import FairMOT model
# =========================
from models.model import create_model, load_model


# =========================
# Config
# =========================
WEIGHT_PATH = "fairmot_dla34.pth"
OUT_ONNX = "fairmot_dla34_no_dcn_opset11.onnx"

ARCH = "dla_34"

HEADS = {
    "hm": 1,
    "wh": 4,
    "id": 128,
    "reg": 2,
}

HEAD_CONV = 256

INPUT_H = 608
INPUT_W = 1088


# =========================
# Wrapper
# =========================
class FairMOTONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)

        # FairMOT usually returns list of dict
        if isinstance(outputs, list):
            outputs = outputs[-1]

        hm = outputs["hm"]
        wh = outputs["wh"]
        identity = outputs["id"]
        reg = outputs["reg"]

        # FairMOT inference usually applies sigmoid to hm
        hm = torch.sigmoid(hm)

        return hm, wh, identity, reg


def main():
    device = torch.device("cpu")

    print("[INFO] Creating model...")
    model = create_model(ARCH, HEADS, HEAD_CONV)

    print("[INFO] Loading weight:", WEIGHT_PATH)
    model = load_model(model, WEIGHT_PATH)

    model.to(device)
    model.eval()

    wrapped_model = FairMOTONNXWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    dummy_input = torch.randn(1, 3, INPUT_H, INPUT_W, device=device)

    print("[INFO] Testing forward...")
    with torch.no_grad():
        outputs = wrapped_model(dummy_input)
        for i, out in enumerate(outputs):
            print(f"[INFO] Output {i} shape:", tuple(out.shape))

    print("[INFO] Exporting ONNX...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        OUT_ONNX,
        opset_version=11,
        input_names=["images"],
        output_names=["hm", "wh", "id", "reg"],
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print("[OK] Export done:", OUT_ONNX)


if __name__ == "__main__":
    main()