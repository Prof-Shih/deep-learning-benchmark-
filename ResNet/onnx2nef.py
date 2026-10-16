import os
import shutil
import onnx
import numpy as np
from PIL import Image
import ktc

# =========================
# Config (改這裡就好)
# =========================
ONNX_PATH = "resnet18_opset11.onnx"   # ✅ 你的模型
DATA_DIR  = "images"                 # 校正圖片資料夾
MODEL_ID  = 12345
MODEL_VER = "0001"
CHIP      = "730"
OUT_DIR   = "resnet18_output"

IMG_H, IMG_W = 224, 224   # ✅ ResNet18 input

MAX_CALIB = 100


# =========================
# Helpers
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                out.append(os.path.join(r, f))
    out.sort()
    return out

def preprocess(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((IMG_W, IMG_H), Image.BILINEAR)

    arr = np.array(im, dtype=np.float32) / 255.0

    # ✅ ImageNet normalize（ResNet 必須）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]

    return arr.astype(np.float32)

def safe_copy(src, dst_dir):
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy(src, dst)
    return dst


# =========================
# Main
# =========================
print("[1] Load ONNX")
model = onnx.load(ONNX_PATH)

print("[2] Optimize ONNX")
model = ktc.onnx_optimizer.onnx2onnx_flow(model)

print("[3] Build ModelConfig")
km = ktc.ModelConfig(MODEL_ID, MODEL_VER, CHIP, onnx_model=model)

print("[4] BIE Calibration")

imgs = list_images(DATA_DIR)
assert imgs, "No images found"

imgs = imgs[:MAX_CALIB]

arrs = [preprocess(p) for p in imgs]

input_name = model.graph.input[0].name

print(f"Input name: {input_name}")
print(f"Using {len(arrs)} images")

bie_path = km.analysis({input_name: arrs})
safe_copy(bie_path, OUT_DIR)

print("BIE done")

print("[5] Compile NEF")
nef_path = ktc.compile([km])
safe_copy(nef_path, OUT_DIR)

print("\n✅ Done!")
print(f"Output folder: {OUT_DIR}")