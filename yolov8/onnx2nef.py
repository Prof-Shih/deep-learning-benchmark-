import os
import shutil
import onnx
import numpy as np
from PIL import Image
import ktc

# =========================
# Config (改這裡就好)
# =========================
ONNX_PATH = "yolov8n.onnx"
DATA_DIR  = "images"
MODEL_ID  = 20099
MODEL_VER = "0001"
CHIP      = "730"
OUT_DIR   = "yolov8n_output"

# YOLOv8 input size (一定要跟你export一致)
IMG_H, IMG_W = 640, 640

MAX_CALIB = 100

# =========================
# Utils
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
    """
    YOLOv8 preprocessing:
    1) RGB
    2) resize to 640x640
    3) /255
    4) HWC -> CHW
    """

    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]

    return arr.astype(np.float32)

def safe_copy(src_path, dst_dir):
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copy(src_path, dst_path)
    return dst_path

def print_onnx_io(model):
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
        print(f"{i}: {inp.name}, shape={dims}")

# =========================
# Main
# =========================
print("[1] Load ONNX")
model = onnx.load(ONNX_PATH)
print_onnx_io(model)

# 👉 修 dynamic shape（YOLO常見問題）
print("\n[Fix] Force static shape")
input_tensor = model.graph.input[0]
input_tensor.type.tensor_type.shape.dim[0].dim_value = 1
input_tensor.type.tensor_type.shape.dim[1].dim_value = 3
input_tensor.type.tensor_type.shape.dim[2].dim_value = IMG_H
input_tensor.type.tensor_type.shape.dim[3].dim_value = IMG_W

# =========================
# Optimize
# =========================
print("\n[2] Optimize ONNX")
model = ktc.onnx_optimizer.onnx2onnx_flow(model)

# =========================
# ModelConfig
# =========================
print("\n[3] Create ModelConfig")
km = ktc.ModelConfig(MODEL_ID, MODEL_VER, CHIP, onnx_model=model)

# =========================
# BIE
# =========================
print("\n[4] BIE Analysis")

imgs = list_images(DATA_DIR)
assert imgs, "No images found!"

imgs = imgs[:MAX_CALIB]
print(f"Use {len(imgs)} images")

arrs = [preprocess(p) for p in imgs]

input_name = model.graph.input[0].name
print("Input name:", input_name)

bie_path = km.analysis({input_name: arrs})
safe_copy(bie_path, OUT_DIR)

# =========================
# Compile NEF
# =========================
print("\n[5] Compile NEF")

nef_path = ktc.compile([km])
safe_copy(nef_path, OUT_DIR)

print("\n✅ DONE → NEF 在:", OUT_DIR)