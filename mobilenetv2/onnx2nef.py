import os
import shutil
import onnx
import numpy as np
from PIL import Image
import ktc

# =========================
# Config
# =========================
ONNX_PATH = "mobilenet_v2_opset11.onnx"
DATA_DIR = "images"

MODEL_ID = 20099
MODEL_VER = "0001"
CHIP = "730"

OUT_DIR = "mobilenetv2_output"

IMG_H, IMG_W = 224, 224
MAX_CALIB = 200

# ImageNet normalization for torchvision MobileNetV2
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def safe_copy(src_path, dst_dir):
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print(f"[copy] Skip (same path): {dst_path}")
        return dst_path
    shutil.copy(src_path, dst_path)
    print(f"[copy] -> {dst_path}")
    return dst_path


def preprocess(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return arr.astype(np.float32)


def inspect_first_input_shape(model):
    if len(model.graph.input) == 0:
        raise RuntimeError("No ONNX input found")
    inp = model.graph.input[0]
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return inp.name, dims


# =========================
# Main
# =========================
print(f"[1/5] Loading ONNX: {ONNX_PATH}")
assert os.path.isfile(ONNX_PATH), f"ONNX not found: {ONNX_PATH}"

model = onnx.load(ONNX_PATH)
print_onnx_io(model)

input_name, input_shape = inspect_first_input_shape(model)
print(f"\n[INFO] First input name: {input_name}")
print(f"[INFO] First input shape: {input_shape}")

if input_shape != [1, 3, 224, 224]:
    print("[WARN] Expected MobileNetV2 input shape usually is [1, 3, 224, 224]")

print("\n[2/5] Optimizing ONNX...")
model = ktc.onnx_optimizer.onnx2onnx_flow(model)
print("   ONNX loaded + optimized")

print("\n[3/5] Create KL730 ModelConfig...")
km = ktc.ModelConfig(MODEL_ID, MODEL_VER, CHIP, onnx_model=model)
print(f"   Model ID: {MODEL_ID}, Version: {MODEL_VER}, Chip: {CHIP}")

print(f"\n[4/5] BIE Quantization with images from: {DATA_DIR}")
imgs = list_images(DATA_DIR)
assert imgs, f"No images found under {DATA_DIR}"

if len(imgs) > MAX_CALIB:
    imgs = imgs[:MAX_CALIB]

print(f"   Using {len(imgs)} images for calibration")

arrs = [preprocess(p) for p in imgs]
print(f"   Input name: {input_name}")
print(f"   Example input shape: {arrs[0].shape}")

bie_path = km.analysis({input_name: arrs})
safe_copy(bie_path, OUT_DIR)
print(f"   BIE OK: {bie_path}")

print("\n[5/5] Compiling NEF...")
nef_path = ktc.compile([km])
safe_copy(nef_path, OUT_DIR)
print(f"   NEF OK: {nef_path}")

print(f"\nDone! Files saved to: {OUT_DIR}/")