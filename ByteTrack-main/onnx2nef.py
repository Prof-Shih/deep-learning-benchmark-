import os
import shutil
import onnx
import numpy as np
from PIL import Image
import ktc

# =========================
# Config
# =========================
# 你的 ONNX。若你的 bytetrack_nano_mot17.onnx 實際上是 YOLOX 偵測網路，就填這個
ONNX_PATH = "bytetrack_nano_mot17.onnx"

# BIE 校正圖片資料夾
DATA_DIR = "images"

# KL730 model config
MODEL_ID = 20099
MODEL_VER = "0001"
CHIP = "730"

# 輸出資料夾
OUT_DIR = "bytetrack_yolox_output"

# 依你的模型輸入大小修改
# 很多 YOLOX nano / ByteTrack 版本會是 608x1088，也有可能是 640x640
IMG_H, IMG_W = 608, 1088

# 最多幾張圖拿來做 calibration
MAX_CALIB = 200

# 是否使用 letterbox
USE_LETTERBOX = True

# padding value，YOLOX 常見是 114
PAD_VALUE = 114.0


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


def letterbox_resize(im, target_w, target_h, pad_value=114):
    """
    YOLOX-style letterbox
    return:
      padded_img (H,W,3) float32
    """
    src_w, src_h = im.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = im.resize((new_w, new_h), Image.BILINEAR)
    resized_np = np.array(resized, dtype=np.float32)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.float32)
    canvas[:new_h, :new_w, :] = resized_np
    return canvas


def direct_resize(im, target_w, target_h):
    resized = im.resize((target_w, target_h), Image.BILINEAR)
    return np.array(resized, dtype=np.float32)


def preprocess(path):
    """
    YOLOX / ByteTrack detector commonly uses:
      - RGB
      - resize or letterbox
      - usually no mean/std normalization
      - float32
      - HWC -> CHW -> NCHW
    """
    im = Image.open(path).convert("RGB")

    if USE_LETTERBOX:
        arr = letterbox_resize(im, IMG_W, IMG_H, pad_value=PAD_VALUE)
    else:
        arr = direct_resize(im, IMG_W, IMG_H)

    # 若你的 ONNX 匯出時是吃 0~1，就打開下一行
    # arr = arr / 255.0

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

# 檢查是否像影像模型
if len(input_shape) != 4:
    raise RuntimeError(
        f"This ONNX first input shape is {input_shape}, not typical image tensor NCHW. "
        "It may not be a pure detector model, so it is likely NOT suitable for direct KL730 NEF compile."
    )

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