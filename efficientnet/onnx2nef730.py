import os
import shutil
import onnx
import numpy as np
from PIL import Image
import ktc

# =========================
# Config
# =========================
ONNX_PATH = "efficientnet_v2_l_768.onnx"   # batch=2 的 ONNX
DATA_DIR  = "images"                       # BIE 校正圖資料夾
MODEL_ID  = 20015                          # 可自行改
MODEL_VER = "0001"
CHIP      = "730"
OUT_DIR   = "efficientnet_v2"

# efficientnet_v2_s input size
IMG_H, IMG_W = 768, 768

# 這次要測的固定 batch
BATCH_SIZE = 1

# 最多拿幾張圖做 calibration
MAX_CALIB = 200

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
    """
    單張前處理，回傳 shape = (3, H, W)
    之後再由 build_calib_batches() 組成 (B, 3, H, W)
    """
    im = Image.open(path).convert("RGB")
    im = im.resize((IMG_W, IMG_H), Image.BILINEAR)

    arr = np.array(im, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))   # CHW
    return arr.astype(np.float32)

def build_calib_batches(img_paths, batch_size):
    """
    將單張圖片前處理後，組成固定 batch 的 calibration 資料。
    每一筆 shape = (batch_size, 3, H, W)
    """
    single_arrs = [preprocess(p) for p in img_paths]

    usable = (len(single_arrs) // batch_size) * batch_size
    dropped = len(single_arrs) - usable

    if usable == 0:
        raise RuntimeError(
            f"Not enough images for batch_size={batch_size}. "
            f"Need at least {batch_size} images, got {len(single_arrs)}."
        )

    if dropped > 0:
        print(f"[warn] Dropping last {dropped} image(s) because they do not fill a complete batch.")

    single_arrs = single_arrs[:usable]

    batched = []
    for i in range(0, usable, batch_size):
        batch_arr = np.stack(single_arrs[i:i + batch_size], axis=0)  # (B, 3, H, W)
        batched.append(batch_arr.astype(np.float32))

    return batched

def safe_copy(src_path, dst_dir):
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print(f"[copy] Skip (same path): {dst_path}")
        return dst_path
    shutil.copy(src_path, dst_path)
    print(f"[copy] -> {dst_path}")
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

def get_first_input_shape(model):
    inp = model.graph.input[0]
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims

# =========================
# Main
# =========================
print(f"[1/5] Loading ONNX: {ONNX_PATH}")
assert os.path.isfile(ONNX_PATH), f"ONNX not found: {ONNX_PATH}"

model = onnx.load(ONNX_PATH)
print_onnx_io(model)

input_shape = get_first_input_shape(model)
print(f"\nDetected ONNX input shape: {input_shape}")

if len(input_shape) != 4:
    raise RuntimeError(f"Unexpected input rank: {input_shape}")

if input_shape[0] != BATCH_SIZE:
    raise RuntimeError(
        f"ONNX batch size mismatch: script expects BATCH_SIZE={BATCH_SIZE}, "
        f"but model input shape is {input_shape}"
    )

if input_shape[1] != 3 or input_shape[2] != IMG_H or input_shape[3] != IMG_W:
    raise RuntimeError(
        f"ONNX input shape mismatch: expected [{BATCH_SIZE}, 3, {IMG_H}, {IMG_W}], "
        f"got {input_shape}"
    )

# toolchain 常需要先做 optimizer / shape 補全
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

print(f"   Found {len(imgs)} images before batching")
print(f"   Batch size = {BATCH_SIZE}")

arrs = build_calib_batches(imgs, BATCH_SIZE)

input_name = model.graph.input[0].name
print(f"   Input name: {input_name}")
print(f"   Number of calibration batches: {len(arrs)}")
print(f"   Example batched input shape: {arrs[0].shape}")

# BIE analysis
#bie_path = km.analysis({input_name: arrs})
#safe_copy(bie_path, OUT_DIR)
#print(f"   BIE OK: {bie_path}")
bie_path = km.analysis({input_name: arrs})
print(f"   BIE OK: {bie_path}")

# compile NEF
print("\n[5/5] Compiling NEF...")
nef_path = ktc.compile([km])
#safe_copy(nef_path, OUT_DIR)
print(f"   NEF OK: {nef_path}")

print(f"\nDone! Files saved to: {OUT_DIR}/")

# =========================
# Remove /data1/kneron_flow/output before moving
# =========================
REMOVE_DIR = "/data1/kneron_flow/output"

print("\n[clean] Removing output folder before move...")
if os.path.exists(REMOVE_DIR):
    shutil.rmtree(REMOVE_DIR)
    print(f"   Removed: {REMOVE_DIR}")
else:
    print(f"   Skip, not found: {REMOVE_DIR}")

# =========================
# Move whole project folder to /docker_mount
# 等同於：
# mv /data1/kneron_flow /docker_mount/kneron_flow
# =========================
SRC_DIR = "/data1/kneron_flow"
DST_PARENT = "/docker_mount"
DST_DIR = os.path.join(DST_PARENT, os.path.basename(SRC_DIR.rstrip("/")))

print("\n[6/6] Moving folder to docker_mount...")
print(f"   From: {SRC_DIR}")
print(f"   To  : {DST_DIR}")

if not os.path.exists(SRC_DIR):
    raise RuntimeError(f"Source folder not found: {SRC_DIR}")

os.makedirs(DST_PARENT, exist_ok=True)

if os.path.exists(DST_DIR):
    print(f"   Remove old destination folder: {DST_DIR}")
    shutil.rmtree(DST_DIR)

shutil.move(SRC_DIR, DST_DIR)

print("   Move OK ✅")
print(f"\nAll done! Folder moved to: {DST_DIR}")

# 結束提示音
os.system('printf "\\a"')