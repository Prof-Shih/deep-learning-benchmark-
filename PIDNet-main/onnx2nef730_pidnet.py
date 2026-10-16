import ktc, onnx, shutil, os
import numpy as np
from PIL import Image

# --- Config ---
ONNX_PATH = "PIDNet_L_Cityscapes_val.onnx"   # PIDNet ONNX 模型
DATA_DIR  = "images"             # 校正圖資料夾
MODEL_ID, MODEL_VER, CHIP = 20089, "0001", "730"  # 使用新的 MODEL_ID
OUT_DIR   = "pidnet_output"     # 輸出 BIE/NEF 文件夾
IMG_H, IMG_W = 1024, 2048          # PIDNet input size (Cityscapes)

# --- Helpers ---
def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def list_images(root):
    return [os.path.join(r, f) for r, _, fs in os.walk(root)
            for f in fs if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

def preprocess(path):
    """
    PIDNet segmentation 預處理：
    1. 讀取 RGB
    2. resize 到 1024x2048
    3. normalize (ImageNet mean/std)
    4. CHW → NCHW
    """
    im = Image.open(path).convert("RGB")
    im = im.resize((IMG_W, IMG_H), Image.BILINEAR)

    arr = np.array(im, dtype=np.float32) / 255.0

    # ImageNet normalization (和 PIDNet 訓練時使用的一樣)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))[None]  # HWC → CHW → NCHW
    return arr

def safe_copy(src_path, dst_dir):
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))

    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print(f"[copy] Skip (same path)")
        return dst_path
    
    shutil.copy(src_path, dst_path)
    print(f"[copy] → {dst_path}")
    return dst_path


# --- Pipeline ---
print(f"[1/4] Loading ONNX: {ONNX_PATH}")
assert os.path.isfile(ONNX_PATH), f"ONNX not found: {ONNX_PATH}"
model = onnx.load(ONNX_PATH)
model = ktc.onnx_optimizer.onnx2onnx_flow(model)
print("   ONNX loaded + optimized")


print("[2/4] Create KL730 ModelConfig...")
km = ktc.ModelConfig(MODEL_ID, MODEL_VER, CHIP, onnx_model=model)
print(f"   Model ID: {MODEL_ID}, Version: {MODEL_VER}, Chip: {CHIP}")


print(f"[3/4] BIE Quantization with images from: {DATA_DIR}")
imgs = list_images(DATA_DIR)
assert imgs, f"No images found under {DATA_DIR}"
print(f"   Found {len(imgs)} images for calibration")

arrs = [preprocess(p) for p in imgs]
input_name = model.graph.input[0].name
print(f"   Input name: {input_name}, Shape: {arrs[0].shape}")

bie_path = km.analysis({input_name: arrs})
safe_copy(bie_path, OUT_DIR)
print(f"   ✔ BIE OK: {bie_path}")


print("[4/4] Compiling NEF...")
nef_path = ktc.compile([km])
safe_copy(nef_path, OUT_DIR)
print(f"   ✔ NEF OK: {nef_path}")

print(f"\n🎉 Done! Files saved to {OUT_DIR}/")

