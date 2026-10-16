import os
import shutil
import onnx
from onnx import shape_inference
import ktc
import numpy as np
from PIL import Image


# ============================================================
# Config
# ============================================================
ONNX_PATH = "yolov5s.onnx"

OUT_DIR = "yolov5s_output"
DATA_DIR = "images"

MODEL_ID = 20090
MODEL_VER = "0001"
CHIP = "730"

# YOLOv5s usually uses 640x640 NCHW
IMG_H = 640
IMG_W = 640
IMG_C = 3

CLEAN_ONNX_PATH = "yolov5s_clean.onnx"
SHAPE_ONNX_PATH = "yolov5s_clean_shape.onnx"
FINAL_ONNX_PATH = "yolov5s_clean_shape_inputs.onnx"


# ============================================================
# Helpers
# ============================================================
def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def list_images(root):
    imgs = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                imgs.append(os.path.join(r, f))
    return sorted(imgs)


def safe_copy(src_path, dst_dir):
    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))

    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print("[copy] Skip same path")
        return dst_path

    shutil.copy(src_path, dst_path)
    print(f"[copy] {src_path} -> {dst_path}")
    return dst_path


def clean_name(name):
    if name is None:
        return name

    return (
        name.replace(":", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
    )


def get_shape(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(d.dim_value)
    return dims


def print_onnx_info(model, title="ONNX"):
    print(f"========== {title} Inputs ==========")
    for inp in model.graph.input:
        print(f"name: {inp.name}")
        print(f"shape: {get_shape(inp)}")

    print(f"========== {title} Outputs ==========")
    for out in model.graph.output:
        print(f"name: {out.name}")
        print(f"shape: {get_shape(out)}")

    print(f"========== {title} Opset ==========")
    for o in model.opset_import:
        print(f"domain={repr(o.domain)}, version={o.version}")

    print(f"========== {title} Node Types ==========")
    ops = sorted(set(n.op_type for n in model.graph.node))
    print(ops)


# ============================================================
# Step 1: clean / fix YOLOv5 ONNX
# ============================================================
def clean_yolov5_onnx(in_path, out_path):
    print(f"[clean] Loading ONNX: {in_path}")
    model = onnx.load(in_path)

    # 1. Remove unsupported ai.onnx.ml opset, if exists
    kept_opsets = []
    for opset in model.opset_import:
        if opset.domain == "ai.onnx.ml":
            print(f"[clean] Remove unsupported opset: ai.onnx.ml v{opset.version}")
            continue

        # KL730 usually prefers opset 11
        if opset.domain == "" and opset.version > 11:
            print(f"[clean] Main ONNX opset {opset.version} -> 11")
            opset.version = 11

        kept_opsets.append(opset)

    del model.opset_import[:]
    model.opset_import.extend(kept_opsets)

    # 2. Rename bad tensor names
    name_map = {}

    def add_name(name):
        if name and clean_name(name) != name:
            name_map[name] = clean_name(name)

    for x in model.graph.input:
        add_name(x.name)

    for x in model.graph.output:
        add_name(x.name)

    for x in model.graph.value_info:
        add_name(x.name)

    for x in model.graph.initializer:
        add_name(x.name)

    for node in model.graph.node:
        add_name(node.name)
        for n in node.input:
            add_name(n)
        for n in node.output:
            add_name(n)

    print(f"[clean] Rename tensor count: {len(name_map)}")

    for x in model.graph.input:
        if x.name in name_map:
            x.name = name_map[x.name]

    for x in model.graph.output:
        if x.name in name_map:
            x.name = name_map[x.name]

    for x in model.graph.value_info:
        if x.name in name_map:
            x.name = name_map[x.name]

    for x in model.graph.initializer:
        if x.name in name_map:
            x.name = name_map[x.name]

    for node in model.graph.node:
        if node.name in name_map:
            node.name = name_map[node.name]

        for i in range(len(node.input)):
            if node.input[i] in name_map:
                node.input[i] = name_map[node.input[i]]

        for i in range(len(node.output)):
            if node.output[i] in name_map:
                node.output[i] = name_map[node.output[i]]

    # 3. Fix input shape to NCHW [1,3,640,640]
    inp = model.graph.input[0]
    shape = inp.type.tensor_type.shape
    fixed_dims = [1, IMG_C, IMG_H, IMG_W]

    for d, v in zip(shape.dim, fixed_dims):
        d.ClearField("dim_param")
        d.dim_value = v

    print(f"[clean] Fixed input name : {model.graph.input[0].name}")
    print(f"[clean] Fixed input shape: {fixed_dims}")

    # 4. Warn if YOLO ONNX has postprocess / NMS
    ops = sorted(set(n.op_type for n in model.graph.node))
    if "NonMaxSuppression" in ops:
        print("[warn] This ONNX contains NonMaxSuppression.")
        print("[warn] KL730 may not support it. Better export raw YOLO head without NMS.")

    try:
        onnx.checker.check_model(model)
        print("[clean] ONNX checker OK")
    except Exception as e:
        print("[warn] ONNX checker failed, continue anyway:")
        print(e)

    onnx.save(model, out_path)
    print(f"[clean] Saved: {out_path}")
    return out_path


# ============================================================
# Step 2: add shape info
# ============================================================
def add_shape_info(in_path, out_path):
    print(f"[shape] Loading ONNX: {in_path}")
    model = onnx.load(in_path)

    print("[shape] Running onnx.shape_inference.infer_shapes...")
    model = shape_inference.infer_shapes(model)

    try:
        onnx.checker.check_model(model)
        print("[shape] ONNX checker OK")
    except Exception as e:
        print("[warn] ONNX checker failed, continue anyway:")
        print(e)

    onnx.save(model, out_path)
    print(f"[shape] Saved: {out_path}")
    return out_path


# ============================================================
# Step 3: remove extra graph inputs
# ============================================================
def clean_graph_inputs(in_path, out_path):
    print(f"[inputs] Loading ONNX: {in_path}")
    model = onnx.load(in_path)

    real_input = model.graph.input[0].name
    initializer_names = set(init.name for init in model.graph.initializer)

    print(f"[inputs] Real input should be: {real_input}")
    print(f"[inputs] Before graph.input count: {len(model.graph.input)}")

    new_inputs = []
    for inp in model.graph.input:
        if inp.name == real_input:
            new_inputs.append(inp)
            continue

        if inp.name in initializer_names:
            continue

        print(f"[warn] Keep non-initializer input: {inp.name}")
        new_inputs.append(inp)

    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

    print(f"[inputs] After graph.input count: {len(model.graph.input)}")
    for inp in model.graph.input:
        print(f"[inputs] input: {inp.name}, shape={get_shape(inp)}")

    try:
        onnx.checker.check_model(model)
        print("[inputs] ONNX checker OK")
    except Exception as e:
        print("[warn] ONNX checker failed, continue anyway:")
        print(e)

    onnx.save(model, out_path)
    print(f"[inputs] Saved: {out_path}")
    return out_path


# ============================================================
# YOLOv5 preprocess
# ============================================================
def letterbox_rgb(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    YOLOv5 letterbox:
    keep aspect ratio, pad to 640x640.
    Input PIL RGB image.
    Return PIL RGB image.
    """
    w, h = im.size
    new_h, new_w = new_shape

    r = min(new_w / w, new_h / h)
    resized_w = int(round(w * r))
    resized_h = int(round(h * r))

    im_resized = im.resize((resized_w, resized_h), Image.BILINEAR)

    canvas = Image.new("RGB", (new_w, new_h), color)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    left = pad_w // 2
    top = pad_h // 2

    canvas.paste(im_resized, (left, top))
    return canvas


def preprocess_yolov5_nchw(path):
    """
    YOLOv5 preprocess:
    RGB
    letterbox 640x640
    /255.0
    NCHW [1,3,640,640]
    """
    im = Image.open(path).convert("RGB")
    im = letterbox_rgb(im, new_shape=(IMG_H, IMG_W))

    arr = np.array(im, dtype=np.float32) / 255.0

    # HWC -> CHW -> NCHW
    arr = np.transpose(arr, (2, 0, 1))[None]
    return arr.astype(np.float32)


# ============================================================
# Main
# ============================================================
print("========== YOLOv5s ONNX -> KL730 NEF ==========")

assert os.path.isfile(ONNX_PATH), f"ONNX not found: {ONNX_PATH}"
assert os.path.isdir(DATA_DIR), f"Calibration image folder not found: {DATA_DIR}"

ensure_dir(OUT_DIR)

print("\n[1/6] Clean YOLOv5 ONNX")
clean_path = clean_yolov5_onnx(ONNX_PATH, CLEAN_ONNX_PATH)
safe_copy(clean_path, OUT_DIR)

print("\n[2/6] Add shape info")
shape_path = add_shape_info(clean_path, SHAPE_ONNX_PATH)
safe_copy(shape_path, OUT_DIR)

print("\n[3/6] Remove extra graph inputs")
final_path = clean_graph_inputs(shape_path, FINAL_ONNX_PATH)
safe_copy(final_path, OUT_DIR)

print("\n[4/6] Load final ONNX")
model = onnx.load(final_path)
print_onnx_info(model, title="Final ONNX")

if len(model.graph.input) != 1:
    raise RuntimeError("Final ONNX should have only one input.")

input_name = model.graph.input[0].name
input_shape = get_shape(model.graph.input[0])

if input_shape != [1, 3, IMG_H, IMG_W]:
    raise RuntimeError(f"Final input shape should be [1,3,{IMG_H},{IMG_W}], got {input_shape}")

# YOLOv5 是 CNN graph，這邊可以先試 Kneron optimizer。
# 如果 optimizer 失敗，就直接跳過，像 DeepLab / SegFormer 那樣用 shape_info ONNX。
try:
    print("[opt] Try Kneron onnx optimizer...")
    model = ktc.onnx_optimizer.onnx2onnx_flow(
        model,
        duplicate_shared_weights=False
    )
    print("[opt] Kneron optimizer OK")
except TypeError:
    print("[warn] This SDK does not support duplicate_shared_weights=False")
    print("[warn] Skip Kneron optimizer")
except Exception as e:
    print("[warn] Kneron optimizer failed:")
    print(e)
    print("[warn] Skip optimizer and use final ONNX directly")

print("\n[5/6] BIE Quantization")
km = ktc.ModelConfig(
    MODEL_ID,
    MODEL_VER,
    CHIP,
    onnx_model=model
)

imgs = list_images(DATA_DIR)
assert imgs, f"No images found under: {DATA_DIR}"

print(f"Calibration images: {len(imgs)}")

arrs = [preprocess_yolov5_nchw(p) for p in imgs]

# 重新抓 input name，避免 optimizer 改過 graph
input_name = model.graph.input[0].name

print(f"Input name: {input_name}")
print(f"Calibration input shape: {arrs[0].shape}")

bie_path = km.analysis({
    input_name: arrs
})

safe_copy(bie_path, OUT_DIR)
print(f"[ok] BIE generated: {bie_path}")

print("\n[6/6] Compile NEF")
nef_path = ktc.compile([km])

safe_copy(nef_path, OUT_DIR)
print(f"[ok] NEF generated: {nef_path}")

print("\n========== Done ==========")
print(f"Output folder: {OUT_DIR}")