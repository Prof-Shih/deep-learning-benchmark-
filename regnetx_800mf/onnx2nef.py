import os
import shutil
import argparse
import onnx
import numpy as np
import ktc
from PIL import Image
from onnx import shape_inference


DEFAULT_ONNX_PATH = "RegNetX-800MF_dds_8gpu.onnx"
DEFAULT_DATA_DIR = "images"
DEFAULT_OUT_DIR = "regnetx800mf_original_output"

MODEL_ID = 20091
MODEL_VER = "0001"
CHIP = "730"

IMG_H = 224
IMG_W = 224
IMG_C = 3

CLEAN_ONNX_PATH = "regnetx800mf_original_clean.onnx"
SHAPE_ONNX_PATH = "regnetx800mf_original_shape.onnx"
FINAL_ONNX_PATH = "regnetx800mf_original_final.onnx"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_copy(src, dst_dir):
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
    return dst


def list_images(root):
    imgs = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                imgs.append(os.path.join(r, f))
    return sorted(imgs)


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


def print_onnx_info(model, title):
    print(f"\n========== {title} Inputs ==========")
    for inp in model.graph.input:
        print("name :", inp.name)
        print("shape:", get_shape(inp))

    print(f"\n========== {title} Outputs ==========")
    for out in model.graph.output:
        print("name :", out.name)
        print("shape:", get_shape(out))

    print(f"\n========== {title} Opset ==========")
    for o in model.opset_import:
        print(f"domain={repr(o.domain)}, version={o.version}")

    print(f"\n========== {title} Node Types ==========")
    print(sorted(set(n.op_type for n in model.graph.node)))


def clean_onnx(in_path, out_path):
    print("[1] Clean ONNX")
    model = onnx.load(in_path)

    print_onnx_info(model, "Original ONNX")

    kept_opsets = []
    for opset in model.opset_import:
        if opset.domain == "ai.onnx.ml":
            print(f"[clean] remove ai.onnx.ml v{opset.version}")
            continue
        kept_opsets.append(opset)

    del model.opset_import[:]
    model.opset_import.extend(kept_opsets)

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

    print("[clean] rename count:", len(name_map))

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

    inp = model.graph.input[0]
    fixed_shape = [1, IMG_C, IMG_H, IMG_W]
    for d, v in zip(inp.type.tensor_type.shape.dim, fixed_shape):
        d.ClearField("dim_param")
        d.dim_value = v

    print("[clean] input name :", model.graph.input[0].name)
    print("[clean] input shape:", fixed_shape)

    try:
        onnx.checker.check_model(model)
        print("[clean] checker OK")
    except Exception as e:
        print("[warn] checker failed, continue:")
        print(e)

    onnx.save(model, out_path)
    print("[clean] saved:", out_path)
    return out_path


def add_shape_info(in_path, out_path):
    print("\n[2] Add shape info")
    model = onnx.load(in_path)

    try:
        model = shape_inference.infer_shapes(model)
        print("[shape] infer_shapes OK")
    except Exception as e:
        print("[warn] shape inference failed, continue:")
        print(e)

    try:
        onnx.checker.check_model(model)
        print("[shape] checker OK")
    except Exception as e:
        print("[warn] checker failed, continue:")
        print(e)

    onnx.save(model, out_path)
    print("[shape] saved:", out_path)
    return out_path


def clean_graph_inputs(in_path, out_path):
    print("\n[3] Clean graph inputs")
    model = onnx.load(in_path)

    real_input = model.graph.input[0].name
    initializer_names = set(init.name for init in model.graph.initializer)

    new_inputs = []
    for inp in model.graph.input:
        if inp.name == real_input:
            new_inputs.append(inp)
        elif inp.name in initializer_names:
            continue
        else:
            new_inputs.append(inp)

    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

    print("[inputs] input count:", len(model.graph.input))
    for inp in model.graph.input:
        print("[inputs]", inp.name, get_shape(inp))

    try:
        onnx.checker.check_model(model)
        print("[inputs] checker OK")
    except Exception as e:
        print("[warn] checker failed, continue:")
        print(e)

    onnx.save(model, out_path)
    print("[inputs] saved:", out_path)
    return out_path


def try_kneron_optimizer(in_path, out_path):
    print("\n[4] Kneron optimizer")
    model = onnx.load(in_path)

    try:
        model = ktc.onnx_optimizer.onnx2onnx_flow(
            model,
            duplicate_shared_weights=False
        )
        print("[opt] optimizer OK")
        onnx.save(model, out_path)
        return out_path

    except TypeError:
        print("[warn] duplicate_shared_weights=False not supported, try default")
        try:
            model = ktc.onnx_optimizer.onnx2onnx_flow(model)
            print("[opt] default optimizer OK")
            onnx.save(model, out_path)
            return out_path
        except Exception as e:
            print("[warn] optimizer failed, use original:")
            print(e)
            shutil.copy(in_path, out_path)
            return out_path

    except Exception as e:
        print("[warn] optimizer failed, use original:")
        print(e)
        shutil.copy(in_path, out_path)
        return out_path


def preprocess_imagenet_nchw(path):
    img = Image.open(path).convert("RGB")

    w, h = img.size
    resize_size = 256

    if w < h:
        new_w = resize_size
        new_h = int(round(h * resize_size / w))
    else:
        new_h = resize_size
        new_w = int(round(w * resize_size / h))

    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - IMG_W) // 2
    top = (new_h - IMG_H) // 2
    img = img.crop((left, top, left + IMG_W, top + IMG_H))

    arr = np.array(img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None]

    return arr.astype(np.float32)


def preprocess_direct224_nchw(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None]

    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser("RegNetX-800MF ONNX to NEF original version")

    parser.add_argument("--onnx", type=str, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--images", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit_calib", type=int, default=0)

    parser.add_argument(
        "--preprocess",
        type=str,
        default="imagenet",
        choices=["imagenet", "direct224"]
    )

    args = parser.parse_args()

    print("========== RegNetX-800MF Original ONNX -> KL730 NEF ==========")
    print("ONNX      :", args.onnx)
    print("Images    :", args.images)
    print("Out dir   :", args.out_dir)
    print("Preprocess:", args.preprocess)
    print("=============================================================")

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(args.onnx)

    if not os.path.isdir(args.images):
        raise FileNotFoundError(args.images)

    ensure_dir(args.out_dir)

    clean_path = clean_onnx(args.onnx, CLEAN_ONNX_PATH)
    safe_copy(clean_path, args.out_dir)

    shape_path = add_shape_info(clean_path, SHAPE_ONNX_PATH)
    safe_copy(shape_path, args.out_dir)

    final_path = clean_graph_inputs(shape_path, FINAL_ONNX_PATH)
    safe_copy(final_path, args.out_dir)

    opt_path = try_kneron_optimizer(final_path, "regnetx800mf_original_opt.onnx")
    safe_copy(opt_path, args.out_dir)

    print("\n[5] Load final ONNX")
    model = onnx.load(opt_path)
    print_onnx_info(model, "Final ONNX")

    km = ktc.ModelConfig(
        MODEL_ID,
        MODEL_VER,
        CHIP,
        onnx_model=model
    )

    imgs = list_images(args.images)
    if args.limit_calib > 0:
        imgs = imgs[:args.limit_calib]

    if len(imgs) == 0:
        raise RuntimeError("No calibration images found")

    print("\n[6] BIE analysis")
    print("Calibration images:", len(imgs))

    if args.preprocess == "imagenet":
        arrs = [preprocess_imagenet_nchw(p) for p in imgs]
    else:
        arrs = [preprocess_direct224_nchw(p) for p in imgs]

    input_name = model.graph.input[0].name
    print("Input name:", input_name)
    print("Input sample shape:", arrs[0].shape)
    print("Input sample min/max:", float(arrs[0].min()), float(arrs[0].max()))

    bie_path = km.analysis({
        input_name: arrs
    })

    safe_copy(bie_path, args.out_dir)
    print("[ok] BIE:", bie_path)

    print("\n[7] Compile NEF")
    nef_path = ktc.compile([km])
    safe_copy(nef_path, args.out_dir)

    print("[ok] NEF:", nef_path)
    print("\n========== Done ==========")
    print("Output dir:", args.out_dir)


if __name__ == "__main__":
    main()