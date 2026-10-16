import os
import shutil
import argparse
import onnx
import numpy as np
from PIL import Image
import ktc

# =========================================================
# EfficientNetV2 S/M/L 多尺寸 ONNX -> NEF 批次轉換腳本
#
# 支援：
#   efficientnet_v2_s_224.onnx
#   efficientnet_v2_s_384.onnx
#   efficientnet_v2_s_480.onnx
#   efficientnet_v2_s_640.onnx
#   efficientnet_v2_s_768.onnx
#   efficientnet_v2_m_224.onnx ...
#   efficientnet_v2_l_768.onnx
#
# 注意：
#   1. ONNX 必須是 fixed shape，例如 [1, 3, 384, 384]
#   2. 不要 dynamic batch / dynamic H / dynamic W
#   3. KL730 / Kneron 建議固定 shape
# =========================================================


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


def safe_copy(src_path, dst_dir, new_name=None):
    ensure_dir(dst_dir)

    if new_name is None:
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    else:
        dst_path = os.path.join(dst_dir, new_name)

    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print(f"[copy] Skip same path: {dst_path}")
        return dst_path

    shutil.copy(src_path, dst_path)
    print(f"[copy] {src_path} -> {dst_path}")
    return dst_path


def get_tensor_shape(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def print_onnx_io(model):
    print("\n[ONNX Inputs]")
    for i, inp in enumerate(model.graph.input):
        dims = get_tensor_shape(inp)
        print(f"  {i}: name={inp.name}, shape={dims}")

    print("\n[ONNX Outputs]")
    for i, out in enumerate(model.graph.output):
        dims = get_tensor_shape(out)
        print(f"  {i}: name={out.name}, shape={dims}")


def get_first_input_shape(model):
    inp = model.graph.input[0]
    return get_tensor_shape(inp)


def check_fixed_nchw_shape(input_shape, onnx_path):
    """
    要求 shape = [B, 3, H, W]
    而且 B/H/W 都必須是 int，不可以是 dynamic。
    """
    if len(input_shape) != 4:
        raise RuntimeError(f"{onnx_path}: unexpected input rank: {input_shape}")

    b, c, h, w = input_shape

    for x in input_shape:
        if not isinstance(x, int):
            raise RuntimeError(
                f"{onnx_path}: dynamic shape is not allowed for Kneron/KL730. "
                f"Got input shape: {input_shape}"
            )

    if c != 3:
        raise RuntimeError(
            f"{onnx_path}: expected NCHW input with C=3, got shape: {input_shape}"
        )

    return b, c, h, w


def preprocess_image(path, img_h, img_w):
    """
    EfficientNetV2 / TorchVision ImageNet 常用前處理：
    RGB -> resize -> /255 -> normalize -> CHW float32
    回傳 shape = (3, H, W)
    """
    im = Image.open(path).convert("RGB")
    im = im.resize((img_w, img_h), Image.BILINEAR)

    arr = np.array(im, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

    return arr.astype(np.float32)


def build_calib_batches(img_paths, batch_size, img_h, img_w):
    """
    將 calibration 圖片組成：
        batch_size = 1 時，每筆 shape = (1, 3, H, W)
    """
    single_arrs = []

    for p in img_paths:
        try:
            single_arrs.append(preprocess_image(p, img_h, img_w))
        except Exception as e:
            print(f"[warn] Skip bad image: {p}, error: {e}")

    usable = (len(single_arrs) // batch_size) * batch_size
    dropped = len(single_arrs) - usable

    if usable == 0:
        raise RuntimeError(
            f"Not enough valid images for batch_size={batch_size}. "
            f"Need at least {batch_size}, got {len(single_arrs)}."
        )

    if dropped > 0:
        print(
            f"[warn] Dropping last {dropped} image(s) because they do not fill a complete batch."
        )

    single_arrs = single_arrs[:usable]

    batched = []
    for i in range(0, usable, batch_size):
        batch_arr = np.stack(single_arrs[i:i + batch_size], axis=0)
        batched.append(batch_arr.astype(np.float32))

    return batched


def find_onnx_files(onnx_dir, variants, sizes):
    """
    只找指定 EfficientNetV2 S/M/L 與指定尺寸。
    檔名格式：
        efficientnet_v2_s_224.onnx
    """
    files = []

    for v in variants:
        for size in sizes:
            name = f"efficientnet_v2_{v}_{size}.onnx"
            path = os.path.join(onnx_dir, name)
            if os.path.isfile(path):
                files.append(path)
            else:
                print(f"[warn] ONNX not found, skip: {path}")

    return files


def convert_one_onnx_to_nef(
    onnx_path,
    data_dir,
    output_root,
    model_id,
    model_ver,
    chip,
    max_calib,
):
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    out_dir = os.path.join(output_root, base)
    ensure_dir(out_dir)

    print("\n" + "=" * 80)
    print(f"[START] {base}")
    print(f"ONNX     : {onnx_path}")
    print(f"OUT DIR  : {out_dir}")
    print(f"MODEL ID : {model_id}")
    print("=" * 80)

    # =========================
    # 1. Load ONNX
    # =========================
    print("\n[1/5] Loading ONNX...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    print_onnx_io(model)

    input_shape = get_first_input_shape(model)
    print(f"\nDetected ONNX input shape: {input_shape}")

    batch_size, c, img_h, img_w = check_fixed_nchw_shape(input_shape, onnx_path)

    print(f"Batch size : {batch_size}")
    print(f"Input H,W  : {img_h}, {img_w}")

    # =========================
    # 2. Optimize ONNX
    # =========================
    print("\n[2/5] Optimizing ONNX...")
    model = ktc.onnx_optimizer.onnx2onnx_flow(model)
    print("ONNX optimized OK")

    # optimizer 後重新抓 input name
    input_name = model.graph.input[0].name
    print(f"Input name after optimize: {input_name}")

    # =========================
    # 3. Create ModelConfig
    # =========================
    print("\n[3/5] Create KL730 ModelConfig...")
    km = ktc.ModelConfig(model_id, model_ver, chip, onnx_model=model)
    print(f"Model ID: {model_id}, Version: {model_ver}, Chip: {chip}")

    # =========================
    # 4. BIE Quantization
    # =========================
    print(f"\n[4/5] BIE Quantization with images from: {data_dir}")
    imgs = list_images(data_dir)
    assert imgs, f"No images found under {data_dir}"

    if len(imgs) > max_calib:
        imgs = imgs[:max_calib]

    print(f"Calibration images : {len(imgs)}")
    print(f"Batch size         : {batch_size}")
    print(f"Image size         : {img_h}x{img_w}")

    arrs = build_calib_batches(imgs, batch_size, img_h, img_w)

    print(f"Number of calibration batches: {len(arrs)}")
    print(f"Example input shape          : {arrs[0].shape}")

    bie_path = km.analysis({input_name: arrs})
    safe_copy(bie_path, out_dir, new_name=f"{base}.bie")
    print(f"BIE OK: {bie_path}")

    # =========================
    # 5. Compile NEF
    # =========================
    print("\n[5/5] Compiling NEF...")
    nef_path = ktc.compile([km])
    safe_copy(nef_path, out_dir, new_name=f"{base}.nef")
    print(f"NEF OK: {nef_path}")

    print(f"\n[DONE] {base}")
    print(f"Saved to: {out_dir}/")

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=".",
        help="ONNX models folder"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="images",
        help="BIE calibration images folder"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="nef_outputs",
        help="Output root folder"
    )

    parser.add_argument(
        "--chip",
        type=str,
        default="730",
        help="Target chip, default: 730"
    )

    parser.add_argument(
        "--model-ver",
        type=str,
        default="0001",
        help="Model version"
    )

    parser.add_argument(
        "--model-id-start",
        type=int,
        default=20000,
        help="Start model id. Each ONNX will use model_id_start + index"
    )

    parser.add_argument(
        "--max-calib",
        type=int,
        default=200,
        help="Max calibration images"
    )

    parser.add_argument(
        "--variants",
        type=str,
        default="s,m,l",
        help="EfficientNetV2 variants, example: s,m,l"
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default="224,384,480,640,768",
        help="Input sizes, example: 224,384,480,640,768"
    )

    args = parser.parse_args()

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    ensure_dir(args.output_root)

    print("========== Batch ONNX -> NEF Config ==========")
    print(f"ONNX dir       : {args.onnx_dir}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output root    : {args.output_root}")
    print(f"Chip           : {args.chip}")
    print(f"Model version  : {args.model_ver}")
    print(f"Model ID start : {args.model_id_start}")
    print(f"Max calib      : {args.max_calib}")
    print(f"Variants       : {variants}")
    print(f"Sizes          : {sizes}")
    print("=============================================")

    onnx_files = find_onnx_files(args.onnx_dir, variants, sizes)

    if not onnx_files:
        raise RuntimeError("No ONNX files found. Please check --onnx-dir, --variants, and --sizes.")

    print("\nONNX files to convert:")
    for p in onnx_files:
        print(f"  - {p}")

    success = []
    failed = []

    for idx, onnx_path in enumerate(onnx_files):
        model_id = args.model_id_start + idx

        try:
            convert_one_onnx_to_nef(
                onnx_path=onnx_path,
                data_dir=args.data_dir,
                output_root=args.output_root,
                model_id=model_id,
                model_ver=args.model_ver,
                chip=args.chip,
                max_calib=args.max_calib,
            )
            success.append(onnx_path)

        except Exception as e:
            print("\n" + "!" * 80)
            print(f"[FAILED] {onnx_path}")
            print(f"Error: {e}")
            print("!" * 80)
            failed.append((onnx_path, str(e)))

    print("\n\n========== Batch Convert Summary ==========")
    print(f"Total ONNX : {len(onnx_files)}")
    print(f"Success    : {len(success)}")
    print(f"Failed     : {len(failed)}")

    if success:
        print("\n[Success]")
        for p in success:
            print(f"  ✅ {p}")

    if failed:
        print("\n[Failed]")
        for p, err in failed:
            print(f"  ❌ {p}")
            print(f"     {err}")

    print("\nAll done.")


if __name__ == "__main__":
    main()