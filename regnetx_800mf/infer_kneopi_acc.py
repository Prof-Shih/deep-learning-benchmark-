import os
import sys
import time
import json
import argparse
import numpy as np
import cv2

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp


DEFAULT_USB_PORT_ID = 5
DEFAULT_SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
DEFAULT_MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')
DEFAULT_IMAGE_DIR_PATH = os.path.join(PWD, '../imagenet-mini/val')

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
RESIZE_SIZE = 256

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def get_image_paths(folder):
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(IMG_EXTS):
                files.append(os.path.join(root, name))
    return sorted(files)


def build_class_map_by_sorted_folders(root_dir):
    classes = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            classes.append(name)

    classes.sort()

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    return class_to_idx, idx_to_class


def build_class_map_from_imagenet_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    class_to_idx = {}
    idx_to_class = {}

    for k, v in data.items():
        idx = int(k)
        wnid = v[0]
        class_to_idx[wnid] = idx
        idx_to_class[idx] = wnid

    return class_to_idx, idx_to_class


def output_to_numpy(output):
    if hasattr(output, 'ndarray'):
        return np.array(output.ndarray, dtype=np.float32)
    if hasattr(output, 'np_array'):
        return np.array(output.np_array, dtype=np.float32)
    if hasattr(output, 'data'):
        return np.array(output.data, dtype=np.float32)
    if hasattr(output, 'float_data'):
        return np.array(output.float_data, dtype=np.float32)
    if hasattr(output, 'value'):
        return np.array(output.value, dtype=np.float32)
    if hasattr(output, 'output'):
        return np.array(output.output, dtype=np.float32)

    return np.array(output, dtype=np.float32)


def topk(logits, k=5):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    k = min(k, logits.size)

    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]

    return idx.astype(np.int64)


def preprocess_bgr565_direct224(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

    img_bgr565 = cv2.cvtColor(img, cv2.COLOR_BGR2BGR565)

    return img_bgr565


def preprocess_bgr565_imagenet_crop(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    if w < h:
        new_w = RESIZE_SIZE
        new_h = int(round(h * RESIZE_SIZE / w))
    else:
        new_h = RESIZE_SIZE
        new_w = int(round(w * RESIZE_SIZE / h))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    left = (new_w - INPUT_WIDTH) // 2
    top = (new_h - INPUT_HEIGHT) // 2

    img = img[top:top + INPUT_HEIGHT, left:left + INPUT_WIDTH, :]

    if img.shape[0] != INPUT_HEIGHT or img.shape[1] != INPUT_WIDTH:
        raise ValueError(f"Unexpected crop shape: {img.shape}")

    img_bgr565 = cv2.cvtColor(img, cv2.COLOR_BGR2BGR565)

    return img_bgr565


def print_descriptor(model_nef_descriptor):
    try:
        model = model_nef_descriptor.models[0]

        print("\n[NEF Descriptor]")
        print("Model ID:", model.id)

        print("Input nodes:")
        for i, node in enumerate(model.input_nodes):
            print(
                f"  [{i}] name={getattr(node, 'name', 'N/A')}, "
                f"layout={getattr(node, 'data_layout', 'N/A')}"
            )

        print("Output nodes:")
        for i, node in enumerate(model.output_nodes):
            print(
                f"  [{i}] name={getattr(node, 'name', 'N/A')}, "
                f"layout={getattr(node, 'data_layout', 'N/A')}"
            )

    except Exception as e:
        print("[warn] Cannot print descriptor:", e)


def main():
    parser = argparse.ArgumentParser("Kneo Pi RegNetX-800MF accuracy original")

    parser.add_argument("--port", "-p", type=int, default=DEFAULT_USB_PORT_ID)
    parser.add_argument("--firmware", type=str, default=DEFAULT_SCPU_FW_PATH)
    parser.add_argument("--nef", type=str, default=DEFAULT_MODEL_FILE_PATH)
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGE_DIR_PATH)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)

    parser.add_argument(
        "--preprocess",
        type=str,
        default="direct224",
        choices=["direct224", "imagenet_crop"]
    )

    parser.add_argument("--class_index_json", type=str, default="")
    parser.add_argument("--label_offset", type=int, default=0)

    args = parser.parse_args()

    print("========== Kneo Pi RegNetX-800MF Accuracy Original ==========")
    print("Port       :", args.port)
    print("Firmware   :", args.firmware)
    print("NEF        :", args.nef)
    print("Images     :", args.images)
    print("Preprocess :", args.preprocess)
    print("Label offset:", args.label_offset)
    print("============================================================")

    if not os.path.isdir(args.images):
        raise FileNotFoundError(args.images)

    if not os.path.isfile(args.nef):
        raise FileNotFoundError(args.nef)

    if not os.path.isfile(args.firmware):
        raise FileNotFoundError(args.firmware)

    if args.class_index_json:
        class_to_idx, idx_to_class = build_class_map_from_imagenet_json(args.class_index_json)
        print("\n[Class Map] imagenet_class_index_json")
    else:
        class_to_idx, idx_to_class = build_class_map_by_sorted_folders(args.images)
        print("\n[Class Map] sorted folders")

    print("Classes:", len(class_to_idx))
    print("First class:", idx_to_class.get(0, "N/A"))
    print("Last class :", idx_to_class.get(len(idx_to_class) - 1, "N/A"))

    print("\n[First 20 classes]")
    for i in range(min(20, len(idx_to_class))):
        print(f"{i:04d}: {idx_to_class[i]}")

    print("\n[Check USB]")
    usb_speed = get_device_usb_speed_by_port_id(usb_port_id=args.port)
    if usb_speed not in (
        kp.UsbSpeed.KP_USB_SPEED_SUPER,
        kp.UsbSpeed.KP_USB_SPEED_HIGH
    ):
        print("[Warning] Device is not running at super/high speed")
    else:
        print(" - Success")

    print("\n[Connect]")
    device_group = kp.core.connect_devices(usb_port_ids=[args.port])
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print(" - Success")

    print("\n[Load Firmware]")
    kp.core.load_firmware_from_file(
        device_group=device_group,
        scpu_fw_path=args.firmware,
        ncpu_fw_path=""
    )
    print(" - Success")

    print("\n[Load Model]")
    model_nef_descriptor = kp.core.load_model_from_file(
        device_group=device_group,
        file_path=args.nef
    )
    print(" - Success")

    print_descriptor(model_nef_descriptor)

    model_id = model_nef_descriptor.models[0].id
    print("\nModel ID:", model_id)

    image_paths = get_image_paths(args.images)

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    print("\nImages:", len(image_paths))

    if len(image_paths) == 0:
        raise RuntimeError("No image found")

    valid = 0
    skipped = 0
    top1_correct = 0
    top5_correct = 0
    total_infer_time = 0.0
    output_checked = False
    output_size = None

    print("\n[Start Inference]")

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            rel_path = os.path.relpath(image_path, args.images).replace("\\", "/")
            gt_class_name = rel_path.split("/")[0]

            if gt_class_name not in class_to_idx:
                raise ValueError(f"class folder not found in map: {gt_class_name}")

            gt_idx = class_to_idx[gt_class_name] + args.label_offset

            if args.preprocess == "direct224":
                img_565 = preprocess_bgr565_direct224(image_path)
            else:
                img_565 = preprocess_bgr565_imagenet_crop(image_path)

            generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
                model_id=model_id,
                inference_number=0,
                input_node_image_list=[
                    kp.GenericInputNodeImage(
                        image=img_565,
                        image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565
                    )
                ]
            )

            t0 = time.perf_counter()

            kp.inference.generic_image_inference_send(
                device_group=device_group,
                generic_inference_input_descriptor=generic_inference_input_descriptor
            )

            raw_result = kp.inference.generic_image_inference_receive(
                device_group=device_group
            )

            t1 = time.perf_counter()
            total_infer_time += (t1 - t0)

            output = kp.inference.generic_inference_retrieve_float_node(
                node_idx=0,
                generic_raw_result=raw_result,
                channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
            )

            logits = output_to_numpy(output).reshape(-1)

            if not output_checked:
                output_size = logits.size
                print("\n[Output Check]")
                print("Output size:", output_size)
                print("Class count:", len(class_to_idx))
                print("First 10 logits:", logits[:10])
                print("Logit min/max:", float(np.min(logits)), float(np.max(logits)))

                if output_size == 1001 and args.label_offset == 0:
                    print("[Warning] output is 1001, maybe try --label_offset 1")

                output_checked = True

            top5_idx = topk(logits, 5)
            pred_top1 = int(top5_idx[0])

            if pred_top1 == gt_idx:
                top1_correct += 1

            if gt_idx in top5_idx:
                top5_correct += 1

            valid += 1

        except Exception as e:
            print(f"[Skip] {image_path} -> {e}")
            skipped += 1
            continue

        if idx % args.print_every == 0 or idx == len(image_paths):
            top1 = top1_correct / valid * 100 if valid else 0.0
            top5 = top5_correct / valid * 100 if valid else 0.0
            avg_ms = total_infer_time / valid * 1000 if valid else 0.0
            fps = valid / total_infer_time if total_infer_time > 0 else 0.0

            print(
                f"{idx:05d}/{len(image_paths):05d} | "
                f"Valid: {valid:05d} | "
                f"Top-1: {top1:.2f}% | "
                f"Top-5: {top5:.2f}% | "
                f"Avg infer: {avg_ms:.2f} ms | "
                f"FPS: {fps:.2f}"
            )

    print("\n========== Final Result ==========")
    print("Valid images  :", valid)
    print("Skipped images:", skipped)
    print("Output size   :", output_size)
    print("Preprocess    :", args.preprocess)

    if valid == 0:
        print("No valid inference")
        return

    top1 = top1_correct / valid * 100
    top5 = top5_correct / valid * 100
    avg_ms = total_infer_time / valid * 1000
    fps = valid / total_infer_time if total_infer_time > 0 else 0.0

    print("Top-1 Correct :", top1_correct)
    print("Top-5 Correct :", top5_correct)
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    print(f"Total infer   : {total_infer_time:.4f} s")
    print(f"Avg infer     : {avg_ms:.2f} ms")
    print(f"FPS           : {fps:.2f}")
    print("==================================")


if __name__ == "__main__":
    main()