import os
import sys
import time
import argparse
import numpy as np
import cv2

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp
from pycocotools.coco import COCO


DEFAULT_USB_PORT_ID = 5
DEFAULT_SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
DEFAULT_MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')

DEFAULT_IMAGE_DIR = '../data/val2017'
DEFAULT_ANN_PATH = '../data/annotations_trainval2017/instances_val2017.json'
DEFAULT_SAVE_DIR = './yolov5_kneopi_raw_npy'

DEFAULT_INPUT_WIDTH = 640
DEFAULT_INPUT_HEIGHT = 640

# 改成 60000，避免連續推論時 USB timeout
TIMEOUT_MS = 60000

# 每張成功後稍微休息，降低 USB / 裝置連續卡住機率
SLEEP_AFTER_SAVE = 0.02


def get_image_id(path):
    name = os.path.basename(path)
    return int(os.path.splitext(name)[0])


def list_coco_images(image_dir, coco, limit=0):
    image_paths = []

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        p = os.path.join(image_dir, info["file_name"])
        if os.path.isfile(p):
            image_paths.append(p)

    image_paths.sort()

    if limit > 0:
        image_paths = image_paths[:limit]

    return image_paths


def letterbox_bgr(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w0, new_h / h0)
    resize_w = int(round(w0 * r))
    resize_h = int(round(h0 * r))

    resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)

    pad_w = (new_w - resize_w) // 2
    pad_h = (new_h - resize_h) // 2

    canvas[pad_h:pad_h + resize_h, pad_w:pad_w + resize_w] = resized

    return canvas


def rgb_to_rgb565_uint8(rgb_img):
    rgb = rgb_img.astype(np.uint16)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

    out = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 2), dtype=np.uint8)
    out[:, :, 0] = rgb565 & 0xFF
    out[:, :, 1] = (rgb565 >> 8) & 0xFF

    return out


def preprocess_to_bgr565(image_path, input_w, input_h):
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_lb = letterbox_bgr(img_bgr, new_shape=(input_w, input_h))

    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_rgb565 = rgb_to_rgb565_uint8(img_rgb)

    return img_rgb565

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


def main():
    parser = argparse.ArgumentParser(description="Kneo Pi YOLOv5 save raw npy outputs")

    parser.add_argument("--port", "-p", type=int, default=DEFAULT_USB_PORT_ID)
    parser.add_argument("--firmware", type=str, default=DEFAULT_SCPU_FW_PATH)
    parser.add_argument("--nef", type=str, default=DEFAULT_MODEL_FILE_PATH)

    parser.add_argument("--images", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ann", type=str, default=DEFAULT_ANN_PATH)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)

    parser.add_argument("--input_width", type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument("--input_height", type=int, default=DEFAULT_INPUT_HEIGHT)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()

    print("========== Kneo Pi YOLOv5 Save Raw NPY ==========")
    print(f"USB Port   : {args.port}")
    print(f"Firmware   : {args.firmware}")
    print(f"NEF        : {args.nef}")
    print(f"Images     : {args.images}")
    print(f"Annotation : {args.ann}")
    print(f"Save dir   : {args.save_dir}")
    print(f"Input      : {args.input_width}x{args.input_height}")
    print(f"Timeout    : {TIMEOUT_MS} ms")
    print(f"Sleep/save : {SLEEP_AFTER_SAVE} s")
    print("=================================================")

    if not os.path.isdir(args.images):
        print(f"Error: image folder not found -> {args.images}")
        sys.exit(1)

    if not os.path.isfile(args.ann):
        print(f"Error: annotation file not found -> {args.ann}")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n[Load COCO Annotation]")
    coco = COCO(args.ann)

    image_paths = list_coco_images(args.images, coco, args.limit)

    if len(image_paths) == 0:
        print("Error: no images found.")
        sys.exit(1)

    print(f"Images found: {len(image_paths)}")

    print("\n[Check Device USB Speed]")
    try:
        usb_speed = get_device_usb_speed_by_port_id(usb_port_id=args.port)

        if usb_speed not in (
            kp.UsbSpeed.KP_USB_SPEED_SUPER,
            kp.UsbSpeed.KP_USB_SPEED_HIGH
        ):
            print("\033[91m[Warning] Device is not running at super/high speed.\033[0m")
        else:
            print(" - Success")

    except Exception as exception:
        print(f"Error: check device USB speed fail, port ID = {args.port}, error = [{exception}]")
        sys.exit(1)

    print("\n[Connect Device]")
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port])
        print(" - Success")
    except kp.ApiKPException as exception:
        print(f"Error: connect device fail, port ID = {args.port}, error = [{exception}]")
        sys.exit(1)

    print("\n[Set Device Timeout]")
    try:
        kp.core.set_timeout(device_group=device_group, milliseconds=TIMEOUT_MS)
        print(f" - Success, timeout = {TIMEOUT_MS} ms")
    except Exception as exception:
        print(f"Error: set timeout fail, error = [{exception}]")
        sys.exit(1)

    print("\n[Upload Firmware]")
    if not os.path.isfile(args.firmware):
        print(f"Error: firmware not found -> {args.firmware}")
        sys.exit(1)

    try:
        kp.core.load_firmware_from_file(
            device_group=device_group,
            scpu_fw_path=args.firmware,
            ncpu_fw_path=""
        )
        print(" - Success")
    except kp.ApiKPException as exception:
        print(f"Error: upload firmware failed, error = '{exception}'")
        sys.exit(1)

    print("\n[Upload Model]")
    if not os.path.isfile(args.nef):
        print(f"Error: NEF not found -> {args.nef}")
        sys.exit(1)

    try:
        model_nef_descriptor = kp.core.load_model_from_file(
            device_group=device_group,
            file_path=args.nef
        )
        print(" - Success")
    except kp.ApiKPException as exception:
        print(f"Error: upload model failed, error = '{exception}'")
        sys.exit(1)

    model_id = model_nef_descriptor.models[0].id
    print(f"Model ID: {model_id}")

    valid_count = 0
    skipped_count = 0
    pure_infer_times = []
    total_start = time.perf_counter()

    output_checked = False

    print("\n[Starting Inference + Save NPY]")

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            image_id = get_image_id(image_path)

            img_bgr565 = preprocess_to_bgr565(
                image_path=image_path,
                input_w=args.input_width,
                input_h=args.input_height
            )

            generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
                model_id=model_id,
                inference_number=0,
                input_node_image_list=[
                    kp.GenericInputNodeImage(
                        image=img_bgr565,
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

            output = kp.inference.generic_inference_retrieve_float_node(
                node_idx=0,
                generic_raw_result=raw_result,
                channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
            )

            pred = output_to_numpy(output).reshape(-1)

            t1 = time.perf_counter()
            pure_infer_times.append(t1 - t0)

            if not output_checked:
                print("\n[Output Check]")
                print(f"Raw output shape : {output_to_numpy(output).shape}")
                print(f"Flatten size     : {pred.size}")
                print("Expected YOLOv5  : 25200 x 85 = 2142000 for 640x640")
                output_checked = True

            save_path = os.path.join(args.save_dir, f"{image_id:012d}.npy")
            np.save(save_path, pred)

            valid_count += 1

            # 降低連續 USB timeout 機率
            if SLEEP_AFTER_SAVE > 0:
                time.sleep(SLEEP_AFTER_SAVE)

        except Exception as exception:
            print(f"[Skip] {image_path} -> {exception}")
            skipped_count += 1
            continue

        if (idx % args.print_every == 0) or (idx == len(image_paths)):
            print(
                f"{idx:05d}/{len(image_paths):05d} | "
                f"Valid: {valid_count:05d} | "
                f"Skipped: {skipped_count:05d}"
            )

    total_time = time.perf_counter() - total_start
    pure_time = sum(pure_infer_times)

    infer_fps = valid_count / pure_time if pure_time > 0 else 0.0
    overall_fps = valid_count / total_time if total_time > 0 else 0.0
    avg_infer_ms = (pure_time / valid_count * 1000) if valid_count > 0 else 0.0

    print("\n========== Kneo Pi Raw Output Save Finished ==========")
    print(f"Total images   : {len(image_paths)}")
    print(f"Valid images   : {valid_count}")
    print(f"Skipped images : {skipped_count}")
    print(f"Save dir       : {args.save_dir}")
    print(f"Total time     : {total_time:.2f} s")
    print(f"Pure infer time: {pure_time:.2f} s")
    print(f"Avg infer time : {avg_infer_ms:.2f} ms")
    print(f"Overall FPS    : {overall_fps:.2f}")
    print(f"Inference FPS  : {infer_fps:.2f}")
    print("======================================================")


if __name__ == "__main__":
    main()