import os
import sys
import time
import statistics
import argparse
import numpy as np
import cv2
import psutil

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp


# =========================
# 固定路徑
# =========================
DEFAULT_USB_PORT_ID = 5
DEFAULT_SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
DEFAULT_MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')
DEFAULT_IMAGE_DIR_PATH = '../data/imagenet-mini/val'

INPUT_WIDTH = 224
INPUT_HEIGHT = 224

TEMP_PATH = '/sys/class/thermal/thermal_zone0/temp'


# =========================
# 工具函式
# =========================
def build_class_map(root_dir):
    classes = []

    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            classes.append(name)

    classes.sort()

    class_to_idx = {}
    idx_to_class = {}

    for idx, cls_name in enumerate(classes):
        class_to_idx[cls_name] = idx
        idx_to_class[idx] = cls_name

    return class_to_idx, idx_to_class


def get_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = []

    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))

    files.sort()
    return files


def read_temp_c():
    try:
        with open(TEMP_PATH, 'r') as f:
            raw = f.read().strip()

        t = float(raw)

        if t > 1000:
            t = t / 1000.0

        return t

    except Exception:
        return None


def preprocess_to_bgr565(image_path, width, height):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f'Cannot read image: {image_path}')

    img = cv2.resize(img, (width, height))
    img_bgr565 = cv2.cvtColor(img, cv2.COLOR_BGR2BGR565)

    return img_bgr565


def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def output_to_numpy(output):
    """
    修正 Kneron SDK 回傳 InferenceFloatNodeOutput 不能直接 np.array 的問題
    """

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


def fmt_temp(v):
    return f'{v:.2f} C' if v is not None else 'N/A'


# =========================
# 主程式
# =========================
def main():
    parser = argparse.ArgumentParser(
        description='Kneo Pi ImageNet-mini Accuracy Test'
    )

    parser.add_argument('--port', '-p', type=int, default=DEFAULT_USB_PORT_ID)
    parser.add_argument('--firmware', type=str, default=DEFAULT_SCPU_FW_PATH)
    parser.add_argument('--nef', type=str, default=DEFAULT_MODEL_FILE_PATH)
    parser.add_argument('--images', type=str, default=DEFAULT_IMAGE_DIR_PATH)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--limit', type=int, default=0)

    args = parser.parse_args()

    print('========== Kneo Pi Accuracy Test ==========')
    print(f'USB Port : {args.port}')
    print(f'Firmware : {args.firmware}')
    print(f'NEF      : {args.nef}')
    print(f'Images   : {args.images}')
    print('===========================================')

    if not os.path.isdir(args.images):
        print(f'Error: image folder not found -> {args.images}')
        sys.exit(1)

    class_to_idx, idx_to_class = build_class_map(args.images)

    if len(class_to_idx) == 0:
        print(f'Error: no class folders found in -> {args.images}')
        sys.exit(1)

    print(f'\n[Class Map]')
    print(f'Classes found: {len(class_to_idx)}')

    print('\n[Check Device USB Speed]')
    try:
        usb_speed = get_device_usb_speed_by_port_id(usb_port_id=args.port)

        if usb_speed not in (
            kp.UsbSpeed.KP_USB_SPEED_SUPER,
            kp.UsbSpeed.KP_USB_SPEED_HIGH
        ):
            print('\033[91m[Warning] Device is not running at super/high speed.\033[0m')
        else:
            print(' - Success')

    except Exception as exception:
        print(f'Error: check device USB speed fail, port ID = {args.port}, error = [{exception}]')
        sys.exit(1)

    print('\n[Connect Device]')
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port])
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f'Error: connect device fail, port ID = {args.port}, error = [{exception}]')
        sys.exit(1)

    print('\n[Set Device Timeout]')
    try:
        kp.core.set_timeout(device_group=device_group, milliseconds=10000)
        print(' - Success')
    except Exception as exception:
        print(f'Error: set timeout fail, error = [{exception}]')
        sys.exit(1)

    print('\n[Upload Firmware]')
    if not os.path.isfile(args.firmware):
        print(f'Error: firmware not found -> {args.firmware}')
        sys.exit(1)

    try:
        kp.core.load_firmware_from_file(
            device_group=device_group,
            scpu_fw_path=args.firmware,
            ncpu_fw_path=""
        )
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f"Error: upload firmware failed, error = '{exception}'")
        sys.exit(1)

    print('\n[Upload Model]')
    if not os.path.isfile(args.nef):
        print(f'Error: NEF not found -> {args.nef}')
        sys.exit(1)

    try:
        model_nef_descriptor = kp.core.load_model_from_file(
            device_group=device_group,
            file_path=args.nef
        )
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f"Error: upload model failed, error = '{exception}'")
        sys.exit(1)

    model_id = model_nef_descriptor.models[0].id
    print(f'Model ID: {model_id}')

    print('\n[Collect Images]')
    image_paths = get_image_paths(args.images)

    if len(image_paths) == 0:
        print(f'Error: no images found in {args.images}')
        sys.exit(1)

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    total_images = len(image_paths)
    print(f' - Found {total_images} images')

    process = psutil.Process(os.getpid())

    temps = []
    ram_list = []
    pure_infer_times = []

    valid_count = 0
    skipped_count = 0

    top1_correct = 0
    top5_correct = 0

    print('\n[Starting Accuracy Test]')
    total_time_start = time.perf_counter()

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            rel_path = os.path.relpath(image_path, args.images).replace('\\', '/')
            gt_class_name = rel_path.split('/')[0]
            gt_idx = class_to_idx[gt_class_name]

            img_bgr565 = preprocess_to_bgr565(
                image_path=image_path,
                width=INPUT_WIDTH,
                height=INPUT_HEIGHT
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

            infer_start = time.perf_counter()

            kp.inference.generic_image_inference_send(
                device_group=device_group,
                generic_inference_input_descriptor=generic_inference_input_descriptor
            )

            raw_result = kp.inference.generic_image_inference_receive(
                device_group=device_group
            )

            infer_end = time.perf_counter()

            infer_time = infer_end - infer_start
            pure_infer_times.append(infer_time)

            output = kp.inference.generic_inference_retrieve_float_node(
                node_idx=0,
                generic_raw_result=raw_result,
                channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
            )

            logits = output_to_numpy(output).reshape(-1)
            prob = softmax(logits)

            top5_idx = prob.argsort()[-5:][::-1]
            pred_top1 = int(top5_idx[0])

            if pred_top1 == gt_idx:
                top1_correct += 1

            if gt_idx in top5_idx:
                top5_correct += 1

            valid_count += 1

            ram_mb = process.memory_info().rss / (1024 * 1024)
            ram_list.append(ram_mb)

            temp_c = read_temp_c()
            if temp_c is not None:
                temps.append(temp_c)

        except Exception as exception:
            print(f'[Skip] {image_path} -> {exception}')
            skipped_count += 1
            continue

        if (idx % args.print_every == 0) or (idx == total_images):
            current_top1 = top1_correct / valid_count * 100 if valid_count > 0 else 0.0
            current_top5 = top5_correct / valid_count * 100 if valid_count > 0 else 0.0

            print(
                f'{idx:05d}/{total_images:05d} | '
                f'Top-1: {current_top1:.2f}% | '
                f'Top-5: {current_top5:.2f}%'
            )

    total_time_end = time.perf_counter()

    total_time = total_time_end - total_time_start
    pure_infer_time = sum(pure_infer_times)

    print('\n========== Final Result ==========')
    print(f'Total images              : {total_images}')
    print(f'Valid images              : {valid_count}')
    print(f'Skipped images            : {skipped_count}')

    if valid_count == 0:
        print('No valid inference completed.')
        print('==================================')
        sys.exit(0)

    top1_acc = top1_correct / valid_count * 100
    top5_acc = top5_correct / valid_count * 100

    avg_infer_time = pure_infer_time / valid_count
    pure_infer_fps = valid_count / pure_infer_time if pure_infer_time > 0 else 0.0
    overall_fps = valid_count / total_time if total_time > 0 else 0.0

    avg_ram = statistics.mean(ram_list) if ram_list else 0.0
    peak_ram = max(ram_list) if ram_list else 0.0

    min_temp = min(temps) if temps else None
    max_temp = max(temps) if temps else None
    avg_temp = statistics.mean(temps) if temps else None

    print(f'\nTop-1 Correct             : {top1_correct}')
    print(f'Top-5 Correct             : {top5_correct}')
    print(f'Top-1 Accuracy            : {top1_acc:.2f}%')
    print(f'Top-5 Accuracy            : {top5_acc:.2f}%')

    print(f'\nTotal elapsed time        : {total_time:.4f} s')
    print(f'Total pure inference time : {pure_infer_time:.4f} s')
    print(f'Average inference time    : {avg_infer_time * 1000:.4f} ms/image')

    print(f'\nPure inference FPS        : {pure_infer_fps:.2f}')
    print(f'Overall FPS               : {overall_fps:.2f}')

    print(f'\nAverage RAM               : {avg_ram:.2f} MB')
    print(f'Peak RAM                  : {peak_ram:.2f} MB')

    print(f'\nMin Temp                  : {fmt_temp(min_temp)}')
    print(f'Max Temp                  : {fmt_temp(max_temp)}')
    print(f'Average Temp              : {fmt_temp(avg_temp)}')

    print('==================================')


if __name__ == '__main__':
    main()