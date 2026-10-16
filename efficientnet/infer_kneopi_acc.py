import os
import sys
import argparse
import numpy as np
import cv2

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp


# =========================
# Default Config
# =========================
DEFAULT_USB_PORT_ID = 5
DEFAULT_SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
DEFAULT_MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')
DEFAULT_IMAGE_DIR_PATH = '../data/imagenet-mini/val'

# EfficientNet V2-S input size
DEFAULT_INPUT_WIDTH = 384
DEFAULT_INPUT_HEIGHT = 384


# =========================
# Helper Functions
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


def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


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


def rgb_to_rgb565_uint8(rgb_img):
    """
    Convert RGB888 image to RGB565 packed uint8 image.

    Input:
        rgb_img: H x W x 3, uint8, RGB order

    Output:
        rgb565: H x W x 2, uint8
    """
    rgb = rgb_img.astype(np.uint16)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

    out = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 2), dtype=np.uint8)
    out[:, :, 0] = rgb565 & 0xFF
    out[:, :, 1] = (rgb565 >> 8) & 0xFF

    return out


def preprocess_to_rgb565(image_path, width, height):
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(f'Cannot read image: {image_path}')

    img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

    # OpenCV is BGR, Kneron generic image input uses RGB565
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_rgb565 = rgb_to_rgb565_uint8(img_rgb)

    return img_rgb565


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description='Kneo Pi EfficientNet V2-S ImageNet-mini Accuracy Test'
    )

    parser.add_argument('--port', '-p', type=int, default=DEFAULT_USB_PORT_ID)
    parser.add_argument('--firmware', type=str, default=DEFAULT_SCPU_FW_PATH)
    parser.add_argument('--nef', type=str, default=DEFAULT_MODEL_FILE_PATH)
    parser.add_argument('--images', type=str, default=DEFAULT_IMAGE_DIR_PATH)

    parser.add_argument('--input_width', type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument('--input_height', type=int, default=DEFAULT_INPUT_HEIGHT)

    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--limit', type=int, default=0)

    # 1000-class ImageNet model 通常用 0
    # 如果你的模型 output 是 1001，而且第 0 類是 background，改成 1
    parser.add_argument('--label_offset', type=int, default=0)

    args = parser.parse_args()

    print('========== Kneo Pi EfficientNet V2-S Accuracy Test ==========')
    print(f'USB Port     : {args.port}')
    print(f'Firmware     : {args.firmware}')
    print(f'NEF          : {args.nef}')
    print(f'Images       : {args.images}')
    print(f'Input        : {args.input_width}x{args.input_height}')
    print(f'Label offset : {args.label_offset}')
    print('=============================================================')

    if not os.path.isdir(args.images):
        print(f'Error: image folder not found -> {args.images}')
        sys.exit(1)

    class_to_idx, idx_to_class = build_class_map(args.images)

    if len(class_to_idx) == 0:
        print(f'Error: no class folders found in -> {args.images}')
        sys.exit(1)

    print('\n[Class Map]')
    print(f'Classes found: {len(class_to_idx)}')
    print(f'First class  : {idx_to_class.get(0, "N/A")}')
    print(f'Last class   : {idx_to_class.get(len(idx_to_class) - 1, "N/A")}')

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

    valid_count = 0
    skipped_count = 0

    top1_correct = 0
    top5_correct = 0

    output_size_checked = False
    output_size = None

    print('\n[Starting Accuracy Test]')

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            rel_path = os.path.relpath(image_path, args.images).replace('\\', '/')
            gt_class_name = rel_path.split('/')[0]

            if gt_class_name not in class_to_idx:
                raise ValueError(f'Class folder not found in class map: {gt_class_name}')

            gt_idx = class_to_idx[gt_class_name] + args.label_offset

            img_rgb565 = preprocess_to_rgb565(
                image_path=image_path,
                width=args.input_width,
                height=args.input_height
            )

            generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
                model_id=model_id,
                inference_number=0,
                input_node_image_list=[
                    kp.GenericInputNodeImage(
                        image=img_rgb565,
                        image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565
                    )
                ]
            )

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

            logits = output_to_numpy(output).reshape(-1)

            if not output_size_checked:
                output_size = logits.size
                print('\n[Output Check]')
                print(f'Output size : {output_size}')
                print(f'Class count : {len(class_to_idx)}')

                if output_size == 1001 and len(class_to_idx) == 1000 and args.label_offset == 0:
                    print('[Warning] Output is 1001 classes but label_offset is 0.')
                    print('[Warning] If Top-1 is very low, try: --label_offset 1')

                if output_size == 1000 and args.label_offset != 0:
                    print('[Warning] Output is 1000 classes but label_offset is not 0.')
                    print('[Warning] Usually 1000-class ImageNet model should use --label_offset 0')

                output_size_checked = True

            prob = softmax(logits)

            top5_idx = prob.argsort()[-5:][::-1]
            pred_top1 = int(top5_idx[0])

            if pred_top1 == gt_idx:
                top1_correct += 1

            if gt_idx in top5_idx:
                top5_correct += 1

            valid_count += 1

        except Exception as exception:
            print(f'[Skip] {image_path} -> {exception}')
            skipped_count += 1
            continue

        if (idx % args.print_every == 0) or (idx == total_images):
            current_top1 = top1_correct / valid_count * 100 if valid_count > 0 else 0.0
            current_top5 = top5_correct / valid_count * 100 if valid_count > 0 else 0.0

            print(
                f'{idx:05d}/{total_images:05d} | '
                f'Valid: {valid_count:05d} | '
                f'Top-1: {current_top1:.2f}% | '
                f'Top-5: {current_top5:.2f}%'
            )

    print('\n========== Final Accuracy Result ==========')
    print(f'Model          : EfficientNet V2-S')
    print(f'Input size     : {args.input_width}x{args.input_height}')
    print(f'Output size    : {output_size if output_size is not None else "N/A"}')
    print(f'Label offset   : {args.label_offset}')

    print(f'\nTotal images   : {total_images}')
    print(f'Valid images   : {valid_count}')
    print(f'Skipped images : {skipped_count}')

    if valid_count == 0:
        print('No valid inference completed.')
        print('==========================================')
        sys.exit(0)

    top1_acc = top1_correct / valid_count * 100
    top5_acc = top5_correct / valid_count * 100

    print(f'\nTop-1 Correct  : {top1_correct}')
    print(f'Top-5 Correct  : {top5_correct}')
    print(f'Top-1 Accuracy : {top1_acc:.2f}%')
    print(f'Top-5 Accuracy : {top5_acc:.2f}%')

    print('==========================================')


if __name__ == '__main__':
    main()