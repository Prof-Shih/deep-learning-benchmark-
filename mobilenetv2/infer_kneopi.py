import os
import sys
import time
import statistics
import argparse

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp
import cv2
import psutil


# =========================
# 預設固定路徑
# firmware / nef / 圖片資料夾路徑都不改
# =========================
DEFAULT_USB_PORT_ID = 5
DEFAULT_SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
DEFAULT_MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')
DEFAULT_IMAGE_DIR_PATH = os.path.join(PWD, '../Test')

# MobileNetV2 官方輸入尺寸 224x224
INPUT_WIDTH = 224
INPUT_HEIGHT = 224

# Kneo Pi 溫度
TEMP_PATH = '/sys/class/thermal/thermal_zone0/temp'


# =========================
# 工具函式
# =========================
def get_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = []
    if not os.path.isdir(folder):
        return files

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            files.append(path)

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


def fmt_temp(v):
    return f'{v:.2f} C' if v is not None else 'N/A'


# =========================
# 主程式
# =========================
def main():
    parser = argparse.ArgumentParser(
        description='MobileNetV2 benchmark script for Kneo Pi / KL730'
    )
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_USB_PORT_ID,
                        help='USB port id')
    parser.add_argument('--firmware', type=str, default=DEFAULT_SCPU_FW_PATH,
                        help='KL730 firmware path')
    parser.add_argument('--nef', type=str, default=DEFAULT_MODEL_FILE_PATH,
                        help='NEF model path')
    parser.add_argument('--images', type=str, default=DEFAULT_IMAGE_DIR_PATH,
                        help='Image folder path')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print progress every N images')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only run first N images, 0 means all')
    args = parser.parse_args()

    print('[Check Device USB Speed]')
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

    print('[Connect Device]')
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[args.port])
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f'Error: connect device fail, port ID = {args.port}, error = [{exception}]')
        sys.exit(1)

    print('[Set Device Timeout]')
    try:
        kp.core.set_timeout(device_group=device_group, milliseconds=10000)
        print(' - Success')
    except Exception as exception:
        print(f'Error: set timeout fail, error = [{exception}]')
        sys.exit(1)

    print('[Upload Firmware]')
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

    print('[Upload Model]')
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

    print('[Collect Images]')
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

    print('[Starting Benchmark]')
    total_time_start = time.perf_counter()

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            img_bgr565 = preprocess_to_bgr565(
                image_path=image_path,
                width=INPUT_WIDTH,
                height=INPUT_HEIGHT
            )
        except Exception as exception:
            print(f'[Skip] {image_path} -> {exception}')
            skipped_count += 1
            continue

        generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
            model_id=model_nef_descriptor.models[0].id,
            inference_number=0,
            input_node_image_list=[
                kp.GenericInputNodeImage(
                    image=img_bgr565,
                    image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565
                )
            ]
        )

        infer_start = time.perf_counter()
        try:
            kp.inference.generic_image_inference_send(
                device_group=device_group,
                generic_inference_input_descriptor=generic_inference_input_descriptor
            )
            _ = kp.inference.generic_image_inference_receive(
                device_group=device_group
            )
        except kp.ApiKPException as exception:
            print(f'[Skip] inference failed on {image_path}, error = {exception}')
            skipped_count += 1
            continue
        infer_end = time.perf_counter()

        infer_time = infer_end - infer_start
        pure_infer_times.append(infer_time)
        valid_count += 1

        try:
            ram_mb = process.memory_info().rss / (1024 * 1024)
            ram_list.append(ram_mb)
        except Exception:
            pass

        temp_c = read_temp_c()
        if temp_c is not None:
            temps.append(temp_c)

        if (idx % args.print_every == 0) or (idx == total_images):
            print(f'{idx:02d}/{total_images:02d}')

    total_time_end = time.perf_counter()

    total_time = total_time_end - total_time_start
    pure_infer_time = sum(pure_infer_times)

    print('\n[Result]')
    print(f'Total images found: {total_images}')
    print(f'Valid images inferred: {valid_count}')
    print(f'Skipped images: {skipped_count}')

    if valid_count == 0:
        print('\nNo valid inference completed.')
        sys.exit(0)

    avg_infer_time = pure_infer_time / valid_count
    pure_infer_fps = valid_count / pure_infer_time if pure_infer_time > 0 else 0.0
    overall_fps = valid_count / total_time if total_time > 0 else 0.0

    avg_ram = statistics.mean(ram_list) if ram_list else 0.0
    peak_ram = max(ram_list) if ram_list else 0.0

    if temps:
        min_temp = min(temps)
        max_temp = max(temps)
        avg_temp = statistics.mean(temps)
    else:
        min_temp = None
        max_temp = None
        avg_temp = None

    print(f'\nTotal time: {total_time:.4f} s')
    print(f'Total pure inference time: {pure_infer_time:.4f} s')
    print(f'Average inference time: {avg_infer_time * 1000:.2f} ms')

    print(f'\nAverage FPS: {pure_infer_fps:.2f}')
    print(f'Overall FPS: {overall_fps:.2f}')

    print(f'\nAverage RAM: {avg_ram:.2f} MB')
    print(f'Peak RAM: {peak_ram:.2f} MB')

    print(f'\nMin Temp: {fmt_temp(min_temp)}')
    print(f'Max Temp: {fmt_temp(max_temp)}')
    print(f'Average Temp: {fmt_temp(avg_temp)}')


if __name__ == '__main__':
    main()