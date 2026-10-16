import os
import sys
import time
import statistics

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp
import cv2
import psutil

# =========================
# 固定設定
# =========================
USB_PORT_ID = 5
SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')
IMAGE_DIR_PATH = os.path.join(PWD, '../Test')

# PIDNet_S_Cityscapes 常見輸入尺寸
# 一定要和你的 NEF 實際輸入一致
INPUT_WIDTH = 2048
INPUT_HEIGHT = 1024

# 幾張印一次進度
PRINT_EVERY = 10

# 溫度檔
TEMP_PATH = '/sys/class/thermal/thermal_zone0/temp'


# =========================
# 工具函式
# =========================
def get_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = []
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


# =========================
# 主程式
# =========================
if __name__ == '__main__':
    print('[Check Device USB Speed]')
    try:
        usb_speed = get_device_usb_speed_by_port_id(usb_port_id=USB_PORT_ID)
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != usb_speed and \
           kp.UsbSpeed.KP_USB_SPEED_HIGH != usb_speed:
            print('\033[91m[Warning] Device is not running at super/high speed.\033[0m')
        else:
            print(' - Success')
    except Exception as exception:
        print(f'Error: check device USB speed fail, port ID = {USB_PORT_ID}, error = [{exception}]')
        sys.exit(0)

    print('[Connect Device]')
    try:
        device_group = kp.core.connect_devices(usb_port_ids=[USB_PORT_ID])
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f'Error: connect device fail, port ID = {USB_PORT_ID}, error = [{exception}]')
        sys.exit(0)

    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print(' - Success')

    print('[Upload Firmware]')
    try:
        kp.core.load_firmware_from_file(
            device_group=device_group,
            scpu_fw_path=SCPU_FW_PATH,
            ncpu_fw_path=""
        )
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f"Error: upload firmware failed, error = '{exception}'")
        sys.exit(0)

    print('[Upload Model]')
    try:
        model_nef_descriptor = kp.core.load_model_from_file(
            device_group=device_group,
            file_path=MODEL_FILE_PATH
        )
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f"Error: upload model failed, error = '{exception}'")
        sys.exit(0)

    print('[Collect Images]')
    image_paths = get_image_paths(IMAGE_DIR_PATH)
    if len(image_paths) == 0:
        print(f'Error: no images found in {IMAGE_DIR_PATH}')
        sys.exit(0)

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

        t0 = time.perf_counter()
        try:
            kp.inference.generic_image_inference_send(
                device_group=device_group,
                generic_inference_input_descriptor=generic_inference_input_descriptor
            )
            _ = kp.inference.generic_image_inference_receive(device_group=device_group)
        except kp.ApiKPException as exception:
            print(f'[Skip] inference failed on {image_path}, error = {exception}')
            skipped_count += 1
            continue
        t1 = time.perf_counter()

        infer_time = t1 - t0
        pure_infer_times.append(infer_time)
        valid_count += 1

        ram_mb = process.memory_info().rss / (1024 * 1024)
        ram_list.append(ram_mb)

        temp_c = read_temp_c()
        if temp_c is not None:
            temps.append(temp_c)

        if idx % PRINT_EVERY == 0 or idx == total_images:
            print(f' - Progress: {idx}/{total_images}')

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
    avg_fps = valid_count / pure_infer_time if pure_infer_time > 0 else 0.0
    overall_fps = valid_count / total_time if total_time > 0 else 0.0

    avg_ram = statistics.mean(ram_list) if ram_list else 0.0
    peak_ram = max(ram_list) if ram_list else 0.0

    if temps:
        min_temp = min(temps)
        max_temp = max(temps)
        avg_temp = statistics.mean(temps)
    else:
        min_temp = max_temp = avg_temp = 0.0

    print(f'\nTotal time: {total_time:.4f} s')
    print(f'Total pure inference time: {pure_infer_time:.4f} s')
    print(f'Average inference time: {avg_infer_time * 1000:.2f} ms')

    print(f'\nAverage FPS: {avg_fps:.2f}')
    print(f'Overall FPS: {overall_fps:.2f}')

    print(f'\nAverage RAM: {avg_ram:.2f} MB')
    print(f'Peak RAM: {peak_ram:.2f} MB')

    print(f'\nMin Temp: {min_temp:.2f} C')
    print(f'Max Temp: {max_temp:.2f} C')
    print(f'Average Temp: {avg_temp:.2f} C')