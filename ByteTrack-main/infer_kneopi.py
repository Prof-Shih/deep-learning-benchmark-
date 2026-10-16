#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import statistics

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id
import kp
import cv2
import psutil
import numpy as np


# =========================
# CLI
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0, help='limit frames (0 = no limit)')
parser.add_argument('--progress', type=int, default=10, help='print every N frames')
args = parser.parse_args()

PRINT_EVERY = args.progress


# =========================
# 固定設定
# =========================
USB_PORT_ID = 5
SCPU_FW_PATH = os.path.join(PWD, '../res/firmware/KL730/kp_firmware.tar')
MODEL_FILE_PATH = os.path.join(PWD, 'models_730.nef')
VIDEO_PATH = os.path.join(PWD, '../test.mp4')

# 這裡一定要改成你的 NEF 實際輸入尺寸
# 如果你的 NEF 是 608x1088 就維持這個
INPUT_HEIGHT = 608
INPUT_WIDTH = 1088

TEMP_PATH = '/sys/class/thermal/thermal_zone0/temp'
PAD_VALUE = 114


# =========================
# 工具函式
# =========================
def read_temp_c():
    try:
        with open(TEMP_PATH, 'r') as f:
            raw = f.read().strip()
        t = float(raw)
        if t > 1000:
            t /= 1000.0
        return t
    except Exception:
        return None


def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_video_total_frames(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total if total > 0 else 0


def format_progress(cur, total):
    return f"{cur}/{total}" if total > 0 else f"{cur}/?"


def letterbox_bgr(img_bgr, target_w, target_h, pad_value=114):
    h, w = img_bgr.shape[:2]
    ratio = min(target_w / float(w), target_h / float(h))
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, ratio


def preprocess_to_bgr565(frame_bgr, width, height):
    padded_bgr, ratio = letterbox_bgr(frame_bgr, width, height, PAD_VALUE)
    img_bgr565 = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2BGR565)
    return img_bgr565, ratio


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

    print('[Open Video]')
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'Error: cannot open video: {VIDEO_PATH}')
        sys.exit(0)

    total_frames = get_video_total_frames(cap)
    if args.limit > 0:
        total_frames = min(total_frames, args.limit) if total_frames > 0 else args.limit

    print(f' - Video path: {VIDEO_PATH}')
    print(f' - Total frames: {total_frames if total_frames > 0 else "unknown"}')

    temps = []
    ram_list = []
    pure_infer_times = []

    frame_idx = 0
    valid_count = 0
    skipped_count = 0

    total_time_start = time.perf_counter()
    print('[Start]')

    while True:
        if args.limit > 0 and frame_idx >= args.limit:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1

        try:
            img_bgr565, _ = preprocess_to_bgr565(
                frame_bgr=frame_bgr,
                width=INPUT_WIDTH,
                height=INPUT_HEIGHT
            )
        except Exception as exception:
            print(f'[Skip] frame {frame_idx}: preprocess failed -> {exception}')
            skipped_count += 1
            continue

        desc = kp.GenericImageInferenceDescriptor(
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
                generic_inference_input_descriptor=desc
            )
            # 即使不做後處理，這裡仍會嘗試接收完成訊號/結果
            _ = kp.inference.generic_image_inference_receive(device_group=device_group)
        except kp.ApiKPException as exception:
            print(f'[Skip] frame {frame_idx}: inference failed -> {exception}')
            skipped_count += 1

            # 如果是資料太大，直接中止，因為後面每幀都一樣不會成功
            msg = str(exception)
            if 'KP_ERROR_RECV_DATA_TOO_LARGE' in msg or 'Error code: 18' in msg:
                print('\n[Stop]')
                print('The current NEF returns too much data for generic_image_inference_receive().')
                print('This cannot be fixed by changing tracking code only.')
                print('You need a smaller-output NEF (for example, decode/NMS done before output).')
                break

            continue
        t1 = time.perf_counter()

        infer_time = t1 - t0
        pure_infer_times.append(infer_time)
        valid_count += 1

        ram_list.append(get_ram_mb())
        temp_c = read_temp_c()
        if temp_c is not None:
            temps.append(temp_c)

        if frame_idx % PRINT_EVERY == 0 or (total_frames > 0 and frame_idx == total_frames):
            print(f'Progress: {format_progress(frame_idx, total_frames)}')

    total_time = time.perf_counter() - total_time_start
    cap.release()

    print('\n===== RESULT =====')
    print(f'Total frames read: {frame_idx}')
    print(f'Valid frames inferred: {valid_count}')
    print(f'Skipped frames: {skipped_count}')

    if valid_count == 0:
        print('\nNo valid inference completed.')
        sys.exit(0)

    pure_infer_total = sum(pure_infer_times)
    avg_infer = pure_infer_total / valid_count
    avg_fps = valid_count / pure_infer_total if pure_infer_total > 0 else 0.0
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
    print(f'Total pure inference time: {pure_infer_total:.4f} s')
    print(f'Average inference time: {avg_infer * 1000:.2f} ms')

    print(f'\nAverage FPS: {avg_fps:.2f}')
    print(f'Overall FPS: {overall_fps:.2f}')

    print(f'\nAverage RAM: {avg_ram:.2f} MB')
    print(f'Peak RAM: {peak_ram:.2f} MB')

    print(f'\nMin Temp: {min_temp:.2f} C')
    print(f'Max Temp: {max_temp:.2f} C')
    print(f'Average Temp: {avg_temp:.2f} C')