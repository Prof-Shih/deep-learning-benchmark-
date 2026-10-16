import os
import time
import glob
import cv2
import numpy as np
import onnxruntime as ort
import psutil
import argparse

# =========================
# 設定
# =========================
MODEL_PATH = "PIDNet_S_Cityscapes_val_dynamic.onnx"
IMG_DIR = "../Test"
INPUT_SIZE = (1024, 2048)  # Cityscapes 預設，可依你的模型調整

# =========================
# 指令參數
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()

# =========================
# 溫度讀取（Raspberry Pi）
# =========================
def get_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            t = int(f.read())
            return t / 1000.0
    except:
        return 0.0

# =========================
# RAM 使用
# =========================
def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

# =========================
# 前處理（PIDNet）
# =========================
def preprocess(img):
    img = cv2.resize(img, INPUT_SIZE[::-1])
    img = img.astype(np.float32)

    # ImageNet normalization
    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    img = img.transpose(2, 0, 1)  # HWC → CHW
    return img.astype(np.float32)

# =========================
# 建立 batch
# =========================
def make_batch(paths):
    batch_imgs = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue

        inp = preprocess(img)
        batch_imgs.append(inp)

    if not batch_imgs:
        return None

    return np.stack(batch_imgs, axis=0).astype(np.float32)

# =========================
# 主程式
# =========================
def main():
    args = parse_args()
    batch_size = args.batch

    # ONNX Runtime
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    total_images = len(image_paths)

    if total_images == 0:
        print("❌ 找不到圖片")
        return

    print(f"Total images: {total_images}")
    print(f"Batch size: {batch_size}")

    # =========================
    # 計時
    # =========================
    total_start = time.time()
    pure_infer_time = 0.0

    temps = []
    rams = []
    processed_images = 0

    # =========================
    # 推論
    # =========================
    for start in range(0, total_images, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        inp = make_batch(batch_paths)

        if inp is None:
            continue

        current_batch = inp.shape[0]

        t0 = time.time()
        _ = session.run(None, {input_name: inp})
        t1 = time.time()

        infer_time = t1 - t0
        pure_infer_time += infer_time

        processed_images += current_batch

        # 收集數據
        temps.append(get_temp())
        rams.append(get_ram())

        print(f"[{processed_images}/{total_images}] done")

    total_end = time.time()
    total_time = total_end - total_start

    # =========================
    # 統計
    # =========================
    total_images = processed_images

    avg_infer = pure_infer_time / total_images

    avg_fps = total_images / pure_infer_time
    overall_fps = total_images / pure_infer_time
    real_fps = total_images / total_time

    avg_temp = sum(temps) / len(temps)
    peak_temp = max(temps)

    avg_ram = sum(rams) / len(rams)
    peak_ram = max(rams)

    # =========================
    # 輸出
    # =========================
    print("\n========== RESULT ==========")
    print(f"Total images: {total_images}")

    print(f"\nTotal time (with IO): {total_time:.3f} s")
    print(f"Total pure inference time: {pure_infer_time:.3f} s")

    print(f"\nAverage inference time: {avg_infer*1000:.2f} ms")

    print(f"\nAverage FPS (pure infer): {avg_fps:.2f}")
    print(f"Overall FPS (pure infer): {overall_fps:.2f}")
    print(f"Real FPS (end-to-end): {real_fps:.2f}")

    print(f"\nAverage RAM: {avg_ram:.2f} MB")
    print(f"Peak RAM: {peak_ram:.2f} MB")

    print(f"\nAverage Temp: {avg_temp:.2f} °C")
    print(f"Peak Temp: {peak_temp:.2f} °C")

    print("============================")

if __name__ == "__main__":
    main()