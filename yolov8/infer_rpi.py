import os
import cv2
import time
import psutil
import numpy as np
import onnxruntime as ort
import argparse

# ====== 設定 ======
MODEL_PATH = "yolov8n_dynamic.onnx"
IMAGE_DIR = "../Test"
IMG_SIZE = 640

# ====== 溫度讀取 ======
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip())
            if temp > 200:
                temp = temp / 1000.0
            return temp
    except:
        return 0.0

# ====== RAM ======
def get_ram():
    mem = psutil.virtual_memory()
    used = (mem.total - mem.available) / (1024 ** 2)
    return used

# ====== 前處理 ======
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

# ====== 建立 batch ======
def make_batch(paths):
    batch_imgs = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        batch_imgs.append(preprocess(img))

    if not batch_imgs:
        return None

    return np.stack(batch_imgs, axis=0)

# ====== 主程式 ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    print("Loading model...")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    total_images = len(image_files)

    print(f"Total images: {total_images}")
    print(f"Batch size: {args.batch}")

    infer_times = []
    ram_usage = []
    temps = []

    processed_images = 0

    start_total = time.time()

    # ===== Batch 推論 =====
    for start in range(0, total_images, args.batch):

        batch_names = image_files[start:start + args.batch]
        batch_paths = [os.path.join(IMAGE_DIR, n) for n in batch_names]

        input_tensor = make_batch(batch_paths)

        if input_tensor is None:
            continue

        current_batch = input_tensor.shape[0]

        # ===== 推論 =====
        t0 = time.time()
        _ = session.run(None, {input_name: input_tensor})
        t1 = time.time()

        infer_time = t1 - t0
        infer_times.append(infer_time)

        processed_images += current_batch

        # ===== 記錄 =====
        ram_usage.append(get_ram())
        temps.append(get_cpu_temp())

        print(f"[{processed_images}/{total_images}] done")

    total_time = time.time() - start_total

    # ===== 統計（修正）=====
    total_infer = sum(infer_times)
    total_images = processed_images

    avg_infer = total_infer / total_images if total_images > 0 else 0

    avg_fps = total_images / total_infer if total_infer > 0 else 0
    overall_fps = total_images / total_infer if total_infer > 0 else 0
    real_fps = total_images / total_time if total_time > 0 else 0

    avg_ram = np.mean(ram_usage)
    peak_ram = np.max(ram_usage)

    avg_temp = np.mean(temps)
    peak_temp = np.max(temps)

    # ===== 輸出 =====
    print("\n========== RESULT ==========")
    print(f"Total images: {total_images}")

    print(f"\nTotal pure inference time: {total_infer:.3f} s")
    print(f"Average inference time: {avg_infer*1000:.2f} ms")

    print(f"\nAverage FPS (pure inference): {avg_fps:.2f}")
    print(f"Overall FPS (pure inference): {overall_fps:.2f}")
    print(f"Real FPS (含前後處理): {real_fps:.2f}")

    print(f"\nAverage RAM: {avg_ram:.2f} MB")
    print(f"Peak RAM: {peak_ram:.2f} MB")

    print(f"\nAverage CPU Temp: {avg_temp:.2f} C")
    print(f"Peak Temp: {peak_temp:.2f} C")


if __name__ == "__main__":
    main()