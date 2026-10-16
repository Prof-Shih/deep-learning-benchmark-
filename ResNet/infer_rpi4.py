import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import psutil

# ====== 設定 ======
MODEL_PATH = "resnet18.onnx"
IMAGE_DIR = "images"   # 你的測試資料夾
INPUT_SIZE = (224, 224)

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

# ====== 讀取圖片 ======
def load_images(image_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(exts):
            paths.append(os.path.join(image_dir, f))
    paths.sort()
    return paths

# ====== 前處理 ======
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

# ====== 主程式 ======
def main():
    print("Loading model...")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_paths = load_images(IMAGE_DIR)
    total_images = len(image_paths)

    print(f"Total images: {total_images}")
    print("Start inference...\n")

    infer_times = []
    temps = []
    ram_usage = []

    total_start = time.time()

    for idx, img_path in enumerate(image_paths):
        input_data = preprocess(img_path)

        # ===== 記錄 RAM =====
        process = psutil.Process(os.getpid())
        ram_usage.append(process.memory_info().rss / (1024 * 1024))  # MB

        # ===== 記錄溫度 =====
        temps.append(get_cpu_temp())

        # ===== 推論 =====
        start = time.time()
        _ = session.run(None, {input_name: input_data})
        end = time.time()

        infer_time = end - start
        infer_times.append(infer_time)

        print(f"Inferencing {idx+1}/{total_images}")

    total_end = time.time()

    # ====== 計算 ======
    total_time = total_end - total_start
    total_infer_time = sum(infer_times)
    avg_infer_time = total_infer_time / total_images

    avg_fps = 1.0 / avg_infer_time
    overall_fps = total_images / total_time

    avg_ram = sum(ram_usage) / len(ram_usage)
    peak_ram = max(ram_usage)

    avg_temp = sum(temps) / len(temps)
    peak_temp = max(temps)

    # ====== 輸出 ======
    print("\n===== RESULT =====")
    print(f"Total images: {total_images}")

    print(f"\nTotal time: {total_time:.3f} s")
    print(f"Total pure inference time: {total_infer_time:.3f} s")
    print(f"Average inference time: {avg_infer_time*1000:.2f} ms")

    print(f"\nAverage FPS (inference only): {avg_fps:.2f}")
    print(f"Overall FPS (end-to-end): {overall_fps:.2f}")

    print(f"\nAverage RAM: {avg_ram:.2f} MB")
    print(f"Peak RAM: {peak_ram:.2f} MB")

    print(f"\nAverage CPU Temp: {avg_temp:.2f} C")
    print(f"Peak Temp: {peak_temp:.2f} C")


if __name__ == "__main__":
    main()