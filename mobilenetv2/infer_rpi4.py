import os
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort
import psutil

# ---------------------------
# 溫度讀取 (Raspberry Pi)
# ---------------------------
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            t = float(f.read().strip())
            if t > 200:
                t /= 1000.0
            return t
    except:
        return 0.0

# ---------------------------
# 取得圖片
# ---------------------------
def load_images(img_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(exts)]

# ---------------------------
# 預處理 (MobileNetV2)
# ---------------------------
def preprocess(img_path, size=224):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))

    return img.astype(np.float32)

# ---------------------------
# 建立 batch
# ---------------------------
def make_batch(paths, size):
    batch_imgs = []

    for p in paths:
        img = preprocess(p, size)
        if img is not None:
            batch_imgs.append(img)

    if not batch_imgs:
        return None

    return np.stack(batch_imgs, axis=0)

# ---------------------------
# 主程式
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet_v2_opset11_dynamic.onnx")
    parser.add_argument("--dir", default="../Test")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    image_paths = load_images(args.dir)
    total_images = len(image_paths)

    print(f"Total images: {total_images}")
    print(f"Batch size: {args.batch}")

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    infer_times = []
    temps = []
    ram_usage = []

    processed_images = 0

    start_total = time.time()

    # ---------------------------
    # Batch 推論
    # ---------------------------
    for start in range(0, total_images, args.batch):
        batch_paths = image_paths[start:start + args.batch]
        batch = make_batch(batch_paths, args.size)

        if batch is None:
            continue

        current_batch = batch.shape[0]

        t0 = time.time()
        _ = session.run(None, {input_name: batch})
        t1 = time.time()

        infer_time = t1 - t0
        infer_times.append(infer_time)

        processed_images += current_batch

        temps.append(get_cpu_temp())
        ram_usage.append(psutil.Process().memory_info().rss / (1024 * 1024))

        print(f"[{processed_images}/{total_images}] done")

    end_total = time.time()

    # ---------------------------
    # 統計（修正）
    # ---------------------------
    total_time = end_total - start_total
    total_infer_time = sum(infer_times)
    total_images = processed_images

    avg_infer = total_infer_time / total_images

    fps_infer = total_images / total_infer_time
    fps_total = total_images / total_time

    avg_ram = sum(ram_usage) / len(ram_usage)
    peak_ram = max(ram_usage)

    avg_temp = sum(temps) / len(temps)
    peak_temp = max(temps)

    # ---------------------------
    # 輸出結果
    # ---------------------------
    print("\n===== RESULT =====")
    print(f"Total images: {total_images}")

    print(f"\nTotal time: {total_time:.3f} s")
    print(f"Total pure inference time: {total_infer_time:.3f} s")

    print(f"\nAverage inference time: {avg_infer*1000:.2f} ms")

    print(f"\nAverage FPS (infer only): {fps_infer:.2f}")
    print(f"Overall FPS (total): {fps_total:.2f}")

    print(f"\nAverage RAM: {avg_ram:.1f} MB")
    print(f"Peak RAM: {peak_ram:.1f} MB")

    print(f"\nAverage CPU Temp: {avg_temp:.1f} C")
    print(f"Peak Temp: {peak_temp:.1f} C")


if __name__ == "__main__":
    main()