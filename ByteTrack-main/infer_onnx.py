#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
import psutil
import os
import sys

# 加入 yolox / bytetrack 路徑
sys.path.append(os.getcwd())

from yolox.data.data_augment import preproc
from yolox.utils import multiclass_nms, demo_postprocess
from yolox.tracker.byte_tracker import BYTETracker


RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            t = float(f.read().strip())
            if t > 200:
                return t / 1000.0
            return t
    except:
        return None


def get_ram():
    return psutil.virtual_memory().used / (1024 * 1024)


class Predictor:
    def __init__(self, session, input_shape, score_thr=0.1, nms_thr=0.7):
        self.session = session
        self.input_shape = input_shape
        self.input_name = session.get_inputs()[0].name
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def preprocess(self, img):
        img, ratio = preproc(img, self.input_shape, RGB_MEAN, RGB_STD)
        return img[None].astype(np.float32), ratio

    def postprocess(self, output, ratio):
        pred = demo_postprocess(output, self.input_shape)[0]

        boxes = pred[:, :4]
        scores = pred[:, 4:5] * pred[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy /= ratio

        dets = multiclass_nms(
            boxes_xyxy, scores,
            nms_thr=self.nms_thr,
            score_thr=self.score_thr
        )

        if dets is None:
            return np.zeros((0, 5), dtype=np.float32)

        return dets[:, :-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="bytetrack_nano_mot17.onnx")
    parser.add_argument("--input", default="../test.mp4")
    parser.add_argument("--input-shape", default="608,1088")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress", type=int, default=10)
    args = parser.parse_args()

    input_shape = tuple(map(int, args.input_shape.split(",")))

    session = ort.InferenceSession(
        args.onnx,
        providers=["CPUExecutionProvider"]
    )

    predictor = Predictor(session, input_shape)
    class TrackArgs:
        track_thresh = 0.5
        track_buffer = 30
        match_thresh = 0.8
        min_box_area = 0.8
        mot20 = False
    tracker = BYTETracker(TrackArgs(), frame_rate=30)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("❌ cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    infer_times = []
    ram_list = []
    temp_list = []

    pure_infer_total = 0
    t_start = time.time()

    while True:

        if args.limit > 0 and frame_id >= args.limit:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % args.progress == 0:
            print(f"Processing frame: {frame_id}")

        img, ratio = predictor.preprocess(frame)

        t0 = time.time()
        outputs = session.run(None, {predictor.input_name: img})
        dt = time.time() - t0

        pure_infer_total += dt
        infer_times.append(dt * 1000)

        dets = predictor.postprocess(outputs[0], ratio)
        tracker.update(dets, frame.shape[:2], frame.shape[:2])

        # RAM / TEMP
        ram_list.append(get_ram())
        temp = get_cpu_temp()
        if temp:
            temp_list.append(temp)

    total_time = time.time() - t_start
    cap.release()

    # ===== 統計 =====
    avg_infer = np.mean(infer_times) if infer_times else 0
    avg_fps = 1000 / avg_infer if avg_infer > 0 else 0
    overall_fps = frame_id / total_time if total_time > 0 else 0

    avg_ram = np.mean(ram_list)
    peak_ram = np.max(ram_list)

    avg_temp = np.mean(temp_list) if temp_list else 0
    peak_temp = np.max(temp_list) if temp_list else 0

    print("\n" + "="*50)
    print(f"Total frames: {frame_id}")
    print("")
    print(f"Total pure inference time: {pure_infer_total:.3f} s")
    print(f"Average inference time: {avg_infer:.2f} ms")
    print("")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Overall FPS: {overall_fps:.1f}")
    print("")
    print(f"Average RAM: {avg_ram:.0f} MB")
    print(f"Peak RAM: {peak_ram:.0f} MB")
    print("")
    print(f"Average CPU Temp: {avg_temp:.1f} C")
    print(f"Peak Temp: {peak_temp:.1f} C")
    print("="*50)


if __name__ == "__main__":
    main()
