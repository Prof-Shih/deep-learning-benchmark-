# infer_bytetrack_onnx_rpi5.py

import os
import re
import cv2
import time
import argparse
import threading
import subprocess
from pathlib import Path

import psutil
import numpy as np
import onnxruntime as ort

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# Raspberry Pi 5 Monitor
# =========================

def read_pmic_power_w():
    try:
        out = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"],
            text=True,
            timeout=1.0
        )
    except Exception:
        return None

    currents = {}
    volts = {}

    for line in out.splitlines():
        m_i = re.match(
            r"(.+?)_A\s+current\(\d+\)=([0-9.]+)A",
            line.strip()
        )
        if m_i:
            currents[m_i.group(1).strip()] = float(m_i.group(2))
            continue

        m_v = re.match(
            r"(.+?)_V\s+volt\(\d+\)=([0-9.]+)V",
            line.strip()
        )
        if m_v:
            volts[m_v.group(1).strip()] = float(m_v.group(2))
            continue

    total_w = 0.0
    for rail, current in currents.items():
        if rail in volts:
            total_w += current * volts[rail]

    return total_w


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


class Monitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.running = False
        self.thread = None
        self.ram_mb = []
        self.power_w = []
        self.temp_c = []
        self.proc = psutil.Process(os.getpid())

    def _loop(self):
        while self.running:
            try:
                self.ram_mb.append(self.proc.memory_info().rss / 1024 / 1024)
            except Exception:
                pass

            p = read_pmic_power_w()
            if p is not None:
                self.power_w.append(p)

            t = read_temp_c()
            if t is not None:
                self.temp_c.append(t)

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    @staticmethod
    def avg(xs):
        return float(np.mean(xs)) if xs else None

    @staticmethod
    def mn(xs):
        return float(np.min(xs)) if xs else None

    @staticmethod
    def mx(xs):
        return float(np.max(xs)) if xs else None


# =========================
# Data
# =========================

def get_images(source):
    p = Path(source)
    imgs = []
    for x in p.rglob("*"):
        if x.suffix.lower() in IMG_EXTS:
            imgs.append(str(x))
    return sorted(imgs)


def letterbox(img, new_shape=(608, 1088), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    r = min(new_h / h, new_w / w)
    resized_w = int(round(w * r))
    resized_h = int(round(h * r))

    dw = new_w - resized_w
    dh = new_h - resized_h
    dw /= 2
    dh /= 2

    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, r, left, top


def preprocess(img, input_h, input_w):
    img_lb, ratio, pad_x, pad_y = letterbox(img, (input_h, input_w))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32)
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x, ratio, pad_x, pad_y


# =========================
# YOLOX Decode / NMS
# =========================

def make_grids(input_h, input_w, strides=(8, 16, 32)):
    grids = []
    expanded_strides = []

    for s in strides:
        hs = input_h // s
        ws = input_w // s
        yv, xv = np.meshgrid(np.arange(hs), np.arange(ws), indexing="ij")
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((1, hs * ws, 1), s))

    grids = np.concatenate(grids, axis=1).astype(np.float32)
    expanded_strides = np.concatenate(expanded_strides, axis=1).astype(np.float32)
    return grids, expanded_strides


def decode_yolox_outputs(outputs, input_h, input_w):
    """
    支援常見 YOLOX raw output:
    [1, N, 85] or [1, N, 6/7]
    """
    if isinstance(outputs, list):
        out = outputs[0]
    else:
        out = outputs

    out = np.asarray(out)

    if out.ndim == 3:
        out = out[0]

    # 已經是 x1,y1,x2,y2,score,class
    if out.shape[-1] in [6, 7]:
        return out

    # YOLOX raw: cx, cy, w, h, obj, cls...
    grids, expanded_strides = make_grids(input_h, input_w)

    pred = out.copy()
    pred[:, 0:2] = (pred[:, 0:2] + grids[0]) * expanded_strides[0]
    pred[:, 2:4] = np.exp(pred[:, 2:4]) * expanded_strides[0]

    boxes = pred[:, 0:4]
    obj = pred[:, 4:5]
    cls_scores = pred[:, 5:]

    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
    scores = obj[:, 0] * cls_conf

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    dets = np.stack([x1, y1, x2, y2, scores, cls_ids], axis=1)
    return dets


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])

    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = iou_matrix(boxes[i:i+1], boxes[order[1:]])[0]
        inds = np.where(ious <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def postprocess(dets, ratio, pad_x, pad_y, orig_w, orig_h,
                conf_thres=0.1, nms_thres=0.7, person_only=True):
    if dets is None or len(dets) == 0:
        return np.empty((0, 5), dtype=np.float32)

    dets = np.asarray(dets)

    if dets.shape[1] < 6:
        return np.empty((0, 5), dtype=np.float32)

    boxes = dets[:, :4].astype(np.float32)
    scores = dets[:, 4].astype(np.float32)
    cls_ids = dets[:, 5].astype(np.int32)

    if person_only:
        mask = cls_ids == 0
        boxes = boxes[mask]
        scores = scores[mask]

    mask = scores >= conf_thres
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    keep = nms(boxes, scores, nms_thres)
    boxes = boxes[keep]
    scores = scores[keep]

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    return np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)


# =========================
# Simple BYTE-like Tracker
# =========================

class Track:
    def __init__(self, box, score, track_id, frame_id):
        self.box = box.astype(np.float32)
        self.score = float(score)
        self.track_id = track_id
        self.last_frame = frame_id
        self.age = 0

    def update(self, box, score, frame_id):
        self.box = box.astype(np.float32)
        self.score = float(score)
        self.last_frame = frame_id
        self.age = 0


class SimpleBYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, low_thresh=0.1, max_lost=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.max_lost = max_lost
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0

    def _match(self, tracks, dets, iou_threshold):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        track_boxes = np.array([t.box for t in tracks], dtype=np.float32)
        det_boxes = dets[:, :4].astype(np.float32)

        ious = iou_matrix(track_boxes, det_boxes)
        cost = 1.0 - ious

        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(dets)))

        if HAS_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if ious[r, c] >= iou_threshold:
                    matches.append((r, c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)
        else:
            pairs = []
            for r in range(ious.shape[0]):
                for c in range(ious.shape[1]):
                    pairs.append((ious[r, c], r, c))
            pairs.sort(reverse=True)

            used_r, used_c = set(), set()
            for val, r, c in pairs:
                if val < iou_threshold:
                    break
                if r not in used_r and c not in used_c:
                    matches.append((r, c))
                    used_r.add(r)
                    used_c.add(c)
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def update(self, detections):
        """
        detections: [N, 5] => x1,y1,x2,y2,score
        """
        self.frame_id += 1

        if detections is None or len(detections) == 0:
            for t in self.tracks:
                t.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_lost]
            return []

        scores = detections[:, 4]
        high = detections[scores >= self.track_thresh]
        low = detections[(scores >= self.low_thresh) & (scores < self.track_thresh)]

        # Stage 1: high score detections
        matches, unmatched_t, unmatched_high = self._match(
            self.tracks, high, self.match_thresh
        )

        for ti, di in matches:
            self.tracks[ti].update(high[di, :4], high[di, 4], self.frame_id)

        remain_tracks = [self.tracks[i] for i in unmatched_t]

        # Stage 2: low score detections
        matches2, unmatched_t2, _ = self._match(
            remain_tracks, low, 0.5
        )

        for ti, di in matches2:
            remain_tracks[ti].update(low[di, :4], low[di, 4], self.frame_id)

        # New tracks from unmatched high detections
        for di in unmatched_high:
            box = high[di, :4]
            score = high[di, 4]
            self.tracks.append(Track(box, score, self.next_id, self.frame_id))
            self.next_id += 1

        for t in self.tracks:
            if t.last_frame != self.frame_id:
                t.age += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_lost]

        active = []
        for t in self.tracks:
            if t.last_frame == self.frame_id:
                active.append([
                    t.track_id,
                    t.box[0], t.box[1], t.box[2], t.box[3],
                    t.score
                ])

        return active


# =========================
# Main
# =========================

def create_session(model_path, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    return sess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bytetrack_nano_mot17.onnx")
    parser.add_argument("--source", type=str, default="../test.mp4",
                        help="image folder, MOT img1 folder, or video path")
    parser.add_argument("--input-h", type=int, default=608)
    parser.add_argument("--input-w", type=int, default=1088)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--low-thresh", type=float, default=0.1)
    parser.add_argument("--max-lost", type=int, default=30)

    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--result-path", type=str, default="bytetrack_result.txt")

    args = parser.parse_args()

    sess = create_session(args.model, args.threads)
    input_name = sess.get_inputs()[0].name

    tracker = SimpleBYTETracker(
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        low_thresh=args.low_thresh,
        max_lost=args.max_lost
    )

    source_path = Path(args.source)

    is_video = source_path.is_file() and source_path.suffix.lower() not in IMG_EXTS

    if is_video:
        cap = cv2.VideoCapture(str(source_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.limit > 0:
            total_frames = min(total_frames, args.limit)
        frame_iter = None
    else:
        images = get_images(args.source)
        if args.limit > 0:
            images = images[:args.limit]
        total_frames = len(images)
        cap = None

    if total_frames <= 0:
        raise RuntimeError("No input frames found.")

    # Warmup
    dummy = np.zeros((1, 3, args.input_h, args.input_w), dtype=np.float32)
    for _ in range(args.warmup):
        _ = sess.run(None, {input_name: dummy})

    monitor = Monitor(interval=0.2)
    monitor.start()

    total_start = time.perf_counter()
    infer_time = 0.0
    processed = 0
    results = []

    while True:
        if is_video:
            if args.limit > 0 and processed >= args.limit:
                break
            ret, img = cap.read()
            if not ret:
                break
        else:
            if processed >= len(images):
                break
            img = cv2.imread(images[processed])
            if img is None:
                processed += 1
                continue

        orig_h, orig_w = img.shape[:2]

        x, ratio, pad_x, pad_y = preprocess(img, args.input_h, args.input_w)

        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: x})
        t1 = time.perf_counter()

        infer_time += (t1 - t0)

        raw_dets = decode_yolox_outputs(outputs, args.input_h, args.input_w)

        dets = postprocess(
            raw_dets,
            ratio=ratio,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=orig_w,
            orig_h=orig_h,
            conf_thres=args.conf,
            nms_thres=args.nms,
            person_only=True
        )

        tracks = tracker.update(dets)

        frame_id = processed + 1
        for tr in tracks:
            tid, x1, y1, x2, y2, score = tr
            w = x2 - x1
            h = y2 - y1
            results.append(
                f"{frame_id},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1"
            )

        processed += 1

        if processed % 10 == 0:
            print(f"Processed {processed}/{total_frames}")

    total_end = time.perf_counter()
    monitor.stop()

    if cap is not None:
        cap.release()

    total_time = total_end - total_start
    total_fps = processed / total_time if total_time > 0 else 0.0
    infer_fps = processed / infer_time if infer_time > 0 else 0.0

    if args.save_result:
        with open(args.result_path, "w") as f:
            f.write("\n".join(results))
        print(f"Saved tracking result: {args.result_path}")

    print("\n========== ByteTrack ONNX Raspberry Pi 5 Result ==========")
    print(f"Model                 : {args.model}")
    print(f"Source                : {args.source}")
    print(f"Input size            : {args.input_h} x {args.input_w}")
    print(f"Frames                : {processed}")

    print("\n[Time / FPS]")
    print(f"Total time            : {total_time:.4f} s")
    print(f"Pure inference time   : {infer_time:.4f} s")
    print(f"Total FPS             : {total_fps:.4f} FPS")
    print(f"Pure inference FPS    : {infer_fps:.4f} FPS")

    print("\n[RAM]")
    print(f"Average RAM           : {Monitor.avg(monitor.ram_mb):.2f} MB")

    print("\n[Power]")
    avg_power = Monitor.avg(monitor.power_w)
    if avg_power is None:
        print("Average Power         : N/A")
        print("Note                  : vcgencmd pmic_read_adc failed or returned no valid rails")
    else:
        print(f"Average Power         : {avg_power:.4f} W")

    print("\n[Temperature]")
    t_min = Monitor.mn(monitor.temp_c)
    t_max = Monitor.mx(monitor.temp_c)
    t_avg = Monitor.avg(monitor.temp_c)

    if t_avg is None:
        print("Temperature           : N/A")
    else:
        print(f"Min Temperature       : {t_min:.2f} °C")
        print(f"Max Temperature       : {t_max:.2f} °C")
        print(f"Avg Temperature       : {t_avg:.2f} °C")

    print("==========================================================\n")


if __name__ == "__main__":
    main()