import os
import cv2
import time
import argparse
import configparser
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import motmetrics as mm

from yolox.tracker.byte_tracker import BYTETracker


# =========================
# TensorRT Wrapper
# =========================
class TRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.new_api = hasattr(self.engine, "num_io_tensors")

        self.input_name = None
        self.output_name = None

        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}
        self.bindings = []

        self._allocate_buffers()

    def _allocate_buffers(self):
        if self.new_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = tuple(self.engine.get_tensor_shape(name))

                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name

                    if -1 in shape:
                        raise RuntimeError(
                            f"目前腳本先支援固定 shape engine，你的 input shape 是 dynamic: {shape}"
                        )

                    size = int(np.prod(shape))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)

                    self.host_inputs[name] = host_mem
                    self.device_inputs[name] = device_mem
                    self.context.set_tensor_address(name, int(device_mem))

                else:
                    self.output_name = name

                    if -1 in shape:
                        raise RuntimeError(
                            f"目前腳本先支援固定 shape engine，你的 output shape 是 dynamic: {shape}"
                        )

                    size = int(np.prod(shape))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)

                    self.host_outputs[name] = host_mem
                    self.device_outputs[name] = device_mem
                    self.context.set_tensor_address(name, int(device_mem))

        else:
            for binding in self.engine:
                idx = self.engine.get_binding_index(binding)
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                shape = tuple(self.engine.get_binding_shape(idx))

                if -1 in shape:
                    raise RuntimeError(
                        f"目前腳本先支援固定 shape engine，你的 binding shape 是 dynamic: {shape}"
                    )

                size = int(np.prod(shape))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                self.bindings.append(int(device_mem))

                if self.engine.binding_is_input(binding):
                    self.input_name = binding
                    self.host_inputs[binding] = host_mem
                    self.device_inputs[binding] = device_mem
                else:
                    self.output_name = binding
                    self.host_outputs[binding] = host_mem
                    self.device_outputs[binding] = device_mem

        print("========== TensorRT Engine Info ==========")
        print(f"Input name  : {self.input_name}")
        print(f"Output name : {self.output_name}")
        print(f"Input shape : {self.input_shape}")
        print(f"Output shape: {self.output_shape}")
        print("==========================================")

    @property
    def input_shape(self):
        if self.new_api:
            return tuple(self.engine.get_tensor_shape(self.input_name))
        else:
            idx = self.engine.get_binding_index(self.input_name)
            return tuple(self.engine.get_binding_shape(idx))

    @property
    def output_shape(self):
        if self.new_api:
            return tuple(self.engine.get_tensor_shape(self.output_name))
        else:
            idx = self.engine.get_binding_index(self.output_name)
            return tuple(self.engine.get_binding_shape(idx))

    def infer(self, x):
        x = np.ascontiguousarray(x)

        np.copyto(self.host_inputs[self.input_name], x.ravel())

        cuda.memcpy_htod_async(
            self.device_inputs[self.input_name],
            self.host_inputs[self.input_name],
            self.stream,
        )

        if self.new_api:
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle,
            )

        cuda.memcpy_dtoh_async(
            self.host_outputs[self.output_name],
            self.device_outputs[self.output_name],
            self.stream,
        )

        self.stream.synchronize()

        out = np.array(self.host_outputs[self.output_name])
        return out.reshape(self.output_shape)


# =========================
# YOLOX preprocess / decode
# =========================
def preprocess_yolox(img, input_size):
    """
    官方 YOLOX / ByteTrack preprocess:
    resize keep ratio + padding 114 + BGR to RGB + CHW
    """
    input_h, input_w = input_size
    h, w = img.shape[:2]

    ratio = min(input_h / h, input_w / w)

    resized = cv2.resize(
        img,
        (int(w * ratio), int(h * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = np.ones((input_h, input_w, 3), dtype=np.uint8) * 114
    padded[: resized.shape[0], : resized.shape[1]] = resized

    padded = padded[:, :, ::-1]
    padded = padded.transpose(2, 0, 1)
    padded = np.ascontiguousarray(padded, dtype=np.float32)

    return padded, ratio


def decode_yolox_outputs(outputs, input_size):
    """
    YOLOX raw output decode.
    如果你的 engine output 已經 decode 過，可以執行時加 --no-decode。
    """
    if outputs.ndim == 3:
        outputs = outputs[0]

    input_h, input_w = input_size
    strides = [8, 16, 32]

    grids = []
    expanded_strides = []

    for stride in strides:
        hsize = input_h // stride
        wsize = input_w // stride

        yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)

        grids.append(grid)
        expanded_strides.append(np.full((1, grid.shape[1], 1), stride))

    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)

    outputs = outputs.copy()

    if outputs.shape[0] != grids.shape[1]:
        print("[WARN] output 數量和 YOLOX grid 數量不同，跳過 decode")
        return outputs

    outputs[:, 0:2] = (outputs[:, 0:2] + grids[0]) * expanded_strides[0]
    outputs[:, 2:4] = np.exp(outputs[:, 2:4]) * expanded_strides[0]

    return outputs


def nms(boxes, scores, nms_thr):
    if boxes.shape[0] == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        ww = np.maximum(0.0, xx2 - xx1 + 1)
        hh = np.maximum(0.0, yy2 - yy1 + 1)

        inter = ww * hh
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def postprocess_yolox(outputs, input_size, conf_thre, nms_thre, decode=True):
    """
    回傳給官方 BYTETracker 的格式：
    Nx5 = x1, y1, x2, y2, score

    注意：
    這裡不要先除 ratio。
    官方 BYTETracker.update() 會自己根據 img_info / img_size scale 回原圖。
    """
    if outputs.ndim == 3:
        outputs = outputs[0]

    if decode:
        outputs = decode_yolox_outputs(outputs, input_size)

    if outputs.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32)

    if outputs.shape[1] < 6:
        return np.empty((0, 5), dtype=np.float32)

    boxes = outputs[:, :4]
    obj_scores = outputs[:, 4]
    cls_scores = outputs[:, 5:]

    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]

    scores = obj_scores * cls_conf

    # MOT17 是 person tracking，所以只取 person class = 0
    keep = (cls_ids == 0) & (scores >= conf_thre)

    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32)

    # cx cy w h -> x1 y1 x2 y2
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep_nms = nms(boxes_xyxy, scores, nms_thre)

    dets = np.concatenate(
        [boxes_xyxy[keep_nms], scores[keep_nms, None]],
        axis=1,
    ).astype(np.float32)

    return dets


# =========================
# MOT17 tools
# =========================
def read_seqinfo(seq_path):
    seqinfo_path = os.path.join(seq_path, "seqinfo.ini")

    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    frame_rate = int(config["Sequence"]["frameRate"])
    seq_length = int(config["Sequence"]["seqLength"])

    return frame_rate, seq_length


def get_sequences(mot_root):
    train_dir = os.path.join(mot_root, "train")

    seqs = []

    for name in sorted(os.listdir(train_dir)):
        seq_path = os.path.join(train_dir, name)

        if not os.path.isdir(seq_path):
            continue

        if not os.path.exists(os.path.join(seq_path, "img1")):
            continue

        if not os.path.exists(os.path.join(seq_path, "gt", "gt.txt")):
            continue

        if not os.path.exists(os.path.join(seq_path, "seqinfo.ini")):
            continue

        seqs.append(name)

    return seqs


def write_mot_results(save_path, results):
    """
    MOTChallenge result format:
    frame, id, x, y, w, h, score, -1, -1, -1
    """
    with open(save_path, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, tid, score in zip(tlwhs, track_ids, scores):
                x, y, w, h = tlwh
                line = (
                    f"{frame_id},{tid},"
                    f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                    f"{score:.4f},-1,-1,-1\n"
                )
                f.write(line)


def load_gt(gt_path):
    data = np.loadtxt(gt_path, delimiter=",")

    if data.ndim == 1:
        data = data[None, :]

    # MOT17 gt:
    # frame,id,x,y,w,h,mark,class,visibility
    # mark == 1 有效
    # class == 1 pedestrian
    valid = (data[:, 6] == 1) & (data[:, 7] == 1)

    return data[valid]


def load_result(result_path):
    if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
        return np.empty((0, 10), dtype=np.float32)

    data = np.loadtxt(result_path, delimiter=",")

    if data.ndim == 1:
        data = data[None, :]

    return data


def evaluate_mot(gt_path, result_path):
    gt = load_gt(gt_path)
    res = load_result(result_path)

    acc = mm.MOTAccumulator(auto_id=True)

    gt_frames = set(gt[:, 0].astype(int).tolist()) if len(gt) else set()
    res_frames = set(res[:, 0].astype(int).tolist()) if len(res) else set()

    frames = sorted(gt_frames | res_frames)

    for frame_id in frames:
        gt_frame = gt[gt[:, 0] == frame_id]
        res_frame = res[res[:, 0] == frame_id]

        gt_ids = gt_frame[:, 1].astype(int).tolist()
        res_ids = res_frame[:, 1].astype(int).tolist()

        gt_boxes = gt_frame[:, 2:6]
        res_boxes = res_frame[:, 2:6]

        distances = mm.distances.iou_matrix(
            gt_boxes,
            res_boxes,
            max_iou=0.5,
        )

        acc.update(gt_ids, res_ids, distances)

    return acc


# =========================
# Run one sequence
# =========================
def run_sequence(args, model, seq_name):
    seq_path = os.path.join(args.mot_root, "train", seq_name)
    img_dir = os.path.join(seq_path, "img1")

    frame_rate, _ = read_seqinfo(seq_path)

    tracker_args = argparse.Namespace(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=False,
    )

    tracker = BYTETracker(tracker_args, frame_rate=frame_rate)

    img_files = sorted(
        [
            x
            for x in os.listdir(img_dir)
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if args.limit > 0:
        img_files = img_files[: args.limit]

    results = []

    total_infer_time = 0.0
    total_track_time = 0.0

    start_total = time.time()

    print(f"\n========== Sequence: {seq_name} ==========")
    print(f"Total frames: {len(img_files)}")

    for idx, img_name in enumerate(img_files, start=1):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        blob, _ = preprocess_yolox(img, args.input_size)
        blob = blob[None, :, :, :]

        t0 = time.time()
        outputs = model.infer(blob)
        infer_time = time.time() - t0
        total_infer_time += infer_time

        dets = postprocess_yolox(
            outputs,
            input_size=args.input_size,
            conf_thre=args.det_conf,
            nms_thre=args.nms,
            decode=not args.no_decode,
        )

        t1 = time.time()
        online_targets = tracker.update(
            dets,
            [img_h, img_w],
            args.input_size,
        )
        track_time = time.time() - t1
        total_track_time += track_time

        online_tlwhs = []
        online_ids = []
        online_scores = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            score = t.score

            vertical = tlwh[2] / tlwh[3] > 1.6

            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(score)

        results.append((idx, online_tlwhs, online_ids, online_scores))

        if idx % args.print_freq == 0 or idx == len(img_files):
            print(
                f"[{idx:04d}/{len(img_files):04d}] "
                f"infer: {infer_time * 1000:.2f} ms | "
                f"dets: {len(dets)} | tracks: {len(online_ids)}"
            )

    elapsed = time.time() - start_total

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{seq_name}.txt")
    write_mot_results(result_path, results)

    frames = len(img_files)

    print(f"\nResult saved: {result_path}")
    print(f"Total frames              : {frames}")
    print(f"Total pure inference time : {total_infer_time:.4f} s")
    print(f"Average inference time    : {total_infer_time / frames * 1000:.4f} ms/frame")
    print(f"Pure inference FPS        : {frames / total_infer_time:.2f}")
    print(f"Total tracking time       : {total_track_time:.4f} s")
    print(f"Total elapsed time        : {elapsed:.4f} s")
    print(f"Overall FPS               : {frames / elapsed:.2f}")

    stats = {
        "frames": frames,
        "infer_time": total_infer_time,
        "track_time": total_track_time,
        "elapsed": elapsed,
    }

    return result_path, stats


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--engine", type=str, default="nano_mot17_608,1088/bytetrack_nano_mot17.engine")
    parser.add_argument("--mot-root", type=str, default="../data/MOT17")
    parser.add_argument("--result-dir", type=str, default="trt_mot17_results")

    parser.add_argument("--seq", type=str, default="", help="例如 MOT17-05-FRCNN，不填就是跑全部 train")
    parser.add_argument("--limit", type=int, default=-1, help="只跑前 N 張，-1 表示全部")

    parser.add_argument("--input-h", type=int, default=-1)
    parser.add_argument("--input-w", type=int, default=-1)

    parser.add_argument("--det-conf", type=float, default=0.01)
    parser.add_argument("--nms", type=float, default=0.7)

    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--min-box-area", type=float, default=10)

    parser.add_argument("--no-decode", action="store_true", help="如果 engine output 已經 decode 過，就加這個")
    parser.add_argument("--print-freq", type=int, default=10)

    args = parser.parse_args()

    print("========== ByteTrack TensorRT MOT17 Accuracy ==========")
    print(f"Engine     : {args.engine}")
    print(f"MOT root   : {args.mot_root}")
    print(f"Result dir : {args.result_dir}")
    print("Metric     : MOTA / IDF1 / FP / FN / IDs")
    print("=======================================================")

    model = TRTModel(args.engine)

    input_shape = model.input_shape

    if len(input_shape) != 4:
        raise RuntimeError(f"不支援的 input shape: {input_shape}")

    _, c, engine_h, engine_w = input_shape

    if c != 3:
        raise RuntimeError(f"input channel 不是 3，目前 input shape: {input_shape}")

    if args.input_h <= 0:
        args.input_h = engine_h

    if args.input_w <= 0:
        args.input_w = engine_w

    args.input_size = (args.input_h, args.input_w)

    print(f"Use input size: {args.input_h}x{args.input_w}")

    if args.seq:
        sequences = [args.seq]
    else:
        sequences = get_sequences(args.mot_root)

    if len(sequences) == 0:
        raise RuntimeError("找不到 MOT17 train sequences，請確認 --mot-root 是否指定到 MOT17 這一層")

    all_accs = []
    names = []
    all_stats = []

    for seq_name in sequences:
        result_path, stats = run_sequence(args, model, seq_name)

        gt_path = os.path.join(args.mot_root, "train", seq_name, "gt", "gt.txt")
        acc = evaluate_mot(gt_path, result_path)

        all_accs.append(acc)
        names.append(seq_name)
        all_stats.append(stats)

    mh = mm.metrics.create()

    summary = mh.compute_many(
        all_accs,
        names=names,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mostly_tracked",
            "mostly_lost",
        ],
        generate_overall=True,
    )

    print("\n========== MOT17 Tracking Accuracy ==========")
    print(
        mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap={
                "num_frames": "Frames",
                "mota": "MOTA",
                "motp": "MOTP",
                "idf1": "IDF1",
                "idp": "IDP",
                "idr": "IDR",
                "num_switches": "IDs",
                "num_false_positives": "FP",
                "num_misses": "FN",
                "mostly_tracked": "MT",
                "mostly_lost": "ML",
            },
        )
    )
    print("=============================================")

    total_frames = sum(x["frames"] for x in all_stats)
    total_infer = sum(x["infer_time"] for x in all_stats)
    total_track = sum(x["track_time"] for x in all_stats)
    total_elapsed = sum(x["elapsed"] for x in all_stats)

    print("\n========== Speed Summary ==========")
    print(f"Total frames              : {total_frames}")
    print(f"Total pure inference time : {total_infer:.4f} s")
    print(f"Average inference time    : {total_infer / total_frames * 1000:.4f} ms/frame")
    print(f"Pure inference FPS        : {total_frames / total_infer:.2f}")
    print(f"Total tracking time       : {total_track:.4f} s")
    print(f"Total elapsed time        : {total_elapsed:.4f} s")
    print(f"Overall FPS               : {total_frames / total_elapsed:.2f}")
    print("===================================")


if __name__ == "__main__":
    main()