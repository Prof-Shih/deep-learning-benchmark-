import os
import json
import argparse
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


YOLO_TO_COCO_CAT_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-7
    return inter / union


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        if len(idxs) == 1:
            break

        ious = box_iou(boxes[current], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_thres]

    return keep


def postprocess_yolov8(pred, conf_thres=0.001, iou_thres=0.7):
    pred = np.squeeze(pred)

    if pred.ndim != 2:
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])

    if pred.shape[0] == 84 and pred.shape[1] != 84:
        pred = pred.T

    if pred.shape[1] == 84:
        boxes_xywh = pred[:, :4]
        cls_conf = pred[:, 4:]

        cls_ids = np.argmax(cls_conf, axis=1)
        scores = cls_conf[np.arange(cls_conf.shape[0]), cls_ids]

        mask = scores > conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        if len(scores) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        boxes_xyxy = xywh2xyxy(boxes_xywh)

        final_dets = []

        for c in np.unique(cls_ids):
            class_mask = cls_ids == c
            b = boxes_xyxy[class_mask]
            s = scores[class_mask]
            cids = cls_ids[class_mask]

            keep = nms(b, s, iou_thres)

            for k in keep:
                x1, y1, x2, y2 = b[k]
                final_dets.append([x1, y1, x2, y2, s[k], cids[k]])

        if len(final_dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        return np.array(final_dets, dtype=np.float32)

    if pred.shape[1] == 6:
        det = pred[pred[:, 4] > conf_thres]
        return det.astype(np.float32)

    raise RuntimeError(f"Unsupported YOLOv8 output shape: {pred.shape}")


def get_letterbox_meta(original_w, original_h, imgsz):
    r = min(imgsz / original_h, imgsz / original_w)

    new_unpad_w = int(round(original_w * r))
    new_unpad_h = int(round(original_h * r))

    dw = imgsz - new_unpad_w
    dh = imgsz - new_unpad_h

    dw /= 2
    dh /= 2

    return r, (dw, dh)


def scale_boxes_to_original(dets, ratio, pad, original_w, original_h):
    if len(dets) == 0:
        return dets

    dw, dh = pad

    dets[:, [0, 2]] -= dw
    dets[:, [1, 3]] -= dh
    dets[:, :4] /= ratio

    dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, original_w)
    dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, original_h)

    return dets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ann", type=str, default="../data/annotations_trainval2017/instances_val2017.json")
    parser.add_argument("--npy_dir", type=str, default="./yolov8_kneopi_raw_npy")
    parser.add_argument("--save-json", type=str, default="yolov8_kneopi_predictions.json")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()

    print("========== PC YOLOv8 COCO mAP from NPY ==========")
    print(f"Annotation : {args.ann}")
    print(f"NPY dir    : {args.npy_dir}")
    print(f"Save JSON  : {args.save_json}")
    print(f"Input size : {args.imgsz}x{args.imgsz}")
    print(f"Conf       : {args.conf}")
    print(f"IoU        : {args.iou}")
    print("=================================================")

    if not os.path.isfile(args.ann):
        print(f"Error: annotation not found -> {args.ann}")
        return

    if not os.path.isdir(args.npy_dir):
        print(f"Error: npy folder not found -> {args.npy_dir}")
        return

    coco_gt = COCO(args.ann)

    img_ids = sorted(coco_gt.getImgIds())
    if args.limit > 0:
        img_ids = img_ids[:args.limit]

    coco_results = []
    eval_img_ids = []

    valid_count = 0
    missing_count = 0
    error_count = 0

    for idx, img_id in enumerate(img_ids, start=1):
        npy_path = os.path.join(args.npy_dir, f"{img_id:012d}.npy")

        if not os.path.isfile(npy_path):
            missing_count += 1
            continue

        try:
            info = coco_gt.loadImgs(img_id)[0]
            original_w = info["width"]
            original_h = info["height"]

            pred = np.load(npy_path)

            dets = postprocess_yolov8(
                pred,
                conf_thres=args.conf,
                iou_thres=args.iou
            )

            ratio, pad = get_letterbox_meta(
                original_w=original_w,
                original_h=original_h,
                imgsz=args.imgsz
            )

            dets = scale_boxes_to_original(
                dets,
                ratio=ratio,
                pad=pad,
                original_w=original_w,
                original_h=original_h
            )

            eval_img_ids.append(img_id)

            for det in dets:
                x1, y1, x2, y2, score, cls_id = det

                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                cls_id = int(cls_id)

                if cls_id < 0 or cls_id >= len(YOLO_TO_COCO_CAT_ID):
                    continue

                coco_results.append({
                    "image_id": int(img_id),
                    "category_id": YOLO_TO_COCO_CAT_ID[cls_id],
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(w),
                        float(h)
                    ],
                    "score": float(score)
                })

            valid_count += 1

        except Exception as e:
            print(f"[Error] image_id={img_id} -> {e}")
            error_count += 1
            continue

        if idx % args.print_every == 0 or idx == len(img_ids):
            print(
                f"[Progress] {idx}/{len(img_ids)} | "
                f"Valid: {valid_count} | "
                f"Missing: {missing_count} | "
                f"Errors: {error_count} | "
                f"Predictions: {len(coco_results)}"
            )

    with open(args.save_json, "w") as f:
        json.dump(coco_results, f)

    print("\n========== Prediction Summary ==========")
    print(f"Valid images      : {valid_count}")
    print(f"Missing npy       : {missing_count}")
    print(f"Errors            : {error_count}")
    print(f"Total predictions : {len(coco_results)}")
    print(f"Prediction JSON   : {args.save_json}")
    print("========================================")

    if len(coco_results) == 0:
        print("[ERROR] No predictions generated. Cannot run COCO evaluation.")
        return

    print("\n========== COCO mAP Evaluation ==========")

    coco_dt = coco_gt.loadRes(args.save_json)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(list(set(eval_img_ids)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("========================================")


if __name__ == "__main__":
    main()