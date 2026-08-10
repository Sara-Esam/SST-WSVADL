"""Patch-level localization metrics (PAUC, PAP, TIoU).

Paper numbers use bounding-box TIoU: thresholded patch maps are converted to a
tight bbox per frame, then IoU is taken against the GT bbox and gated by the
temporal score.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


DEFAULT_BBOX_ANNOTATIONS = {
    "ucf": "annotations/merged_ucf_crime_0.25_quantized.json",
    "msad": "annotations/merged_MSAD_0.25_quantized.json",
    "xdviolence": "annotations/merged_xd-v_0.25_quantized.json",
}

DEFAULT_VIDEO_SIZES = {
    "ucf": "annotations/ucf_video_sizes.json",
    "msad": "annotations/msad_video_sizes.json",
    "xdviolence": "annotations/xdviolence_video_sizes.json",
}

DEFAULT_NATIVE_SIZE = {
    "ucf": (240, 320),
    "msad": (720, 1280),
    "xdviolence": (320, 640),
}


def find_optimal_threshold_youden(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def patches_to_bbox(patches, patch_size=16, image_size=128):
    """Tight axis-aligned bbox covering all active patches on the grid."""
    patches = np.asarray(patches, dtype=np.uint8).ravel()
    side = image_size // patch_size
    grid = patches.reshape(side, side)
    ys, xs = np.where(grid > 0)
    if len(xs) == 0:
        return None
    x1 = int(xs.min()) * patch_size
    y1 = int(ys.min()) * patch_size
    x2 = int(xs.max() + 1) * patch_size
    y2 = int(ys.max() + 1) * patch_size
    return [x1, y1, min(x2, image_size), min(y2, image_size)]


def compute_bbox_iou(bbox1, bbox2):
    if bbox1 is None or bbox2 is None:
        return 0.0
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def compute_iou(preds, gt, anomaly_only=True):
    """Binary patch IoU for one frame. Returns -1 when anomaly_only and gt is empty."""
    preds = np.asarray(preds, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=np.float64).ravel()
    if anomaly_only:
        if gt.sum() == 0:
            return -1.0
        return float(np.sum(preds * gt) / (np.sum(preds + gt - preds * gt) + 1e-6))
    if preds.sum() == 0 and gt.sum() == 0:
        return 1.0
    return float(np.sum(preds * gt) / (np.sum(preds + gt - preds * gt) + 1e-6))


def compute_pauc_pap(patch_scores, frame_scores, patch_gt, patches_per_frame):
    """Patch AUC / AP with temporal scores expanded to patch resolution."""
    frame_exp = np.repeat(np.asarray(frame_scores, dtype=np.float64), patches_per_frame)
    scores = np.asarray(patch_scores, dtype=np.float64) * frame_exp
    gt = np.asarray(patch_gt)

    fpr, tpr, _ = roc_curve(gt, scores)
    pauc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(gt, scores)
    pap = float(auc(recall, precision))
    return pauc, pap


def corners_to_xyxy(corners):
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def scale_box_to_image(box, vh, vw, image_size=128):
    sx, sy = image_size / float(vw), image_size / float(vh)
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def video_name_to_file_name(name, dataset="ucf"):
    name = str(name)
    if name.endswith(".mp4"):
        return name
    if dataset == "ucf":
        base = name.replace("_x264", "")
        return f"{base}_x264.mp4"
    return f"{name}.mp4"


def index_bbox_annotations(annotations):
    by_file = defaultdict(list)
    for entry in annotations:
        by_file[entry["file_name"]].append(entry)
    return by_file


def load_bbox_annotations(path):
    with open(path) as f:
        data = json.load(f)
    return index_bbox_annotations(data)


def load_video_sizes(path, default_size=None):
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        return {k: (int(v[0]), int(v[1])) for k, v in raw.items()}
    return {}


def union_boxes(boxes):
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def native_boxes_for_frame(entries, frame_idx):
    """Union of all annotation boxes covering frame_idx (native video coords)."""
    boxes = []
    for entry in entries:
        for box in entry.get("bounding_boxes", []):
            st, ed = int(box["start_frame"]), int(box["end_frame"])
            if frame_idx < st or frame_idx > ed:
                continue
            kfs = box["keyframes"]
            key = str(st) if str(st) in kfs else next(iter(kfs))
            boxes.append(corners_to_xyxy(kfs[key]))
    return union_boxes(boxes)


def build_gt_bboxes_from_annotations(
    ann_index,
    video_names,
    frames_count_per_video,
    dataset="ucf",
    video_sizes=None,
    default_size=(240, 320),
    image_size=128,
):
    """
    Per-frame GT boxes in image_size coords (list aligned with concatenated frames).
    None means no GT box on that frame.
    """
    video_sizes = video_sizes or {}
    out = []
    for name, n_frames in zip(video_names, frames_count_per_video):
        n_frames = int(n_frames)
        file_name = video_name_to_file_name(name, dataset=dataset)
        entries = ann_index.get(file_name, [])
        if not entries and file_name.replace("_x264", "") in ann_index:
            entries = ann_index[file_name.replace("_x264", "")]
        vh, vw = video_sizes.get(file_name, default_size)
        for f in range(n_frames):
            native = native_boxes_for_frame(entries, f)
            if native is None:
                out.append(None)
            else:
                out.append(scale_box_to_image(native, vh, vw, image_size=image_size))
    return out


def _frame_ious(
    preds,
    gts,
    mode="bbox",
    patch_size=16,
    image_size=128,
    gt_bboxes_frame=None,
):
    """Per-frame IoU; -1 where GT has no active patches / no GT box."""
    n_frames = preds.shape[0]
    ious = np.empty(n_frames, dtype=np.float64)
    gt_sum = gts.sum(axis=1) if gts is not None else None
    for j in range(n_frames):
        if mode == "bbox" and gt_bboxes_frame is not None:
            gt_box = gt_bboxes_frame[j]
            if gt_box is None:
                ious[j] = -1.0
                continue
            ious[j] = compute_bbox_iou(
                patches_to_bbox(preds[j], patch_size, image_size),
                gt_box,
            )
            continue

        if gt_sum[j] == 0:
            ious[j] = -1.0
            continue
        if mode == "bbox":
            ious[j] = compute_bbox_iou(
                patches_to_bbox(preds[j], patch_size, image_size),
                patches_to_bbox(gts[j], patch_size, image_size),
            )
        else:
            p = preds[j]
            g = gts[j]
            ious[j] = np.sum(p * g) / (np.sum(p + g - p * g) + 1e-6)
    return ious


def compute_tiou(
    patch_scores,
    frame_scores,
    patch_gt,
    frames_count_per_video,
    video_names,
    temporal_threshold,
    patch_threshold=0.5,
    patches_per_frame=64,
    anomaly_frames_only=False,
    iou_mode="bbox",
    patch_size=16,
    image_size=128,
    gt_bboxes=None,
):
    """
    Temporal IoU as in patch_evaluation_topk.py.

    iou_mode:
      - "bbox" (paper default): convert pred patches to tight bboxes, then box IoU
      - "patch": binary patch-set IoU

    If gt_bboxes is provided (list of xyxy or None, length = sum(frames)), those
    boxes are used as GT for bbox mode (from quantized JSON annotations).
    Otherwise GT boxes are reconstructed from patch_gt.

    If anomaly_frames_only is True, only frames with a GT box enter the mean
    (temporal misses contribute 0). Default False matches the paper table
    (legacy UR-DMU protocol that can include zero contributions from
    non-anomaly frames when the temporal score is below threshold).
    """
    patch_scores = np.asarray(patch_scores, dtype=np.float64)
    frame_scores = np.asarray(frame_scores, dtype=np.float64)
    patch_gt = np.asarray(patch_gt, dtype=np.float64)
    thresholded = (patch_scores >= patch_threshold).astype(np.float64)

    tiou_values = []
    patch_cursor = 0
    frame_cursor = 0

    for n_frames, name in zip(frames_count_per_video, video_names):
        n_patches = int(n_frames) * patches_per_frame
        if "Normal" in name or "normal" in name:
            patch_cursor += n_patches
            frame_cursor += int(n_frames)
            continue

        preds = thresholded[patch_cursor:patch_cursor + n_patches].reshape(int(n_frames), patches_per_frame)
        gts = patch_gt[patch_cursor:patch_cursor + n_patches].reshape(int(n_frames), patches_per_frame)
        temporal = frame_scores[frame_cursor:frame_cursor + int(n_frames)].copy()
        temporal_bin = (temporal >= temporal_threshold).astype(np.float64)

        gt_slice = None
        if gt_bboxes is not None:
            gt_slice = gt_bboxes[frame_cursor:frame_cursor + int(n_frames)]

        ious = _frame_ious(
            preds,
            gts,
            mode=iou_mode,
            patch_size=patch_size,
            image_size=image_size,
            gt_bboxes_frame=gt_slice,
        )

        if anomaly_frames_only:
            valid = ious >= 0
            if valid.any():
                tiou_values.extend((ious[valid] * temporal_bin[valid]).tolist())
        else:
            scored = ious * temporal_bin
            tiou_values.extend(scored[scored >= 0].tolist())

        patch_cursor += n_patches
        frame_cursor += int(n_frames)

    if not tiou_values:
        return 0.0, 0
    return float(np.mean(tiou_values)), len(tiou_values)


def evaluate_patch_localization(
    patch_scores,
    frame_scores,
    patch_gt,
    frames_count_per_video,
    video_names,
    temporal_threshold,
    patches_per_frame,
    patch_thresholds=(0.5, 0.1, 0.7),
    anomaly_frames_only=False,
    iou_mode="bbox",
    patch_size=16,
    image_size=128,
    gt_bboxes=None,
):
    """Report PAUC/PAP and TIoU at several patch score thresholds."""
    pauc, pap = compute_pauc_pap(patch_scores, frame_scores, patch_gt, patches_per_frame)
    print(f"PAUC: {pauc:.4f}, PAP: {pap:.4f}")

    results = {"pauc": pauc, "pap": pap, "tiou": {}}
    for thr in patch_thresholds:
        tiou, n = compute_tiou(
            patch_scores,
            frame_scores,
            patch_gt,
            frames_count_per_video,
            video_names,
            temporal_threshold=temporal_threshold,
            patch_threshold=thr,
            patches_per_frame=patches_per_frame,
            anomaly_frames_only=anomaly_frames_only,
            iou_mode=iou_mode,
            patch_size=patch_size,
            image_size=image_size,
            gt_bboxes=gt_bboxes,
        )
        mode = "anomaly frames" if anomaly_frames_only else "legacy"
        src = "json-gt" if (gt_bboxes is not None and iou_mode == "bbox") else "patch-gt"
        print(f"TIoU-{iou_mode} (%) @ {thr} ({mode}, {src}): {tiou * 100:.4f}  [n={n}]")
        results["tiou"][thr] = tiou
    return results
