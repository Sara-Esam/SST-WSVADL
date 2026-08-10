"""
Build patch-quantized bbox annotations in the same schema as annotations/merged_*.json.

For each original box, map it to the 8x8 (16px) patch grid with the same 0.25
overlap rule used for patch GT, then replace the keyframe corners with the
tight axis-aligned box covering those patches (scaled back to video resolution).

Originals are backed up to annotations/*_original.json on first run.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.patch_metrics import patches_to_bbox


DATASET_CFG = {
    "ucf": {
        "stem": "annotations/merged_ucf_crime",
        "original_backup": "annotations/merged_ucf_crime_original.json",
        "sizes_sidecar": "annotations/ucf_video_sizes.json",
        "video_root": "/projects/0/prjs1250/feature_extraction/Videos/Videos/all_videos_test_only",
        "default_size": (240, 320),  # H, W
    },
    "msad": {
        "stem": "annotations/merged_MSAD",
        "original_backup": "annotations/merged_MSAD_original.json",
        "sizes_sidecar": "annotations/msad_video_sizes.json",
        "video_root": "/projects/0/prjs1250/video_datasets/MSAD/all_videos",
        "default_size": (720, 1280),
    },
    "xdviolence": {
        "stem": "annotations/merged_xd-v",
        "original_backup": "annotations/merged_xd-v_original.json",
        "sizes_sidecar": "annotations/xdviolence_video_sizes.json",
        "video_root": "/projects/0/prjs1250/video_datasets/xd-violence/all_videos",
        "default_size": (320, 640),
    },
}


def thr_tag(overlap_threshold):
    """Format overlap threshold for filenames (0.25, 0.50, 0.75, 1.0)."""
    t = float(overlap_threshold)
    if abs(t - 1.0) < 1e-9:
        return "1.0"
    return f"{t:.2f}"


def quantized_output_path(cfg, overlap_threshold):
    return f"{cfg['stem']}_{thr_tag(overlap_threshold)}_quantized.json"


def corners_to_xyxy(corners):
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def corners_from_xyxy(box):
    x1, y1, x2, y2 = box
    return [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]


def probe_size(video_path, default_size):
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return int(h), int(w)
    except Exception:
        pass
    return default_size


def box_to_patch_mask(x1, y1, x2, y2, patch_size=16, image_size=128, overlap_threshold=0.25, tol=5):
    """Same overlap rule as UR-DMU generate_patch_gt.py (incl. ±5 tolerance)."""
    side = image_size // patch_size
    mask = np.zeros(side * side, dtype=np.uint8)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return mask
    for i in range(side * side):
        px = (i % side) * patch_size
        py = (i // side) * patch_size
        patch_x2 = px + patch_size
        patch_y2 = py + patch_size
        overlap_x = max(0, min(patch_x2, x2 + tol) - max(px, x1 - tol))
        overlap_y = max(0, min(patch_y2, y2 + tol) - max(py, y1 - tol))
        overlap_area = overlap_x * overlap_y
        if overlap_area / (patch_size * patch_size) >= overlap_threshold:
            mask[i] = 1
    return mask


def quantize_corners(corners, vh, vw, image_size=128, patch_size=16, overlap_threshold=0.25):
    """Native-resolution corners → patch-grid tight box → native-resolution corners."""
    x1, y1, x2, y2 = corners_to_xyxy(corners)
    sx, sy = image_size / float(vw), image_size / float(vh)
    rx1, ry1, rx2, ry2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    mask = box_to_patch_mask(
        rx1, ry1, rx2, ry2,
        patch_size=patch_size,
        image_size=image_size,
        overlap_threshold=overlap_threshold,
    )
    if mask.sum() == 0:
        return None
    box128 = patches_to_bbox(mask, patch_size=patch_size, image_size=image_size)
    if box128 is None:
        return None
    inv_sx, inv_sy = vw / float(image_size), vh / float(image_size)
    q = [
        int(round(box128[0] * inv_sx)),
        int(round(box128[1] * inv_sy)),
        int(round(box128[2] * inv_sx)),
        int(round(box128[3] * inv_sy)),
    ]
    # Clamp to frame
    q[0] = max(0, min(q[0], vw - 1))
    q[1] = max(0, min(q[1], vh - 1))
    q[2] = max(q[0] + 1, min(q[2], vw))
    q[3] = max(q[1] + 1, min(q[3], vh))
    return corners_from_xyxy(q)


def resolve_original_path(cfg, repo_root):
    backup = os.path.join(repo_root, cfg["original_backup"])
    if os.path.exists(backup):
        return backup
    # Legacy un-suffixed name
    legacy = os.path.join(repo_root, cfg["stem"] + ".json")
    if os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(f"No original annotations found for {cfg['original_backup']}")


def backup_original(cfg, repo_root, source_path):
    backup = os.path.join(repo_root, cfg["original_backup"])
    if os.path.exists(backup):
        print(f"Original backup already present: {backup}")
        return backup
    legacy = os.path.join(repo_root, cfg["stem"] + ".json")
    if os.path.exists(legacy) and os.path.abspath(source_path) == os.path.abspath(legacy):
        os.rename(legacy, backup)
        print(f"Backed up original annotations to {backup}")
        return backup
    return source_path


def build_quantized(dataset, repo_root, image_size=128, patch_size=16, overlap_threshold=0.25):
    cfg = DATASET_CFG[dataset]
    source = resolve_original_path(cfg, repo_root)
    backup_original(cfg, repo_root, source)
    source = os.path.join(repo_root, cfg["original_backup"])
    if not os.path.exists(source):
        source = resolve_original_path(cfg, repo_root)

    data = json.load(open(source))
    sizes = {}
    skipped_boxes = 0
    kept_boxes = 0

    # Cache size per file_name
    file_names = sorted({e["file_name"] for e in data})
    for fn in file_names:
        path = os.path.join(cfg["video_root"], fn)
        if not os.path.exists(path):
            # UCF sometimes lives without _x264 in root variants
            alt = os.path.join(cfg["video_root"], fn.replace("_x264", ""))
            path = alt if os.path.exists(alt) else path
        sizes[fn] = list(probe_size(path, cfg["default_size"]))

    out = []
    for entry in data:
        vh, vw = sizes[entry["file_name"]]
        new_boxes = []
        for box in entry["bounding_boxes"]:
            new_kfs = {}
            empty = False
            for k, corners in box["keyframes"].items():
                q = quantize_corners(
                    corners, vh, vw,
                    image_size=image_size,
                    patch_size=patch_size,
                    overlap_threshold=overlap_threshold,
                )
                if q is None:
                    empty = True
                    break
                new_kfs[k] = q
            if empty:
                skipped_boxes += 1
                continue
            new_boxes.append(
                {
                    "id": box["id"],
                    "start_frame": box["start_frame"],
                    "end_frame": box["end_frame"],
                    "keyframes": new_kfs,
                }
            )
            kept_boxes += 1
        if not new_boxes:
            # Keep entry shell only if it had no boxes (normals); else drop empty
            if not entry["bounding_boxes"]:
                out.append(entry)
            continue
        out.append(
            {
                "id": entry["id"],
                "file_name": entry["file_name"],
                "start_frame": entry["start_frame"],
                "end_frame": entry["end_frame"],
                "bounding_boxes": new_boxes,
            }
        )

    out_path = os.path.join(repo_root, quantized_output_path(cfg, overlap_threshold))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    sizes_path = os.path.join(repo_root, cfg["sizes_sidecar"])
    with open(sizes_path, "w") as f:
        json.dump(sizes, f)
    print(
        f"Wrote {out_path} ({len(out)} entries, {kept_boxes} boxes kept, "
        f"{skipped_boxes} boxes skipped; sizes → {sizes_path})"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate patch-quantized bbox JSON annotations")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["ucf", "msad", "xdviolence"],
        choices=list(DATASET_CFG.keys()) + ["all"],
    )
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        nargs="+",
        default=[0.25],
        help="One or more patch–bbox overlap thresholds (writes *_<thr>_quantized.json)",
    )
    args = parser.parse_args()
    repo_root = args.repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets = list(DATASET_CFG.keys()) if "all" in args.dataset else args.dataset
    for ds in datasets:
        for thr in args.overlap_threshold:
            build_quantized(ds, repo_root, overlap_threshold=thr)


if __name__ == "__main__":
    main()
