"""
Re-derive every spatial ground truth for letterbox-cropped UCF-Crime videos.

Each native (320x240) annotation box is moved into the crop (shift by the crop
origin, clip to the crop) and then goes through exactly the rules that built
the original files, with the crop size in place of the frame size:

  frame_label/<variant>/ucf_gt_patches_0.25.npy     per-frame 8x8 patch GT (PAUC/PAP/TIoU)
      rule of UR-DMU generate_patch_gt.py: int-truncated 128x128 corners, +-5 px
      tolerance, >=25% patch overlap, and a later box overwrites earlier ones
      inside its frame range
  annotations/<variant>/merged_ucf_crime_0.25_quantized.json + ucf_video_sizes.json
      rule of eval/generate_quantized_annotations.py
  frame_label/<variant>/Liu_Ma_Test_{gt,subset_frames}_0.25.pkl
      rule of UR-DMU create_gt_Liu_Ma.py

    python tools/noletterbox_gt.py --validate          # full-frame crops must reproduce the originals
    python tools/noletterbox_gt.py --crops annotations/noletterbox/ucf_crops.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.generate_quantized_annotations import corners_from_xyxy, corners_to_xyxy, quantize_corners


UR_DMU = "/path/to/UR-DMU"
NATIVE_W, NATIVE_H = 320, 240
GRID, PATCH, IMAGE = 8, 16, 128


def crop_box(box, crop):
    """Native xyxy -> xyxy inside the crop (crop-local pixels), or None if it falls outside."""
    cx1, cy1, cx2, cy2 = crop
    x1 = min(max(box[0] - cx1, 0), cx2 - cx1)
    y1 = min(max(box[1] - cy1, 0), cy2 - cy1)
    x2 = min(max(box[2] - cx1, 0), cx2 - cx1)
    y2 = min(max(box[3] - cy1, 0), cy2 - cy1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def crop_of(crops, file_name):
    entry = crops.get(file_name)
    return entry["crop"] if entry else [0, 0, NATIVE_W, NATIVE_H]


# --------------------------------------------------------------------------- patch GT

def patch_gt_for_video(entries, n_frames, crop, stats):
    labels = np.zeros((n_frames, GRID * GRID), dtype=np.int64)
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    rw, rh = IMAGE / cw, IMAGE / ch
    px = np.arange(GRID * GRID) % GRID * PATCH
    py = np.arange(GRID * GRID) // GRID * PATCH
    for entry in entries:
        for box in entry["bounding_boxes"]:
            st, ed = box["start_frame"], box["end_frame"]
            native = [box["keyframes"][str(st)][0][0], box["keyframes"][str(st)][0][1],
                      box["keyframes"][str(ed)][2][0], box["keyframes"][str(ed)][2][1]]
            local = crop_box(native, crop)
            if local is None:
                stats["boxes_outside_crop"] += 1
                continue
            x1, y1 = int(local[0] * rw), int(local[1] * rh)
            x2, y2 = int(local[2] * rw), int(local[3] * rh)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                stats["boxes_collapsed"] += 1
                continue
            ox = np.maximum(0, np.minimum(px + PATCH, x2 + 5) - np.maximum(px, x1 - 5))
            oy = np.maximum(0, np.minimum(py + PATCH, y2 + 5) - np.maximum(py, y1 - 5))
            mask = ((ox * oy) / (PATCH * PATCH) >= 0.25).astype(np.int64)
            lo, hi = max(st, 0), min(ed, n_frames - 1)
            if lo <= hi:
                labels[lo:hi + 1] = mask   # later boxes overwrite, as in generate_patch_gt.py
    return labels.reshape(-1)


def build_patch_gt(crops, video_names, frames_count, annotations):
    by_file = {}
    for entry in annotations:
        by_file.setdefault(entry["file_name"], []).append(entry)
    stats = {"boxes_outside_crop": 0, "boxes_collapsed": 0}
    parts = []
    for name, n_frames in zip(video_names, frames_count):
        file_name = f"{name}_x264.mp4"
        entries = [] if "Normal" in name else by_file.get(file_name, [])
        parts.append(patch_gt_for_video(entries, int(n_frames), crop_of(crops, file_name), stats))
    return np.concatenate(parts), stats


# --------------------------------------------------------------------------- quantized json

def build_quantized(crops, annotations):
    out, sizes = [], {}
    kept = skipped = 0
    for entry in annotations:
        crop = crop_of(crops, entry["file_name"])
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        sizes[entry["file_name"]] = [ch, cw]
        new_boxes = []
        for box in entry["bounding_boxes"]:
            new_kfs = {}
            for k, corners in box["keyframes"].items():
                local = crop_box(corners_to_xyxy(corners), crop)
                q = None if local is None else quantize_corners(corners_from_xyxy(local), ch, cw)
                if q is None:
                    new_kfs = None
                    break
                new_kfs[k] = q
            if new_kfs is None:
                skipped += 1
                continue
            new_boxes.append({"id": box["id"], "start_frame": box["start_frame"],
                              "end_frame": box["end_frame"], "keyframes": new_kfs})
            kept += 1
        if not new_boxes:
            if not entry["bounding_boxes"]:
                out.append(entry)
            continue
        out.append({"id": entry["id"], "file_name": entry["file_name"],
                    "start_frame": entry["start_frame"], "end_frame": entry["end_frame"],
                    "bounding_boxes": new_boxes})
    return out, sizes, {"boxes_kept": kept, "boxes_skipped": skipped}


# --------------------------------------------------------------------------- Liu & Ma

def overlaps_with_tolerance(patch_box, gt_box, min_frac=0.25):
    ix1, iy1 = max(patch_box[0], gt_box[0]), max(patch_box[1], gt_box[1])
    ix2, iy2 = min(patch_box[2], gt_box[2]), min(patch_box[3], gt_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    return (ix2 - ix1) * (iy2 - iy1) / (PATCH * PATCH) >= min_frac


def build_liu_ma(crops, test_annotations):
    gt, subset = {}, {}
    outside = 0
    for video, annos in test_annotations.items():
        crop = crop_of(crops, f"{video}.mp4")
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        subset[video], labels = [], []
        for anno in annos:
            subset[video].append(anno[0].split("/")[-1].split("_")[1].split(".")[0])
            local = crop_box(list(anno[1:]), crop)
            if local is None:
                outside += 1
                labels.extend([0] * (GRID * GRID))
                continue
            sx, sy = IMAGE / float(cw), IMAGE / float(ch)
            box = (min(max(local[0] * sx, 0), IMAGE), min(max(local[1] * sy, 0), IMAGE),
                   min(max(local[2] * sx, 0), IMAGE), min(max(local[3] * sy, 0), IMAGE))
            for j in range(GRID * GRID):
                r, c = j // GRID, j % GRID
                labels.append(int(overlaps_with_tolerance(
                    (c * PATCH, r * PATCH, c * PATCH + PATCH, r * PATCH + PATCH), box)))
        gt[video] = np.array(labels)
    return gt, subset, {"boxes_outside_crop": outside}


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Spatial GT for letterbox-cropped UCF-Crime")
    p.add_argument("--crops", default="annotations/noletterbox/ucf_crops.json")
    p.add_argument("--variant", default="noletterbox")
    p.add_argument("--validate", action="store_true",
                   help="Use full-frame crops and compare against the existing GT files")
    p.add_argument("--metadata", default="outputs/patch_eval_40_abl_rgb_all/metadata.json",
                   help="Any patch-eval metadata.json: supplies the test video order and frame counts")
    args = p.parse_args()

    crops = {} if args.validate else json.load(open(args.crops))
    meta = json.load(open(args.metadata))
    annotations = json.load(open("annotations/merged_ucf_crime_original.json"))
    with open(os.path.join(UR_DMU, "frame_label", "Test_annotation.pkl"), "rb") as f:
        test_annotations = pickle.load(f)

    patch_gt, patch_stats = build_patch_gt(crops, meta["video_names"], meta["frames_count_per_video"], annotations)
    quantized, sizes, q_stats = build_quantized(crops, annotations)
    liu_gt, liu_subset, liu_stats = build_liu_ma(crops, test_annotations)
    print("patch GT:", patch_stats, "| quantized:", q_stats, "| Liu & Ma:", liu_stats)

    if args.validate:
        ref_patch = np.load("frame_label/ucf_gt_patches_0.25.npy")
        ref_q = json.load(open("annotations/merged_ucf_crime_0.25_quantized.json"))
        ref_liu = np.load(os.path.join(UR_DMU, "frame_label", "Liu_Ma_Test_gt_0.25.pkl"), allow_pickle=True)
        ref_sub = np.load(os.path.join(UR_DMU, "frame_label", "Liu_Ma_Test_subset_frames_0.25.pkl"), allow_pickle=True)
        diff = int((ref_patch != patch_gt).sum()) if len(ref_patch) == len(patch_gt) else None
        print(f"patch GT identical: {diff == 0} (differing entries: {diff}, length {len(patch_gt)} vs {len(ref_patch)})")
        print(f"quantized json identical: {quantized == ref_q}")
        liu_same = ref_liu.keys() == liu_gt.keys() and all(np.array_equal(ref_liu[k], liu_gt[k]) for k in ref_liu)
        print(f"Liu & Ma GT identical: {liu_same}; subset frames identical: {dict(ref_sub) == liu_subset}")
        return

    label_dir = os.path.join("frame_label", args.variant)
    ann_dir = os.path.join("annotations", args.variant)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    np.save(os.path.join(label_dir, "ucf_gt_patches_0.25.npy"), patch_gt)
    with open(os.path.join(ann_dir, "merged_ucf_crime_0.25_quantized.json"), "w") as f:
        json.dump(quantized, f)
    with open(os.path.join(ann_dir, "ucf_video_sizes.json"), "w") as f:
        json.dump(sizes, f)
    with open(os.path.join(label_dir, "Liu_Ma_Test_gt_0.25.pkl"), "wb") as f:
        pickle.dump(liu_gt, f)
    with open(os.path.join(label_dir, "Liu_Ma_Test_subset_frames_0.25.pkl"), "wb") as f:
        pickle.dump(liu_subset, f)
    print(f"Wrote {label_dir}/ and {ann_dir}/")


if __name__ == "__main__":
    main()
