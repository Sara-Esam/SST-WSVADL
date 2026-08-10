"""
Generic patch-level evaluation for arbitrary spatial grids.

Consumes:
  - patch scores (flat array, frame-major, patches_per_frame contiguous)
  - frame scores JSON from export_frame_scores.py
  - patch ground-truth array aligned with the same video order

Works for any H x W patch layout (e.g. 8x8, 7x7, 14x14).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.patch_metrics import (
    DEFAULT_BBOX_ANNOTATIONS,
    DEFAULT_NATIVE_SIZE,
    DEFAULT_VIDEO_SIZES,
    build_gt_bboxes_from_annotations,
    evaluate_patch_localization,
    find_optimal_threshold_youden,
    load_bbox_annotations,
    load_video_sizes,
)


def load_frame_scores(path):
    with open(path) as f:
        data = json.load(f)
    frame_scores = np.concatenate(
        [np.asarray(s, dtype=np.float64) for s in data["frame_scores"]], axis=0
    )
    return data, frame_scores


def resolve_grid(args):
    if args.grid_h is not None and args.grid_w is not None:
        return args.grid_h, args.grid_w
    if args.image_size is not None and args.patch_size is not None:
        h, w = args.image_size
        return h // args.patch_size, w // args.patch_size
    raise ValueError("Provide --grid_h/--grid_w or --image_size with --patch_size")


def main():
    parser = argparse.ArgumentParser(description="Generic patch-level TIoU / PAUC evaluation")
    parser.add_argument("--patch_preds", type=str, required=True,
                        help="Flat .npy of patch scores (N_frames * H * W)")
    parser.add_argument("--frame_scores", type=str, required=True,
                        help="JSON from export_frame_scores.py")
    parser.add_argument("--patch_gt", type=str,
                        default="frame_label/ucf_gt_patches_0.25.npy",
                        help="Flat .npy patch ground truth (0.25 = patch–GT-bbox overlap threshold)")
    parser.add_argument("--grid_h", type=int, default=None)
    parser.add_argument("--grid_w", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--temporal_threshold", type=float, default=None,
                        help="Override Youden threshold stored in the frame-scores JSON")
    parser.add_argument("--patch_thresholds", type=float, nargs="+", default=[0.5, 0.1, 0.7])
    parser.add_argument(
        "--anomaly_frames_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, average TIoU only over frames with GT boxes. "
             "Default (False) matches the paper table.",
    )
    parser.add_argument(
        "--iou_mode",
        type=str,
        default="bbox",
        choices=["bbox", "patch"],
        help="Paper metrics use bbox IoU (default).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ucf",
        choices=["ucf", "xdviolence", "msad"],
        help="Used to resolve default bbox annotation / size paths.",
    )
    parser.add_argument(
        "--bbox_annotations",
        type=str,
        default=None,
        help="Quantized bbox JSON for bbox TIoU GT. Empty string → reconstruct from patch GT.",
    )
    parser.add_argument("--video_sizes", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    grid_h, grid_w = resolve_grid(args)
    patches_per_frame = grid_h * grid_w
    print(f"Grid: {grid_h}x{grid_w} ({patches_per_frame} patches/frame)")

    meta, frame_scores = load_frame_scores(args.frame_scores)
    patch_preds = np.load(args.patch_preds)
    patch_gt = np.load(args.patch_gt)

    video_names = meta["video_names"]
    frames_count = meta["frames_count_per_video"]
    temporal_threshold = (
        args.temporal_threshold
        if args.temporal_threshold is not None
        else float(meta["optimal_threshold"])
    )

    if args.subset is not None:
        video_names = video_names[: args.subset]
        frames_count = frames_count[: args.subset]
        n_frames = int(np.sum(frames_count))
        frame_scores = frame_scores[:n_frames]
        patch_preds = patch_preds[: n_frames * patches_per_frame]
        patch_gt = patch_gt[: n_frames * patches_per_frame]

    expected_patches = int(np.sum(frames_count)) * patches_per_frame
    if len(patch_preds) != expected_patches:
        raise ValueError(
            f"patch_preds length {len(patch_preds)} != expected {expected_patches} "
            f"(sum(frames)={int(np.sum(frames_count))} x {patches_per_frame})"
        )
    if len(patch_gt) < expected_patches:
        raise ValueError(f"patch_gt length {len(patch_gt)} < expected {expected_patches}")
    patch_gt = patch_gt[:expected_patches]

    if len(frame_scores) != int(np.sum(frames_count)):
        raise ValueError(
            f"frame_scores length {len(frame_scores)} != sum(frames_count) {int(np.sum(frames_count))}"
        )

    opt_patch = find_optimal_threshold_youden(patch_gt, patch_preds)
    print(f"Temporal threshold: {temporal_threshold}")
    print(f"Optimal patch threshold (Youden): {opt_patch:.6g}")

    thresholds = list(args.patch_thresholds)
    if opt_patch not in thresholds:
        thresholds = [opt_patch] + thresholds

    image_size = args.image_size[0] if args.image_size is not None else grid_h * 16
    patch_size = args.patch_size if args.patch_size is not None else (
        (args.image_size[0] // grid_h) if args.image_size is not None else 16
    )

    gt_bboxes = None
    bbox_path = args.bbox_annotations
    if bbox_path is None:
        bbox_path = DEFAULT_BBOX_ANNOTATIONS.get(args.dataset)
    if args.iou_mode == "bbox" and bbox_path:
        if not os.path.exists(bbox_path):
            raise FileNotFoundError(f"bbox annotations not found: {bbox_path}")
        sizes_path = args.video_sizes or DEFAULT_VIDEO_SIZES.get(args.dataset)
        ann_index = load_bbox_annotations(bbox_path)
        video_sizes = load_video_sizes(sizes_path)
        gt_bboxes = build_gt_bboxes_from_annotations(
            ann_index,
            video_names,
            frames_count,
            dataset=args.dataset,
            video_sizes=video_sizes,
            default_size=DEFAULT_NATIVE_SIZE.get(args.dataset, (240, 320)),
            image_size=image_size,
        )
        print(f"GT bboxes loaded from {bbox_path}")

    evaluate_patch_localization(
        patch_scores=patch_preds,
        frame_scores=frame_scores,
        patch_gt=patch_gt,
        frames_count_per_video=frames_count,
        video_names=video_names,
        temporal_threshold=temporal_threshold,
        patches_per_frame=patches_per_frame,
        patch_thresholds=thresholds,
        anomaly_frames_only=args.anomaly_frames_only,
        iou_mode=args.iou_mode,
        patch_size=patch_size,
        image_size=image_size,
        gt_bboxes=gt_bboxes,
    )


if __name__ == "__main__":
    main()
