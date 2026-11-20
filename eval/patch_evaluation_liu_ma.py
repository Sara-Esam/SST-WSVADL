"""
TIoU_sub: patch-level localization on the Liu & Ma UCF-Crime bbox subset.

This is the protocol behind the TIoU_sub column of the paper: instead of scoring
every frame of every anomalous test video, only the frames that carry a
Liu & Ma bounding-box annotation are scored. The frame list and the quantized
8x8 patch GT come from two pickles:

    Liu_Ma_Test_subset_frames_<n>.pkl : {video_x264: [frame numbers, 1-indexed]}
    Liu_Ma_Test_gt_<n>.pkl            : {video_x264: flat uint8, 64 per listed frame}

GT patches are laid out in the *order of the subset-frame list*, not in frame
order, so entry `i` of the list pairs with `gt[i*64:(i+1)*64]`.

Predictions are read from a cached patch_preds.npy (eval/patch_evaluation_sst.py
--from_cache writes one) plus the temporal JSON from eval/export_frame_scores.py;
nothing is re-inferred here.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.patch_metrics import (
    add_bool_arg,
    compute_bbox_iou,
    find_optimal_threshold_youden,
    patches_to_bbox,
)
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PATCHES_PER_FRAME = 64
PATCH_SIZE = 16
IMAGE_SIZE = 128

LIU_MA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "frame_label")


def load_pickle(path):
    return np.load(path, allow_pickle=True)


def gt_key_for(video_name, gt_dict):
    """Test-list names drop the _x264 suffix the annotation keys carry."""
    for candidate in (video_name, f"{video_name}_x264", video_name.replace("_x264", "")):
        if candidate in gt_dict:
            return candidate
    return None


def binarize(scores, patch_threshold=None, topk=None):
    """Active-patch mask for one frame: score threshold, or top-k ranking."""
    if topk is not None:
        mask = np.zeros(len(scores), dtype=np.uint8)
        if topk > 0:
            mask[np.argsort(scores)[-topk:]] = 1
        return mask
    return (np.asarray(scores) >= patch_threshold).astype(np.uint8)


def collect_subset(
    patch_preds,
    frame_scores,
    frames_count_per_video,
    video_names,
    subset_frames,
    gt_dict,
):
    """
    Gather per-frame (patch scores, patch GT, temporal score) for every annotated
    frame, in the order the subset pickle lists them.

    Returns (scores[N,64], gts[N,64], temporal[N], video_of_frame[N]).
    """
    out_scores, out_gt, out_temporal, out_video = [], [], [], []
    skipped = []

    patch_cursor = 0
    frame_cursor = 0
    for n_frames, name in zip(frames_count_per_video, video_names):
        n_frames = int(n_frames)
        n_patches = n_frames * PATCHES_PER_FRAME

        key = gt_key_for(name, subset_frames)
        if key is None or gt_key_for(name, gt_dict) is None:
            patch_cursor += n_patches
            frame_cursor += n_frames
            continue

        video_patches = np.asarray(
            patch_preds[patch_cursor:patch_cursor + n_patches], dtype=np.float64
        ).reshape(n_frames, PATCHES_PER_FRAME)
        video_temporal = np.asarray(
            frame_scores[frame_cursor:frame_cursor + n_frames], dtype=np.float64
        )
        video_gt = np.asarray(gt_dict[gt_key_for(name, gt_dict)], dtype=np.uint8)

        # Annotation frame numbers are 1-indexed.
        indices = [int(f) - 1 for f in subset_frames[key]]
        for i, frame_idx in enumerate(indices):
            if frame_idx < 0 or frame_idx >= n_frames:
                skipped.append((name, frame_idx))
                continue
            out_scores.append(video_patches[frame_idx])
            out_gt.append(video_gt[i * PATCHES_PER_FRAME:(i + 1) * PATCHES_PER_FRAME])
            out_temporal.append(video_temporal[frame_idx])
            out_video.append(name)

        patch_cursor += n_patches
        frame_cursor += n_frames

    if skipped:
        print(f"Warning: {len(skipped)} annotated frames fell outside the decoded "
              f"frame count (e.g. {skipped[:3]}) and were dropped")

    if not out_scores:
        raise RuntimeError("No annotated frames matched the cached predictions")

    return (
        np.stack(out_scores),
        np.stack(out_gt),
        np.asarray(out_temporal),
        np.asarray(out_video),
    )


def tiou_at(scores, gts, temporal_bin, patch_threshold=None, topk=None,
            iou_mode="bbox", anomaly_frames_only=False):
    """
    Mean temporally-gated IoU over the annotated frames.

    anomaly_frames_only=False reproduces patch_evaluation_Liu_Ma.py: frames whose
    GT is empty stay in the mean and contribute 0. True drops them instead.
    """
    ious = np.zeros(len(scores), dtype=np.float64)
    has_gt = gts.sum(axis=1) > 0

    for i in range(len(scores)):
        if not has_gt[i]:
            continue
        pred = binarize(scores[i], patch_threshold, topk)
        if iou_mode == "bbox":
            ious[i] = compute_bbox_iou(
                patches_to_bbox(pred, PATCH_SIZE, IMAGE_SIZE),
                patches_to_bbox(gts[i], PATCH_SIZE, IMAGE_SIZE),
            )
        else:
            g = gts[i].astype(np.float64)
            p = pred.astype(np.float64)
            ious[i] = np.sum(p * g) / (np.sum(p + g - p * g) + 1e-6)

    gated = ious * temporal_bin
    if anomaly_frames_only:
        gated = gated[has_gt]
    return float(np.mean(gated)), int(len(gated))


def subset_pauc_pap(scores, gts, temporal):
    """PAUC/PAP over the annotated frames only, with the temporal gate applied as a weight."""
    flat_gt = gts.reshape(-1)
    if flat_gt.sum() == 0 or flat_gt.sum() == len(flat_gt):
        return float("nan"), float("nan")
    weighted = (scores * temporal[:, None]).reshape(-1)
    fpr, tpr, _ = roc_curve(flat_gt, weighted)
    precision, recall, _ = precision_recall_curve(flat_gt, weighted)
    return float(auc(fpr, tpr)), float(auc(recall, precision))


def main():
    p = argparse.ArgumentParser(description="Liu & Ma subset patch localization (TIoU_sub)")
    p.add_argument("--patch_preds", type=str, required=True,
                   help="Cached patch_preds.npy (flat, 64 scores per frame)")
    p.add_argument("--frame_scores", type=str, required=True,
                   help="JSON from eval/export_frame_scores.py")
    p.add_argument("--metadata", type=str, default=None,
                   help="metadata.json from patch evaluation; overrides the video "
                        "order in --frame_scores when patch inference ran on a subset")
    p.add_argument("--overlap", type=str, default="0.25",
                   choices=["0.25", "0.5", "0.75", "1.0", "legacy"],
                   help="Patch-to-bbox overlap quantization of the Liu & Ma GT")
    p.add_argument("--liu_ma_dir", type=str, default=LIU_MA_DIR)
    p.add_argument("--subset_frames", type=str, default=None, help="Override GT frame-list pickle")
    p.add_argument("--gt", type=str, default=None, help="Override GT patch pickle")
    p.add_argument("--temporal_threshold", type=float, default=None,
                   help="Default: optimal_threshold from the frame-scores JSON")
    p.add_argument("--patch_thresholds", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--topk", type=int, nargs="*", default=None,
                   help="Also report TIoU keeping the top-k patches per frame")
    p.add_argument("--iou_mode", type=str, default="both", choices=["bbox", "patch", "both"])
    add_bool_arg(p, "anomaly_frames_only", False,
                 help="Drop annotated frames with empty GT instead of scoring them 0 "
                      "(default False matches patch_evaluation_Liu_Ma.py)")
    p.add_argument("--normalize", action="store_true",
                   help="Min-max normalize patch scores per frame before thresholding")
    p.add_argument("--videos_file", type=str, default=None,
                   help="Text file with one video name per line: score only these videos "
                        "(e.g. the held-out half of the supervised probe split)")
    p.add_argument("--output", type=str, default=None, help="Write results to this JSON")
    args = p.parse_args()

    with open(args.frame_scores) as f:
        meta = json.load(f)
    temporal_threshold = (
        args.temporal_threshold
        if args.temporal_threshold is not None
        else float(meta["optimal_threshold"])
    )

    name_to_scores = {
        n: np.asarray(s, dtype=np.float64)
        for n, s in zip(meta["video_names"], meta["frame_scores"])
    }

    if args.metadata:
        cached = json.load(open(args.metadata))
        video_names = list(cached["video_names"])
        frames_count = [int(c) for c in cached["frames_count_per_video"]]
    else:
        video_names = list(meta["video_names"])
        frames_count = [int(c) for c in meta["frames_count_per_video"]]

    frame_scores = np.concatenate(
        [name_to_scores[n][:c] for n, c in zip(video_names, frames_count)], axis=0
    )

    patch_preds = np.load(args.patch_preds, mmap_mode="r")
    expected = int(np.sum(frames_count)) * PATCHES_PER_FRAME
    if len(patch_preds) != expected:
        raise ValueError(
            f"patch_preds has {len(patch_preds)} entries, expected {expected} "
            f"({np.sum(frames_count)} frames x {PATCHES_PER_FRAME})"
        )

    suffix = "" if args.overlap == "legacy" else f"_{args.overlap}"
    subset_path = args.subset_frames or os.path.join(
        args.liu_ma_dir, f"Liu_Ma_Test_subset_frames{suffix}.pkl"
    )
    gt_path = args.gt or os.path.join(args.liu_ma_dir, f"Liu_Ma_Test_gt{suffix}.pkl")

    subset_frames = load_pickle(subset_path)
    gt_dict = load_pickle(gt_path)

    if args.videos_file:
        keep = {line.strip().replace("_x264", "") for line in open(args.videos_file) if line.strip()}
        subset_frames = {k: v for k, v in subset_frames.items() if k.replace("_x264", "") in keep}
        print(f"Restricted to {len(subset_frames)} videos from {args.videos_file}")

    scores, gts, temporal, videos = collect_subset(
        patch_preds, frame_scores, frames_count, video_names, subset_frames, gt_dict
    )

    if args.normalize:
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
        scores = (scores - lo) / np.maximum(hi - lo, 1e-12)

    temporal_bin = (temporal >= temporal_threshold).astype(np.float64)
    has_gt = gts.sum(axis=1) > 0

    print(f"GT frame list : {subset_path}")
    print(f"GT patches    : {gt_path}")
    print(f"Predictions   : {args.patch_preds}")
    print(f"Annotated frames: {len(scores)} across {len(np.unique(videos))} videos "
          f"({int(has_gt.sum())} with non-empty GT)")
    print(f"Temporal threshold: {temporal_threshold:.6g} "
          f"-> {temporal_bin.mean() * 100:.1f}% of frames pass the gate")
    print(f"Patch score range: [{scores.min():.4g}, {scores.max():.4g}], "
          f"mean {scores.mean():.4g}")

    opt_patch = find_optimal_threshold_youden(gts.reshape(-1), scores.reshape(-1))
    print(f"Optimal patch threshold (Youden, subset): {opt_patch:.6g}")

    pauc, pap = subset_pauc_pap(scores, gts, temporal)
    print(f"PAUC_sub: {pauc:.4f}  PAP_sub: {pap:.4f}")

    # Spatial head alone (no temporal weighting), and what trivial predictions score.
    spatial_auc = float(roc_auc_score(gts.reshape(-1), scores.reshape(-1)))
    spatial_ap = float(average_precision_score(gts.reshape(-1), scores.reshape(-1)))
    positive_rate = float(gts.mean())
    rng = np.random.default_rng(0)
    full_frame, _ = tiou_at(np.ones_like(scores), gts, temporal_bin, patch_threshold=0.5,
                            anomaly_frames_only=args.anomaly_frames_only)
    random_top10, _ = tiou_at(rng.random(scores.shape), gts, temporal_bin, topk=10,
                              anomaly_frames_only=args.anomaly_frames_only)
    print(f"Spatial-only AUC_sub: {spatial_auc:.4f} (chance 0.5), AP_sub: {spatial_ap:.4f} "
          f"(chance {positive_rate:.4f})")
    print(f"Baselines (bbox TIoU_sub): full-frame box {full_frame * 100:.2f}%, "
          f"random top-10 {random_top10 * 100:.2f}%")

    modes = ["bbox", "patch"] if args.iou_mode == "both" else [args.iou_mode]
    gating = "anomaly frames" if args.anomaly_frames_only else "legacy"
    results = {
        "patch_preds": os.path.abspath(args.patch_preds),
        "frame_scores": os.path.abspath(args.frame_scores),
        "subset_frames": subset_path,
        "gt": gt_path,
        "n_frames": int(len(scores)),
        "n_frames_with_gt": int(has_gt.sum()),
        "n_videos": int(len(np.unique(videos))),
        "videos_file": args.videos_file,
        "temporal_threshold": temporal_threshold,
        "optimal_patch_threshold": float(opt_patch),
        "pauc_sub": pauc,
        "pap_sub": pap,
        "spatial_auc_sub": spatial_auc,
        "spatial_ap_sub": spatial_ap,
        "spatial_ap_sub_chance": positive_rate,
        "baseline_full_frame_tiou": full_frame,
        "baseline_random_top10_tiou": random_top10,
        "anomaly_frames_only": bool(args.anomaly_frames_only),
        "tiou": {},
    }

    thresholds = list(args.patch_thresholds)
    if not any(abs(t - opt_patch) < 1e-12 for t in thresholds):
        thresholds = [float(opt_patch)] + thresholds

    for mode in modes:
        results["tiou"][mode] = {"threshold": {}, "topk": {}}
        print(f"\n--- TIoU_sub ({mode} IoU, {gating}) ---")
        for thr in thresholds:
            tiou, n = tiou_at(scores, gts, temporal_bin, patch_threshold=thr,
                              iou_mode=mode, anomaly_frames_only=args.anomaly_frames_only)
            tag = " (Youden)" if abs(thr - opt_patch) < 1e-12 else ""
            print(f"  thr {thr:<10.6g} : {tiou * 100:7.4f} %  [n={n}]{tag}")
            results["tiou"][mode]["threshold"][f"{thr:.6g}"] = tiou

        for k in args.topk or []:
            tiou, n = tiou_at(scores, gts, temporal_bin, topk=k,
                              iou_mode=mode, anomaly_frames_only=args.anomaly_frames_only)
            print(f"  top-{k:<7d} : {tiou * 100:7.4f} %  [n={n}]")
            results["tiou"][mode]["topk"][str(k)] = tiou

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
