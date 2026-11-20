"""
Patch-score heatmaps overlaid on test videos, written straight to mp4.

Renders from cached predictions (no model inference), so several experiments can
be shown side by side on the same frames. The default is the reference model:
41_nlb_rgb_base (best checkpoint) on the letterbox-free test videos.

    python eval/visualize_heatmap.py --per_class 3 --pred_topk 10
    python eval/visualize_heatmap.py --all_anomalous --num_normal 5 --pred_topk 10
    python eval/visualize_heatmap.py --data_variant "" \
        --exps 37_ucf_random_agg 38_ucf_spatial_from_tokens --output_dir vis/heatmaps

Each experiment resolves to outputs/patch_eval_<exp>/{patch_preds.npy,metadata.json}
and predictions/<exp>_frame_scores.json, as written by eval_liu_ma.sh.

Every panel shows the smoothed 8x8 patch-score map, the quantized GT box (green),
optionally the top-k prediction box that TIoU_sub scores (red, --pred_topk), and
the temporal score; a strip under the panels plots the temporal score over
the whole video with a cursor. The spatial head's scores sit in a narrow band
near 1, so maps are min-max normalized per frame by default (--norm) and the raw
score range is printed on the panel.
"""

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.patch_metrics import (
    add_bool_arg,
    build_gt_bboxes_from_annotations,
    load_bbox_annotations,
    load_video_sizes,
)


IMAGE_SIZE = 128
GRID = 8
PATCHES_PER_FRAME = GRID * GRID
STRIP_H = 48
HEADER_H = 22


def parse_args():
    p = argparse.ArgumentParser(description="Overlay cached patch-score heatmaps on test videos")
    p.add_argument("--exps", type=str, nargs="+", default=["41_nlb_rgb_base"],
                   help="Experiment names with cached outputs/patch_eval_<exp>/patch_preds.npy "
                        "(default: the reference model, best checkpoint)")
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="Panel titles (default: experiment names)")
    p.add_argument("--data_variant", type=str, default="noletterbox",
                   help="Picks the default video folder and GT files: 'noletterbox' or '' (original)")
    p.add_argument("--video_root", type=str, default=None,
                   help="Default: the 128x128 video folder of --data_variant")
    p.add_argument("--output_dir", type=str, default="vis/heatmaps")
    p.add_argument("--videos", type=str, nargs="*", default=None,
                   help="Video names (e.g. Arson022). Default: first anomalous video of each class")
    p.add_argument("--per_class", type=int, default=1,
                   help="Videos per anomaly class when --videos is not given")
    add_bool_arg(p, "include_normal", False, help="Also render one normal video")
    p.add_argument("--all_anomalous", action="store_true",
                   help="Render every anomalous test video instead of --per_class picks")
    p.add_argument("--num_normal", type=int, default=0,
                   help="Number of normal test videos to add (overrides --include_normal)")
    p.add_argument("--bbox_annotations", type=str, default=None,
                   help="GT boxes to draw (default: quantized boxes of --data_variant); pass '' to skip")
    p.add_argument("--video_sizes", type=str, default=None,
                   help="{file_name: [H, W]} of the annotation coordinate frame (default: from --data_variant)")
    p.add_argument("--norm", type=str, default="frame", choices=["frame", "video", "none"],
                   help="Heatmap scaling: per-frame min-max, per-video min-max, or raw [0,1]")
    add_bool_arg(p, "gate", True,
                 help="Hide the heatmap on frames whose temporal score is below the "
                      "experiment's Youden threshold (--no-gate shows it everywhere)")
    p.add_argument("--display_size", type=int, default=320, help="Panel side in pixels")
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--stride", type=int, default=1, help="Render every n-th frame")
    p.add_argument("--start", type=int, default=0, help="First frame to render")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--pred_topk", type=int, default=0,
                   help="Draw the bbox of the top-k patches in red, i.e. the box TIoU_sub scores "
                        "(0 = off; only on frames above the temporal threshold)")
    args = p.parse_args()

    variant = args.data_variant
    if args.video_root is None:
        folder = f"all_videos_{variant}_resized128x128" if variant else "all_videos_resized128x128"
        root = os.environ.get("UCF_VIDEO_ROOT")
        if not root:
            p.error("--video_root is required (or set UCF_VIDEO_ROOT): the parent of "
                    f"{folder}/, holding the 128x128 test videos")
        args.video_root = os.path.join(root, folder)
    if args.bbox_annotations is None:
        args.bbox_annotations = os.path.join("annotations", variant, "merged_ucf_crime_0.25_quantized.json")
    if args.video_sizes is None and variant:
        args.video_sizes = os.path.join("annotations", variant, "ucf_video_sizes.json")
    return args


class Run:
    """Cached patch and temporal scores of one experiment, indexed by video name."""

    def __init__(self, exp, label):
        self.label = label
        eval_dir = os.path.join("outputs", f"patch_eval_{exp}")
        meta = json.load(open(os.path.join(eval_dir, "metadata.json")))
        frame_meta = json.load(open(os.path.join("predictions", f"{exp}_frame_scores.json")))

        self.patch_preds = np.load(os.path.join(eval_dir, "patch_preds.npy"), mmap_mode="r")
        self.threshold = float(frame_meta["optimal_threshold"])
        self.frame_scores = {
            n: np.asarray(s, dtype=np.float32)
            for n, s in zip(frame_meta["video_names"], frame_meta["frame_scores"])
        }
        self.offsets = {}
        cursor = 0
        for name, count in zip(meta["video_names"], meta["frames_count_per_video"]):
            self.offsets[name] = (cursor, int(count))
            cursor += int(count) * PATCHES_PER_FRAME

    def video(self, name):
        start, n_frames = self.offsets[name]
        patches = np.asarray(
            self.patch_preds[start:start + n_frames * PATCHES_PER_FRAME], dtype=np.float32
        ).reshape(n_frames, GRID, GRID)
        return patches, self.frame_scores[name][:n_frames]


def default_videos(names, per_class, num_normal, all_anomalous=False):
    by_class = {}
    for name in names:
        if "Normal" in name:
            continue
        by_class.setdefault(re.sub(r"\d+$", "", name), []).append(name)
    limit = None if all_anomalous else per_class
    picked = [n for cls in sorted(by_class) for n in sorted(by_class[cls])[:limit]]
    picked += sorted(n for n in names if "Normal" in n)[:num_normal]
    return picked


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frames.append(frame)
    cap.release()
    return frames


def heat_image(grid, size, lo, hi):
    """8x8 scores -> smooth [0,1] map at display resolution."""
    heat = (grid - lo) / max(hi - lo, 1e-8)
    heat = np.clip(heat, 0.0, 1.0).astype(np.float32)
    heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=size / GRID / 3.0)
    return np.clip(heat, 0.0, 1.0)


def overlay(frame, heat, alpha):
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (alpha * heat)[..., None]
    return (frame.astype(np.float32) * (1 - a) + color.astype(np.float32) * a).astype(np.uint8)


def put_text(img, text, org, scale=0.42, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def temporal_strip(scores_per_run, thresholds, width, n_frames):
    """Static background of the temporal-score plot; the cursor is drawn per frame."""
    strip = np.full((STRIP_H, width, 3), 30, np.uint8)
    colors = [(80, 200, 255), (255, 160, 80), (140, 255, 140), (220, 120, 255)]
    xs = np.linspace(0, width - 1, n_frames).astype(np.int32)
    for i, (scores, thr) in enumerate(zip(scores_per_run, thresholds)):
        color = colors[i % len(colors)]
        ys = (STRIP_H - 4 - np.clip(scores, 0, 1) * (STRIP_H - 8)).astype(np.int32)
        cv2.polylines(strip, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        thr_y = int(STRIP_H - 4 - np.clip(thr, 0, 1) * (STRIP_H - 8))
        cv2.line(strip, (0, thr_y), (width - 1, thr_y), tuple(c // 2 for c in color), 1)
    return strip


class Mp4Writer:
    """H.264 via imageio-ffmpeg when available (browser-playable), else OpenCV mp4v."""

    def __init__(self, path, width, height, fps):
        self._gen = None
        try:
            import imageio_ffmpeg
            self._gen = imageio_ffmpeg.write_frames(
                path, (width, height), fps=fps, codec="libx264",
                pix_fmt_in="rgb24", output_params=["-pix_fmt", "yuv420p", "-crf", "20"],
                macro_block_size=1,
            )
            self._gen.send(None)
        except ImportError:
            self._cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, bgr):
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(bgr[..., ::-1]))
        else:
            self._cv.write(bgr)

    def close(self):
        if self._gen is not None:
            self._gen.close()
        else:
            self._cv.release()


def render_video(name, runs, args, gt_boxes):
    path = os.path.join(args.video_root, f"{name}_x264.mp4")
    frames = read_frames(path)
    if not frames:
        print(f"  skip {name}: no frames read from {path}")
        return None

    per_run = [run.video(name) for run in runs]
    n_frames = min(len(frames), *(len(p) for p, _ in per_run))
    stop = n_frames if args.max_frames is None else min(n_frames, args.start + args.max_frames)
    indices = range(args.start, stop, args.stride)
    if len(indices) == 0:
        print(f"  skip {name}: no frames in range")
        return None

    size = args.display_size
    width = size * len(runs)
    height = HEADER_H + size + STRIP_H
    strip_bg = temporal_strip([t[:n_frames] for _, t in per_run],
                              [r.threshold for r in runs], width, n_frames)
    scale = size / IMAGE_SIZE

    out_path = os.path.join(args.output_dir, f"{name}.mp4")
    writer = Mp4Writer(out_path, width, height, args.fps / args.stride)
    try:
        for f in indices:
            canvas = np.zeros((height, width, 3), np.uint8)
            base = cv2.resize(frames[f], (size, size), interpolation=cv2.INTER_LINEAR)
            box = gt_boxes[f] if gt_boxes is not None and f < len(gt_boxes) else None

            for i, (run, (patches, temporal)) in enumerate(zip(runs, per_run)):
                grid = patches[f]
                if args.norm == "frame":
                    lo, hi = float(grid.min()), float(grid.max())
                elif args.norm == "video":
                    lo, hi = float(patches[:n_frames].min()), float(patches[:n_frames].max())
                else:
                    lo, hi = 0.0, 1.0

                above = temporal[f] >= run.threshold
                panel = base.copy()
                if above or not args.gate:
                    panel = overlay(panel, heat_image(grid, size, lo, hi), args.alpha)
                if args.pred_topk > 0 and above:
                    active = np.zeros(PATCHES_PER_FRAME, np.uint8)
                    active[np.argsort(grid.ravel())[-args.pred_topk:]] = 1
                    ys, xs = np.nonzero(active.reshape(GRID, GRID))
                    cell = size / GRID
                    cv2.rectangle(panel, (int(xs.min() * cell), int(ys.min() * cell)),
                                  (int((xs.max() + 1) * cell) - 1, int((ys.max() + 1) * cell) - 1),
                                  (0, 0, 255), 2)
                if box is not None:
                    x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
                    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)

                put_text(panel, f"t={temporal[f]:.2e}" + ("" if above else " (below thr)"),
                         (6, size - 24), color=(255, 255, 255) if above else (160, 160, 160))
                put_text(panel, f"patch [{grid.min():.4f}, {grid.max():.4f}]", (6, size - 8))
                canvas[HEADER_H:HEADER_H + size, i * size:(i + 1) * size] = panel

                put_text(canvas, run.label, (i * size + 6, HEADER_H - 7), scale=0.45)
                if i > 0:
                    cv2.line(canvas, (i * size, 0), (i * size, HEADER_H + size), (90, 90, 90), 1)

            strip = strip_bg.copy()
            cx = int(f / max(n_frames - 1, 1) * (width - 1))
            cv2.line(strip, (cx, 0), (cx, STRIP_H - 1), (255, 255, 255), 1)
            put_text(strip, f"{name}  frame {f}/{n_frames}", (6, 14), scale=0.38,
                     color=(200, 200, 200))
            canvas[HEADER_H + size:] = strip
            writer.write(canvas)
    finally:
        writer.close()
    return out_path, len(indices)


def main():
    args = parse_args()
    labels = args.labels or args.exps
    if len(labels) != len(args.exps):
        raise ValueError("--labels must match --exps one-to-one")

    runs = [Run(exp, label) for exp, label in zip(args.exps, labels)]
    common = [n for n in runs[0].offsets if all(n in r.offsets for r in runs[1:])]
    num_normal = args.num_normal or (1 if args.include_normal else 0)
    videos = args.videos or default_videos(common, args.per_class, num_normal, args.all_anomalous)
    missing = [v for v in videos if v not in common]
    if missing:
        raise ValueError(f"Not in cached predictions: {missing}")

    ann_index = load_bbox_annotations(args.bbox_annotations) if args.bbox_annotations else None
    video_sizes = load_video_sizes(args.video_sizes)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Rendering {len(videos)} videos x {len(runs)} runs -> {args.output_dir}")

    for name in videos:
        gt_boxes = None
        if ann_index is not None:
            _, count = runs[0].offsets[name]
            gt_boxes = build_gt_bboxes_from_annotations(ann_index, [name], [count], video_sizes=video_sizes,
                                                        image_size=IMAGE_SIZE)
        result = render_video(name, runs, args, gt_boxes)
        if result:
            print(f"  wrote {result[0]} ({result[1]} frames)")


if __name__ == "__main__":
    main()
