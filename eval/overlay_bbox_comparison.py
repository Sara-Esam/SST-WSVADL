"""
Overlay GT bboxes on videos for visual comparison.

Modes:
  1) Original vs one quantized JSON (default):
       green = original, magenta = quantized file
  2) Multi-threshold quantization (recommended for paper figs):
       white = original
       magenta / cyan / orange / yellow = 0.25 / 0.50 / 0.75 / 1.0
       Boxes are quantized on the fly from the original JSON (native video coords).

Examples:
  python eval/overlay_bbox_comparison.py --dataset ucf --thresholds 0.25 0.50 0.75 1.0 \\
      --video_names Vandalism007_x264.mp4 Assault010_x264.mp4 Arson022_x264.mp4
  python eval/overlay_bbox_comparison.py --dataset ucf --num_videos 3
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def patches_to_bbox(patches, patch_size=16, image_size=128):
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


def box_to_patch_mask(x1, y1, x2, y2, patch_size=16, image_size=128, overlap_threshold=0.25, tol=5):
    side = image_size // patch_size
    mask = np.zeros(side * side, dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask
    for i in range(side * side):
        px = (i % side) * patch_size
        py = (i // side) * patch_size
        overlap_x = max(0, min(px + patch_size, x2 + tol) - max(px, x1 - tol))
        overlap_y = max(0, min(py + patch_size, y2 + tol) - max(py, y1 - tol))
        if overlap_x * overlap_y / (patch_size * patch_size) >= overlap_threshold:
            mask[i] = 1
    return mask


def quantize_corners(corners, vh, vw, image_size=128, patch_size=16, overlap_threshold=0.25):
    x1, y1, x2, y2 = corners_to_xyxy(corners)
    sx, sy = image_size / float(vw), image_size / float(vh)
    mask = box_to_patch_mask(
        x1 * sx, y1 * sy, x2 * sx, y2 * sy,
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
    q[0] = max(0, min(q[0], vw - 1))
    q[1] = max(0, min(q[1], vh - 1))
    q[2] = max(q[0] + 1, min(q[2], vw))
    q[3] = max(q[1] + 1, min(q[3], vh))
    return corners_from_xyxy(q)


try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False


DATASET_CFG = {
    "ucf": {
        "original": "annotations/merged_ucf_crime_original.json",
        "quantized": "annotations/merged_ucf_crime_0.25_quantized.json",
        "video_root": "/projects/0/prjs1250/feature_extraction/Videos/Videos/all_videos_test_only",
        "fallback_root": "/projects/0/prjs1250/feature_extraction/Videos/Videos/all_videos",
        "default_size": (240, 320),
    },
    "msad": {
        "original": "annotations/merged_MSAD_original.json",
        "quantized": "annotations/merged_MSAD_0.25_quantized.json",
        "video_root": "/projects/0/prjs1250/video_datasets/MSAD/all_videos",
        "fallback_root": None,
        "default_size": (720, 1280),
    },
    "xdviolence": {
        "original": "annotations/merged_xd-v_original.json",
        "quantized": "annotations/merged_xd-v_0.25_quantized.json",
        "video_root": "/projects/0/prjs1250/video_datasets/xd-violence/all_videos",
        "fallback_root": None,
        "default_size": (320, 640),
    },
}

# Distinct colors for multi-threshold overlays (RGB)
THRESHOLD_COLORS = {
    0.25: (255, 0, 255),     # magenta
    0.50: (0, 220, 255),     # cyan
    0.5: (0, 220, 255),
    0.75: (255, 140, 0),     # orange
    1.0: (255, 255, 0),      # yellow
    1.00: (255, 255, 0),
}
ORIGINAL_COLOR = (0, 220, 0)  # green
FALLBACK_COLORS = [
    (255, 0, 255),
    (0, 220, 255),
    (255, 140, 0),
    (255, 255, 0),
    (100, 100, 255),
    (255, 100, 100),
]


def index_annotations(path):
    data = json.load(open(path))
    by_file = defaultdict(list)
    for e in data:
        by_file[e["file_name"]].append(e)
    return by_file


def resolve_video(cfg, file_name):
    for root in (cfg["video_root"], cfg.get("fallback_root")):
        if not root:
            continue
        path = os.path.join(root, file_name)
        if os.path.exists(path):
            return path
        alt = os.path.join(root, file_name.replace("_x264", ""))
        if os.path.exists(alt):
            return alt
    return None


def probe_size(video_path, default_size):
    if HAS_CV2 and video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return h, w
    return default_size


def all_boxes_on_frame(entries, frame_idx):
    boxes = []
    for entry in entries:
        for box in entry.get("bounding_boxes", []):
            st, ed = int(box["start_frame"]), int(box["end_frame"])
            if frame_idx < st or frame_idx > ed:
                continue
            kfs = box["keyframes"]
            key = str(st) if str(st) in kfs else next(iter(kfs))
            boxes.append(corners_to_xyxy(kfs[key]))
    return boxes


def quantized_boxes_on_frame(entries, frame_idx, vh, vw, threshold):
    boxes = []
    for entry in entries:
        for box in entry.get("bounding_boxes", []):
            st, ed = int(box["start_frame"]), int(box["end_frame"])
            if frame_idx < st or frame_idx > ed:
                continue
            kfs = box["keyframes"]
            key = str(st) if str(st) in kfs else next(iter(kfs))
            q = quantize_corners(kfs[key], vh, vw, overlap_threshold=threshold)
            if q is not None:
                boxes.append(corners_to_xyxy(q))
    return boxes


def draw_boxes(img, boxes, color, width=3):
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        for t in range(width):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
    return img


def color_for_threshold(thr, idx):
    if thr in THRESHOLD_COLORS:
        return THRESHOLD_COLORS[thr]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def legend_text(thresholds):
    parts = ["green=original"]
    names = {0.25: "magenta=0.25", 0.5: "cyan=0.50", 0.50: "cyan=0.50",
             0.75: "orange=0.75", 1.0: "yellow=1.0", 1.00: "yellow=1.0"}
    for t in thresholds:
        parts.append(names.get(t, f"t={t:g}"))
    return "  ".join(parts)


def annotate_multi(frame_rgb, orig_entries, frame_idx, vh, vw, thresholds):
    img = Image.fromarray(frame_rgb)
    draw_boxes(img, all_boxes_on_frame(orig_entries, frame_idx), ORIGINAL_COLOR, width=3)
    for i, thr in enumerate(thresholds):
        boxes = quantized_boxes_on_frame(orig_entries, frame_idx, vh, vw, thr)
        draw_boxes(img, boxes, color_for_threshold(thr, i), width=2)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # legend with colored swatches
    x, y = 10, 10
    swatch = 12
    items = [("original", ORIGINAL_COLOR)] + [
        (f"{ thr:g}", color_for_threshold(thr, i)) for i, thr in enumerate(thresholds)
    ]
    for label, color in items:
        draw.rectangle([x, y, x + swatch, y + swatch], fill=color, outline=(255, 255, 255))
        draw.text((x + swatch + 4, y), label, fill=(255, 255, 255), font=font)
        x += swatch + 4 + max(8 * len(label), 40) + 10
    draw.text((10, 28), f"frame {frame_idx}", fill=(255, 255, 255), font=font)
    return np.asarray(img)


def annotate_pair(frame_rgb, orig_boxes, quan_boxes, frame_idx):
    img = Image.fromarray(frame_rgb)
    draw_boxes(img, orig_boxes, ORIGINAL_COLOR, width=3)
    draw_boxes(img, quan_boxes, (255, 0, 255), width=3)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 10), "green=original  magenta=quantized", fill=(255, 255, 255), font=font)
    draw.text((10, 28), f"frame {frame_idx}", fill=(255, 255, 255), font=font)
    return np.asarray(img)


def interesting_span(entries_list):
    interesting = set()
    for entries in entries_list:
        for e in entries:
            for b in e.get("bounding_boxes", []):
                for f in range(int(b["start_frame"]), int(b["end_frame"]) + 1):
                    interesting.add(f)
    if not interesting:
        return None
    return min(interesting), max(interesting)


def pick_videos(orig_index, video_names, num_videos):
    if video_names:
        return video_names
    scored = []
    for fn, entries in orig_index.items():
        n = sum(len(e.get("bounding_boxes", [])) for e in entries)
        if n > 0:
            scored.append((n, fn))
    scored.sort(reverse=True)
    return [fn for _, fn in scored[:num_videos]]


def iter_video_frames(video_path):
    if HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fps
            idx += 1
        cap.release()
        return

    if not HAS_IMAGEIO:
        raise RuntimeError("Need OpenCV (cv2) or imageio[ffmpeg] to read videos")
    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 25.0) or 25.0)
    except Exception:
        fps = 25.0
    for i, frame in enumerate(reader):
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        yield i, np.asarray(frame), fps
    reader.close()


def write_video(path, frames, fps):
    if HAS_CV2:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    if not HAS_IMAGEIO:
        raise RuntimeError("Need OpenCV (cv2) or imageio to write videos")
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def overlay_multi(video_path, orig_entries, out_path, thresholds, default_size,
                  max_frames=None, stride=1):
    span = interesting_span([orig_entries])
    if span is None:
        print(f"  skip (no boxes): {os.path.basename(video_path)}")
        return None
    fmin, fmax = span
    if max_frames is not None:
        fmax = min(fmax, fmin + max_frames - 1)
    vh, vw = probe_size(video_path, default_size)

    frames_out = []
    fps = 25.0
    for fidx, frame, fps in iter_video_frames(video_path):
        if fidx < fmin:
            continue
        if fidx > fmax:
            break
        if (fidx - fmin) % stride != 0:
            continue
        frames_out.append(annotate_multi(frame, orig_entries, fidx, vh, vw, thresholds))

    if not frames_out:
        print(f"  skip (no frames written): {os.path.basename(video_path)}")
        return None
    write_video(out_path, frames_out, max(fps / stride, 1.0))
    print(f"  wrote {out_path} ({len(frames_out)} frames, span {fmin}-{fmax}, size {vw}x{vh})")
    return out_path


def overlay_pair(video_path, orig_entries, quan_entries, out_path, max_frames=None, stride=1):
    span = interesting_span([orig_entries, quan_entries])
    if span is None:
        print(f"  skip (no boxes): {os.path.basename(video_path)}")
        return None
    fmin, fmax = span
    if max_frames is not None:
        fmax = min(fmax, fmin + max_frames - 1)

    frames_out = []
    fps = 25.0
    for fidx, frame, fps in iter_video_frames(video_path):
        if fidx < fmin:
            continue
        if fidx > fmax:
            break
        if (fidx - fmin) % stride != 0:
            continue
        frames_out.append(
            annotate_pair(
                frame,
                all_boxes_on_frame(orig_entries, fidx),
                all_boxes_on_frame(quan_entries, fidx),
                fidx,
            )
        )
    if not frames_out:
        return None
    write_video(out_path, frames_out, max(fps / stride, 1.0))
    print(f"  wrote {out_path} ({len(frames_out)} frames, span {fmin}-{fmax})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Overlay original vs quantized GT bboxes")
    parser.add_argument("--dataset", type=str, default="ucf", choices=list(DATASET_CFG.keys()))
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--original", type=str, default=None)
    parser.add_argument("--quantized", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="vis/bbox_compare")
    parser.add_argument("--video_names", type=str, nargs="+", default=None)
    parser.add_argument("--num_videos", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="If set, overlay original + quantized boxes at each overlap threshold "
             "(e.g. 0.25 0.50 0.75 1.0). Computed on the fly from original JSON.",
    )
    parser.add_argument("--also_side_by_side", action="store_true",
                        help="Pairwise mode only: also save original | quantized side-by-side")
    args = parser.parse_args()

    repo_root = args.repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = dict(DATASET_CFG[args.dataset])
    if args.video_root:
        cfg["video_root"] = args.video_root

    orig_path = os.path.join(repo_root, args.original or cfg["original"])
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f"Original annotations missing: {orig_path}")
    orig_index = index_annotations(orig_path)
    videos = pick_videos(orig_index, args.video_names, args.num_videos)

    if args.thresholds is not None:
        out_dir = os.path.join(repo_root, args.output_dir, args.dataset, "thresholds")
        os.makedirs(out_dir, exist_ok=True)
        thr = args.thresholds
        print(f"Original:   {orig_path}")
        print(f"Thresholds: {thr}")
        print(f"Colors:     green=original, magenta=0.25, cyan=0.50, orange=0.75, yellow=1.0")
        print(f"Videos:     {videos}")
        print(f"Out:        {out_dir}")
        for fn in videos:
            path = resolve_video(cfg, fn)
            if path is None:
                print(f"  missing video: {fn}")
                continue
            tag = "_".join(f"{t:g}" for t in thr)
            out_path = os.path.join(out_dir, fn.replace(".mp4", f"_thr{tag}.mp4"))
            overlay_multi(
                path,
                orig_index.get(fn, []),
                out_path,
                thr,
                cfg["default_size"],
                max_frames=args.max_frames,
                stride=args.stride,
            )
        return

    # Pairwise: original vs released quantized JSON
    quan_path = os.path.join(repo_root, args.quantized or cfg["quantized"])
    if not os.path.exists(quan_path):
        raise FileNotFoundError(f"Quantized annotations missing: {quan_path}")
    quan_index = index_annotations(quan_path)
    out_dir = os.path.join(repo_root, args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Original:  {orig_path}")
    print(f"Quantized: {quan_path}")
    print(f"Videos:    {videos}")
    for fn in videos:
        path = resolve_video(cfg, fn)
        if path is None:
            print(f"  missing video: {fn}")
            continue
        out_path = os.path.join(out_dir, fn.replace(".mp4", "_compare.mp4"))
        overlay_pair(
            path,
            orig_index.get(fn, []),
            quan_index.get(fn, []),
            out_path,
            max_frames=args.max_frames,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
