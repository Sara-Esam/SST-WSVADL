"""
Per-video letterbox detection for UCF-Crime.

Bars are not consistent across the dataset (top/bottom, left/right, different
thickness, burned-in timestamps inside them), so each video gets its own crop.
A row (or column) counts as bar only if, in most sampled frames, it is almost
entirely near-black and flat, and it is connected to the frame edge. That keeps
dark scene content (night sky, shadows) that has texture or changes over time.

    python tools/letterbox.py detect --out annotations/noletterbox/ucf_crops.json
    python tools/letterbox.py sheet  --crops annotations/noletterbox/ucf_crops.json --out vis/letterbox_sheet.jpg

Crops are [x1, y1, x2, y2] in native pixels (x2/y2 exclusive). A crop equal to the
full frame means no letterbox.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


VIDEO_ROOT = "/path/to/all_videos"


def sample_frames(path, n_samples):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, max(total - 1, 0), n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    h, w = frames[0].shape if frames else (int(cap.get(4)), int(cap.get(3)))
    cap.release()
    return np.stack(frames) if frames else None, total, h, w


def bar_lines(gray, axis, dark_level, dark_frac, flat_std):
    """
    Per sampled frame, which lines along `axis` look like bar.
    gray is (frames, H, W): axis=2 averages over x -> one value per row,
    axis=1 averages over y -> one value per column.
    """
    dark = (gray < dark_level).mean(axis=axis)          # (frames, lines)
    flat = gray.astype(np.float32).std(axis=axis) < flat_std
    return (dark >= dark_frac) & flat


def edge_run(is_bar):
    """Length of the run of True values starting at index 0."""
    run = 0
    for v in is_bar:
        if not v:
            break
        run += 1
    return run


def detect_one(name, video_root, n_samples=64, dark_level=28, dark_frac=0.97, flat_std=8.0,
               persist=0.9, max_bar_frac=0.4, margin=1):
    path = os.path.join(video_root, name)
    gray, total, h, w = sample_frames(path, n_samples)
    if gray is None:
        return name, {"error": "unreadable", "frames": total}

    result = {"frames": total, "size": [h, w]}
    extents = {}
    for axis, length, lo_key, hi_key in ((2, h, "top", "bottom"), (1, w, "left", "right")):
        per_frame = bar_lines(gray, axis, dark_level, dark_frac, flat_std)   # (F, length)
        persistent = per_frame.mean(axis=0) >= persist
        lo = edge_run(persistent)
        hi = edge_run(persistent[::-1])
        # Frame-wise extents, to flag bars that change during the video.
        lo_f = np.array([edge_run(r) for r in per_frame])
        hi_f = np.array([edge_run(r[::-1]) for r in per_frame])
        limit = int(length * max_bar_frac)
        if lo > limit or hi > limit:
            result.setdefault("flags", []).append(f"{lo_key}/{hi_key} bar over {max_bar_frac:.0%}, ignored")
            lo, hi = min(lo, 0 if lo > limit else lo), min(hi, 0 if hi > limit else hi)
        extents[lo_key], extents[hi_key] = lo, hi
        unstable = np.mean((np.abs(lo_f - lo) > 4) | (np.abs(hi_f - hi) > 4))
        if unstable > 0.1:
            result.setdefault("flags", []).append(f"{lo_key}/{hi_key} bars vary in {unstable:.0%} of samples")

    # Trim a small margin past each detected bar to drop the soft edge line.
    top = extents["top"] + (margin if extents["top"] else 0)
    bottom = h - extents["bottom"] - (margin if extents["bottom"] else 0)
    left = extents["left"] + (margin if extents["left"] else 0)
    right = w - extents["right"] - (margin if extents["right"] else 0)
    # Even dimensions for H.264: shave one more line off a side that has a bar.
    # (Native frames are 320x240, so an axis without bars is already even.)
    if (bottom - top) % 2:
        if extents["bottom"]:
            bottom -= 1
        else:
            top += 1
    if (right - left) % 2:
        if extents["right"]:
            right -= 1
        else:
            left += 1
    result["crop"] = [int(left), int(top), int(right), int(bottom)]
    result["bars"] = extents
    result["letterboxed"] = result["crop"] != [0, 0, w, h]
    return name, result


def cmd_detect(args):
    names = sorted(n for n in os.listdir(args.video_root) if n.endswith(".mp4"))
    if args.limit:
        names = names[:args.limit]
    if args.videos:
        names = [n if n.endswith(".mp4") else f"{n}_x264.mp4" for n in args.videos]
    out = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (name, res) in enumerate(pool.map(detect_one, names, [args.video_root] * len(names))):
            out[name] = res
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)
    boxed = [n for n, r in out.items() if r.get("letterboxed")]
    flagged = [n for n, r in out.items() if r.get("flags")]
    print(f"{len(boxed)}/{len(out)} videos letterboxed, {len(flagged)} flagged for review")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {args.out}")


def cmd_sheet(args):
    """Contact sheet: middle frame of each video with the crop drawn (green) or flagged (red)."""
    crops = json.load(open(args.crops))
    names = [n for n, r in sorted(crops.items())
             if (r.get("letterboxed") or r.get("flags")) and (not args.only_flagged or r.get("flags"))]
    names = names[args.offset:args.offset + args.count]
    tiles = []
    for name in names:
        r = crops[name]
        cap = cv2.VideoCapture(os.path.join(args.video_root, name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frames"] * args.position)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        x1, y1, x2, y2 = r["crop"]
        color = (0, 0, 255) if r.get("flags") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        cv2.putText(frame, name.replace("_x264.mp4", ""), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (240, 180)))
    cols = args.cols
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"Wrote {args.out} ({len(names)} videos)")


def main():
    p = argparse.ArgumentParser(description="Per-video letterbox detection")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--video_root", default=VIDEO_ROOT)
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--videos", nargs="*", default=None)
    s = sub.add_parser("sheet")
    s.add_argument("--video_root", default=VIDEO_ROOT)
    s.add_argument("--crops", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--count", type=int, default=48)
    s.add_argument("--cols", type=int, default=6)
    s.add_argument("--position", type=float, default=0.5)
    s.add_argument("--only_flagged", action="store_true")
    args = p.parse_args()
    {"detect": cmd_detect, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    main()
