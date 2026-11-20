"""
Encode saved visualization frames into one .mp4 per clip.

Supports both layouts from eval/visualize_topk.py:

  vis/clip_name/clip_name_000123.jpg     (default per-video folders)
  vis/clip_name_000123.jpg               (--flat)

    python eval/frames_to_video.py --frames_dir ./vis/topk
    python eval/frames_to_video.py --frames_dir ./vis/topk --fps 25 --output_dir ./vis/topk/videos
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile

import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
FRAME_RE = re.compile(r"^(?P<stem>.+)_(?P<idx>\d+)\.(?P<ext>jpg|jpeg|png|bmp)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert saved frames to mp4 videos")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Root folder written by visualize_topk.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write mp4s (default: <frames_dir>/videos)")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--clip", type=str, default=None,
                        help="Only encode this clip name (folder or flat prefix)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_image(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS


def parse_frame_name(filename):
    match = FRAME_RE.match(filename)
    if not match:
        return None, None
    return match.group("stem"), int(match.group("idx"))


def collect_from_subdir(folder):
    frames = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not is_image(path):
            continue
        _stem, idx = parse_frame_name(name)
        if idx is None:
            idx = 10**12
        frames.append((idx, path))
    frames.sort(key=lambda x: (x[0], x[1]))
    return [p for _, p in frames]


def collect_flat_groups(root):
    groups = {}
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not is_image(path):
            continue
        stem, idx = parse_frame_name(name)
        if stem is None:
            continue
        groups.setdefault(stem, []).append((idx, path))
    for stem in groups:
        groups[stem].sort(key=lambda x: (x[0], x[1]))
        groups[stem] = [p for _, p in groups[stem]]
    return groups


def discover_clips(frames_dir, clip_filter=None):
    """Return {clip_name: [frame_paths_sorted]}."""
    clips = {}
    subdirs = [
        os.path.join(frames_dir, name)
        for name in sorted(os.listdir(frames_dir))
        if os.path.isdir(os.path.join(frames_dir, name)) and name != "videos"
    ]
    for folder in subdirs:
        name = os.path.basename(folder)
        frames = collect_from_subdir(folder)
        if frames:
            clips[name] = frames

    flat = collect_flat_groups(frames_dir)
    for name, frames in flat.items():
        if name not in clips:
            clips[name] = frames

    if clip_filter:
        if clip_filter not in clips:
            raise FileNotFoundError(
                f"Clip '{clip_filter}' not found under {frames_dir}. "
                f"Available: {list(clips)[:8]}"
            )
        clips = {clip_filter: clips[clip_filter]}
    if not clips:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    return clips


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def encode_ffmpeg(frame_paths, out_path, fps):
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Could not read {frame_paths[0]}")
    height, width = first.shape[:2]

    list_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            list_path = handle.name
            for path in frame_paths:
                escaped = os.path.abspath(path).replace("'", r"'\''")
                handle.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-r", str(fps),
            "-i", list_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale=trunc({width}/2)*2:trunc({height}/2)*2",
            "-movflags", "+faststart",
            out_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
    finally:
        if list_path and os.path.isfile(list_path):
            os.remove(list_path)


def encode_opencv(frame_paths, out_path, fps):
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Could not read {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open writer for {out_path}")
    try:
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is None:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


def encode_clip(frame_paths, out_path, fps, overwrite):
    if os.path.isfile(out_path) and not overwrite:
        print(f"skip existing: {out_path}")
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if ffmpeg_available():
        encode_ffmpeg(frame_paths, out_path, fps)
    else:
        encode_opencv(frame_paths, out_path, fps)


def main():
    args = parse_args()
    frames_dir = os.path.abspath(args.frames_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(frames_dir, "videos"))
    clips = discover_clips(frames_dir, args.clip)

    backend = "ffmpeg/libx264" if ffmpeg_available() else "opencv/mp4v"
    print(f"Found {len(clips)} clip(s) in {frames_dir}")
    print(f"Encoder: {backend}  fps={args.fps}  output={output_dir}")

    for name, frames in clips.items():
        out_path = os.path.join(output_dir, f"{name}.mp4")
        print(f"{name}: {len(frames)} frames -> {out_path}")
        encode_clip(frames, out_path, args.fps, args.overwrite)

    print(f"Done. Videos saved under {output_dir}")


if __name__ == "__main__":
    main()
