"""
Write letterbox-free copies of the UCF-Crime videos from tools/letterbox.py crops.

For every video two outputs are produced, mirroring the existing folders:
  <out_native>/<name>   cropped at native resolution (input to VideoMAEv2)
  <out_128>/<name>      cropped, then bilinear-resized to 128x128 (input to DTFE)
Videos without bars are symlinked to the originals, so their pixels (and
VideoMAEv2 features) are unchanged. Every written video is checked to have the
same frame count as its source under both OpenCV and decord.

    python tools/crop_videos.py --crops annotations/noletterbox/ucf_crops.json --workers 18
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import decord
import imageio_ffmpeg
import numpy as np


ROOT = "/path/to/Videos"


def frame_counts(path):
    cap = cv2.VideoCapture(path)
    n_cv = 0
    while cap.grab():
        n_cv += 1
    cap.release()
    return n_cv, len(decord.VideoReader(path))


def open_writer(path, width, height, fps):
    gen = imageio_ffmpeg.write_frames(
        path, (width, height), fps=fps, codec="libx264", pix_fmt_in="rgb24",
        output_params=["-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium"],
        macro_block_size=1,
    )
    gen.send(None)
    return gen


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def process(name, crop, args):
    src_native = os.path.join(args.src_native, name)
    src_128 = os.path.join(args.src_128, name)
    dst_native = os.path.join(args.out_native, name)
    dst_128 = os.path.join(args.out_128, name)

    x1, y1, x2, y2 = crop
    cap = cv2.VideoCapture(src_native)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    if [x1, y1, x2, y2] == [0, 0, w, h]:
        cap.release()
        link(src_native, dst_native)
        link(src_128, dst_128)
        return name, {"status": "linked"}

    if (not args.overwrite and os.path.exists(dst_native) and not os.path.islink(dst_native)
            and os.path.exists(dst_128) and not os.path.islink(dst_128)):
        cap.release()
        return name, {"status": "exists"}

    tmp_native, tmp_128 = dst_native + ".part.mp4", dst_128 + ".part.mp4"
    native_writer = open_writer(tmp_native, x2 - x1, y2 - y1, fps)
    small_writer = open_writer(tmp_128, 128, 128, fps)
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(np.ascontiguousarray(frame[y1:y2, x1:x2]), cv2.COLOR_BGR2RGB)
        native_writer.send(rgb)
        small_writer.send(cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR))
        written += 1
    cap.release()
    native_writer.close()
    small_writer.close()

    src_counts = frame_counts(src_native)
    out_counts = (frame_counts(tmp_native), frame_counts(tmp_128))
    if out_counts[0] != src_counts or out_counts[1] != src_counts:
        return name, {"status": "count_mismatch", "source": src_counts,
                      "native": out_counts[0], "128": out_counts[1], "written": written}
    for tmp, dst in ((tmp_native, dst_native), (tmp_128, dst_128)):
        if os.path.lexists(dst):
            os.remove(dst)
        os.rename(tmp, dst)
    return name, {"status": "cropped", "frames": written}


def main():
    p = argparse.ArgumentParser(description="Crop letterbox bars from UCF-Crime videos")
    p.add_argument("--crops", required=True)
    p.add_argument("--src_native", default=os.path.join(ROOT, "all_videos"))
    p.add_argument("--src_128", default=os.path.join(ROOT, "all_videos_resized128x128"))
    p.add_argument("--out_native", default=os.path.join(ROOT, "all_videos_noletterbox"))
    p.add_argument("--out_128", default=os.path.join(ROOT, "all_videos_noletterbox_resized128x128"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report", default=None, help="Default: <crops dir>/crop_report.json")
    args = p.parse_args()

    crops = json.load(open(args.crops))
    names = sorted(crops) if not args.videos else [
        v if v.endswith(".mp4") else f"{v}_x264.mp4" for v in args.videos]
    missing = [n for n in names if "crop" not in crops[n]]
    if missing:
        raise ValueError(f"No crop for {missing[:5]} ({len(missing)} total)")
    os.makedirs(args.out_native, exist_ok=True)
    os.makedirs(args.out_128, exist_ok=True)

    report = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process, n, crops[n]["crop"], args) for n in names]
        for i, fut in enumerate(futures):
            name, res = fut.result()
            report[name] = res
            if res["status"] == "count_mismatch":
                print(f"  MISMATCH {name}: {res}", flush=True)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)

    summary = {}
    for r in report.values():
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    print("summary:", summary)
    out = args.report or os.path.join(os.path.dirname(args.crops), "crop_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
