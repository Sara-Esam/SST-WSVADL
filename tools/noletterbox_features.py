"""
VideoMAEv2 features for the letterbox-free UCF-Crime videos.

  prepare  link the existing feature of every video without bars (re-extraction of
           an unchanged video is bit-identical, so this is exact), and split the
           cropped videos into shards balanced by frame count
  check    after extraction: every video has a feature file with the same number of
           snippets as the original feature

    python tools/noletterbox_features.py prepare --shards 20 [--partial --round 1]
    sbatch --array=0-19 --export=ALL,ROUND=1 tools/extract_noletterbox_features.sh
    python tools/noletterbox_features.py check
"""

import argparse
import json
import os

import numpy as np


FEAT_SRC = "/path/to/videos_videomaev2"
FEAT_DST = "/path/to/videos_videomaev2_noletterbox"
VIDEOS = "/path/to/all_videos_noletterbox"
SHARDS = "/path/to/noletterbox_extraction_shards"


def cmd_prepare(args):
    crops = json.load(open(args.crops))
    if args.partial:
        # Crop step still running: only take videos whose cropped files are final
        # (crop_videos.py renames them into place after the frame-count check).
        ready = {n for n in crops if os.path.isfile(os.path.join(VIDEOS, n))
                 and not os.path.islink(os.path.join(VIDEOS, n))
                 and os.path.exists(os.path.join(VIDEOS.replace("all_videos_noletterbox", "all_videos_noletterbox_resized128x128"), n))}
    else:
        report = json.load(open(args.report))
        bad = {n: r for n, r in report.items() if r["status"] not in ("cropped", "linked", "exists")}
        if bad or len(report) != len(crops):
            raise RuntimeError(f"Crop step incomplete: {len(report)}/{len(crops)} videos, problems: {list(bad)[:5]}")
        ready = set(crops)
    assigned_path = os.path.join(SHARDS, "assigned.json")
    assigned = json.load(open(assigned_path)) if os.path.exists(assigned_path) else {}

    os.makedirs(FEAT_DST, exist_ok=True)
    todo = []
    for name, r in sorted(crops.items()):
        feat = name.replace(".mp4", ".npy")
        dst = os.path.join(FEAT_DST, feat)
        if not r["letterboxed"]:
            if not os.path.lexists(dst):
                os.symlink(os.path.join(FEAT_SRC, feat), dst)
        elif name in ready and name not in assigned and not (os.path.exists(dst) and not os.path.islink(dst)):
            todo.append((name, r["frames"]))

    # Greedy balance by frame count.
    loads = [0] * args.shards
    members = [[] for _ in range(args.shards)]
    for name, frames in sorted(todo, key=lambda x: -x[1]):
        i = int(np.argmin(loads))
        loads[i] += frames
        members[i].append(name)
    for i, names in enumerate(members):
        shard = os.path.join(SHARDS, f"round{args.round}_shard_{i:02d}")
        os.makedirs(shard, exist_ok=True)
        for name in names:
            os.symlink(os.path.join(VIDEOS, name), os.path.join(shard, name))
            assigned[name] = os.path.basename(shard)
    with open(assigned_path, "w") as f:
        json.dump(assigned, f, indent=1)
    linked = sum(not r["letterboxed"] for r in crops.values())
    print(f"linked {linked} unchanged features; {len(todo)} cropped videos to extract in {args.shards} shards "
          f"({min(loads) / 16:.0f}-{max(loads) / 16:.0f} snippets each)")


def cmd_check(args):
    crops = json.load(open(args.crops))
    missing, mismatch = [], []
    for name in sorted(crops):
        feat = name.replace(".mp4", ".npy")
        dst = os.path.join(FEAT_DST, feat)
        if not os.path.exists(dst):
            missing.append(name)
            continue
        new = np.load(dst, mmap_mode="r")
        old = np.load(os.path.join(FEAT_SRC, feat), mmap_mode="r")
        if new.shape != old.shape:
            mismatch.append((name, new.shape, old.shape))
        elif crops[name]["letterboxed"] and not np.any(np.asarray(new[-1])):
            # The extractor writes into a preallocated file; a job killed mid-video
            # leaves trailing all-zero snippets that a rerun would skip.
            mismatch.append((name, "trailing all-zero snippets", old.shape))
    print(f"{len(crops) - len(missing) - len(mismatch)}/{len(crops)} feature files present with matching shape")
    if missing:
        print(f"missing ({len(missing)}): {missing[:10]}")
    if mismatch:
        print(f"shape mismatch ({len(mismatch)}): {mismatch[:10]}")


def main():
    p = argparse.ArgumentParser(description="Letterbox-free VideoMAEv2 features")
    p.add_argument("--crops", default="annotations/noletterbox/ucf_crops.json")
    p.add_argument("--report", default="annotations/noletterbox/crop_report.json")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--shards", type=int, default=20)
    pr.add_argument("--round", type=int, default=1, help="Shard folders are named round<k>_shard_<i>")
    pr.add_argument("--partial", action="store_true", help="Crop step still running; take finished videos only")
    sub.add_parser("check")
    args = p.parse_args()
    {"prepare": cmd_prepare, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
