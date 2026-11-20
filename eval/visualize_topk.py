"""
Overlay patch-level anomaly heatmaps on video frames.

    python eval/visualize_topk.py \
        --model_path ./models/exp_x \
        --list list/UCF_Test.list \
        --video_root /path/to/videos \
        --feature_root /path/to/features \
        --output_dir ./vis/topk \
        --cross_attention --motion_loss --snippet_agg random \
        --resize 224 224 --patch_size 16 --num_tubelet 8 --segment_length 16 \
        --token_ratio 1.0 1.0 1.0 --mode topk --k 10

Frames land in <output_dir>/<clip_name>/; eval/frames_to_video.py encodes them to mp4.
--mode bottomk highlights the lowest-scoring patches instead.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DTFEModel, WSAD_temporal, WSVAD_spatial
from utils.utils import SNIPPET_AGG_CHOICES, aggregate_snippets, set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov")
FEATURE_EXTS = (".npy",)

# Feature-dir -> video-dir replacements, tried in order when --video_root is not
# given and the list holds feature paths. Extend for your own layout.
FEATURE_TO_VIDEO = (
    ("all_videomaev2/", "all_videos/anomaly/"),
    ("all_videomaev2/", "all_videos/all/"),
    ("videos_videomaev2/", "Videos/Videos/all_videos/"),
)

SST_CKPT = {
    "temporal": "urdmu_model_{seed}.pkl",
    "spatial": "spatial_urdmu_model_{seed}.pkl",
    "dtfe": "dtfe_model_{seed}.pkl",
}
LEGACY_CKPT = {
    "temporal": "ucf_trans_{seed}.pkl",
    "spatial": "stpvad_model_{seed}.pkl",
    "dtfe": "stp_model_{seed}.pkl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SST-WSVADL top-k / bottom-k patch heatmaps"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Checkpoint directory (or a single .pkl if using --*_ckpt)")
    parser.add_argument("--list", type=str, required=True,
                        help="Text file: one video or feature path per line")
    parser.add_argument("--output_dir", type=str, default="./vis")
    parser.add_argument("--video_root", type=str, default="",
                        help="Directory of mp4/mkv files (used when the list has names or .npy paths)")
    parser.add_argument("--feature_root", type=str, default="",
                        help="Directory of .npy features (used when the list has video names/paths)")

    parser.add_argument("--temporal_ckpt", type=str, default=None)
    parser.add_argument("--spatial_ckpt", type=str, default=None)
    parser.add_argument("--dtfe_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2022)

    parser.add_argument("--len_feature", type=int, default=1408)
    parser.add_argument("--resize", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_tubelet", type=int, default=8)
    parser.add_argument("--segment_length", type=int, default=16)
    parser.add_argument("--token_ratio", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--disable_pruning", action="store_true")
    parser.add_argument("--cross_attention", action="store_true")
    parser.add_argument("--motion_loss", action="store_true")
    parser.add_argument("--motion_loss_weight", type=float, default=0.01)
    parser.add_argument("--motion_aware_type", type=str, default="time-reversal")
    parser.add_argument("--spatial_from_tokens", action="store_true",
                        help="Match training: spatial head scores 768-d DTFE tokens")
    parser.add_argument("--num_segments", type=int, default=200)
    parser.add_argument("--snippet_agg", type=str, default="none",
                        choices=["none"] + list(SNIPPET_AGG_CHOICES),
                        help="Match training aggregation. none = full snippet sequence (test-style). "
                             "random = pick a method independently per video")
    parser.add_argument("--snippet_drop", type=int, default=8)

    parser.add_argument("--mode", type=str, default="topk", choices=["topk", "bottomk", "both"])
    parser.add_argument("--k", type=int, default=10, help="Number of patches to highlight")
    parser.add_argument("--invert_scores", action="store_true",
                        help="Invert selected scores (on by default for --mode bottomk)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cell_sigma", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.8)

    parser.add_argument("--include_normal", action="store_true",
                        help="Also visualize clips whose name looks normal (skipped by default)")
    parser.add_argument("--score_threshold", type=float, default=None,
                        help="Skip segments whose temporal score is below this")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--save_video", action="store_true",
                        help="Also write an mp4 per clip (25 fps)")
    parser.add_argument("--flat", action="store_true",
                        help="Save frames in one folder instead of per-video subdirs")
    return parser.parse_args()


def read_list(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line.split()[0])
    if not entries:
        raise ValueError(f"No entries found in {path}")
    return entries


def stem_name(path):
    name = os.path.basename(path.rstrip("/"))
    if name.endswith(".mkv.npy"):
        return name[:-8]
    root, ext = os.path.splitext(name)
    if ext.lower() in FEATURE_EXTS or ext.lower() in VIDEO_EXTS:
        return root
    return name


def is_normal_name(name):
    lower = name.lower()
    if "abnormal" in lower or "anomaly" in lower:
        return False
    return "normal" in lower


def first_existing(candidates):
    seen = set()
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p
    return None


def feature_candidates(raw, feature_root):
    stem = stem_name(raw)
    names = [f"{stem}.npy", f"{stem}.mkv.npy", f"{os.path.basename(raw)}.npy"]
    cands = []
    if feature_root:
        cands.extend(os.path.join(feature_root, n) for n in names)
    if raw.endswith(".npy"):
        cands.append(raw)
    else:
        root, ext = os.path.splitext(raw)
        if ext.lower() in VIDEO_EXTS:
            cands.append(root + ".npy")
            cands.append(raw + ".npy")
    for feat_dir, vid_dir in FEATURE_TO_VIDEO:
        if vid_dir in raw:
            cands.append(raw.replace(vid_dir, feat_dir))
            root, ext = os.path.splitext(raw.replace(vid_dir, feat_dir))
            if ext.lower() in VIDEO_EXTS:
                cands.append(root + ".npy")
    return cands


def video_candidates(raw, video_root):
    stem = stem_name(raw)
    names = [f"{stem}.mp4", f"{stem}.mkv", f"{stem}.avi"]
    basename = os.path.basename(raw)
    if os.path.splitext(basename)[1].lower() in VIDEO_EXTS:
        names.insert(0, basename)
    cands = []
    if video_root:
        cands.extend(os.path.join(video_root, n) for n in names)
    if os.path.splitext(raw)[1].lower() in VIDEO_EXTS:
        cands.append(raw)
    for feat_dir, vid_dir in FEATURE_TO_VIDEO:
        if feat_dir in raw:
            swapped = raw.replace(feat_dir, vid_dir)
            root, ext = os.path.splitext(swapped)
            if ext.lower() in FEATURE_EXTS:
                cands.extend([root + e for e in VIDEO_EXTS])
            else:
                cands.append(swapped)
                if ext.lower() not in VIDEO_EXTS:
                    cands.extend([swapped + e for e in VIDEO_EXTS])
    # Same directory, swap extension.
    root, ext = os.path.splitext(raw)
    if ext.lower() in FEATURE_EXTS:
        cands.extend([root + e for e in VIDEO_EXTS])
    return cands


def resolve_sample(raw, video_root, feature_root):
    feat = first_existing(feature_candidates(raw, feature_root))
    video = first_existing(video_candidates(raw, video_root))
    if feat is None or video is None:
        tried_f = feature_candidates(raw, feature_root)
        tried_v = video_candidates(raw, video_root)
        raise FileNotFoundError(
            f"Could not resolve list entry:\n  {raw}\n"
            f"  feature found: {feat}\n  video found: {video}\n"
            f"  feature tried: {tried_f[:6]}\n  video tried: {tried_v[:6]}"
        )
    return feat, video


def discover_seed(model_dir, seed):
    pattern = os.path.join(model_dir, "urdmu_model_*.pkl")
    hits = sorted(glob.glob(pattern))
    if hits:
        name = os.path.basename(hits[0])
        try:
            return int(name.replace("urdmu_model_", "").replace(".pkl", ""))
        except ValueError:
            pass
    return seed


def resolve_model_dir(model_path):
    """Accept a dir/file given from repo root or from eval/."""
    if os.path.isfile(model_path) or os.path.isdir(model_path):
        return os.path.abspath(model_path)
    candidates = [
        os.path.join(os.getcwd(), model_path),
        os.path.join(PROJECT_ROOT, model_path),
        os.path.join(PROJECT_ROOT, os.path.basename(model_path)),
    ]
    for path in candidates:
        if os.path.isfile(path) or os.path.isdir(path):
            return os.path.abspath(path)
    return model_path


def resolve_checkpoints(args):
    model_dir = resolve_model_dir(args.model_path)
    if os.path.isfile(model_dir):
        model_dir = os.path.dirname(model_dir)

    seed = args.seed
    if not args.temporal_ckpt:
        seed = discover_seed(model_dir, seed)

    def pick(key, explicit):
        if explicit:
            path = explicit if os.path.isabs(explicit) else os.path.join(model_dir, explicit)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing {key} checkpoint: {path}")
            return path
        for table in (SST_CKPT, LEGACY_CKPT):
            path = os.path.join(model_dir, table[key].format(seed=seed))
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"Could not find {key} checkpoint in {model_dir} "
            f"(looked for {SST_CKPT[key].format(seed=seed)} and {LEGACY_CKPT[key].format(seed=seed)})"
        )

    return {
        "temporal": pick("temporal", args.temporal_ckpt),
        "spatial": pick("spatial", args.spatial_ckpt),
        "dtfe": pick("dtfe", args.dtfe_ckpt),
    }


def infer_resize_from_dtfe(state, patch_size, num_tubelet, segment_length):
    """pos_embed is [1, (H/p)*(W/p)*(T/tubelet), 768]. Recover square HxW."""
    pos = state.get("pos_embed")
    if pos is None:
        return None
    n = int(pos.shape[1])
    n_tubes = max(1, int(segment_length) // int(num_tubelet))
    if n % n_tubes != 0:
        return None
    spatial = n // n_tubes
    side = int(round(spatial ** 0.5))
    if side * side != spatial:
        return None
    hw = side * int(patch_size)
    return [hw, hw]


def infer_spatial_from_tokens(state):
    weight = state.get("embedding.conv_1.0.weight")
    if weight is None:
        return False
    return int(weight.shape[1]) == 768


def load_models(args, ckpts, device):
    dtfe_state = torch.load(ckpts["dtfe"], map_location="cpu")
    spatial_state = torch.load(ckpts["spatial"], map_location="cpu")
    temporal_state = torch.load(ckpts["temporal"], map_location="cpu")

    inferred = infer_resize_from_dtfe(
        dtfe_state, args.patch_size, args.num_tubelet, args.segment_length
    )
    if inferred is not None and list(args.resize) != inferred:
        print(f"Checkpoint DTFE pos_embed implies resize {inferred[0]}x{inferred[1]} "
              f"(got --resize {args.resize[0]} {args.resize[1]}). Using checkpoint size.")
        args.resize = inferred

    use_tokens = args.spatial_from_tokens or infer_spatial_from_tokens(spatial_state)
    if use_tokens and not args.spatial_from_tokens:
        print("Checkpoint spatial head expects 768-d tokens; enabling --spatial_from_tokens.")
        args.spatial_from_tokens = True
    spatial_in = 768 if args.spatial_from_tokens else (
        args.patch_size * args.patch_size * args.num_tubelet * 3
    )

    temporal = WSAD_temporal(
        input_size=args.len_feature, flag="Test", a_nums=60, n_nums=60
    ).to(device)
    spatial = WSVAD_spatial(
        input_size=spatial_in,
        flag="Test", a_nums=60, n_nums=60,
    ).to(device)
    dtfe = DTFEModel(
        img_size=list(args.resize),
        patch_size=args.patch_size,
        tubelet_size=args.num_tubelet,
        all_frames=args.segment_length,
        in_chans=3, num_classes=1, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, pruning_loc=[3, 6, 9], token_ratio=list(args.token_ratio),
        distill=False, disable_pruning=args.disable_pruning,
        motion_loss=args.motion_loss, motion_loss_weight=args.motion_loss_weight,
        motion_aware_type=args.motion_aware_type,
        spatial_from_tokens=args.spatial_from_tokens,
    ).to(device)

    print("Loading checkpoints:")
    for k, p in ckpts.items():
        print(f"  {k}: {p}")

    temporal.load_state_dict(temporal_state, strict=False)
    spatial.load_state_dict(spatial_state, strict=False)
    dtfe.load_state_dict(dtfe_state, strict=False)

    temporal.eval()
    spatial.eval()
    dtfe.eval()
    temporal.flag = "Test"
    spatial.flag = "Test"
    return temporal, spatial, dtfe


def load_features(path):
    feat = np.load(path).astype(np.float32)
    if feat.ndim == 3:
        feat = feat[0]
    if feat.ndim != 2:
        raise ValueError(f"Unexpected feature shape {feat.shape} in {path}")
    return feat


def read_video_frames(path, resize):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (resize[1], resize[0])))
    cap.release()
    return frames


def overlay_soft_splat_heatmap(
    frame, height, width, patch_size, coords, scores,
    alpha=0.5, cell_sigma=0.15, blur_sigma_px=None, gamma=0.8,
):
    if coords is None or len(coords) == 0:
        return frame

    gh, gw = height // patch_size, width // patch_size
    grid = np.zeros((gh, gw), np.float32)
    weight = np.zeros((gh, gw), np.float32)
    radius = max(1, int(np.ceil(2 * cell_sigma)))

    for (x, y), score in zip(coords, scores):
        gy, gx = y // patch_size, x // patch_size
        y0, y1 = max(0, gy - radius), min(gh, gy + radius + 1)
        x0, x1 = max(0, gx - radius), min(gw, gx + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dy = (yy - gy).astype(np.float32)
        dx = (xx - gx).astype(np.float32)
        g = np.exp(-(dx * dx + dy * dy) / (2.0 * cell_sigma * cell_sigma)).astype(np.float32)
        grid[y0:y1, x0:x1] += float(score) * g
        weight[y0:y1, x0:x1] += g

    heat_grid = np.divide(grid, weight, out=np.zeros_like(grid), where=weight > 1e-8)
    heat = cv2.resize(heat_grid, (width, height), interpolation=cv2.INTER_CUBIC)
    if blur_sigma_px is None:
        blur_sigma_px = patch_size / 3.0
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=blur_sigma_px, sigmaY=blur_sigma_px)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    if gamma != 1.0:
        heat = np.power(heat, gamma)

    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    final_alpha = (alpha * heat)[..., None]
    return (
        frame.astype(np.float32) * (1.0 - final_alpha)
        + heat_color.astype(np.float32) * final_alpha
    ).astype(np.uint8)


def selected_patch_coords(preserve_index, scores, height, width, patch_size, k, mode):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    preserve = np.asarray(preserve_index).reshape(-1)
    n = min(len(scores), len(preserve))
    scores, preserve = scores[:n], preserve[:n]

    order = np.argsort(scores)
    if mode == "topk":
        idx = order[-k:]
    elif mode == "bottomk":
        idx = order[:k]
    else:
        idx = np.concatenate([order[-k:], order[:k]])

    sel_scores = scores[idx]
    sel_preserve = preserve[idx]
    patches_per_frame = (height // patch_size) * (width // patch_size)
    grid_w = width // patch_size

    coords, uniq_scores = [], []
    best = {}
    for s, orig in zip(sel_scores, sel_preserve):
        spatial = int(orig) % patches_per_frame
        if spatial not in best or s > best[spatial]:
            best[spatial] = float(s)
    for spatial, s in best.items():
        gy = spatial // grid_w
        gx = spatial % grid_w
        coords.append((gx * patch_size, gy * patch_size))
        uniq_scores.append(s)
    return coords, np.asarray(uniq_scores, dtype=np.float32)


def prepare_segment_tensor(frames, start, end, segment_length, device):
    segment = frames[start:end]
    if not segment:
        return None
    if len(segment) < segment_length:
        segment = segment + [segment[-1]] * (segment_length - len(segment))
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in segment], axis=0)
    tensor = torch.from_numpy(rgb.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(device) / 255.0
    return tensor, segment


def draw_score(frame, text):
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (6, 6), (12 + tw, 14 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 10 + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


@torch.no_grad()
def visualize_video(args, temporal, spatial, dtfe, feature_path, video_path, name, out_dir, device):
    features = load_features(feature_path)
    frames = read_video_frames(video_path, args.resize)
    if not frames:
        print(f"  skip (no frames): {video_path}")
        return

    feat_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    orig_to_bin = np.arange(features.shape[0], dtype=np.int32)
    agg_method = "none"
    if getattr(args, "snippet_agg", "none") not in (None, "none"):
        agg_feat, orig_to_bin, agg_method = aggregate_snippets(
            features, args.num_segments, args.snippet_agg, args.snippet_drop,
            return_mapping=True,
        )
        print(f"  snippet_agg={agg_method} ({features.shape[0]} snippets -> {agg_feat.shape[0]} bins)")
        feat_tensor = torch.from_numpy(agg_feat).unsqueeze(0).to(device)

    temporal_out = temporal(feat_tensor)
    segment_scores = temporal_out["frame"].detach().cpu().numpy().flatten()
    snippet_features = temporal_out.get("snippet_features")

    n_segments = features.shape[0]
    height, width = args.resize
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            os.path.join(out_dir, f"{name}.mp4"),
            fourcc, 25.0, (width, height),
        )

    invert = args.invert_scores or args.mode == "bottomk"

    for seg_idx in tqdm(range(n_segments), desc=name, leave=False):
        bin_i = int(orig_to_bin[seg_idx]) if seg_idx < len(orig_to_bin) else -1
        if bin_i >= 0 and bin_i < len(segment_scores):
            score = float(segment_scores[bin_i])
        else:
            score = 0.0
        if args.score_threshold is not None and score < args.score_threshold:
            continue

        start = seg_idx * args.segment_length
        end = min(start + args.segment_length, len(frames))
        prepared = prepare_segment_tensor(frames, start, end, args.segment_length, device)
        if prepared is None:
            continue
        inp, segment_frames = prepared

        stp_x, _, preserve_index = dtfe(inp)
        if args.cross_attention and snippet_features is not None and bin_i >= 0:
            snip = snippet_features[:, bin_i, :].unsqueeze(0)
            spatial_out = spatial(stp_x, snippet_features=snip)
        else:
            spatial_out = spatial(stp_x)

        patch_scores = spatial_out["frame"].detach().cpu().numpy().flatten()
        preserved = preserve_index[0].detach().cpu().numpy()
        coords, sel_scores = selected_patch_coords(
            preserved, patch_scores, height, width, args.patch_size, args.k, args.mode
        )
        vis_scores = 1.0 - sel_scores if invert and len(sel_scores) else sel_scores

        for t, frame in enumerate(segment_frames[: end - start]):
            overlayed = overlay_soft_splat_heatmap(
                frame, height, width, args.patch_size, coords, vis_scores,
                alpha=args.alpha, cell_sigma=args.cell_sigma,
                blur_sigma_px=args.patch_size / 3.0, gamma=args.gamma,
            )
            overlayed = draw_score(overlayed, f"pred {score:.3f}")
            frame_idx = start + t
            out_path = os.path.join(out_dir, f"{name}_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, overlayed)
            if writer is not None:
                writer.write(overlayed)

    if writer is not None:
        writer.release()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = resolve_checkpoints(args)
    temporal, spatial, dtfe = load_models(args, ckpts, device)

    entries = read_list(args.list)
    if args.max_videos is not None:
        entries = entries[: args.max_videos]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Visualizing {len(entries)} entries from {args.list}")
    print(f"Mode={args.mode} k={args.k} device={device} snippet_agg={args.snippet_agg}")

    for raw in entries:
        name = stem_name(raw)
        if not args.include_normal and is_normal_name(name):
            print(f"skip normal: {name}")
            continue
        try:
            feat_path, video_path = resolve_sample(raw, args.video_root, args.feature_root)
        except FileNotFoundError as exc:
            print(exc)
            continue

        out_dir = args.output_dir if args.flat else os.path.join(args.output_dir, name)
        print(f"{name}\n  features: {feat_path}\n  video:    {video_path}")
        visualize_video(
            args, temporal, spatial, dtfe, feat_path, video_path, name, out_dir, device
        )

    print(f"Done. Frames saved under {args.output_dir}")


if __name__ == "__main__":
    main()
