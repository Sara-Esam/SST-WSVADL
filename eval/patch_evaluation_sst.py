"""
SST-WSVADL patch-level evaluation (8x8 grid on 128x128 frames).

Temporal scores are loaded from export_frame_scores.py (not recomputed here).
Spatial scores come from DTFEModel + WSVAD_spatial on RGB segments.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import WSVAD_spatial, DTFEModel
from data.datasets import UCFCrime, XDViolence, MSAD
from eval.patch_metrics import (
    DEFAULT_BBOX_ANNOTATIONS,
    add_bool_arg,
    DEFAULT_NATIVE_SIZE,
    DEFAULT_VIDEO_SIZES,
    build_gt_bboxes_from_annotations,
    evaluate_patch_localization,
    find_optimal_threshold_youden,
    load_bbox_annotations,
    load_patch_gt,
    load_video_sizes,
)
from utils.utils import set_seed


# No token pruning at test time. For qualitative overlays, ranking patches and
# keeping the top-10 per frame gives the clearest visualizations reported in the paper.
TOKEN_RATIO = (1.0, 1.0, 1.0)
TOPK_VIS = 10

IMAGE_SIZE = (128, 128)
PATCH_SIZE = 16
SEGMENT_LENGTH = 16
NUM_TUBELET = 8
GRID_H = IMAGE_SIZE[0] // PATCH_SIZE
GRID_W = IMAGE_SIZE[1] // PATCH_SIZE
PATCHES_PER_FRAME = GRID_H * GRID_W


def build_dataset(dataset, len_feature, i3d, data_variant=''):
    kwargs = dict(root_dir=None, mode="Test", modal="rgb", num_segments=200,
                  len_feature=len_feature, is_normal=None, i3d=i3d)
    if dataset == "ucf":
        return UCFCrime(data_variant=data_variant, **kwargs)
    if dataset == "xdviolence":
        return XDViolence(**kwargs)
    if dataset == "msad":
        return MSAD(**kwargs)
    raise ValueError(f"Unsupported dataset: {dataset}")


def video_path_for(dataset, video_root, name):
    if dataset == "ucf":
        return os.path.join(video_root, f"{name}_x264.mp4")
    return os.path.join(video_root, f"{name}.mp4")


def read_video_frames(path, resize=IMAGE_SIZE):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (resize[1], resize[0])))
    cap.release()
    return frames


def pool_to_eval_grid(grid, src_h, src_w, dst_h=GRID_H, dst_w=GRID_W):
    """
    Area-average a (src_h x src_w) score map onto the (dst_h x dst_w) evaluation grid.

    Maps from other input sizes (e.g. 14x14 tokens at 224x224) are compared on the same 8x8
    grid and GT as every other result: each 8x8 cell takes the mean of the model cells it
    covers, weighted by overlap area. This equals resizing the map to 128x128 with area
    interpolation and averaging each 16x16 cell.
    """
    grid = np.asarray(grid, dtype=np.float64).reshape(src_h, src_w)
    if (src_h, src_w) == (dst_h, dst_w):
        return grid.reshape(-1)

    def overlap(n_src, n_dst):
        edges_s = np.linspace(0, 1, n_src + 1)
        edges_d = np.linspace(0, 1, n_dst + 1)
        w = np.zeros((n_dst, n_src))
        for i in range(n_dst):
            lo = np.maximum(edges_d[i], edges_s[:-1])
            hi = np.minimum(edges_d[i + 1], edges_s[1:])
            w[i] = np.clip(hi - lo, 0, None) / (edges_d[i + 1] - edges_d[i])
        return w

    wy, wx = overlap(src_h, dst_h), overlap(src_w, dst_w)
    return (wy @ grid @ wx.T).reshape(-1)


def scatter_patch_scores(scores, preserved_index, patches_per_frame=PATCHES_PER_FRAME):
    """Map (possibly pruned) tubelet scores back onto the spatial grid; keep max on collisions."""
    full = {}
    for i, idx in enumerate(preserved_index):
        patch_idx = int(idx) % patches_per_frame
        score = float(scores[i])
        if patch_idx not in full or score > full[patch_idx]:
            full[patch_idx] = score
    return np.array([full.get(j, 0.0) for j in range(patches_per_frame)], dtype=np.float64)


def load_frame_scores_json(path):
    with open(path) as f:
        data = json.load(f)
    flat = np.concatenate([np.asarray(s, dtype=np.float64) for s in data["frame_scores"]], axis=0)
    return data, flat


def load_snippet_features(path):
    if path is None or not os.path.exists(path):
        return None
    return dict(np.load(path))


@torch.no_grad()
def infer_segment(segment, stp_model, spatial_model, device, snippet_features=None,
                  use_cross_attention=False, input_range="0_1", model_grid=(GRID_H, GRID_W)):
    """
    8x8 patch scores (flat, 64) for one BGR snippet of SEGMENT_LENGTH frames.

    input_range="0_1" matches training (torchvision ToTensor). "0_255" reproduces
    the evaluation before 2026-09-15, which fed raw pixel values to DTFE.
    """
    spatial_model.flag = "Test"
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in segment], axis=0)
    inp = torch.from_numpy(rgb.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(device)
    if input_range == "0_1":
        inp = inp / 255.0
    elif input_range != "0_255":
        raise ValueError(f"input_range must be '0_1' or '0_255', got {input_range}")

    stp_features, _, preserved_index = stp_model(inp)
    if use_cross_attention and snippet_features is not None:
        feat = torch.from_numpy(snippet_features).float().to(device)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        result = spatial_model(stp_features, snippet_features=feat)
    else:
        result = spatial_model(stp_features)

    scores = result["frame"].detach().cpu().numpy()[0]
    preserved = preserved_index[0].detach().cpu().numpy().reshape(-1)
    gh, gw = model_grid
    return pool_to_eval_grid(scatter_patch_scores(scores, preserved, gh * gw), gh, gw)


@torch.no_grad()
def infer_video_patches(
    frames,
    n_snippets,
    stp_model,
    spatial_model,
    device,
    snippet_features=None,
    use_cross_attention=False,
    input_range="0_1",
    model_grid=(GRID_H, GRID_W),
    image_size=IMAGE_SIZE,
):
    video_patch_scores = []

    for seg_idx in range(n_snippets):
        start = seg_idx * SEGMENT_LENGTH
        end = min(start + SEGMENT_LENGTH, len(frames))
        segment = frames[start:end]
        if len(segment) < SEGMENT_LENGTH:
            pad = [segment[0]] * (SEGMENT_LENGTH - len(segment)) if segment else []
            segment = pad + segment
            if len(segment) < SEGMENT_LENGTH:
                segment = segment + [np.zeros((*image_size, 3), dtype=np.uint8)] * (
                    SEGMENT_LENGTH - len(segment)
                )

        grid_scores = infer_segment(segment, stp_model, spatial_model, device,
                                    snippet_features, use_cross_attention, input_range, model_grid)
        # One copy of the whole grid per frame. np.repeat would instead stretch
        # each patch over 16 consecutive entries, i.e. 2 rows of 4 different frames.
        video_patch_scores.extend(np.tile(grid_scores, SEGMENT_LENGTH).tolist())

    expected = n_snippets * SEGMENT_LENGTH * PATCHES_PER_FRAME
    if len(video_patch_scores) != expected:
        raise RuntimeError(
            f"Patch count mismatch: got {len(video_patch_scores)}, expected {expected}"
        )
    return video_patch_scores


def load_models(model_dir, stp_file, spatial_file, device, motion_loss=True,
                motion_aware_type="time-reversal", spatial_from_tokens=False, token_order="sorted",
                image_size=None):
    """
    DTFE + spatial head in eval mode. Token input is detected from the checkpoint.

    token_order="sorted" is the original evaluation: DTFE's eval branch reorders the
    tokens of each tubelet-frame by its keep score (even at token_ratio 1.0) before the
    spatial head, whose first layer is a Conv1d over the token sequence. During
    training tokens are only masked, never reordered, so the head is trained on raster
    order. token_order="original" skips the reordering at test time to match training.
    """
    if token_order not in ("sorted", "original"):
        raise ValueError(f"token_order must be 'sorted' or 'original', got {token_order}")

    # Input size and tubelet come from the DTFE checkpoint itself (128x128 / tubelet 8 for the
    # original models, e.g. 224x224 / tubelet 16 for VideoMAEv2-initialised ones).
    stp_state = torch.load(os.path.join(model_dir, stp_file), map_location="cpu")
    num_tubelet = int(stp_state["patch_embed.proj.weight"].shape[2])
    tokens = int(stp_state["pos_embed"].shape[1]) // (SEGMENT_LENGTH // num_tubelet)
    side = int(round(tokens ** 0.5))
    if image_size is None:
        image_size = (side * PATCH_SIZE, side * PATCH_SIZE)
    model_grid = (image_size[0] // PATCH_SIZE, image_size[1] // PATCH_SIZE)
    spatial_ckpt = torch.load(os.path.join(model_dir, spatial_file), map_location=device)
    weight = spatial_ckpt.get("embedding.conv_1.0.weight")
    if weight is not None and int(weight.shape[1]) == 768 and not spatial_from_tokens:
        print("Checkpoint spatial head expects 768-d tokens; enabling --spatial_from_tokens.")
        spatial_from_tokens = True
    spatial_in = 768 if spatial_from_tokens else PATCH_SIZE * PATCH_SIZE * num_tubelet * 3
    spatial = WSVAD_spatial(
        input_size=spatial_in,
        flag="Test",
        a_nums=60,
        n_nums=60,
    ).to(device)
    stp = DTFEModel(
        img_size=list(image_size),
        patch_size=PATCH_SIZE,
        tubelet_size=num_tubelet,
        all_frames=SEGMENT_LENGTH,
        in_chans=3,
        num_classes=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        pruning_loc=[3, 6, 9],
        token_ratio=list(TOKEN_RATIO),
        distill=False,
        disable_pruning=(token_order == "original"),
        motion_loss=motion_loss,
        motion_aware_type=motion_aware_type,
        spatial_from_tokens=spatial_from_tokens,
    ).to(device)

    spatial.load_state_dict(spatial_ckpt, strict=False)
    stp.load_state_dict(stp_state, strict=False)
    spatial.eval()
    stp.eval()
    stp.training = False
    stp.eval_image_size, stp.eval_model_grid = tuple(image_size), model_grid
    if model_grid != (GRID_H, GRID_W):
        print(f"DTFE input {image_size[0]}x{image_size[1]}, tubelet {num_tubelet}: "
              f"{model_grid[0]}x{model_grid[1]} token grid, area-pooled to {GRID_H}x{GRID_W} for evaluation")
    return stp, spatial, spatial_from_tokens


def main():
    parser = argparse.ArgumentParser(description="SST-WSVADL 8x8 patch evaluation")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--stp_file", type=str, default="stp_model_2022.pkl")
    parser.add_argument("--spatial_file", type=str, default="stpvad_model_2022.pkl")
    parser.add_argument("--frame_scores", type=str, required=True,
                        help="JSON produced by export_frame_scores.py")
    parser.add_argument("--snippet_features", type=str, default=None,
                        help="Optional .npz from export_frame_scores.py --save_snippet_features")
    parser.add_argument("--dataset", type=str, default="ucf", choices=["ucf", "xdviolence", "msad"])
    parser.add_argument("--len_feature", type=int, default=1408)
    parser.add_argument("--i3d", action="store_true")
    parser.add_argument("--data_variant", type=str, default="", help="UCF only, e.g. noletterbox")
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--patch_gt", type=str,
                        default="frame_label/ucf_gt_patches_0.25.npy",
                        help="Patch GT; 0.25 is the patch–GT-bbox overlap threshold (paper)")
    parser.add_argument("--output_dir", type=str, default="outputs/patch_eval")
    parser.add_argument("--temporal_threshold", type=float, default=None)
    parser.add_argument("--patch_thresholds", type=float, nargs="+", default=[0.5, 0.1, 0.7])
    add_bool_arg(parser, "cross_attention", True)
    add_bool_arg(parser, "motion_loss", True)
    parser.add_argument("--motion_aware_type", type=str, default="time-reversal")
    parser.add_argument("--input_range", type=str, default="0_1", choices=["0_1", "0_255"],
                        help="Pixel range fed to DTFE. Training uses [0, 1] (ToTensor), the default; "
                             "0_255 reproduces the evaluation before 2026-09-15")
    parser.add_argument("--token_order", type=str, default="sorted", choices=["sorted", "original"],
                        help="sorted: original evaluation (DTFE reorders tokens by keep score); "
                             "original: keep raster token order, as seen by the spatial head in training")
    parser.add_argument("--spatial_from_tokens", action="store_true",
                        help="Match training: spatial head scores 768-d DTFE tokens")
    parser.add_argument("--from_cache", action="store_true",
                        help="Skip inference; load patch_preds.npy from output_dir")
    add_bool_arg(
        parser,
        "anomaly_frames_only",
        False,
        help="If set, average TIoU only over frames with GT boxes. "
             "Default (False) matches the paper table (legacy UR-DMU protocol).",
    )
    parser.add_argument(
        "--iou_mode",
        type=str,
        default="bbox",
        choices=["bbox", "patch"],
        help="Paper metrics use bbox IoU (default). Use patch for binary patch-set IoU.",
    )
    parser.add_argument(
        "--bbox_annotations",
        type=str,
        default=None,
        help="Quantized bbox JSON (default: annotations/merged_*.json for --dataset). "
             "Used as GT for bbox TIoU. Pass empty string to reconstruct GT from patch maps.",
    )
    parser.add_argument(
        "--video_sizes",
        type=str,
        default=None,
        help="Optional {file_name: [H,W]} JSON written by generate_quantized_annotations.py",
    )
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    meta, frame_scores = load_frame_scores_json(args.frame_scores)
    temporal_threshold = (
        args.temporal_threshold
        if args.temporal_threshold is not None
        else float(meta["optimal_threshold"])
    )
    video_names = list(meta["video_names"])
    frames_count = list(meta["frames_count_per_video"])
    snippet_feats = load_snippet_features(args.snippet_features) if args.cross_attention else None

    cache_preds = os.path.join(args.output_dir, "patch_preds.npy")
    cache_meta = os.path.join(args.output_dir, "metadata.json")

    if args.from_cache and os.path.exists(cache_preds):
        print(f"Loading cached patch scores from {cache_preds}")
        patch_predictions = np.load(cache_preds)
        if os.path.exists(cache_meta):
            cached = json.load(open(cache_meta))
            video_names = cached["video_names"]
            frames_count = cached["frames_count_per_video"]
    else:
        dataset = build_dataset(args.dataset, args.len_feature, args.i3d, args.data_variant)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        stp, spatial, args.spatial_from_tokens = load_models(
            args.model_dir, args.stp_file, args.spatial_file, device,
            motion_loss=args.motion_loss, motion_aware_type=args.motion_aware_type,
            spatial_from_tokens=args.spatial_from_tokens, token_order=args.token_order,
        )

        limit = args.subset if args.subset is not None else len(loader.dataset)
        it = iter(loader)
        patch_predictions = []
        out_names = []
        out_counts = []

        for i in tqdm(range(limit), desc="Patch inference"):
            _data, _label, name = next(it)
            name = name[0] if isinstance(name, (list, tuple)) else str(name)
            n_snippets = int(_data.shape[1]) if _data.dim() == 3 else int(_data.shape[2])
            n_frames = n_snippets * SEGMENT_LENGTH

            path = video_path_for(args.dataset, args.video_root, name)
            frames = read_video_frames(path, resize=stp.eval_image_size)
            if len(frames) == 0:
                raise FileNotFoundError(f"No frames read from {path}")

            feats = None
            if snippet_feats is not None and name in snippet_feats:
                feats = snippet_feats[name]

            scores = infer_video_patches(
                frames,
                n_snippets,
                stp,
                spatial,
                device,
                snippet_features=feats,
                use_cross_attention=args.cross_attention,
                input_range=args.input_range,
                model_grid=stp.eval_model_grid,
                image_size=stp.eval_image_size,
            )
            patch_predictions.extend(scores)
            out_names.append(name)
            out_counts.append(n_frames)

        patch_predictions = np.asarray(patch_predictions, dtype=np.float32)  # model outputs are float32
        video_names = out_names
        frames_count = out_counts

        # Align frame scores to the videos we actually ran
        name_to_scores = {
            n: np.asarray(s, dtype=np.float64)
            for n, s in zip(meta["video_names"], meta["frame_scores"])
        }
        aligned = []
        for n, c in zip(video_names, frames_count):
            s = name_to_scores[n]
            if len(s) != c:
                raise ValueError(f"{n}: frame_scores={len(s)} vs inferred frames={c}")
            aligned.append(s)
        frame_scores = np.concatenate(aligned, axis=0)

        np.save(cache_preds, patch_predictions)
        with open(cache_meta, "w") as f:
            json.dump(
                {
                    "video_names": video_names,
                    "frames_count_per_video": frames_count,
                    "patches_per_frame": PATCHES_PER_FRAME,
                    "topk_vis": TOPK_VIS,
                },
                f,
            )
        print(f"Cached patch scores to {cache_preds}")

    if args.subset is not None:
        video_names = video_names[: args.subset]
        frames_count = frames_count[: args.subset]
        n_frames = int(np.sum(frames_count))
        frame_scores = frame_scores[:n_frames]
        patch_predictions = patch_predictions[: n_frames * PATCHES_PER_FRAME]

    patch_gt = load_patch_gt(args.patch_gt)
    expected = int(np.sum(frames_count)) * PATCHES_PER_FRAME
    if len(patch_gt) < expected:
        raise ValueError(f"patch_gt length {len(patch_gt)} < expected {expected}")
    patch_gt = patch_gt[:expected]
    if len(patch_predictions) != expected:
        raise ValueError(f"patch_preds length {len(patch_predictions)} != expected {expected}")

    opt_patch = find_optimal_threshold_youden(patch_gt, patch_predictions)
    print(f"Grid: {GRID_H}x{GRID_W}")
    print(f"Temporal threshold: {temporal_threshold}")
    print(f"Optimal patch threshold (Youden): {opt_patch:.6g}")

    thresholds = list(args.patch_thresholds)
    if opt_patch not in thresholds:
        thresholds = [opt_patch] + thresholds

    gt_bboxes = None
    bbox_path = args.bbox_annotations
    if bbox_path is None:
        bbox_path = DEFAULT_BBOX_ANNOTATIONS.get(args.dataset)
    if args.iou_mode == "bbox" and bbox_path:
        if not os.path.exists(bbox_path):
            raise FileNotFoundError(
                f"bbox annotations not found: {bbox_path}. "
                "Run eval/generate_quantized_annotations.py or pass --bbox_annotations ''"
            )
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
            image_size=args.image_size[0],
        )
        print(f"GT bboxes loaded from {bbox_path}")

    results = evaluate_patch_localization(
        patch_scores=patch_predictions,
        frame_scores=frame_scores,
        patch_gt=patch_gt,
        frames_count_per_video=frames_count,
        video_names=video_names,
        temporal_threshold=temporal_threshold,
        patches_per_frame=PATCHES_PER_FRAME,
        patch_thresholds=thresholds,
        anomaly_frames_only=args.anomaly_frames_only,
        iou_mode=args.iou_mode,
        patch_size=PATCH_SIZE,
        image_size=args.image_size[0],
        gt_bboxes=gt_bboxes,
    )
    results["temporal_threshold"] = temporal_threshold
    results["tiou"] = {f"{k:.6g}": v for k, v in results["tiou"].items()}
    with open(os.path.join(args.output_dir, "paper_protocol.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
