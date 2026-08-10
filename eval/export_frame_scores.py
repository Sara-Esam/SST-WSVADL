"""
Export frame-level anomaly scores from the temporal model.

Run once for a checkpoint, then pass the JSON to patch evaluation so the
temporal branch is not re-run during spatial localization.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import WSAD_temporal
from data.datasets import UCFCrime, XDViolence, MSAD
from eval.patch_metrics import find_optimal_threshold_youden
from utils.utils import set_seed


SNIPPET_LENGTH = 16


def build_dataset(dataset, len_feature, i3d):
    kwargs = dict(root_dir=None, mode="Test", modal="rgb", num_segments=200,
               len_feature=len_feature, is_normal=None, i3d=i3d)
    if dataset == "ucf":
        return UCFCrime(**kwargs)
    if dataset == "xdviolence":
        return XDViolence(**kwargs)
    if dataset == "msad":
        return MSAD(**kwargs)
    raise ValueError(f"Unsupported dataset: {dataset}")


def frame_gt_path(dataset, i3d, label_dir):
    if i3d:
        return os.path.join(label_dir, f"{dataset}_gt_i3d.npy")
    if dataset == "ucf":
        return os.path.join(label_dir, f"{dataset}_gt_videomaev2.npy")
    return os.path.join(label_dir, f"{dataset}_gt.npy")


@torch.no_grad()
def export_scores(model, loader, device, subset=None):
    model.eval()
    model.flag = "Test"

    video_names = []
    cls_labels = []
    frames_count = []
    frame_scores = []
    snippet_features = {}

    n = subset if subset is not None else len(loader.dataset)
    it = iter(loader)
    for _ in range(n):
        data, label, name = next(it)
        data = data.cuda(non_blocking=True) if device == "cuda" else data
        out = model(data)
        snippet_scores = out["frame"].detach().cpu().numpy()[0]
        expanded = np.repeat(snippet_scores, SNIPPET_LENGTH)

        video_names.append(name[0] if isinstance(name, (list, tuple)) else str(name))
        cls_labels.append(int(label))
        frames_count.append(int(len(expanded)))
        frame_scores.append(expanded.astype(np.float32))

        if "snippet_features" in out:
            snippet_features[video_names[-1]] = out["snippet_features"].detach().cpu().numpy()

    return {
        "video_names": video_names,
        "cls_labels": cls_labels,
        "frames_count_per_video": frames_count,
        "frame_scores": frame_scores,
        "snippet_features": snippet_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Export temporal frame scores for patch evaluation")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_file", type=str, default="ucf_trans_2022.pkl")
    parser.add_argument("--dataset", type=str, default="ucf", choices=["ucf", "xdviolence", "msad"])
    parser.add_argument("--len_feature", type=int, default=1408)
    parser.add_argument("--i3d", action="store_true")
    parser.add_argument("--label_dir", type=str, default="frame_label")
    parser.add_argument("--output", type=str, default="predictions/frame_scores.json")
    parser.add_argument("--save_snippet_features", action="store_true",
                        help="Also save snippet features for cross-attention at patch eval")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(args.model_dir, args.model_file)
    model = WSAD_temporal(input_size=args.len_feature, flag="Test", a_nums=60, n_nums=60).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    dataset = build_dataset(args.dataset, args.len_feature, args.i3d)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    pack = export_scores(model, loader, device, subset=args.subset)
    flat_scores = np.concatenate(pack["frame_scores"], axis=0)

    gt_path = frame_gt_path(args.dataset, args.i3d, args.label_dir)
    if os.path.exists(gt_path):
        frame_gt = np.load(gt_path)
        n = min(len(frame_gt), len(flat_scores))
        fpr, tpr, _ = roc_curve(frame_gt[:n], flat_scores[:n])
        print(f"Frame AUC: {auc(fpr, tpr):.4f}")
        optimal_threshold = find_optimal_threshold_youden(frame_gt[:n], flat_scores[:n])
    else:
        print(f"Warning: missing {gt_path}; optimal_threshold set to 0.5")
        optimal_threshold = 0.5

    print(f"Optimal temporal threshold (Youden): {optimal_threshold}")

    payload = {
        "dataset": args.dataset,
        "model_file": model_path,
        "snippet_length": SNIPPET_LENGTH,
        "optimal_threshold": float(optimal_threshold),
        "video_names": pack["video_names"],
        "cls_labels": pack["cls_labels"],
        "frames_count_per_video": pack["frames_count_per_video"],
        "frame_scores": [s.tolist() for s in pack["frame_scores"]],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {args.output}")

    if args.save_snippet_features and pack["snippet_features"]:
        feat_path = os.path.splitext(args.output)[0] + "_snippet_features.npz"
        np.savez_compressed(feat_path, **pack["snippet_features"])
        print(f"Wrote {feat_path}")


if __name__ == "__main__":
    main()
