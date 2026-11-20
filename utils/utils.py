"""
Utility functions for SST-WSVADL
"""
import os
import torch
import numpy as np
import random
from decord import VideoReader, cpu
import torchvision.transforms.functional as TF


def _open_video_reader(src, resize=None):
    """Open a VideoReader from a path, or return an already-open reader.

    Decoding at `resize` (H, W) avoids keeping native-resolution frames in RAM.
    Returns (reader, owned) so callers can free readers they opened.
    """
    if not isinstance(src, str):
        return src, False
    kwargs = dict(ctx=cpu(0), num_threads=1)
    if resize is not None:
        height, width = int(resize[0]), int(resize[1])
        kwargs['height'] = height
        kwargs['width'] = width
    return VideoReader(src, **kwargs), True


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def norm(data):
    """L2 normalization"""
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)


def save_best_record(test_info, file_path):
    """Save best test record to file"""
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("auc: {:.4f}\n".format(test_info["auc"][-1]))
    fo.write("ap: {:.4f}\n".format(test_info["ap"][-1]))
    fo.write("ac: {:.4f}\n".format(test_info["ac"][-1]))
    fo.close()


def random_perturb(feature_len, length):
    """Generate random perturbation indices"""
    r = np.linspace(0, feature_len, length + 1, dtype=np.uint16)
    return r


SNIPPET_AGG_METHODS = ("mean", "drop_ends", "subsample", "reverse", "combine")
SNIPPET_AGG_CHOICES = SNIPPET_AGG_METHODS + ("random",)


def pick_snippet_agg(method):
    """If method is `random`, draw one of the concrete aggregators for this sample."""
    if method == "random":
        return str(np.random.choice(SNIPPET_AGG_METHODS))
    return method


def aggregate_snippets(feat, num_segments, method="mean", drop=8, return_mapping=False):
    """Map a variable-length snippet sequence [T, D] onto [num_segments, D].

    mean       : current behaviour — average snippets inside each of 200 bins
    drop_ends  : drop `drop` snippets from the start and end, then mean-pool
    subsample  : take the centre snippet of each bin instead of averaging
    reverse    : reverse snippet order, then mean-pool
    combine    : drop_ends + reverse + subsample
    random     : pick one of the five above independently on this call

    If return_mapping is True, also return (orig_to_bin, method) where orig_to_bin[t]
    is the bin index for original snippet t, or -1 if it was dropped.
    """
    method = pick_snippet_agg(method)
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError(f"Expected [T, D] features, got {feat.shape}")
    orig_t = feat.shape[0]
    if orig_t == 0:
        raise ValueError("Empty feature sequence")

    orig_idx = np.arange(orig_t)
    if method in ("drop_ends", "combine"):
        d = min(int(drop), max(0, (orig_t - 1) // 2))
        if orig_t - 2 * d >= 1:
            feat = feat[d:orig_t - d]
            orig_idx = orig_idx[d:orig_t - d]

    if method in ("reverse", "combine"):
        feat = feat[::-1].copy()
        orig_idx = orig_idx[::-1].copy()

    t = feat.shape[0]
    use_subsample = method in ("subsample", "combine")
    new_feat = np.zeros((num_segments, feat.shape[1]), dtype=np.float32)
    orig_to_bin = np.full(orig_t, -1, dtype=np.int32)
    edges = np.linspace(0, t, num_segments + 1, dtype=int)
    for i in range(num_segments):
        a, b = int(edges[i]), int(edges[i + 1])
        if a >= t:
            members = [t - 1]
            new_feat[i] = feat[-1]
        elif a == b:
            members = [min(a, t - 1)]
            new_feat[i] = feat[members[0]]
        else:
            members = list(range(a, b))
            if use_subsample:
                new_feat[i] = feat[a + (b - a) // 2]
            else:
                new_feat[i] = feat[a:b].mean(0)
        for m in members:
            orig_to_bin[int(orig_idx[m])] = i

    if return_mapping:
        return new_feat, orig_to_bin, method
    return new_feat


def map_topk_snippets_to_frames(topk_indices, original_length, snippet_len=16, new_seg_size=32, video_segment_length=200):
    """
    Map top-k snippet indices to frame indices.
    Returns a list of frame indices, each frame index is a tuple of (start, end).
    """
    aggregation_length = [(ori_len // video_segment_length) if (ori_len // video_segment_length) > 0 else 1 for ori_len in original_length]
    snippet_to_frame_indices = {i: i for i in range(video_segment_length)}
    
    topk_frame_indices = []
    for i, idx in enumerate(topk_indices):
        if (original_length[i] // video_segment_length == 0) and (snippet_to_frame_indices[idx] > original_length[i]):
            start = original_length[i] * new_seg_size - new_seg_size
            end = start + new_seg_size
            topk_frame_indices.append([start, end])
        else:
            one_sframe = snippet_to_frame_indices[idx]
            start = one_sframe * snippet_len * aggregation_length[i]
            start = start if isinstance(start, int) else start.cpu().numpy()
            start = start - snippet_len if start > snippet_len else 0
            end = start + new_seg_size
            topk_frame_indices.append([start, end])
    
    return topk_frame_indices


def extract_frames_from_video(video_path, frame_indices, transform=None, resize=None):
    """
    Extract frames from video using decord.
    `video_path` is either a list of paths/readers, or ([paths/readers], lengths).
    Opens at most one video at a time. Returns tensor of shape [B, C, T, H, W].
    """
    if (
        isinstance(video_path, (list, tuple))
        and len(video_path) == 2
        and isinstance(video_path[0], (list, tuple))
    ):
        video_path, _lengths = video_path

    all_video_tensors = []
    for i, video_path_item in enumerate(video_path):
        vr, owned = _open_video_reader(video_path_item, resize=resize)
        try:
            total_frames = len(vr)

            # Clamp indices to available frames
            start, end = frame_indices[i]
            start = max(0, min(int(start), total_frames - 1))
            end = max(start + 1, min(int(end), total_frames))

            frame_range = range(start, end)
            if len(frame_range) == 0:
                frame_range = range(0, min(16, total_frames))

            frames = vr.get_batch(list(frame_range)).asnumpy().astype(np.uint8)
            frame_tensors = [transform(frame) for frame in frames]
            video_tensor = torch.stack(frame_tensors, dim=1)
            all_video_tensors.append(video_tensor.unsqueeze(0))
        finally:
            if owned:
                del vr

    return torch.cat(all_video_tensors, dim=0)  # [B, C, T, H, W]
