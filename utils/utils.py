"""
Utility functions for SST-WSVADL
"""
import os
import torch
import numpy as np
import random
from decord import VideoReader
import torchvision.transforms.functional as TF


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


def extract_frames_from_video(video_path, frame_indices, transform=None):
    """
    Extract frames from video using decord.
    Returns tensor of shape [B, C, T, H, W]
    """
    video_path, lengths = video_path
    all_video_tensors = []
    for i, video_path_item in enumerate(video_path):
        if isinstance(video_path_item, str):
            vr = VideoReader(video_path_item)
        else:
            vr = video_path_item
        total_frames = len(vr)

        # Clamp indices to available frames
        start, end = frame_indices[i]
        start = max(0, min(start, total_frames - 1))
        end = max(start + 1, min(end, total_frames))
        
        frame_range = range(start, end)
        if len(frame_range) == 0:
            frame_range = range(0, min(16, total_frames))  # Default to first 16 frames if empty
        
        frames = vr.get_batch(list(frame_range)).asnumpy().astype(np.uint8)        
        frame_tensors = [TRANSFORM(frame) for frame in frames]  # list of [C, H, W]
        video_tensor = torch.stack(frame_tensors, dim=1)  # [C, T, H, W]
        all_video_tensors.append(video_tensor.unsqueeze(0))

    return torch.cat(all_video_tensors, dim=0)  # [B, C, T, H, W]
