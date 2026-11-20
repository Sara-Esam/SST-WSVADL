"""
Initialise DTFEModel from a VideoMAEv2 ViT-B checkpoint (e.g. vit_b_k710_dl_from_giant.pth).

DTFE and VideoMAEv2 ViT-B share the block layout (768-d, 12 blocks, 12 heads, patch 16),
so most weights copy directly. Three things need adapting:
  * attention bias: VideoMAEv2 stores q_bias / v_bias; DTFE has one qkv bias (k bias = 0).
  * tubelet size: the checkpoint's 3D patch conv spans 2 frames. For other temporal
    sizes the kernel is resampled along time and rescaled so a static clip gives the
    same response.
  * position embedding: VideoMAEv2 uses a fixed 1-D sinusoid table over (t, h*w) tokens,
    not stored in the checkpoint. It is rebuilt for the pretrained 8x14x14 grid and
    resampled to DTFE's grid, then kept learnable.
Pruning predictors and the reconstruction head stay randomly initialised.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sinusoid_table(n_position, dim):
    pos = np.arange(n_position)[:, None]
    j = np.arange(dim)[None, :]
    table = pos / np.power(10000, 2 * (j // 2) / dim)
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32)


def _resample_pos_embed(dim, src_t, src_hw, dst_t, dst_h, dst_w):
    table = sinusoid_table(src_t * src_hw * src_hw, dim).reshape(src_t, src_hw, src_hw, dim)
    t_idx = [min(src_t - 1, int((t + 0.5) * src_t / dst_t)) for t in range(dst_t)]
    table = table[t_idx]                                          # [dst_t, h, w, D]
    if (dst_h, dst_w) != (src_hw, src_hw):
        table = F.interpolate(table.permute(0, 3, 1, 2), size=(dst_h, dst_w),
                              mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    return table.reshape(1, dst_t * dst_h * dst_w, dim)


def _resample_patch_kernel(weight, dst_t):
    """[D, 3, src_t, p, p] -> [D, 3, dst_t, p, p], preserving the summed response."""
    src_t = weight.shape[2]
    if src_t == dst_t:
        return weight
    idx = [min(src_t - 1, int(t * src_t / dst_t)) for t in range(dst_t)]
    return weight[:, :, idx] * (src_t / dst_t)


def load_videomae_into_dtfe(dtfe, ckpt_path, verbose=True):
    state = torch.load(ckpt_path, map_location="cpu")
    for key in ("model", "module", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break

    own = dtfe.state_dict()
    new = {}
    for k, v in state.items():
        if k.startswith("blocks.") and (k.endswith("attn.q_bias") or k.endswith("attn.v_bias")):
            continue
        if k.startswith(("head.", "fc_norm.", "norm.")):
            continue
        if k == "patch_embed.proj.weight":
            v = _resample_patch_kernel(v, own[k].shape[2])
        new[k] = v

    n_blocks = len(dtfe.blocks)
    for i in range(n_blocks):
        q = state.get(f"blocks.{i}.attn.q_bias")
        vb = state.get(f"blocks.{i}.attn.v_bias")
        if q is not None and vb is not None:
            new[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q, torch.zeros_like(q), vb])

    patch = dtfe.patch_embed.patch_size
    ph, pw = (patch, patch) if isinstance(patch, int) else patch
    dst_h, dst_w = dtfe.img_size[0] // ph, dtfe.img_size[1] // pw
    src_t = 8  # VideoMAEv2 ViT-B: 16 frames, tubelet 2
    new["pos_embed"] = _resample_pos_embed(dtfe.embed_dim, src_t, 14, dtfe.num_tubelet, dst_h, dst_w)

    bad = [k for k, v in new.items() if k in own and own[k].shape != v.shape]
    for k in bad:
        new.pop(k)
    missing, unexpected = dtfe.load_state_dict(new, strict=False)
    if verbose:
        loaded = len(new) - len([k for k in new if k not in own])
        print(f"[DTFE] loaded {loaded} tensors from {ckpt_path}")
        if bad:
            print(f"[DTFE] skipped (shape mismatch): {bad}")
        backbone_missing = [k for k in missing if k.startswith(("blocks.", "patch_embed."))]
        if backbone_missing:
            print(f"[DTFE] WARNING backbone tensors not loaded: {backbone_missing[:8]}")
        if unexpected:
            print(f"[DTFE] unexpected keys: {unexpected[:8]}")
    dtfe.input_norm.fill_(1.0)
    return dtfe


def freeze_dtfe_backbone(dtfe):
    """Freeze patch embedding and transformer blocks. Pruning predictors, the final norm
    and the reconstruction head stay trainable."""
    for name, p in dtfe.named_parameters():
        if name.startswith(("patch_embed.", "blocks.")) or name == "pos_embed":
            p.requires_grad = False
