"""
This file contains models copied from UR-DMU and STPrivacy.
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from einops import rearrange, repeat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .memory import Memory_Unit
from .transformer import Transformer
from utils.utils import norm
from cross_attention import CrossAttentionBlock


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x)


class WSAD_temporal(Module):
    """
    UR-DMU Model: Weakly Supervised Anomaly Detection with Dual Memory Units
    """
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        self.embedding = Temporal(input_size,512)
        self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ADCLS_head(1024, 1)
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout = 0.5)
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))
        return kl_loss

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        if self.flag == "Train":
            N_x = x[:b*n//2]  # Normal part
            A_x = x[b*n//2:]  # Abnormal part
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)

            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)
    
            _, A_index = torch.topk(A_att, t//16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)
            
            _, N_index = torch.topk(N_att, t//16 + 1, dim=-1)
            anchor_nx=torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, P_index = torch.topk(N_Aatt, t//16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)
               
            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            
            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            A_aug_new = self.encoder_mu(A_aug)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)
            
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)
          
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            snippet_features = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(snippet_features).reshape((b, n, -1)).mean(1)
    
            return {
                    "frame": pre_att,
                    'triplet_margin': triplet_margin_loss,
                    'kl_loss': kl_loss, 
                    'distance': distance,
                    'A_att': A_att.reshape((b//2, n, -1)).mean(1),
                    "N_att": N_att.reshape((b//2, n, -1)).mean(1),
                    "A_Natt": A_Natt.reshape((b//2, n, -1)).mean(1),
                    "N_Aatt": N_Aatt.reshape((b//2, n, -1)).mean(1),
                    "A_index": A_index,
                    "N_index": N_index,
                    "P_index": P_index,
                    "snippet_features": snippet_features,
                }
        else:           
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)  

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            snippet_features = torch.cat([x, A_aug + N_aug], dim=-1)
            
            pre_att = self.cls_head(snippet_features).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att, "snippet_features": snippet_features}


class WSVAD_spatial(Module):
    """
    STPVAD Model: Weakly Supervised Video Anomaly Detection with Spatio-Temporal Privacy
    Patch-level anomaly detection model
    """
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums
        self.embedding = Temporal(input_size, 512)
        self.cls_head = ADCLS_head(1024, 1)
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout = 0.5)
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()
        self.triplet = nn.TripletMarginLoss(margin=1)
        self.psa = CrossAttentionBlock(dim=1024, heads=8, dim_head=64, mlp_dim=512, dropout=0.5)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))
        return kl_loss

    def forward(self, x, snippet_features=None):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        if self.flag == "Train":
            N_x = x[:b*n//2]  # Normal part
            A_x = x[b*n//2:]  # Abnormal part
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)

            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)
            _, A_index = torch.topk(A_att, t//16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, N_index = torch.topk(N_att, t//16 + 1, dim=-1)
            anchor_nx=torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)
            _, P_index = torch.topk(N_Aatt, t//16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)
            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            
            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            A_aug_new = self.encoder_mu(A_aug)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)
            
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)
          
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            patch_features = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))), dim=-1)

            # Apply cross-attention: patch features (query) attend to snippet features (key-value)
            if snippet_features is not None:
                cross_attended_features = self.psa(
                    query=patch_features,
                    key_value=snippet_features
                )
            else:
                cross_attended_features = patch_features
            pre_att = self.cls_head(cross_attended_features).reshape((b, n, -1)).mean(1)
            
            # Extract patch-level scores
            patch_scores = self.cls_head(cross_attended_features).reshape((b, n, -1))
            
            return {
                    "frame": pre_att,
                    'triplet_margin': triplet_margin_loss,
                    'kl_loss': kl_loss, 
                    'distance': distance,
                    'A_att': A_att.reshape((b//2, n, -1)).mean(1),
                    "N_att": N_att.reshape((b//2, n, -1)).mean(1),
                    "A_Natt": A_Natt.reshape((b//2, n, -1)).mean(1),
                    "N_Aatt": N_Aatt.reshape((b//2, n, -1)).mean(1),
                    "A_index": A_index,
                    "N_index": N_index,
                    "P_index": P_index,
                    "patch_features": patch_features,
                    "cross_attended_features": cross_attended_features,
                    "patch_scores": patch_scores,
                }
        else:           
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)  

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)
           
            if snippet_features is not None:
                cross_attended_features = self.cross_attention(
                    query=x,
                    key_value=snippet_features
                )
            else:
                cross_attended_features = x
            
            pre_att = self.cls_head(cross_attended_features).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None):
        x = x + self.drop_path(self.attn(self.norm1(x), policy=policy))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(128,128), patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0], patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = rearrange(x, 'b c t h w -> b c t (h w)')
        x = rearrange(x, 'b c t n -> b c (t n)').transpose(1, 2)
        return x


class MotionAwarePatchEmbed(nn.Module):
    """Motion-aware patch embedding with time reversal technique"""
    def __init__(self, img_size=(128,128), patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0], patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def compute_motion_scores(self, tubelet_features, tubelet_features_rev):
        """
        Compute motion scores using time reversal technique on tubelet features
        tubelet_features: [B, N, C] - forward time tubelet features
        tubelet_features_rev: [B, N, C] - reversed time tubelet features
        """
        # Separate appearance and motion subspaces
        z_app = 0.5 * (tubelet_features + tubelet_features_rev)  # appearance subspace
        z_mot = 0.5 * (tubelet_features - tubelet_features_rev)  # motion subspace
        
        # Compute motion magnitude for each tubelet
        motion_scores = z_mot.norm(dim=-1)  # [B, N]
        
        return motion_scores, z_app, z_mot

    def forward(self, x, x_rev=None, motion_keep_ratio=1.0):
        """
        x: [B, C, T, H, W] - input video
        x_rev: [B, C, T, H, W] - time-reversed video
        motion_keep_ratio: if provided, select top motion tubelets
        """
        B, C, T, H, W = x.shape
        
        # Standard tubelet embedding (same as original PatchEmbed)
        x = self.proj(x)
        x = rearrange(x, 'b c t h w -> b c t (h w)')
        x = rearrange(x, 'b c t n -> b c (t n)').transpose(1, 2)  # [B, N, C]
        
        # if x_rev is not None:
        # Process reversed video
        x_rev = self.proj(x_rev)
        x_rev = rearrange(x_rev, 'b c t h w -> b c t (h w)')
        x_rev = rearrange(x_rev, 'b c t n -> b c (t n)').transpose(1, 2)  # [B, N, C]
        
        # Compute motion scores
        motion_scores, z_app, z_mot = self.compute_motion_scores(x, x_rev)
        
        # if motion_keep_ratio is not None:
        # Select top motion tubelets
        B, N = motion_scores.shape
        k = int(N * motion_keep_ratio)
        
        # Get top-k motion tubelets
        _, top_indices = torch.topk(motion_scores, k, dim=-1)  # [B, k]
        ori_x = x
        # Select tubelets based on motion scores
        selected_tubelets = torch.gather(x, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [B, k, C]
        
        return selected_tubelets, top_indices, motion_scores, ori_x


class MotionVariencePatchEmbed(nn.Module):
    """
    Motion-aware patch embedding using temporal feature difference.
    This approach computes motion intensity based on the magnitude of the 
    difference between temporally adjacent tubelet features, aligning with 
    the "feature difference" method for motion estimation in models like RefineVAD.
    """
    def __init__(self, img_size=(128,128), patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        # Assuming to_2tuple is available for patch_size conversion
        patch_size = to_2tuple(patch_size) if not isinstance(patch_size, tuple) else patch_size 
        self.tubelet_size = int(tubelet_size)
        self.num_frames = num_frames
        
        T_tubes = num_frames // self.tubelet_size
        N_spatial = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        num_patches = N_spatial * T_tubes
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_tubelet = T_tubes 
        self.num_spatial_patches = N_spatial

        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size = (self.tubelet_size,  patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def compute_motion_scores(self, tubelet_features):
        """
        Compute motion scores based on the magnitude of the difference 
        between adjacent tubelet features in time.
        
        tubelet_features: [B, N, C] - concatenated tubelet features, where N = T * N_spatial
        """
        B, N, C = tubelet_features.shape
        T_tubes = self.num_tubelet
        N_spatial = self.num_spatial_patches
        
        # Reshape from [B, N, C] to [B, T, N_spatial, C]
        x_reshaped = tubelet_features.reshape(B, T_tubes, N_spatial, C)

        # Calculate temporal difference: x_t - x_{t-1}. [B, T-1, N_spatial, C]
        # This is the feature difference between the current tubelet and the previous one.
        diff = x_reshaped[:, 1:, :, :] - x_reshaped[:, :-1, :, :]
        
        # Calculate magnitude (motion score) of the differences. [B, T-1, N_spatial]
        motion_mag = torch.norm(diff, dim=-1)

        # Pad the first time step (t=0) with zeros to match the original T dimension [B, T, N_spatial]
        # The first time step has no 'previous' tubelet for comparison, so its difference score is zero.
        padding = torch.zeros(B, 1, N_spatial, device=tubelet_features.device)
        motion_scores_padded = torch.cat([motion_mag, motion_mag], dim=1) #! or maybe just repeat it, since here we care more about the spatial locations
        
        # Flatten back to [B, N] to get a motion score per tubelet
        motion_scores = motion_scores_padded.reshape(B, N)
        
        return motion_scores

    def forward(self, x, motion_keep_ratio=1.0, **kwargs):
        """
        x: [B, C, T_raw, H, W] - input video
        This method returns the full feature set along with motion scores 
        for the downstream pruning and motion loss computation.
        """
        B, C, T_raw, H, W = x.shape
        
        # Standard tubelet embedding (3D convolution)
        x_proj = self.proj(x)
        
        # Flatten and transpose to get tubelet features [B, N, C]
        x_proj = rearrange(x_proj, 'b c t h w -> b c t (h w)')
        tubelet_features = rearrange(x_proj, 'b c t n -> b c (t n)').transpose(1, 2) # [B, N, C]

        # Compute motion scores
        motion_scores = self.compute_motion_scores(tubelet_features)
        
        # Match the expected output signature (selected_tubelets, top_indices, motion_scores, ori_x)
        # Assuming k=N (ratio 1.0) since selection is done later in the main block.
        N = tubelet_features.shape[1]
        selected_tubelets = tubelet_features 
        top_indices = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)

        # Return the original/full feature set, a placeholder for indices, the computed scores, and the original features again.
        return selected_tubelets, top_indices, motion_scores, tubelet_features


class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384, num_tubelet=8):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.num_tubelet = num_tubelet
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        t = self.num_tubelet
        x = rearrange(x, 'b (t n) c -> b t n c', t=t)
        policy = rearrange(policy, 'b (t n) c -> b t n c', t=t)

        local_x = x[:,:,:, :C//3]
        frame_x = (x[:,:,:, C//3:C//3*2] * policy).sum(dim=2, keepdim=True) / torch.sum(policy, dim=2, keepdim=True)
        video_x = (x[:,:,:, C//3*2:] * policy).sum(dim=(1, 2), keepdim=True) / torch.sum(policy, dim=(1, 2), keepdim=True)
        x = torch.cat([local_x, frame_x.expand(B, t, N//t, C//3), video_x.expand(B, t, N//t, C//3)], dim=-1)
        x = rearrange(x, 'b t n c -> b (t n) c')
        return self.out_conv(x)


class PredictorLGSoft(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384, num_tubelet=8):
        super().__init__()
        self.num_tubelet = num_tubelet
        self.keep_threshold_base = torch.tensor(0.486)
        self.keep_threshold = nn.Parameter(
                torch.zeros_like(self.keep_threshold_base),
                requires_grad=False,
        )
        self.temperature = 1e-1
        self.mask = None

        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, policy):
        keep_threshold = self.keep_threshold + self.keep_threshold_base
        x = self.in_conv(x)
        B, N, C = x.size()
        t = self.num_tubelet

        x = rearrange(x, 'b (t n) c -> b t n c', t=t)
        policy = rearrange(policy, 'b (t n) c -> b t n c', t=t)
        
        local_x = x[:,:,:, :C//3]
        frame_x = (x[:,:,:, C//3:C//3*2] * policy).sum(dim=2, keepdim=True) / torch.sum(policy, dim=2, keepdim=True)
        video_x = (x[:,:,:, C//3*2:] * policy).sum(dim=(1, 2), keepdim=True) / torch.sum(policy, dim=(1, 2), keepdim=True)
        
        x = torch.cat([local_x, frame_x.expand(B, t, N//t, C//3), video_x.expand(B, t, N//t, C//3)], dim=-1)
        x = rearrange(x, 'b t n c -> b (t n) c')
        policy = rearrange(policy, 'b t n c -> b (t n) c')
        x = self.out_conv(x)

        if self.training:
            mask = torch.sigmoid((x[:,:,0:1] - keep_threshold) / self.temperature)
            # print(mask)
        else:
            # mask = torch.sigmoid((x[:,:,0:1] - keep_threshold) / self.temperature)
            mask = torch.ones(x[:,:,0:1].shape, device=x.device)
            mask[x[:,:,0:1] < keep_threshold] = 0

        self.mask = mask*policy

        return x, self.mask, keep_threshold


class DTFEModel(nn.Module):
    def __init__(self, img_size=(128,128), patch_size=16, tubelet_size=2, all_frames=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, 
                 pruning_loc=None, token_ratio=None, distill=False, disable_pruning=False, clip_features=False, 
                 sparse_loss=False, motion_loss=False, motion_loss_weight=0.01, adjacency_loss=False, adjacency_loss_weight=0.01, 
                 motion_aware_type='time-reversal'):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.disable_pruning = disable_pruning
        self.sparse_loss = sparse_loss
        self.clip_features = clip_features
        self.motion_loss = motion_loss
        self.motion_loss_weight = motion_loss_weight
        self.motion_aware_type = motion_aware_type
        self.adjacency_loss = adjacency_loss
        self.adjacency_loss_weight = adjacency_loss_weight
        self.global_step = 0
        if self.clip_features:
            self.clip_weight = nn.Linear(768, 64, bias=False)

        # import pdb; pdb.set_trace()
        if self.motion_loss:
            if self.motion_aware_type == 'time-reversal':
                self.patch_embed = MotionAwarePatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
                    tubelet_size=tubelet_size)
            elif self.motion_aware_type == 'varience':
                self.patch_embed = MotionVariencePatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
                    tubelet_size=tubelet_size)
        else:    
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
                tubelet_size=tubelet_size)

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (all_frames // tubelet_size)
        self.num_tubelet = all_frames // tubelet_size
        self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_recon = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Reconstruction head
        num_classes = patch_size * patch_size * tubelet_size * 3
        self.head_recon = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Only create predictors if pruning is enabled
        if not self.disable_pruning:
            predictor_list = [PredictorLG(embed_dim, self.num_tubelet) for _ in range(len(pruning_loc))]
            self.score_predictor = nn.ModuleList(predictor_list)
        else:
            self.score_predictor = None

        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        if self.motion_loss:
            if self.motion_aware_type == 'varience':
                selected_tubelets, top_indices, motion_scores, x = self.patch_embed(x)
            else:
                selected_tubelets, top_indices, motion_scores, x = self.patch_embed(x, x_rev=x.flip(dims=[2]))
        else:
            x = self.patch_embed(x)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        init_n = self.patch_embed.num_patches
        preserve_index = torch.arange(init_n, dtype=torch.long, device=x.device).reshape(1, -1, 1).expand(B, init_n, 1)
        
        if self.disable_pruning:
            # Process all blocks without pruning - keep all patches
            for i, blk in enumerate(self.blocks):
                x = blk(x)  # No policy parameter needed
            out_pred_prob = []
        else:
            # Original pruning logic
            p_count = 0
            out_pred_prob = []
            prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
            policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
            spars_losses = []
            motion_loss = 0.0
            for i, blk in enumerate(self.blocks):
                if i in self.pruning_loc:
                    """
                    Note that the gumble softmax here is used to sample discrete values (keep/drop), allowing the gradients 
                    to flow through the sampled values only.

                    Softmax gives continuious probabilities, without allowing propagation through discrete values.

                    """
                    spatial_x = x
                    pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                    if self.training:
                        hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                        ################ Motion preference loss computation ################
                        if self.motion_loss: 
                            motion_preference_loss = -torch.sum(hard_keep_decision * motion_scores.unsqueeze(-1)) 
                            motion_loss += motion_preference_loss
                        ###########################################################
                        out_pred_prob.append(rearrange(hard_keep_decision.reshape(B, init_n), 'b (t n) -> b t n', t=self.num_tubelet))
                        policy = hard_keep_decision
                        x = blk(x, policy=policy)
                        prev_decision = hard_keep_decision
                    else:
                        score = rearrange(pred_score[:, :, 0], 'b (t n) -> b t n', t=self.num_tubelet)
                        num_keep_node_per_frame = int(init_n // self.num_tubelet * self.token_ratio[p_count])
                        keep_policy = torch.argsort(score, dim=2, descending=True)[:, :, :num_keep_node_per_frame]
                        now_policy = keep_policy
                        offset = torch.arange(self.num_tubelet, dtype=torch.long, device=x.device).view(1, self.num_tubelet, 1) * score.shape[-1]
                        now_policy = now_policy + offset
                        now_policy = rearrange(now_policy, 'b t c -> b (t c)')
                        x = batch_index_select(x, now_policy)
                        prev_decision = batch_index_select(prev_decision, now_policy)
                        preserve_index = batch_index_select(preserve_index, now_policy)
                        x = blk(x)
                    p_count += 1
                else:
                    if self.training:
                        x = blk(x, policy)
                    else:
                        x = blk(x)

        x = self.norm_recon(x)
        x = self.head_recon(x)
        motion_loss = self.motion_loss_weight* motion_loss.mean() if (self.motion_loss and self.training) else 0

        
        if self.training and self.motion_loss:
            return x, out_pred_prob, preserve_index, motion_loss    
        else:
            return x, out_pred_prob, preserve_index


class DTFEModelSoft(nn.Module):
    def __init__(self, img_size=(128,128), patch_size=16, tubelet_size=2, all_frames=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, 
                 pruning_loc=None, token_ratio=None, distill=False, disable_pruning=False, clip_features=False, 
                 sparse_loss=False, motion_loss=False, motion_loss_weight=0.01, adjacency_loss=False, adjacency_loss_weight=0.01):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.disable_pruning = disable_pruning
        self.sparse_loss = sparse_loss
        self.clip_features = clip_features
        self.motion_loss = motion_loss
        self.motion_loss_weight = motion_loss_weight
        self.adjacency_loss = adjacency_loss
        self.adjacency_loss_weight = adjacency_loss_weight
        self.global_step = 0
        if self.clip_features:
            self.clip_weight = nn.Linear(768, 64, bias=False)

        # import pdb; pdb.set_trace()
        if self.motion_loss:
            self.patch_embed = MotionAwarePatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
                tubelet_size=tubelet_size)
        else:    
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
                tubelet_size=tubelet_size)

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (all_frames // tubelet_size)
        self.num_tubelet = all_frames // tubelet_size
        self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_recon = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Reconstruction head
        num_classes = patch_size * patch_size * tubelet_size * 3
        self.head_recon = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Only create predictors if pruning is enabled
        if not self.disable_pruning:
            predictor_list = [PredictorLGSoft(embed_dim, self.num_tubelet) for _ in range(len(pruning_loc))]
            self.score_predictor = nn.ModuleList(predictor_list)
        else:
            self.score_predictor = None

        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        if self.motion_loss:
            selected_tubelets, top_indices, motion_scores, x = self.patch_embed(x, x_rev=x.flip(dims=[2]))
        else:
            x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        init_n = self.patch_embed.num_patches
        preserve_index = torch.arange(init_n, dtype=torch.long, device=x.device).reshape(1, -1, 1).expand(B, init_n, 1)
        
        if self.disable_pruning:
            # Process all blocks without pruning - keep all patches
            for i, blk in enumerate(self.blocks):
                x = blk(x)  # No policy parameter needed
            out_pred_prob = []
        else:
            # Original pruning logic
            p_count = 0
            out_pred_prob = []
            prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
            policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
            spars_losses = []
            motion_loss = 0.0
            for i, blk in enumerate(self.blocks):
                if i in self.pruning_loc:
                    spatial_x = x
                    pred_score, current_mask, threshold = self.score_predictor[p_count](spatial_x, prev_decision)
                    if self.training:
                        soft_decision = current_mask

                        ################ Motion preference loss computation ################
                        if self.motion_loss: 
                        #! How to improve: random without motion, collapse with motion ?
                            # motion_preference_loss = -torch.sum(soft_decision * motion_scores.unsqueeze(-1)) 
                            # motion_loss += motion_preference_loss

                            # Trying new approach: turn the motion into a distribution and let it match the one from current mask / soft decisions. 
                            p_keep = current_mask.squeeze(-1)   # (B,N)
                            p_keep = p_keep / (p_keep.sum(dim=-1, keepdim=True) + 1e-6) # normalize per sample 
                            m = motion_scores.detach()
                            m = (m - m.mean(-1, keepdim=True)) / (m.std(-1, keepdim=True) + 1e-6)
                            p_m = F.softmax(m / 0.3, dim=-1)
                            L_motion = F.kl_div((p_keep + 1e-8).log(), p_m, reduction="batchmean")
                            motion_loss += L_motion
                        ###########################################################
                        out_pred_prob.append(rearrange(soft_decision.reshape(B, init_n), 'b (t n) -> b t n', t=self.num_tubelet))
                        policy = soft_decision
                        x = blk(x, policy=policy)
                        prev_decision = soft_decision
                    else:
                        now_policy = current_mask
                        now_policy = now_policy.repeat(1,1,x.shape[2])
                        x = blk(x*now_policy)
                        prev_decision = current_mask
                        
                    p_count += 1
                else:
                    if self.training:
                        x = blk(x, policy)
                    else:
                        x = blk(x)
        x = self.norm_recon(x)
        x = self.head_recon(x)
        motion_loss = self.motion_loss_weight* motion_loss.mean() if (self.motion_loss and self.training) else 0  
        if self.training and self.motion_loss:
            return x, out_pred_prob, preserve_index, motion_loss
        else:
            return x, out_pred_prob, preserve_index

