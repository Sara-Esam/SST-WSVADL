"""
SST-WSVADL: Simplified Two-Stage Video Anomaly Detection Training Script

This is a simplified, refactored version of the two-stage VAD training pipeline.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Local imports
from models import WSAD_temporal, WSVAD_spatial, DTFEModel
from models.pretrained import load_videomae_into_dtfe, freeze_dtfe_backbone
from losses.vad_loss import AD_Loss, SpatialAD_Loss
from data.datasets import UCFCrime, XDViolence, MSAD, PoM
from eval.test import test
from utils.utils import set_seed, save_best_record
from utils.training_utils import two_stage_train
from config import Config, parse_args


def _spatial_input_size(config):
    if config.spatial_from_tokens:
        return 768
    return config.patch_size * config.patch_size * config.num_tubelet * 3


def _dataset_kwargs(config, **extra):
    kwargs = dict(
        root_dir=None, modal=config.modal,
        num_segments=config.num_segments, len_feature=config.len_feature,
        snippet_agg=config.snippet_agg, snippet_drop=config.snippet_drop,
    )
    kwargs.update(extra)
    return kwargs


def main():
    """Main training function"""
    args = parse_args()
    config = Config(args)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # Set random seed
    if config.seed >= 0:
        set_seed(config.seed)
    
    # Load datasets
    if config.dataset == 'ucf':
        ds_kw = _dataset_kwargs(config, i3d=config.i3d, data_variant=config.data_variant,
                                video_resolution=config.video_resolution)
        normal_dataset = UCFCrime(mode='Train', is_normal=True, **ds_kw)
        abnormal_dataset = UCFCrime(mode='Train', is_normal=False, **ds_kw)
        test_dataset = UCFCrime(mode='Test', is_normal=None, **ds_kw)
    elif config.dataset == 'xdviolence':
        ds_kw = _dataset_kwargs(
            config, i3d=config.i3d,
            xdviolence_random_sampling=getattr(config, 'xdviolence_random_sampling', False),
        )
        normal_dataset = XDViolence(mode='Train', is_normal=True, **ds_kw)
        abnormal_dataset = XDViolence(mode='Train', is_normal=False, **ds_kw)
        test_dataset = XDViolence(mode='Test', is_normal=None, **ds_kw)
    elif config.dataset == 'msad':
        ds_kw = _dataset_kwargs(config, i3d=config.i3d)
        normal_dataset = MSAD(mode='Train', is_normal=True, **ds_kw)
        abnormal_dataset = MSAD(mode='Train', is_normal=False, **ds_kw)
        test_dataset = MSAD(mode='Test', is_normal=None, **ds_kw)
    elif config.dataset == 'pom':
        ds_kw = _dataset_kwargs(config)
        normal_dataset = PoM(mode='Train', is_normal=True, **ds_kw)
        abnormal_dataset = PoM(mode='Train', is_normal=False, **ds_kw)
        test_dataset = PoM(mode='Test', is_normal=None, **ds_kw)
    else:
        raise ValueError(f"Dataset {config.dataset} not supported. Supported datasets: ucf, xdviolence, msad")
    
    normal_loader = DataLoader(normal_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    abnormal_loader = DataLoader(abnormal_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    
    # Initialize models
    temporal_urdmu = WSAD_temporal(input_size=config.len_feature, flag='Train', a_nums=60, n_nums=60).to(device)
    spatial_urdmu = WSVAD_spatial(
        input_size=_spatial_input_size(config),
        flag='Train', a_nums=60, n_nums=60
    ).to(device)
    dtfe_model = DTFEModel(
        img_size=config.resize, patch_size=config.patch_size,
        tubelet_size=config.num_tubelet, all_frames=config.segment_length,
        in_chans=3, num_classes=1, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4., pruning_loc=[3, 6, 9], token_ratio=config.token_ratio,
        distill=False, disable_pruning=config.disable_pruning, 
        motion_loss=config.motion_loss, motion_loss_weight=config.motion_loss_weight,
        motion_aware_type=config.motion_aware_type,
        spatial_from_tokens=config.spatial_from_tokens,
    ).to(device)
    if config.dtfe_pretrained:
        load_videomae_into_dtfe(dtfe_model, config.dtfe_pretrained)
        if config.dtfe_freeze:
            freeze_dtfe_backbone(dtfe_model)
    elif config.dtfe_freeze:
        raise ValueError("--dtfe_freeze needs --dtfe_pretrained")

    # Optimizer
    params = list(temporal_urdmu.parameters()) + list(spatial_urdmu.parameters())
    if dtfe_model is not None:
        params += [p for p in dtfe_model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(
        params,
        lr=config.lr[0] if isinstance(config.lr, list) else config.lr,
        betas=(0.9, 0.999),
        weight_decay=0.00005
    )
    
    # Loss functions
    temporal_vad_criterion = AD_Loss()
    grid_shape = (
        config.segment_length // config.num_tubelet,
        config.resize[0] // config.patch_size,
        config.resize[1] // config.patch_size,
    )
    spatial_vad_criterion = SpatialAD_Loss(
        grid_shape,
        pool=config.spatial_pool,
        window=config.pool_window,
        contrast_weight=config.contrast_loss_weight,
        contrast_margin=config.contrast_margin,
        bottom_k=config.contrast_bottom_k,
        sparsity_weight=config.sparsity_loss_weight,
    )
    
    # Load pretrained model if specified
    if config.pretrained_point and os.path.exists(config.pretrained_path):
        print(f"Loading pretrained model from {config.pretrained_path}")
        temporal_urdmu.load_state_dict(torch.load(config.pretrained_path), strict=False)
    
    # Training loop
    test_info = {"step": [], "auc": [], "ap": [], "ac": []}
    best_auc = 0.0
    
    normal_loader_iter = iter(normal_loader)
    abnormal_loader_iter = iter(abnormal_loader)
    
    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Dataset: {config.dataset}")
    print(f"Spatial input: {'DTFE tokens (768-d)' if config.spatial_from_tokens else 'reconstructed RGB tubelets'}")
    print(f"Snippet aggregation: {config.snippet_agg}")
    print(f"Data variant: {config.data_variant or 'original'} ({config.video_resolution} videos) | seed {config.seed}")
    print(f"DTFE init: {config.dtfe_pretrained or 'random'}{' (frozen backbone)' if config.dtfe_freeze else ''}")
    print(f"Spatial pooling: {config.spatial_pool}"
          + (f" ({config.pool_window}x{config.pool_window})" if config.spatial_pool == 'window' else "")
          + f" | contrast w={config.contrast_loss_weight} m={config.contrast_margin}"
          + f" | sparsity w={config.sparsity_loss_weight}"
          + f" | flip equivariance w={config.flip_equivariance_weight}")
    
    for step in tqdm(range(1, config.num_epochs + 1), total=config.num_epochs):
        # Reset iterators
        if (step - 1) % len(normal_loader) == 0:
            normal_loader_iter = iter(normal_loader)
        if (step - 1) % len(abnormal_loader) == 0:
            abnormal_loader_iter = iter(abnormal_loader)
        
        # Update learning rate
        if isinstance(config.lr, list) and step > 1 and len(config.lr) > step - 1:
            if config.lr[step - 1] != config.lr[step - 2]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr[step - 1]
        
        # Training step
        losses = two_stage_train(
                temporal_urdmu, spatial_urdmu, dtfe_model,
                normal_loader_iter, abnormal_loader_iter,
                normal_dataset, abnormal_dataset,
                optimizer, temporal_vad_criterion, spatial_vad_criterion, config
            )

        temporal_urdmu_cost, spatial_urdmu_cost = losses[:2]
        # Print progress
        if step % 10 == 0:
            print(f'\nStep {step}:')
            print(f'  Temporal URDMU Loss: {temporal_urdmu_cost.item():.4f}')
            print(f'  Spatial URDMU Loss: {spatial_urdmu_cost.item():.4f}')
            spatial_terms = losses[3]
            for key in ('contrast', 'sparsity', 'flip_equivariance'):
                if key in spatial_terms:
                    print(f'  Spatial {key}: {spatial_terms[key].item():.4f}')
        
        # Evaluation
        if step % 10 == 0 and step > 20:
            try:
                test(temporal_urdmu, config, None, test_loader, test_info, step,
                     model_file=None, test_file=args.test_file, i3d=config.i3d)
                
                if len(test_info["auc"]) > 0 and test_info["auc"][-1] > best_auc:
                    best_auc = test_info["auc"][-1]
                    save_best_record(test_info, os.path.join(config.output_path, f"best_record_{config.seed}.txt"))
                    torch.save(temporal_urdmu.state_dict(),
                              os.path.join(config.model_path, f"urdmu_model_{config.seed}.pkl"))
                    if dtfe_model is not None:
                        torch.save(dtfe_model.state_dict(),
                                  os.path.join(config.model_path, f"dtfe_model_{config.seed}.pkl"))
                    torch.save(spatial_urdmu.state_dict(),
                              os.path.join(config.model_path, f"spatial_urdmu_model_{config.seed}.pkl"))
            except Exception as e:
                print(f"Error in evaluation: {e}")
    
    # Checkpoints above are picked by temporal AUC, which says nothing about the
    # spatial head; keep the final step too, under the same file names.
    last_dir = config.model_path.rstrip("/") + "_last"
    os.makedirs(last_dir, exist_ok=True)
    torch.save(temporal_urdmu.state_dict(), os.path.join(last_dir, f"urdmu_model_{config.seed}.pkl"))
    torch.save(dtfe_model.state_dict(), os.path.join(last_dir, f"dtfe_model_{config.seed}.pkl"))
    torch.save(spatial_urdmu.state_dict(), os.path.join(last_dir, f"spatial_urdmu_model_{config.seed}.pkl"))
    print(f"Final-step checkpoints saved to {last_dir}")

    print(f"Training completed! Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
