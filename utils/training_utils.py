"""
Training utilities for two-stage VAD training
"""
import torch
import numpy as np
from decord import VideoReader
import sys
import os
import torchvision.transforms as T

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import map_topk_snippets_to_frames, extract_frames_from_video


def two_stage_train(urdmu_model, stpvad_model, stp_model,
                   normal_loader, abnormal_loader, normal_dataset, abnormal_dataset,
                   optimizer, criterion, stpvad_criterion, config):
    """
    Two-stage training function
    
    Stage 1: Temporal-level anomaly detection using UR-DMU
    Stage 2: Patch-level anomaly detection using STPVAD
    """
    urdmu_model.train()
    urdmu_model.flag = "Train"
    stpvad_model.train()
    stpvad_model.flag = "Train"
    stp_model.train()
    stp_model.flag = "Train"
    
    # Get data
    ninput, nlabel, nvr = next(normal_loader)
    ainput, alabel, avr = next(abnormal_loader)
    ninput, nlabel = ninput.cuda(), nlabel.cuda()
    ainput, alabel = ainput.cuda(), alabel.cuda()
    
    # Handle different dataset return formats
    if config.dataset == 'xdviolence':
        nnames, nidx, nlen, normal_sampled_indices = nvr
        anames, aidx, alen, anomaly_sampled_indices = avr
    else:
        nnames, nidx, nlen = nvr
        anames, aidx, alen = avr
    
    # Extract video readers
    nvrs = [VideoReader(normal_dataset.v_list[i][0]) for i in nidx]
    avrs = [VideoReader(abnormal_dataset.v_list[i][0]) for i in aidx]
    
    # Stage 1: Temporal-level detection (UR-DMU)
    data = torch.cat((ninput, ainput), 0)
    result = urdmu_model(data)
    
    # Get top-k snippet indices
    A_index = result['A_index']  # Abnormal top snippets
    N_index = result['N_index']  # Normal top snippets
    
    # instead of always picking the top 1 snippet, choose a random one. 
    if config.random_top:
        k_indx = np.random.randint(0, A_index.shape[1])
        A_index, N_index = A_index[:,k_indx], N_index[:,k_indx] 
    elif config.multi_k and config.k> 1:   
        A_idx_k = A_index[:, :config.k].contiguous()   # [B, k]
        N_idx_k = N_index[:, :config.k].contiguous()   # [B, k]
    elif config.second_topk:
        A_index, N_index = A_index[:,1], N_index[:,1] # TOP 2 snippet for each sample
    else: # Top-1
        A_index, N_index = A_index[:,0], N_index[:,0] # TOP 1 snippet for each sample
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(config.resize),
        T.ToTensor(),
    ])
    if config.multi_k:
        A_index = A_idx_k
        N_index = N_idx_k
        # Extract video tensors: [B,k,C,T,H,W] and per-segment indices [B,k,T]
        stpvad_results = []
        for i in range(config.k):
            # Map each per-sample k indices to k frame ranges
            n_frame_ranges = map_topk_snippets_to_frames(N_idx_k[:,i].cpu().numpy(), nlen, new_seg_size=config.segment_length) 
            a_frame_ranges = map_topk_snippets_to_frames(A_idx_k[:,i].cpu().numpy(), alen, new_seg_size=config.segment_length)
            normal_video_tensor = extract_frames_from_video(nvrs, n_frame_ranges, transform=transform)
            anomaly_video_tensor = extract_frames_from_video(avrs, a_frame_ranges, transform=transform)
            normal_video_tensor = normal_video_tensor.cuda(non_blocking=True)
            anomaly_video_tensor = anomaly_video_tensor.cuda(non_blocking=True)

            # DTFE Model 
            if config.motion_loss:
                stp_features_normal, s_normal, preserve_index_normal, motion_loss = stp_model(normal_video_tensor)
                stp_features_anomaly, s_anomaly, preserve_index_anomaly, motion_loss = stp_model(anomaly_video_tensor)
            else:
                stp_features_normal, s_normal, preserve_index_normal = stp_model(normal_video_tensor)
                stp_features_anomaly, s_anomaly, preserve_index_anomaly = stp_model(anomaly_video_tensor)
                
            # Concatenate normal+abnormal on batch axis for the patch VAD
            data_stpvad = torch.cat([stp_features_normal, stp_features_anomaly], dim=0)           # [2B,k,Tp,Dp]
            preserve_index_stpvad = torch.cat([preserve_index_normal, preserve_index_anomaly], 0)  # [2B,k,Tp]
            label_stpvad = torch.cat((nlabel, alabel), 0)     # [2B]
            
            # Stage 2: Spatial Detection
            if config.cross_attention:
                # mask out the non-topk snippets
                gather_idx = torch.cat([A_index, N_index], dim=0)
                masked_snippet_features = result['snippet_features'][:, gather_idx]
                stpvad_result = stpvad_model(data_stpvad, snippet_features=masked_snippet_features[:,:,i,:])
            else:
                stpvad_result = stpvad_model(data_stpvad) 
            
            stpvad_results.append(stpvad_result)
    else:
        anomaly_topk_snippets = A_index.cpu().numpy()  # indices of top-k abnormal snippets
        normal_topk_snippets = N_index.cpu().numpy()    # indices of top-k normal snippets
        
        # get the frame indices
        if config.dataset == 'xdviolence' and config.xdviolence_random_sampling:
            anomaly_frame_indices = anomaly_sampled_indices
            normal_frame_indices = normal_sampled_indices
        else:
            anomaly_frame_indices = map_topk_snippets_to_frames(anomaly_topk_snippets, alen, new_seg_size = config.segment_length)
            normal_frame_indices = map_topk_snippets_to_frames(normal_topk_snippets, nlen, new_seg_size = config.segment_length)
        
        anomaly_video_tensor = extract_frames_from_video([avrs, alen], anomaly_frame_indices, resize=config.resize)
        normal_video_tensor = extract_frames_from_video([nvrs, nlen], normal_frame_indices, resize=config.resize)
        
        anomaly_video_tensor = anomaly_video_tensor.float().cuda()
        normal_video_tensor = normal_video_tensor.float().cuda()

        # DTFE Model
        if config.motion_loss:
            stp_features_normal, s_normal, preserve_index_normal, motion_loss = stp_model(normal_video_tensor)
            stp_features_anomaly, s_anomaly, preserve_index_anomaly, motion_loss = stp_model(anomaly_video_tensor)
        else:
            stp_features_normal, s_normal, preserve_index_normal = stp_model(normal_video_tensor)
            stp_features_anomaly, s_anomaly, preserve_index_anomaly = stp_model(anomaly_video_tensor)

        data_stpvad = torch.cat((stp_features_normal, 
                                stp_features_anomaly), 0)
        label_stpvad = torch.cat((nlabel, alabel), 0) 

        # Concatenate preserve indices
        preserve_index_stpvad = torch.cat((preserve_index_normal, preserve_index_anomaly), 0).cuda()
        
        # Stage 2: Spatial Detection
        if config.cross_attention:
            # mask out the non-topk snippets
            gather_idx = torch.cat([A_index, N_index], dim=0)
            masked_snippet_features = result['snippet_features'][:, gather_idx]
            stpvad_result = stpvad_model(data_stpvad, snippet_features=masked_snippet_features)
        else:
            stpvad_result = stpvad_model(data_stpvad)
    
    urdmu_cost, urdmu_loss = criterion(result, torch.cat((nlabel, alabel), 0))
    
    if config.multi_k:
        stpvad_costs = []
        for i in range(config.k):
            c, stpvad_loss = stpvad_criterion(stpvad_results[i], label_stpvad, 
                                               patch_features=data_stpvad, 
                                               preserve_index=preserve_index_stpvad,
                                               num_tubelet=config.num_tubelet)
            stpvad_costs.append(c)
        stpvad_cost = torch.stack(stpvad_costs).mean()
    else:
        stpvad_cost, stpvad_loss = stpvad_criterion(stpvad_result, label_stpvad, 
                                               patch_features=data_stpvad, 
                                               preserve_index=preserve_index_stpvad,
                                               num_tubelet=config.num_tubelet)
    
    # Accumulate the cost and loss
    if config.motion_loss:
        total_cost = urdmu_cost + stpvad_cost + motion_loss
    else:
        total_cost = urdmu_cost + stpvad_cost
    
    # Backward and update the parameters
    optimizer.zero_grad()
    total_cost.backward()
    optimizer.step()
    
    if config.motion_loss:
        return urdmu_cost, stpvad_cost, urdmu_loss, stpvad_loss, motion_loss
    else:
        return urdmu_cost, stpvad_cost, urdmu_loss, stpvad_loss

































    # # Patch-level detection
    # # Map snippet indices to frame indices
    # if config.dataset == 'xdviolence' and hasattr(config, 'xdviolence_random_sampling') and config.xdviolence_random_sampling:
    #     # For XDViolence with random sampling, sampled_indices maps segment index to original snippet index
    #     # A_index/N_index are indices into the 200 segments, we need to map to frame indices
    #     anomaly_frame_indices = []
    #     normal_frame_indices = []
        
    #     A_idx_np = A_index.cpu().numpy() if torch.is_tensor(A_index) else A_index
    #     N_idx_np = N_index.cpu().numpy() if torch.is_tensor(N_index) else N_index
        
    #     for i in range(len(A_idx_np)):
    #         # topk_idx is index into the 200 segments (0-199)
    #         topk_idx = int(A_idx_np[i])
    #         # Map to original snippet index
    #         if topk_idx < len(anomaly_sampled_indices[i]):
    #             original_snippet_idx = int(anomaly_sampled_indices[i][topk_idx])
    #             # Convert snippet index to frame range (each snippet is 16 frames)
    #             snippet_start = original_snippet_idx * 16
    #             snippet_end = snippet_start + config.segment_length
    #         else:
    #             # Fallback: use last available snippet
    #             original_snippet_idx = int(anomaly_sampled_indices[i][-1])
    #             snippet_start = original_snippet_idx * 16
    #             snippet_end = snippet_start + config.segment_length
    #         anomaly_frame_indices.append([snippet_start, snippet_end])
        
    #     for i in range(len(N_idx_np)):
    #         topk_idx = int(N_idx_np[i])
    #         if topk_idx < len(normal_sampled_indices[i]):
    #             original_snippet_idx = int(normal_sampled_indices[i][topk_idx])
    #             snippet_start = original_snippet_idx * 16
    #             snippet_end = snippet_start + config.segment_length
    #         else:
    #             original_snippet_idx = int(normal_sampled_indices[i][-1])
    #             snippet_start = original_snippet_idx * 16
    #             snippet_end = snippet_start + config.segment_length
    #         normal_frame_indices.append([snippet_start, snippet_end])
    # else:
    #     anomaly_frame_indices = map_topk_snippets_to_frames(
    #         A_index.cpu().numpy() if torch.is_tensor(A_index) else A_index, alen, 
    #         snippet_len=16, new_seg_size=config.segment_length, 
    #         video_segment_length=config.num_segments
    #     )
    #     normal_frame_indices = map_topk_snippets_to_frames(
    #         N_index.cpu().numpy() if torch.is_tensor(N_index) else N_index, nlen,
    #         snippet_len=16, new_seg_size=config.segment_length,
    #         video_segment_length=config.num_segments
    #     )
    
    # # Extract video frames
    # anomaly_video_tensor = extract_frames_from_video(
    #     [avrs, alen], anomaly_frame_indices, resize=config.resize
    # )
    # normal_video_tensor = extract_frames_from_video(
    #     [nvrs, nlen], normal_frame_indices, resize=config.resize
    # )
    
    # anomaly_video_tensor = anomaly_video_tensor.float().cuda()
    # normal_video_tensor = normal_video_tensor.float().cuda()
    
    # # Process through STPrivacy model
    # # STPrivacy expects [B, C, T, H, W] input
    # # extract_frames_from_video returns [B, C, T, H, W], so it should be correct
    # # Values from ToTensor() are already in [0, 1] range
    
    # # Process through STPrivacy
    # # STPrivacy returns: (features, pred_prob, preserve_index) or with additional losses
    # # For simplified version, we ignore pred_prob (second return value)
    # stp_features_normal, _, preserve_index_normal = stp_model(normal_video_tensor)
    # stp_features_anomaly, _, preserve_index_anomaly = stp_model(anomaly_video_tensor)
    
    # # Concatenate for batch processing
    # data_stpvad = torch.cat((stp_features_normal, stp_features_anomaly), 0)
    # preserve_index_stpvad = torch.cat((preserve_index_normal, preserve_index_anomaly), 0).cuda()
    # label_stpvad = torch.cat((nlabel, alabel), 0)
    
    # # Apply cross-attention if enabled
    # if hasattr(config, 'cross_attention') and config.cross_attention:
    #     gather_idx = torch.cat([A_index, N_index], dim=0)
    #     masked_snippet_features = result['snippet_features'][:, gather_idx]
    #     stpvad_result = stpvad_model(data_stpvad, snippet_features=masked_snippet_features)
    # else:
    #     stpvad_result = stpvad_model(data_stpvad)
    
    # # Compute losses
    # urdmu_cost, urdmu_loss = criterion(result, torch.cat((nlabel, alabel), 0))
    # stpvad_cost, stpvad_loss = stpvad_criterion(
    #     stpvad_result, label_stpvad,
    #     patch_features=data_stpvad,
    #     preserve_index=preserve_index_stpvad,
    #     num_tubelet=config.num_tubelet
    # )
    
    # # Total loss
    # total_cost = urdmu_cost + stpvad_cost
    
    # # Backward pass
    # optimizer.zero_grad()
    # total_cost.backward()
    # optimizer.step()
    
    # return urdmu_cost, stpvad_cost, urdmu_loss, stpvad_loss

