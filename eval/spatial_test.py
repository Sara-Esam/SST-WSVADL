import sys
from tkinter import N
sys.path.append('..')
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
from dataset_loader import *
from torch.utils.data import DataLoader
import os
import decord
from model import *
from utils import extract_clip_video_features
import os
from dataset_loader import *
from tqdm import tqdm
from stprivacy.stprivacy import STPrivacy, STPrivacySoft
from stprivacy.stprivacy_motion_alternatives import STPrivacyMotionBased, STPrivacyMotionFiltered, STPrivacyModifiedPureMotionBased
from config import *
from video_segment_loader import VideoSegmentDataset, UCFCrime , MSAD, XDViolence
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import time
import argparse
import json 

warnings.filterwarnings("ignore")


def get_video_frames(video_path, resize):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (resize[1], resize[0]))
        frames.append(frame)
    cap.release()
    return frames


def evaluate_iou_scores(net_urdmu, net_stpvad, stp_model, test_loader, video_root=None, 
                                    config=None, subset=None, iou_threshold=0.5, 
                                    optimal_threshold=None, bbox_iou=False, one_video_flag=False):

    # Set models to evaluation mode
    net_urdmu.eval()
    net_stpvad.eval()
    stp_model.eval()
    net_urdmu.flag = "Test"
    net_stpvad.flag = "Test"
    stp_model.flag = "Test"
    
    load_iter = iter(test_loader)
    
    frame_gt = np.load(f"frame_label/{config.dataset}_gt_patches.npy")  # patch level ground truth (spatial detection)
    frame_scores_gt = np.load(f"frame_label/{config.dataset}_gt.npy") # frame level ground truth (temporal detection)
       
    patch_predictions = []
    cls_label = []
    cls_pre = []
    normal_patches_predictions = [] #! for false alarm rate computation
    num_patches_per_frame = (128 // 16) * (128 // 16)
    iou_per_frame = []
    frames_count_per_video = []
    videos_names = []
    frame_scores_preds = []
    limit = subset if subset else len(test_loader.dataset)
    count=0
    print(f"Evaluating {limit} videos")
    for i in range(limit):          
        _data, _label, name = next(load_iter)
        _data = _data.cuda()
        _label = _label.cuda()
        cls_label.append(int(_label))
        
        # fixes for file names
        if config.dataset == 'ucf':
            video_path = os.path.join(video_root, name[0] + '_x264.mp4')
        elif config.dataset == 'xdviolence':
            video_path = os.path.join(video_root, name[0] + '.mp4')
            print(video_path)
        elif config.dataset == 'msad':
            video_path = os.path.join(video_root, name[0] + '.mp4')
        else:
            raise ValueError(f"Dataset {config.dataset} not supported")
        


        video_frames = get_video_frames(video_path, (128, 128))
        video = video_frames

        # accumulate the patch ground truth per video for the topk and bottomk metrics 
        frames_count_per_video.append( _data.shape[1]*16) #! double check that shape[1] is for #snippets
        videos_names.append(name[0])

        with torch.no_grad():
            urdmu_result = net_urdmu(_data)
        frame_predict = urdmu_result['frame']  # This contains snippet-level anomaly scores
        snippet_scores = frame_predict.cpu().numpy()[0]
        frame_scores_preds.extend(np.repeat(snippet_scores, 16))
        
        segments_to_process = list(range(len(snippet_scores))) # all

        video_patch_scores = []
        total_expected_patches = len(snippet_scores) * 16 * num_patches_per_frame
        for seg_idx in range(len(segments_to_process)):
            
            start, end = seg_idx * 16, min((seg_idx + 1) * 16, len(video))

            if (end - start)<16:
                print('padding < 16 segment !! double check the video length')
                pad_frames = [video[start]]*(16 - (end - start))
                video_segment = pad_frames + video[start:end]
            else:
                video_segment = video[start:end] 

            input_frames = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video_segment], axis=0)
            input_frames = input_frames.transpose(3, 0, 1, 2) # [C, T, H, W]
            input_frames = torch.from_numpy(input_frames).unsqueeze(0).float().cuda() #/ 255.0

            # Process through STP model (spatial feaure extraction)
            with torch.no_grad():
                stp_features, s, preserved_index = stp_model(input_frames)
                stpvad_result = net_stpvad(stp_features) 
            
            patch_scores = stpvad_result.get('frame', torch.zeros(1)).cpu().numpy()[0]

            full_patch_scores = np.zeros(num_patches_per_frame)
            patch_scores_dict = {}
            updated_indices = []
            for i, preserved_idx in enumerate(preserved_index[0]): # Loop over all the patch indices that were kept by the STP model for anomaly detection
                
                # if the patch_idx is > 64, then get the patch_idx - 64
                # We need to handle the case when preserved_idx is repeated. 
                # In that case, we need to take the one with the highest score. 
                if preserved_idx.item() >= 64:
                    patch_idx = preserved_idx.item() % 64
                else:
                    patch_idx = preserved_idx.item()
                full_patch_scores[patch_idx] =  patch_scores[i] # still we dont know if redundant and being overwritten.
                updated_indices.append(patch_idx)

            # patch_scores = full_patch_scores
            # update the patch scores in case of redundancy
            patch_scores_dict = {}
            for i, patch_idx in enumerate(updated_indices):
                if patch_idx in patch_scores_dict: #! correction step in case of redundancy
                    if patch_scores_dict[patch_idx] < patch_scores[i]:
                        patch_scores_dict[patch_idx] = patch_scores[i]
                
                else:
                    patch_scores_dict[patch_idx] = patch_scores[i]
            
            patch_scores = [patch_scores_dict[idx] if idx in patch_scores_dict else 0 for idx in range(64)]                    

            
            # Since a segment has 16 frames, we need to repeat the patch scores for each frame
            patch_scores = np.repeat(patch_scores, 16)
            video_patch_scores.extend(patch_scores)
        
        assert len(video_patch_scores) == total_expected_patches, f"Number of patches in a video does not match. Expected {total_expected_patches}, got {len(video_patch_scores)}"        
        
        patch_predictions.extend(video_patch_scores)
        normal_patches_predictions.extend(video_patch_scores)
        
        print(f'processed {count}/{limit} videos')
        count+=1
    


    frame_scores_preds_expanded = np.repeat(frame_scores_preds, 64)
    
    print('################################################################################')

    #!###################################### AUC & AP computation ##########################
    #!######################################################################################
    fpr, tpr, _ = roc_curve(frame_gt, patch_predictions*frame_scores_preds_expanded) #!
    auc_score = auc(fpr, tpr)

    optimal_patch_threshold = find_optimal_threshold_youden(frame_gt, patch_predictions)
    print(f"Optimal patch threshold: {optimal_patch_threshold:.4f}")
    
    precision, recall, th = precision_recall_curve(frame_gt, patch_predictions*frame_scores_preds_expanded) #!
    ap_score = auc(recall, precision)
    
    print(f"PAUC: {auc_score:.4f}, PAP: {ap_score:.4f}")

    #!###################################### Temporal IoU Computation #########################################
    #!#########################################################################################################
    
    # Get anomaly indices by frame_gt scores > 0.0
    previous_video_count = 0
    anomaly_only_patch_predictions = []
    anomaly_only_patch_ground_truth = []
    for i, (video_name, video_frames_count) in enumerate(zip(videos_names, frames_count_per_video)):
        if 'Normal' in video_name or 'normal' in video_name:
            previous_video_count += video_frames_count * 64
            continue
        start_idx = int(np.sum(frames_count_per_video[:i]) * 64)
        end_idx = int(start_idx + video_frames_count * 64)
        start_f_idx = int(np.sum(frames_count_per_video[:i])) 
        end_f_idx = int(start_f_idx + video_frames_count)
        
        # one video scores 
        f_patch_pr = patch_predictions[start_idx:end_idx]
        f_patch_gt = frame_gt[start_idx:end_idx]
        f_scores_gt = np.array(frame_scores_gt[start_f_idx:end_f_idx])
        for j in range(video_frames_count):
            if f_scores_gt[j] > 0.0:
                anomaly_only_patch_predictions.extend(f_patch_pr[j*64:(j+1)*64])
                anomaly_only_patch_ground_truth.extend(f_patch_gt[j*64:(j+1)*64])
        previous_video_count += video_frames_count * 64

    fpr, tpr, _ = roc_curve(anomaly_only_patch_ground_truth, anomaly_only_patch_predictions)
    anomaly_only_auc = auc(fpr, tpr)

    precision, recall, th = precision_recall_curve(anomaly_only_patch_ground_truth, anomaly_only_patch_predictions)
    anomaly_only_ap = auc(recall, precision)
    print(f"Anomaly only AUC: {anomaly_only_auc:.4f}, Anomaly only AP: {anomaly_only_ap:.4f}")
    print('################################################################################')
    

    print('Calculating iou metrics')
    max_patch_score = max(patch_predictions)
    min_patch_score = min(patch_predictions)
    for iou_threshold in [optimal_patch_threshold, 0.1, 0.5, 0.7]:
        cls_pre = [1 if p >= iou_threshold else 0 for p in patch_predictions]
        thresholded_patch_predictions = np.array(cls_pre)
        frame_gt = np.array(frame_gt)
        iou_per_video = []
        previous_video_count = 0

        for video_idx, (video_frames_count, video_name) in enumerate(zip(frames_count_per_video, videos_names)): #! move video by video
            if 'Normal' in video_name or 'normal' in video_name:
                #! sanity check if order is correct
                # start_idx = previous_video_count
                # end_idx = start_idx + video_frames_count*64
                # sum_f_gt_per_video = 0
                # for i in range(start_idx, end_idx, 64):
                #     sum_f_gt_per_video += np.sum(frame_gt[i:i+64])
                # assert sum_f_gt_per_video == 0, f'WARNING! gt patch sum is not 0 for a normal video {video_name} {sum_f_gt_per_video} !'
                previous_video_count += video_frames_count*64
                continue
            start_idx = previous_video_count
            end_idx = start_idx + video_frames_count*64
            
            video_ious = []
            video_pos = frames_count_per_video[:video_idx] if video_idx != 0 else 0 # no 64 patches per frame
            video_pos_idx = np.sum(video_pos)
            video_frame_scores_preds = np.array(frame_scores_preds[video_pos_idx:video_pos_idx + video_frames_count])
            # TODO: compute the TPAUC 
            debug_frame_predictions = []
            debug_frame_gt = []

            video_frame_scores_0_1 = video_frame_scores_preds # temporal scores
            # optimal threshold taken from ucf_infer.py 
            # this threshold is the 'best' threshold computed for the temporal detection.
            video_frame_scores_0_1[video_frame_scores_0_1 >= optimal_threshold] = 1
            video_frame_scores_0_1[video_frame_scores_0_1 < optimal_threshold] = 0

            if bbox_iou:
                for i in range(start_idx, end_idx, 64): #! move frame by frame
                    frame_predictions = thresholded_patch_predictions[i:i+64] 
                    debug_frame_predictions.append(frame_predictions)
                    debug_frame_gt.append(frame_gt[i:i+64])
                    
                    # Convert patches to bounding boxes
                    pred_bbox = patches_to_bbox(frame_predictions, patch_size=16)
                    gt_bbox = patches_to_bbox(frame_gt[i:i+64])

                    if np.sum(frame_gt[i:i+64]) == 0:
                        iou = -1
                    else:
                        # Compute IoU between bounding boxes
                        iou = compute_bbox_iou(pred_bbox, gt_bbox)
                    video_ious.append(iou)
                    iou_per_frame.append(iou)
                

            else:
                #! patch-based iou 
                # we only want anomaly frames
                allgts = frame_gt[start_idx:end_idx*64]
                if np.sum(allgts) == 0:
                    print(video_name)
                    print(allgts)
                
                for i in range(start_idx, end_idx, 64): #! move frame by frame
                    frame_patch_predictions = thresholded_patch_predictions[i:i+64]
                    f_preds = frame_patch_predictions #* 256
                    f_gt = frame_gt[i:i+64]#* 256
                    
                    iou = compute_iou(f_preds, f_gt, anomaly_only=True)
                    video_ious.append(iou) #! only for the anomaly frames
                    iou_per_frame.append(iou)
            
            video_ious = np.array(video_ious) # previously 

            temporal_ious = video_ious * video_frame_scores_0_1 
            temporal_ious = temporal_ious[temporal_ious >= 0] # remove the -ve values
            iou_per_video.extend(temporal_ious)
            previous_video_count += video_frames_count*64
            
        overall_iou = np.mean(iou_per_video)
        print(f"Overall IoU (%) @ {iou_threshold} (anomaly only): {overall_iou*100:.4f}")
    
    return {
        "overall_iou": overall_iou,
    }


def patches_to_bbox(thresholded_patches, patch_size=16, image_size=128, return_all=False):
    patches = np.array(thresholded_patches, dtype=np.uint8).flatten()
    num_side = image_size // patch_size
    grid = patches.reshape(num_side, num_side)
    if grid.sum() == 0:
        return [] if return_all else None

    patch_kernel = np.ones((patch_size, patch_size), dtype=np.uint8)
    mask = np.kron(grid, patch_kernel) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [] if return_all else None

    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1 = (x // patch_size) * patch_size
        y1 = (y // patch_size) * patch_size
        x2 = min(image_size, ((x + w + patch_size - 1) // patch_size) * patch_size)
        y2 = min(image_size, ((y + h + patch_size - 1) // patch_size) * patch_size)
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    if return_all:
        return bboxes

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def compute_bbox_iou(bbox1, bbox2, anomaly_only=False):
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value
    """

    
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0



def find_optimal_threshold_youden(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]


def extract_frames_from_video(video_path, segment, resize=128, num_segments=200):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        segment: Segment index to extract
        resize: Resize factor for the frames

    Returns:
        numpy array of shape (num_segments, resize, resize, 3)
    """
    # Load video with decord 
    video = decord.VideoReader(video_path)
    video_frames = video.get_batch([segment])
    video_frames = video_frames.asnumpy()
    video_frames = video_frames.transpose(0, 2, 3, 1)
    video_frames = video_frames.reshape(1, -1, resize, resize, 3)
    return video_frames

def compute_iou(preds, gt, anomaly_only=False):
    # preds and gt are arrays of shape (1, 64)
    # compute the iou between the preds and gt
    if anomaly_only:
        if np.sum(gt) == 0:
            iou = -1
        else:
            iou = np.sum(preds * gt) / (np.sum(preds + gt - preds * gt) + 1e-6)
        return iou

    if np.sum(preds) == 0 and np.sum(gt) == 0:
        iou = 1
    else:
        iou = np.sum(preds * gt) / (np.sum(preds + gt - preds * gt) + 1e-6)
    return iou


def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--output_path', type = str, default = 'outputs/')
    parser.add_argument('--root_dir', type = str, default = 'outputs/')
    parser.add_argument('--log_path', type = str, default = 'logs/')
    parser.add_argument('--modal', type = str, default = 'rgb',choices = ["rgb,flow,both"])
    parser.add_argument('--dataset', type = str, default = 'ucf')
    parser.add_argument('--model_path', type = str, default = 'models/')
    parser.add_argument('--lr', type = str, default = '[0.0001]*3000', help = 'learning rates for steps(list form)')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--num_segments', type = int, default = 200)
    parser.add_argument('--seed', type = int, default = 2022, help = 'random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type = str, default = "trans_{}.pkl".format(2022), help = 'the path of pre-trained model file')
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--resize', type=int, nargs=2, default=[128, 128], help='Resize frames to (H, W)')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_tubelet', type=int, default=8)
    parser.add_argument('--segment_length', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--len_feature', type=int, default=1408)
    parser.add_argument('--cross_attention', action = 'store_true')
    parser.add_argument('--enhanced_loss', action = 'store_true')
    parser.add_argument('--pretrained_point', action = 'store_true')
    parser.add_argument('--video_root', type=str, default='/projects/0/prjs1250/feature_extraction/Videos/Videos/all_videos_test_only')
    parser.add_argument('--disable_pruning', action = 'store_true')
    parser.add_argument("--eval_loc_flag", action = 'store_true')
    parser.add_argument("--anomaly_only_flag", action = 'store_true')
    parser.add_argument("--exp_num", type=str, default="14")
    parser.add_argument('--token_ratio', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--feature_root", type=str, default="/gpfs/scratch1/nodespecific/gcn11/sabdulaziz1.14107495/clip_frames_features/")
    parser.add_argument("--clip_features", action = 'store_true')
    parser.add_argument("--with_har", action = 'store_true')
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument('--alternate_training', action='store_true',
                       help='Alternate between VAD and action recognition training')
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--random_top", type=int, default=None)
    parser.add_argument("--tiou", action = 'store_true')
    parser.add_argument("--co_attention", action = 'store_true')
    parser.add_argument("--multi_k", action = 'store_true')
    parser.add_argument("--optimal_threshold", type=float, default=None)
    parser.add_argument("--with_residual", action = 'store_true')
    parser.add_argument("--supervised", action = 'store_true')
    parser.add_argument("--motion_keep_ratio", type=float, default=0.75)
    parser.add_argument("--motion_based_pruning", action = 'store_true')
    parser.add_argument("--motion_filtering", action = 'store_true')
    parser.add_argument("--pure_motion_based_pruning", action = 'store_true')
    parser.add_argument('--modified_pure_motion_based_pruning', action='store_true', help='Use modified pure motion-based pruning')
    parser.add_argument("--sparse_loss", action = 'store_true')
    parser.add_argument("--motion_based_urdmu", action = 'store_true')
    parser.add_argument("--motion_loss", action = 'store_true')
    parser.add_argument("--motion_loss_weight", type=float, default=0.01)
    parser.add_argument("--adjacency_loss", action = 'store_true')
    parser.add_argument("--adjacency_loss_weight", type=float, default=0.01)
    parser.add_argument("--xdviolence_random_sampling", action = 'store_true')
    parser.add_argument('--i3d', action = 'store_true')
    parser.add_argument("--soft_pruning", action = 'store_true')
    parser.add_argument("--random_topk", action = 'store_true')
    parser.add_argument("--second_topk", action = 'store_true')
    parser.add_argument("--bbox_iou", action = 'store_true')
    parser.add_argument("--use_bbox_gt", action = 'store_true')
    parser.add_argument("--rgb_thermal_fusion", action = 'store_true')
    parser.add_argument("--motion_aware_type", type=str, default='time-reversal')
    parser.add_argument("--mae_as_snippet_features", action='store_true', help='Use VideoMAE snippet features instead of features from urdmu classifier!!!')
    parser.add_argument("--depth_stpvadmodel", type=int, default=1, help='Depth of blocks in STPVAD model')
    parser.add_argument("--depth_stpmodel", type=int, default=12, help='Depth of blocks in STP model')
    parser.add_argument("--remove_bias", action = 'store_true')
    parser.add_argument("--compute_token_entropy", action = 'store_true')
    return init_args(parser.parse_args())



def init_args(args):    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args


# Example usage function
def main():
    from config import Config
    # add arguments to the train_args
    test_args = parse_args()

    print('Evaluating exp', test_args.exp_num)
    urdmu_model_path = f"./models/stpvad_exp{test_args.exp_num}/ucf_trans_2022.pkl"
    stpvad_model_path = f"./models/stpvad_exp{test_args.exp_num}/stpvad_model_2022.pkl"
    stp_model_path = f"./models/stpvad_exp{test_args.exp_num}/stp_model_2022.pkl"
    video_root = test_args.video_root
    if test_args.dataset == "msad":
        test_dataset = MSAD(root_dir = None, 
                                mode = 'Test', modal = 'rgb', 
                                num_segments = 200, len_feature = 1024, 
                                is_normal = None, i3d = test_args.i3d)
    elif test_args.dataset == "xdviolence":
        test_dataset = XDViolence(root_dir = None, 
                                mode = 'Test', modal = 'rgb', 
                                num_segments = 200, len_feature = 1024, 
                                is_normal = None, i3d = test_args.i3d)
    else:
        test_dataset = UCFCrime(root_dir = None, 
                                mode = 'Test', modal = 'rgb', 
                                num_segments = 200, len_feature = 1408, 
                                is_normal = None, i3d = test_args.i3d)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    # Initialize test_info
    test_info = {
        "step": [],
        "auc": [],
        "ap": [],
        "ac": []
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_args.seed = 2022
    config = Config(test_args)
    
    # load models 
    urdmu_model = WSAD(input_size=config.len_feature, flag='Train', a_nums=60, n_nums=60).to(device)
    stpvad_model = WSVAD_STP(input_size=config.patch_size*config.patch_size*config.num_tubelet*3, 
                            flag='Train', a_nums=60, n_nums=60,
                            depth=config.depth_stpvadmodel, 
                            mae_as_snippet_features=config.mae_as_snippet_features, 
                            remove_bias=config.remove_bias
                        ).to(device)
    if config.motion_based_pruning:
        stp_model = STPrivacyMotionBased(
            img_size=config.resize, patch_size=config.patch_size, tubelet_size=config.num_tubelet, all_frames=config.segment_length, in_chans=3,
            num_classes=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
            norm_layer=None, pruning_loc=[3, 6, 9], token_ratio=config.token_ratio, distill=False, 
            disable_pruning=config.disable_pruning, clip_features=config.clip_features,
            motion_keep_ratio=config.motion_keep_ratio
        ).to(device)
    elif config.motion_filtering:
        stp_model = STPrivacyMotionFiltered(
            img_size=config.resize, patch_size=config.patch_size, tubelet_size=config.num_tubelet, all_frames=config.segment_length, in_chans=3,
            num_classes=1, embed_dim=768, depth=config.depth_stpmodel, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
            norm_layer=None, pruning_loc=[3, 6, 9], token_ratio=config.token_ratio, distill=False, 
            disable_pruning=config.disable_pruning, clip_features=config.clip_features,
            motion_keep_ratio=config.motion_keep_ratio
        ).to(device)
    elif config.modified_pure_motion_based_pruning:
        stp_model = STPrivacyModifiedPureMotionBased(img_size=config.resize, patch_size=config.patch_size, tubelet_size=config.num_tubelet, all_frames=config.segment_length, in_chans=3,
                                            num_classes=1, embed_dim=768, depth=config.depth_stpmodel, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                            representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                                            norm_layer=None, pruning_loc=[3, 6, 9], token_ratio=config.token_ratio, distill=False, 
                                            disable_pruning=config.disable_pruning, clip_features=config.clip_features,
                                            motion_keep_ratio=config.motion_keep_ratio
                            ).to(device)

    
    elif config.soft_pruning:
        stp_model = STPrivacySoft(img_size=config.resize, patch_size=config.patch_size, tubelet_size=config.num_tubelet, all_frames=config.segment_length, in_chans=3,
                                            num_classes=1, embed_dim=768, depth=config.depth_stpmodel, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                            representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                                            norm_layer=None, pruning_loc=[3, 6, 9], token_ratio=config.token_ratio, distill=False, 
                                            disable_pruning=config.disable_pruning, clip_features=config.clip_features
                            ).to(device)
    else:
        stp_model = STPrivacy(
        img_size=config.resize, patch_size=config.patch_size, tubelet_size=config.num_tubelet, all_frames=config.segment_length, in_chans=3,
        num_classes=1, embed_dim=768, depth=config.depth_stpmodel, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
        norm_layer=None, pruning_loc=[3, 6, 9], token_ratio=config.token_ratio, distill=False, 
        disable_pruning=config.disable_pruning, motion_aware_type=config.motion_aware_type, compute_token_entropy=config.compute_token_entropy
    ).to(device)

    # load model weights
    print(f"Loading URDMU model from {urdmu_model_path}")
    print(f"Loading STPVAD model from {stpvad_model_path}")
    print(f"Loading STP model from {stp_model_path}")
    urdmu_model.load_state_dict(torch.load(urdmu_model_path), strict=False)
    stp_model.load_state_dict(torch.load(stp_model_path), strict=False)
    urdmu_model.eval()
    stp_model.eval()
    stpvad_model.eval()
    stp_model.training = False

    stpvad_model.load_state_dict(torch.load(stpvad_model_path), strict=False)
    try:
        stpvad_model.load_state_dict(torch.load(stpvad_model_path))
    except:
        print("Error loading STPVAD model, trying to fix it")
        stpvad_checkpoint = torch.load(stpvad_model_path, map_location=device)
        state_dict = stpvad_checkpoint['model'] if 'model' in stpvad_checkpoint else stpvad_checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("cross_attention.layers.0.", "cross_attention.")
            new_state_dict[name] = v
        stpvad_model.load_state_dict(new_state_dict, strict=True)
    
    results = evaluate_iou_scores(urdmu_model, 
                        stpvad_model, 
                        stp_model, 
                        test_loader, 
                        video_root=video_root,
                        config=config,
                        eval_loc_flag=test_args.eval_loc_flag,
                        subset=test_args.subset,
                        iou_threshold=test_args.iou_threshold,
                        optimal_threshold=test_args.optimal_threshold,
                        bbox_iou=test_args.bbox_iou,
                        )



if __name__ == "__main__":
    main() 
