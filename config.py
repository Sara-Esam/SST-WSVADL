"""
Simplified configuration management for SST-WSVADL
"""
import argparse
import os


class Config:
    """Configuration class for SST-WSVADL training"""
    
    def __init__(self, args):
        # Paths
        self.output_path = args.output_path
        self.model_path = args.model_path
        self.root_dir = args.root_dir
        self.video_root = args.video_root
        self.test_file = args.test_file
        self.feature_root = getattr(args, 'feature_root', None)
        self.pretrained_path = getattr(args, 'pretrained_path', None)
        
        # Dataset
        self.dataset = args.dataset
        self.modal = getattr(args, 'modal', 'rgb')
        self.num_segments = args.num_segments
        self.len_feature = args.len_feature
        
        # Training
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epochs = args.num_epochs
        self.seed = args.seed
        self.lr = eval(args.lr) if isinstance(args.lr, str) else args.lr
        
        # Model
        self.resize = args.resize
        self.patch_size = args.patch_size
        self.num_tubelet = args.num_tubelet
        self.segment_length = args.segment_length
        self.token_ratio = args.token_ratio
        
        # Features
        self.cross_attention = getattr(args, 'cross_attention', False)
        self.pretrained_point = getattr(args, 'pretrained_point', False)
        self.disable_pruning = getattr(args, 'disable_pruning', False)
        self.i3d = getattr(args, 'i3d', False)
        self.xdviolence_random_sampling = getattr(args, 'xdviolence_random_sampling', False)
        self.motion_loss = getattr(args, 'motion_loss', False)
        self.motion_loss_weight = getattr(args, 'motion_loss_weight', 0.01)
        self.motion_aware_type = getattr(args, 'motion_aware_type', 'time-reversal')
        
        # Random top
        self.random_top = getattr(args, 'random_top', False)
        self.multi_k = getattr(args, 'multi_k', False)
        self.k = getattr(args, 'k', 1)
        self.second_topk = getattr(args, 'second_topk', False)
        self.spatial_from_tokens = getattr(args, 'spatial_from_tokens', False)
        self.snippet_agg = getattr(args, 'snippet_agg', 'mean')
        self.data_variant = getattr(args, 'data_variant', '')
        self.video_resolution = getattr(args, 'video_resolution', '128')
        self.dtfe_pretrained = getattr(args, 'dtfe_pretrained', '')
        self.dtfe_freeze = getattr(args, 'dtfe_freeze', False)
        self.snippet_drop = getattr(args, 'snippet_drop', 8)
        self.spatial_pool = getattr(args, 'spatial_pool', 'topk')
        self.pool_window = getattr(args, 'pool_window', 3)
        self.contrast_loss_weight = getattr(args, 'contrast_loss_weight', 0.0)
        self.contrast_margin = getattr(args, 'contrast_margin', 0.3)
        self.contrast_bottom_k = getattr(args, 'contrast_bottom_k', 16)
        self.sparsity_loss_weight = getattr(args, 'sparsity_loss_weight', 0.0)
        self.flip_equivariance_weight = getattr(args, 'flip_equivariance_weight', 0.0)
        
        # Debug
        self.debug = getattr(args, 'debug', False)
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"Config({attrs})"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SST-WSVADL: Two-stage Video Anomaly Detection')
    
    # ==================== Paths and Directories ====================
    parser.add_argument('--output_path', type=str, default='outputs/')
    parser.add_argument('--root_dir', type=str, default='outputs/')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--video_root', type=str, default='')
    parser.add_argument('--feature_root', type=str, default='')
    parser.add_argument('--test_file', type=str, default='frame_label/ucf_gt_videomaev2.npy')
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')

    # ==================== Dataset Configuration ====================
    parser.add_argument('--dataset', type=str, default='ucf')
    parser.add_argument('--modal', type=str, default='rgb', choices=["rgb", "flow", "both"])
    parser.add_argument('--num_segments', type=int, default=200)
    parser.add_argument('--len_feature', type=int, default=1408)
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--i3d', action='store_true', help='Use i3d features')
    parser.add_argument('--video_resolution', type=str, default='128', choices=['128', 'native'],
                        help='UCF: read frames from the 128x128 copies (default) or the native-resolution '
                             'videos (use with --resize 224 224)')
    parser.add_argument('--dtfe_pretrained', type=str, default='',
                        help='Initialise DTFE from a VideoMAEv2 ViT-B checkpoint, e.g. '
                             '/path/to/VideoMAEv2/checkpoints/vit_b_k710_dl_from_giant.pth')
    parser.add_argument('--dtfe_freeze', action='store_true',
                        help='With --dtfe_pretrained: freeze the DTFE patch embedding and blocks')
    parser.add_argument('--data_variant', type=str, default='',
                        help="UCF only: e.g. 'noletterbox' reads all_videos_noletterbox_resized128x128/ "
                             "and videos_videomaev2_noletterbox/ instead of the original folders")
    parser.add_argument('--clip_features', action='store_true')
    parser.add_argument('--xdviolence_random_sampling', action='store_true', help='Use random sampling for xdviolence')

    # ==================== Training Configuration ====================
    parser.add_argument('--lr', type=str, default='[0.0001]*4000', help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=2022, help='random seed (-1 for no manual seed)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrained_point', action='store_true')

    # ==================== Model Architecture ====================
    parser.add_argument('--resize', type=int, nargs=2, default=[240, 320], help='Resize frames to (H, W)')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_tubelet', type=int, default=2)
    parser.add_argument('--segment_length', type=int, default=16)
    parser.add_argument('--token_ratio', type=float, nargs=3, default=[0.7, 0.5, 0.3])
    parser.add_argument('--disable_pruning', action='store_true')
    parser.add_argument('--soft_pruning', action='store_true', help='Use soft pruning')

    # ==================== Model Features/Flags ====================
    parser.add_argument('--cross_attention', action='store_true')
    parser.add_argument('--random_top', action='store_true', help='Randomly select a snippet for each sample')
    parser.add_argument('--multi_k', action='store_true', help='Use multi-k')
    parser.add_argument('--k', type=int, default=3, help='Number of snippets to select')
    parser.add_argument('--second_topk', action='store_true', help='Use second topk snippet')
    parser.add_argument('--rgb_thermal_fusion', action='store_true', help='Use rgb-thermal fusion')

    # ==================== Motion-related Features ====================
    parser.add_argument('--motion_loss', action='store_true', help='Use motion loss in STP model')
    parser.add_argument('--motion_loss_weight', type=float, default=0.01, help='Motion loss weight')
    parser.add_argument('--motion_aware_type', type=str, default='time-reversal', help='Motion aware type')
    parser.add_argument('--spatial_from_tokens', action='store_true',
                        help='Score DTFE 768-d tokens with the spatial head instead of reconstructed RGB tubelets')
    parser.add_argument('--snippet_agg', type=str, default='mean',
                        choices=['mean', 'drop_ends', 'subsample', 'reverse', 'combine', 'random'],
                        help='How to map a variable-length snippet sequence onto --num_segments during training. '
                             'random: pick one method independently each time a sample is drawn')
    parser.add_argument('--snippet_drop', type=int, default=8,
                        help='Snippets dropped from each end when snippet_agg is drop_ends or combine')

    # ==================== Spatial Localization Losses ====================
    parser.add_argument('--spatial_pool', type=str, default='topk', choices=['topk', 'window'],
                        help='Patch-branch MIL pooling: top t//16+1 patches, or the best '
                             '--pool_window x --pool_window region of the patch grid')
    parser.add_argument('--pool_window', type=int, default=3)
    parser.add_argument('--contrast_loss_weight', type=float, default=0.0,
                        help='Weight of the within-frame contrast hinge on anomalous clips (0 = off)')
    parser.add_argument('--contrast_margin', type=float, default=0.3,
                        help='Required gap between the best region and the lowest patches')
    parser.add_argument('--contrast_bottom_k', type=int, default=16,
                        help='Lowest-scoring patches per frame used as the contrast floor')
    parser.add_argument('--sparsity_loss_weight', type=float, default=0.0,
                        help='Weight of the mean patch score on anomalous clips (0 = off)')
    parser.add_argument('--flip_equivariance_weight', type=float, default=0.0,
                        help='Weight of |map(flip(clip)) - flip(map(clip))|; penalizes heads that '
                             'score patch positions instead of content (0 = off, costs a second '
                             'DTFE forward pass)')

    # ==================== Experimental/Other ====================
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exp_num', type=str, default=None, help='Experiment number')
    
    return parser.parse_args()


