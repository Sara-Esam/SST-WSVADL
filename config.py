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
        self.enhanced_loss = getattr(args, 'enhanced_loss', False)
        self.pretrained_point = getattr(args, 'pretrained_point', False)
        self.disable_pruning = getattr(args, 'disable_pruning', False)
        self.i3d = getattr(args, 'i3d', False)
        self.xdviolence_random_sampling = getattr(args, 'xdviolence_random_sampling', False)

        # Random top
        self.random_top = getattr(args, 'random_top', False)
        self.multi_k = getattr(args, 'multi_k', False)
        self.k = getattr(args, 'k', 1)
        self.second_topk = getattr(args, 'second_topk', False)
        
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
    parser.add_argument('--supervised', action='store_true', help='Use supervised training')
    parser.add_argument('--alternate_training', action='store_true',
                       help='Alternate between VAD and action recognition training')

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
    parser.add_argument('--co_attention', action='store_true', help='Use co-attention')
    parser.add_argument('--with_residual', action='store_true', help='Add residual after cross-attention')
    parser.add_argument('--with_har', action='store_true')
    parser.add_argument('--random_top', action='store_true', help='Randomly select a snippet for each sample')
    parser.add_argument('--multi_k', action='store_true', help='Use multi-k')
    parser.add_argument('--k', type=int, default=3, help='Number of snippets to select')
    parser.add_argument('--second_topk', action='store_true', help='Use second topk snippet')
    parser.add_argument('--rgb_thermal_fusion', action='store_true', help='Use rgb-thermal fusion')

    # ==================== Motion-related Features ====================
    parser.add_argument('--motion_keep_ratio', type=float, default=0.75, help='Motion keep ratio')
    parser.add_argument('--motion_based_pruning', action='store_true', help='Use motion-based pruning')
    parser.add_argument('--motion_filtering', action='store_true', help='Use motion filtering')
    parser.add_argument('--pure_motion_based_pruning', action='store_true', help='Use pure motion-based pruning')
    parser.add_argument('--modified_pure_motion_based_pruning', action='store_true', help='modification to make it purely motion based, no MLPs or so')
    parser.add_argument('--motion_based_urdmu', action='store_true', help='Use motion-based pruning for urdmu in the patch branch')
    parser.add_argument('--motion_loss', action='store_true', help='Use motion loss in STP model')
    parser.add_argument('--motion_loss_weight', type=float, default=0.01, help='Motion loss weight')
    parser.add_argument('--motion_aware_type', type=str, default='time-reversal', help='Motion aware type')

    # ==================== Loss Configuration ====================
    parser.add_argument('--enhanced_loss', action='store_true')
    parser.add_argument('--sparse_loss', action='store_true', help='Use sparse loss')
    parser.add_argument('--adjacency_loss', action='store_true', help='Use adjacency loss')
    parser.add_argument('--adjacency_loss_weight', type=float, default=0.01, help='Adjacency loss weight')

    # ==================== Experimental/Other ====================
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exp_num', type=str, default=None, help='Experiment number')
    
    return parser.parse_args()


