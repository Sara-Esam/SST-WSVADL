"""
PyTorch Dataset for loading video segments using decord.

Example usage:
    dataset = VideoSegmentDataset(
        video_paths=[...],
        labels=[...],
        segment_length=48,  # number of frames
        sample_strategy='middle',
        resize=(112, 112),
        frame_rate=1,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for video_tensor, label in loader:
        # video_tensor: [B, C, T, H, W]
        ...
"""
import torch
from torch.utils.data import Dataset
import decord
from decord import VideoReader
import numpy as np
import random
import torchvision.transforms as T
import os
import sys

# Add parent directory to path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from utils.utils import aggregate_snippets, pick_snippet_agg


def _mp4_has_moov_atom(path, chunk_size=1024 * 1024):
    """Return True if an MP4/MOV file contains a moov atom.

    Truncated recordings often have only mdat and fail with 'moov atom not found'.
    The atom is typically at the start (faststart) or the end of the file.
    """
    try:
        size = os.path.getsize(path)
        if size < 8:
            return False
        with open(path, 'rb') as f:
            if b'moov' in f.read(min(chunk_size, size)):
                return True
            if size > chunk_size:
                f.seek(max(0, size - chunk_size))
                if b'moov' in f.read(chunk_size):
                    return True
        return False
    except OSError:
        return False


def _is_readable_video(path):
    """Return True if the video file exists and is not a truncated MP4.

    Training opens these files with decord VideoReader. A missing file or an
    MP4 without a moov atom (incomplete copy) will crash the training loop.
    """
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mp4', '.mov', '.m4v', '.m4a'):
        return _mp4_has_moov_atom(path)
    return os.path.getsize(path) > 0


def _pom_feature_to_video_path(feature_path, mode):
    if mode == "Train":
        if 'initial_data_recordings' in feature_path:
            return feature_path.replace('pom_dataset_videomaev2/', 'pom_dataset/all_mp4_resized340x620/').replace('.npy', '.mp4')
        if 'ruesel_' in feature_path:
            return feature_path.replace('ruesel_videomaev2/', 'ruesel_anomaly/').replace('.npy', '.mp4')
        return feature_path.replace('all_videomaev2/', 'all_videos/all/').replace('.npy', '.mp4')
    return feature_path.replace('pom_dataset_videomaev2/', 'pom_dataset/all_mp4_resized340x620/').replace('.npy', '.mp4')


def _filter_unreadable_videos(vid_list, v_list, mode):
    """Drop feature/video pairs whose mp4 is missing or unreadable. Keep lists aligned."""
    if not v_list:
        return vid_list, v_list
    kept_vid, kept_v, skipped = [], [], []
    print(f"[PoM {mode}] Checking {len(v_list)} videos for missing/corrupt files...")
    for feat, vid in zip(vid_list, v_list):
        vpath = vid[0]
        if _is_readable_video(vpath):
            kept_vid.append(feat)
            kept_v.append(vid)
        else:
            skipped.append(vpath)
    if skipped:
        print(f"[PoM {mode}] Skipping {len(skipped)}/{len(v_list)} unreadable videos:")
        for p in skipped:
            print(f"  {p}")
    else:
        print(f"[PoM {mode}] All {len(v_list)} videos are readable.")
    print(f"[PoM {mode}] Kept {len(kept_vid)} samples.")
    return kept_vid, kept_v


class UCFCrime(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, i3d=False,
                 snippet_agg='mean', snippet_drop=8, data_variant='', video_resolution='128'):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.snippet_agg = snippet_agg
        self.snippet_drop = snippet_drop
        split_path = os.path.join('list','UCF_{}.list'.format(self.mode)) if not i3d else os.path.join('list','UCF_{}_i3d.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        self.i3d = i3d
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[:800]
            elif is_normal is False:
                self.vid_list = self.vid_list[800:]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list=[]
        if data_variant:
            # e.g. 'noletterbox': all_videos_noletterbox_resized128x128/ + videos_videomaev2_noletterbox/
            if self.i3d:
                raise ValueError("data_variant is only available for VideoMAEv2 features")
            self.vid_list = [[path[0].replace('all_videos_resized128x128', f'all_videos_{data_variant}_resized128x128')]
                             for path in self.vid_list]
        if video_resolution == 'native':
            # Frames are decoded from the native-resolution copies (resized in training by --resize).
            self.v_list = [[path[0].replace('_resized128x128/', '/')] for path in self.vid_list]
        else:
            self.v_list = self.vid_list
        video_dir = f'Videos/Videos/all_videos_{data_variant}_resized128x128' if data_variant else 'Videos/Videos/all_videos_resized128x128'
        feature_dir = f'videos_videomaev2_{data_variant}/' if data_variant else 'videos_videomaev2/'
        if self.i3d:
            self.vid_list = [] if len(self.vid_list) == 0 else [[path[0].replace('Videos/Videos/all_videos_resized128x128', 'online_features/all_combined').replace('.mp4', '.npy')] for path in self.vid_list]
        else:
            self.vid_list = [] if len(self.vid_list) == 0 else [[path[0].replace(video_dir, feature_dir).replace('.mp4', '.npy')] for path in self.vid_list]

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data,label,(name, index, length) = self.get_data(index)
            return data,label, (name, index, length)

    def get_data(self, index):
        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)
        length = video_feature.shape[0]
        if "Normal" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        
        if self.mode == "Train":
            if len(video_feature.shape) > 2 and self.i3d:
                random_idx = np.random.randint(0, video_feature.shape[0])
                video_feature = video_feature[random_idx]
            video_feature = aggregate_snippets(
                video_feature, self.num_segments, self.snippet_agg, self.snippet_drop
            )
        
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label, (name, index, length) 


class XDViolence(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, xdviolence_random_sampling=False,
                 i3d=False, snippet_agg='mean', snippet_drop=8):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.xdviolence_random_sampling = xdviolence_random_sampling
        self.i3d = i3d
        self.snippet_agg = snippet_agg
        self.snippet_drop = snippet_drop
        split_path = os.path.join('list','XD_{}.list'.format(self.mode)) if not self.i3d else os.path.join('list','XD_{}_i3d.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":

            if is_normal is True:
                self.vid_list = self.vid_list[:2047] 
            elif is_normal is False:
                self.vid_list = self.vid_list[2047:]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list=[]
        if self.i3d:
            self.v_list = [] if len(self.vid_list) == 0 else [[path[0].replace('xd_i3d', 'all_videos').replace('.npy', '.mp4')] for path in self.vid_list]
        else:
            self.v_list = [] if len(self.vid_list) == 0 else [[path[0].replace('xd_videomaev2', 'all_videos').replace('.npy', '.mp4')] for path in self.vid_list]
 
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data,label,(name, index, length, sampled_indices) = self.get_data(index)
            return data,label, (name, index, length, sampled_indices)

    def get_data(self, index):
        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split(".npy")[0]
        if 'label_A' in name:
            label = 0
        else:
            label = 1
        # print(vid_info)
        video_feature = np.load(vid_info).astype(np.float32)
        length = video_feature.shape[0] #! 16 is the snippet length
        if self.mode == "Train":
            if len(video_feature.shape) > 2 and self.i3d:
                random_idx = np.random.randint(0, video_feature.shape[0])
                video_feature = video_feature[random_idx]
            if self.xdviolence_random_sampling:
                new_feature = np.zeros((self.num_segments,self.len_feature)).astype(np.float32)
                sample_index = utils.random_perturb(video_feature.shape[0], self.num_segments)
                sampled_indices = [sample_index[i] for i in range(len(sample_index))]
                for i in range(len(sample_index)-1):
                    if sample_index[i] == sample_index[i+1]:
                        new_feature[i,:] = video_feature[sample_index[i],:]
                    else:
                        new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                video_feature = new_feature
                sampled_indices = sample_index
            else:
                video_feature = aggregate_snippets(
                    video_feature, self.num_segments, self.snippet_agg, self.snippet_drop
                )
                sampled_indices = np.linspace(0, length, self.num_segments + 1, dtype=int)
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label, (name, index, length, np.array(sampled_indices, dtype=np.int32))


class MSAD(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, i3d=False,
                 snippet_agg='mean', snippet_drop=8):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.snippet_agg = snippet_agg
        self.snippet_drop = snippet_drop
        split_path = os.path.join('list','MSAD_{}.list'.format(self.mode)) if not i3d else os.path.join('list','MSAD_{}_i3d.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        self.i3d = i3d
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[:360]
            elif is_normal is False:
                self.vid_list = self.vid_list[360:]
            else:
                assert (is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list=[]
        if self.i3d:
            self.v_list = [] if len(self.vid_list) == 0 else [[path[0].replace('msad_i3d', 'all_videos').replace('.npy', '.mp4')] for path in self.vid_list]
        else:
            self.v_list = [] if len(self.vid_list) == 0 else [[path[0].replace('msad_videomaev2', 'all_videos').replace('.npy', '.mp4')] for path in self.vid_list]
    
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data,label,(name, index, length) = self.get_data(index)
            return data,label, (name, index, length)

    def get_data(self, index):
        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split(".npy")[0]
        if 'normal' in name:
            label = 0
        else:
            label = 1
        video_feature = np.load(vid_info).astype(np.float32)
        length = video_feature.shape[0]
        if self.mode == "Train":
            if len(video_feature.shape) > 2:
                random_idx = np.random.randint(0, video_feature.shape[0])
                video_feature = video_feature[random_idx]
            video_feature = aggregate_snippets(
                video_feature, self.num_segments, self.snippet_agg, self.snippet_drop
            )
        
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label, (name, index, length)

class PoM(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, rgb_thermal_fusion=False,
                 snippet_agg='mean', snippet_drop=8):
        self.rgb_thermal_fusion = rgb_thermal_fusion
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.snippet_agg = snippet_agg
        self.snippet_drop = snippet_drop
        split_path = os.path.join('list/pom/','pom_{}_deployment.list'.format(self.mode)) #!
        split_thermal_path = os.path.join('list/pom/','pom_thermal_{}.list'.format(self.mode))
        split_file = open(split_path, 'r')
        split_thermal_file = open(split_thermal_path, 'r')
        self.vid_list = []
        self.vid_list_thermal = []
        # RGB 
        for line in split_file:
            self.vid_list.append(line.split())
        # Thermal 
        for line in split_thermal_file:
            self.vid_list_thermal.append(line.split())
        split_file.close()
        split_thermal_file.close()
        
        self.vid_dict_rgb = {}
        self.vid_dict_thermal = {}
        if self.rgb_thermal_fusion:
            for v_rgb, v_thermal in zip(self.vid_list, self.vid_list_thermal):
                thermal_info_list = v_thermal[0].split("/")[-1].split("_")
                m_split = thermal_info_list[4].split('m')[0]+'m'
                thermal_key = thermal_info_list[0] + '_' \
                            + thermal_info_list[1] + '_' \
                            + thermal_info_list[3] + '_' \
                            + m_split + '_' + thermal_info_list[5] + '_' + thermal_info_list[6]
                self.vid_dict_thermal[thermal_key] = v_thermal 

                rgb_info_list = v_rgb[0].split("/")[-1].split("_")
                m_split = rgb_info_list[4].split('m')[0]+'m'
                rgb_key = rgb_info_list[0] + '_' \
                            + rgb_info_list[1] + '_' \
                            + rgb_info_list[3] + '_' \
                            + m_split + '_' + rgb_info_list[5] + '_' + rgb_info_list[6]
                self.vid_dict_rgb[rgb_key] = v_rgb 

                assert thermal_key == rgb_key, "Thermal and RGB keys do not match"

        # self.vid_dict_thermal = {vid_info[0].split("/")[-1].split("_")[0]+'': vid_info[1] for v_rgb, v_thermal in zip(self.vid_list, self.vid_list_thermal)}
        if self.mode == "Train":
            if 'pom' in split_path:
                if is_normal is True:
                    self.vid_list = self.vid_list[:1747] 
                elif is_normal is False:
                    self.vid_list = self.vid_list[1747:]  
                else:
                    assert (is_normal == None)
                    print("Please sure is_normal=[True/False]")
                    self.vid_list=[]
            elif 'PoM' in split_path:
                if is_normal is True:
                    self.vid_list = self.vid_list[:39] # until exp6 was 38 ! which is incorrect! 
                elif is_normal is False:
                    self.vid_list = self.vid_list[39:]  
                else:
                    assert (is_normal == None)
                    print("Please sure is_normal=[True/False]")
                    self.vid_list=[]
        
        if is_normal is True:
            # shuffle the vid_list and select only the first 250
            np.random.shuffle(self.vid_list)
            self.vid_list = self.vid_list[:250]

        if len(self.vid_list) == 0:
            self.v_list = []
        else:
            self.v_list = [[_pom_feature_to_video_path(path[0], self.mode)] for path in self.vid_list]
            if self.mode == "Train":
                split_name = self.mode
                if is_normal is True:
                    split_name = 'Train-normal'
                elif is_normal is False:
                    split_name = 'Train-abnormal'
                # self.vid_list, self.v_list = _filter_unreadable_videos(
                #     self.vid_list, self.v_list, split_name
                # )
    
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        
        if self.mode == "Test":
            data,label,name = self.get_data(index)
            return data,label,name
        else:
            data, label, (name, index, length) = self.get_data(index)
            return data, label, (name, index, length)  

    def get_data(self, index):
        if self.rgb_thermal_fusion:
            th_vid_info = self.vid_list_thermal[index][0]
            th_video_feature = np.load(th_vid_info).astype(np.float32)
            th_video_feature = th_video_feature[:-1] if (self.mode == "Train" and th_video_feature.shape[0] %16 !=0 ) else th_video_feature

        vid_info = self.vid_list[index][0]  
        name = vid_info.split("/")[-1].split(".")[0]  
        video_feature = np.load(vid_info).astype(np.float32)
        video_feature = video_feature[:-1] if (self.mode == "Train" and video_feature.shape[0] %16 !=0 ) else video_feature
        length = video_feature.shape[0] #! 16 is the snippet length
        # if "normal" in vid_info.split("/")[-1]: #! WOW this has been incorrect for a long time ! 
        #     label = 0
        if "abnormal" not in vid_info.split("/")[-1] and "anomaly" not in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            random_idx = np.random.randint(0, video_feature.shape[0])
            if len(video_feature.shape) > 2:
                video_feature = video_feature[random_idx]
            if self.rgb_thermal_fusion:
                if len(th_video_feature.shape) > 2:
                    th_video_feature = th_video_feature[random_idx]
                method = pick_snippet_agg(self.snippet_agg)
                video_feature = aggregate_snippets(
                    video_feature, self.num_segments, method, self.snippet_drop
                )
                th_video_feature = aggregate_snippets(
                    th_video_feature, self.num_segments, method, self.snippet_drop
                )
            else:
                video_feature = aggregate_snippets(
                    video_feature, self.num_segments, self.snippet_agg, self.snippet_drop
                )
        
        
        if self.mode == "Test":
            if self.rgb_thermal_fusion:
                return [video_feature, th_video_feature], label, name
            else:
                return video_feature, label, name
        else:
            if self.rgb_thermal_fusion:
                return [video_feature, th_video_feature], label, (name, index, length)
            else:
                return video_feature, label, (name, index, length)  