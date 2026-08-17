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


class UCFCrime(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, i3d=False):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
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
        self.v_list = self.vid_list
        if self.i3d:
            self.vid_list = [] if len(self.vid_list) == 0 else [[path[0].replace('Videos/Videos/all_videos', 'online_features/all_combined').replace('.mp4', '.npy')] for path in self.vid_list]
        else:
            self.vid_list = [] if len(self.vid_list) == 0 else [[path[0].replace('Videos/Videos/all_videos', 'videos_videomaev2').replace('.mp4', '.npy')] for path in self.vid_list]

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

            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = int)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            video_feature = new_feat
        
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label, (name, index, length) 


class XDViolence(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, xdviolence_random_sampling=False,
                 i3d=False):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.xdviolence_random_sampling = xdviolence_random_sampling
        self.i3d = i3d
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
                new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
                r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = int)
                for i in range(self.num_segments):
                    if r[i] != r[i+1]:
                        new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                    else:
                        new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
                video_feature = new_feat
                sampled_indices = r
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label, (name, index, length, np.array(sampled_indices, dtype=np.int32))


class MSAD(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, i3d=False):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
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
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = int)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            video_feature = new_feat
        
        if self.mode == "Test":
            return video_feature, label, name      
        else:
            return video_feature, label, (name, index, length)
