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

class VideoSegmentDataset(Dataset):
    def __init__(self, segment_length=3000, 
                        sample_strategy='middle', resize=(240, 320), 
                        frame_rate=1, transform=None, is_normal=None, split='Train'):
        """
        Args:
            segment_length (int): Number of frames to sample per segment.
            sample_strategy (str): 'middle' or 'random'.
            resize (tuple): (H, W) to resize frames.
            frame_rate (int): Sample every Nth frame.
            transform: Optional torchvision transform to apply to frames.
        """
        self.segment_length = segment_length
        self.sample_strategy = sample_strategy
        self.resize = resize
        self.frame_rate = frame_rate
        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.ToTensor(),
        ])
        self.is_normal = is_normal
        self.split = split
        split_path = os.path.join('list','UCF_{}.list'.format(self.split))
        split_file = open(split_path, 'r')
        
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.split == 'Train':
            if is_normal is True:
                self.vid_list = self.vid_list[:800]
            elif is_normal is False:
                self.vid_list = self.vid_list[800:]
            else:
                assert (self.is_normal == None)
                print("Please sure is_normal=[True/False]")
                self.vid_list=[]
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        vid_info = self.vid_list[idx][0]  
        name = vid_info.split("/")[-1].split("_x264")[0]
        if "Normal" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        if self.split == 'Train':
            vr = VideoReader(vid_info)
            total_frames = len(vr)
            # Compute indices to sample
            effective_length = self.segment_length * self.frame_rate
            if total_frames < effective_length:
                # Pad by looping video if too short
                indices = np.arange(0, total_frames, self.frame_rate)
                indices = np.pad(indices, (0, self.segment_length - len(indices)), mode='wrap')
            else:
                if self.sample_strategy == 'middle':
                    start = max((total_frames - effective_length) // 2, 0)
                elif self.sample_strategy == 'random':
                    start = random.randint(0, total_frames - effective_length)
                else:
                    raise ValueError('Unknown sample_strategy')
                indices = np.arange(start, start + effective_length, self.frame_rate)
            # Read and process frames
            frames = vr.get_batch(indices).asnumpy().astype(np.float32)  # [segment_length, H, W, C]
            frames = [self.transform(frame) for frame in frames]  # list of [C, H, W]
            video_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
            # video_tensor = video_tensor.permute(0, 2, 3, 1)
        else:
            vr = VideoReader(vid_info)
            video_feature = vr.get_batch(np.arange(0, len(vr), self.frame_rate)).asnumpy() 
            video_feature = video_feature.astype(np.float32)
            video_tensor = torch.from_numpy(video_feature)
            video_tensor = video_tensor.permute(3, 0, 1, 2)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)
        


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
        
        # self.vrs_list = [VideoReader(vid_info[0]) for vid_info in self.v_list]

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
        length = video_feature.shape[0] #! 16 is the snippet length
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
                self.vid_list = self.vid_list[:2047] #! change if the missing samples are returned. 
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
        # self.v_list = self.vid_list
        # self.vid_list = [] if len(self.vid_list) == 0 else [[path[0]] for path in self.vid_list]
    
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
        # self.v_list = self.vid_list
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
        length = video_feature.shape[0] #! 16 is the snippet length
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


class UCF101(Dataset):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, vad_mode=True, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.vad_mode = vad_mode
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list','UCF101_{}.list'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        self.label_map = {}
        label_map_file = open(os.path.join('frame_label','ucf101_cls_ind.list'), 'r')
        for line in label_map_file:
            self.label_map[line.split()[1]] = int(line.split()[0])
        label_map_file.close()
        if self.mode == "Train":
            self.vid_list = self.vid_list[0:]
        self.v_list = [] if len(self.vid_list) == 0 else [[path[0].replace('ucf101_videomaev2', 'all_videos').replace('.npy', '')] for path in self.vid_list]
    
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
        
        if self.vad_mode:
            name = vid_info.split("/")[-1].split(".npy")[0]
            label = 0 
            video_feature = np.load(vid_info).astype(np.float32)
            length = video_feature.shape[0] #! 16 is the snippet length
            if self.mode == "Train":
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
        
        else:
            name = vid_info.split("/")[-1].split(".avi.npy")[0]
            categ_label = name.split("_")[1]
            label = self.label_map[categ_label] - 1
            # get the video! 
            vid_info = vid_info.replace('.avi.npy', '.avi').replace('ucf101_videomaev2', 'all_videos')
            video_frames = VideoReader(vid_info)
            length = len(video_frames)
            video_frames = video_frames.get_batch(np.arange(0, len(video_frames))).asnumpy()
            # select random 200 frames. If the video is less than 200, pad with frames from the beginning
            if length < 16:
                # repeat the video until 200 frames
                video_frames = np.concatenate([video_frames]*16)
                video_frames = video_frames[:16]
            else:
                video_frames = video_frames[np.random.choice(length, 16, replace=False)]
            video_frames = video_frames.astype(np.float32)
            
            video_frames = torch.from_numpy(video_frames)
            video_frames = video_frames.permute(3, 0, 1, 2)
            # resize the video frames to 128x128
            video_frames = T.Resize((128, 128))(video_frames)
            return video_frames, label, (name, index, len(video_frames))
        


class PoM(Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None, rgb_thermal_fusion=False):
        self.rgb_thermal_fusion = rgb_thermal_fusion
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list','pom_{}.list'.format(self.mode)) #!
        split_thermal_path = os.path.join('list','pom_thermal_{}.list'.format(self.mode))
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
                    self.vid_list = self.vid_list[:412] 
                elif is_normal is False:
                    self.vid_list = self.vid_list[412:]  
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
        self.v_list = [] if len(self.vid_list) == 0 else [[path[0].replace('pom_dataset_videomaev2/', 'pom_dataset/all_mp4_resized340x620/').replace('.npy', '.mp4')] for path in self.vid_list]
            
        print('Training list is ... ', len(self.vid_list))
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
        if "normal" in vid_info.split("/")[-1]: #! WOW this has been incorrect for a long time ! 
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            random_idx = np.random.randint(0, video_feature.shape[0])
            if len(video_feature.shape) > 2:
                video_feature = video_feature[random_idx]
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype = int)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i,:] = np.mean(video_feature[r[i]:r[i+1],:], 0)
                else:
                    new_feat[i:i+1,:] = video_feature[r[i]:r[i]+1,:]
            video_feature = new_feat
            
            if self.rgb_thermal_fusion:
                if len(th_video_feature.shape) > 2:
                    th_video_feature = th_video_feature[random_idx]
                new_feat_th = np.zeros((self.num_segments, th_video_feature.shape[1])).astype(np.float32)
                r = np.linspace(0, len(th_video_feature), self.num_segments + 1, dtype = int)
                for i in range(self.num_segments):
                    if r[i] != r[i+1]:
                        new_feat_th[i,:] = np.mean(th_video_feature[r[i]:r[i+1],:], 0)
                    else:
                        new_feat_th[i:i+1,:] = th_video_feature[r[i]:r[i]+1,:]
                th_video_feature = new_feat_th
        
        
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

