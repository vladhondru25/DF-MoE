from torch.utils.data import DataLoader
import torchvision.ops as ops
from collections.abc import Callable
from datasets import load_from_disk, Dataset, Features, Value, List
from tqdm import tqdm
import pickle
import os
import numpy as np
from PIL import Image
import torch
import time
from dataset.utils import align_audio_to_frames_by_fps, get_fps_opencv
import copy
from datasets import load_from_disk, load_dataset
from typing import Union
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, DINOv3ViTImageProcessorFast
from einops import rearrange
import glob
import random
class MavosFeatureDataset:
    def __init__(self,
                 data_features_path: str,
                 ):

       
        samples = {}
        video_paths = []
        languages = []
        generative_methods = []
        input_dir = "/home/biodeep/alin/BiodeepDetection/interim_outputs2"
        for language in ["english"]:
            list_videos = glob.glob(f"{input_dir}/{language}/*/*")

            random.shuffle(list_videos)
            for video in list_videos:
                video_dir = os.path.join(input_dir, video)
                video_path = video_dir.split("/")[-3:]
                video_path[-1]+=".mp4"
                # video_paths.append("/".join(video_path))
                # languages.append(language)
                # generative_methods.append(video_path[-2])
                data = pickle.load(open(os.path.join(video_dir, "data.pkl"), "rb"))
                for identity_frame in data:
                    identity = identity_frame.split("_")[0]
                    rPPG = data[identity_frame]['rPPG'][0]
                    if rPPG is not None:
                        if '/'.join(video_path) in samples:
                            if identity in samples['/'.join(video_path)]:
                                samples['/'.join(video_path)][identity].append(rPPG)
                            else:
                                samples['/'.join(video_path)][identity] = [rPPG]
                        else:
                            samples["/".join(video_path)] = {identity:[rPPG]}

        # rppg = torch.Tensor(np.array([sample[1] for sample in samples]))

        # self.samples = rearrange(rppg, 'b c d->(b c) d')
        self.true_layer = torch.nn.Linear(512, 224)

        self.features_data = load_from_disk(data_features_path, keep_in_memory=False)#.filter(lambda sample: sample['language']=='arabic').filter(lambda sample: sample['generative_method']=='liveportrait').filter(lambda sample: sample['video_path']=="arabic/liveportrait/nBVWOmPTk-4_90_1--SzhK7Q-pY0U_1_6_511.mp4")
        print(self.features_data)
        # video_paths = set([sample[0] for sample in samples])
        inputs = []
        targets = []
        dataset_size = len(self.features_data)
        batch_size=5000
        print(len(set(self.features_data['video_path'])), dataset_size)
        for j in tqdm(range(0, dataset_size, batch_size), desc="Processing dataset"):
            start_index = j
            end_index = min(j+batch_size, dataset_size)
            batch = self.features_data.select(range(start_index, end_index))
            video_paths_batch = batch['video_path']
            for i, entry in enumerate(video_paths_batch):
                if  entry in samples and str(batch[i]['identity']) in samples[entry] and np.array(batch[i]['rPPG']).max() > np.array(batch[i]['rPPG']).min():
                    usable_rppg =  np.array([b for b in batch[i]['rPPG'] if np.array(b).max() > np.array(b).min()])
                    current_inputs = np.array(samples[entry][str(batch[i]['identity'])])
                    if current_inputs.shape[0] == usable_rppg.shape[0] and current_inputs.shape[1] == usable_rppg.shape[1]:
                        inputs.extend(current_inputs)
                        targets.extend(usable_rppg)
        print(np.array(inputs).shape)
        self.inputs = rearrange(np.array(inputs), 'b c d->(b c) d')
        self.targets = rearrange(np.array(targets), 'b c d->(b c) d')


    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def stack_sequence(sequence, sequence_length):

        # print(key)
        data = sequence
        # print(key, data)
        data = np.stack(data, axis=0)
        # try:
            # print(len(data))
        T = data.shape[0]
        # except:
        #     print(key, data.shape, data)
        #     exit()
        feat_shape = data.shape[1:]
        if T >= sequence_length:
            data = data[:sequence_length]
            padding_mask = np.zeros(sequence_length, dtype=bool) # True indicates that the respective position in not allowed to attend.
        elif T < sequence_length:
            pad_shape = (sequence_length - T,) + feat_shape
            pad = np.zeros(pad_shape, dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)
            padding_mask = np.zeros(sequence_length, dtype=bool)
            padding_mask[T:] = True # True indicates that the respective position in not allowed to attend.

        return data
    
    def __getitem__(self, idx):
        
        # frames = self.features_data[idx]['frames'][:].transpose(1, -1).transpose(1, 2)
        # frames =  self.stack_sequence(frames, 11)
        # output_frames = []
        # for frame in frames:
        #     output_frames.append(self.transform(Image.fromarray(frame)))
        # output_frames = torch.stack(output_frames)
        # # position = random.randint()
        # rPPG = self.features_data[idx]['rPPG']
        # # print(np.array(rPPG).shape)
        # return output_frames, np.array(rPPG[1]) 
        # with torch.no_grad():
        #     sample = self.samples[idx]
        #     gt = self.true_layer(sample[None, :])
        # return sample, gt.squeeze()
        return self.inputs[idx], self.targets[idx]

