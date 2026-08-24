from torch.utils.data import DataLoader
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
from datasets import load_from_disk
from typing import Union
class MavosFeatureDataset:

    def __init__(self, data_video_path: str,
                 data_features_path: str,
                 audio_features_path: str,
                 split: str,
                 features_to_keep: list[str],
                 sequence_length: int,
                 hop_length: Union[int,str],
                 frame_resolution: int,
                 transforms_dictionary: list[Callable] = None,
                 synchronize_audio_features: bool = False,
                 return_video_path: bool = False):

        self.video_data = load_from_disk(data_video_path, keep_in_memory=True).filter(lambda sample: sample['split'] == split)
        self.features_data = load_from_disk(data_features_path, keep_in_memory=False)#.select(range(2000))#.filter(lambda sample: generative_method in sample['video_path']).select(range(1000))#.select(range(20000, 30000))
        self.audio_features_data = load_from_disk(audio_features_path, keep_in_memory=False)
        self.video_labels = {video_path:0 if self.video_data['video_fake'][i] else 1 for i, video_path in enumerate(self.video_data['video_path'])}
        self.audio_labels = {video_path:0 if self.video_data['audio_fake'][i] else 1 for i, video_path in enumerate(self.video_data['video_path'])}
        self.audio_indices = {}
        for i, video_path in enumerate(self.audio_features_data['video_path']):
            self.audio_indices[video_path] = i
        
        self.indices = {"start":[], "ds_position":[]}
        batch_size = 5000
        dataset_size = len(self.features_data)
        for j in tqdm(range(0, dataset_size, batch_size), desc="Processing dataset"):
            start_index = j
            end_index = min(j+batch_size, dataset_size)
            batch = self.features_data.select(range(start_index, end_index))
            video_paths = batch['video_path']
            emotions = batch['emotion_video']
            for i, entry in enumerate(emotions):
                if video_paths[i] not in self.video_labels:
                    continue # if the video is not in the desired split then skip it
                current_sequence_length = len(entry)
                if hop_length == 'sequence_length':
                    start_pos = np.arange(0, current_sequence_length, current_sequence_length, dtype=int) 
                else:
                    start_pos = np.arange(0, current_sequence_length - hop_length, hop_length, dtype=int)

                self.indices['start'].append(start_pos)
                self.indices['ds_position'].append((j+i)*np.ones((len(start_pos,)), dtype=int))
        for key in self.indices:
            self.indices[key] = np.concatenate(self.indices[key], axis=0, dtype=np.int32)
        self.sequence_length = sequence_length
        self.features_to_keep = set(features_to_keep)
        self.frame_resolution = frame_resolution
        self.transforms_dictionary = transforms_dictionary
        self.synchronize_audio_features = synchronize_audio_features
        self.return_video_path = return_video_path
        self.emotion_to_idx = {
                "Anger": 0,
                "Disgust": 1,
                "Fear": 2,
                "Happiness": 3,
                "Neutral": 4,
                "Sadness": 5,
                "Surprise": 6,
                "Contempt": 7
            }
        
        self.emotion_audio_to_idx = {
            'happy':3, 'anger': 0, 'neutral':4, 'disgust':1, 'fear':2, 'sad':5
        }
    def __len__(self):
        return len(self.indices['start'])
    
    def __getitem__(self, idx):
        total_start_time = time.time()
        ds_position, start, length = int(self.indices['ds_position'][idx]), int(self.indices['start'][idx]), self.sequence_length
        # print(ds_position, start)
        video_path = self.features_data['video_path'][ds_position]
        label = self.video_labels[video_path]
        if self.return_video_path:
            # print(video_path)
            output_dictionary = {'video_path': video_path}
        else:
            output_dictionary = {}
        start_time = time.time()
        entry = self.features_data[ds_position]
        available_features = set(self.features_data.column_names)
        features_to_keep = self.features_to_keep.intersection(available_features)
        for feature_type in features_to_keep:
            if 'emotion_video' == feature_type:
                output_dictionary[feature_type] = [self.emotion_to_idx[emotion[0]] for emotion in entry[feature_type]][start:]
            elif 'emotion_audio' == feature_type:
                output_dictionary[feature_type] = [self.emotion_audio_to_idx[emotion] for emotion in entry[feature_type]][start:]
            else:
                output_dictionary['length'] = len(entry[feature_type])
                output_dictionary[feature_type] = entry[feature_type][start:]
        end_time = time.time()
        if "bbox_mouth" in features_to_keep:
            output_dictionary["bbox_mouth"] = [bbox if bbox != None else [0, 0, 0, 0] for bbox in output_dictionary["bbox_mouth"]]
        # print("output dict", end_time-start_time)
        output_dictionary = self.stack_sequence(output_dictionary, length)
        if 'audio_features' in self.features_to_keep:
            audio_features = np.array(self.audio_features_data[self.audio_indices[video_path]]['audio_features'])
            if self.synchronize_audio_features:
                audio_features = align_audio_to_frames_by_fps(audio_features, self.features_data['frame_idx_in_original_video'][ds_position][start:], self.features_data['fps'][ds_position])
                # print(audio_features.shape)
                T = audio_features.shape[0]
                feat_shape = audio_features.shape[1:]
                if T > length:
                    audio_features = audio_features[:length]
                elif T < length:
                    pad_shape = (length - T,) + feat_shape
                    pad = np.zeros(pad_shape, dtype=audio_features.dtype)
                    audio_features = np.concatenate([audio_features, pad], axis=0)
            output_dictionary['audio_features'] = audio_features
            label = min(self.audio_labels[video_path], self.video_labels[video_path])
        else:
            label = self.video_labels[video_path]
        transform_start_time = time.time()
        if self.transforms_dictionary is not None:
            for feature_type in output_dictionary:
                if feature_type in self.transforms_dictionary:
                    transformed_features = self.transforms_dictionary[feature_type](output_dictionary[feature_type])
                    output_dictionary[feature_type] = np.concatenate((transformed_features, output_dictionary[feature_type]), axis=-1)
        output_dictionary['label'] = label
        transform_end_time = time.time()
        # print("transform time", transform_end_time-transform_start_time)
        total_end_time = time.time()
        # print("total time", total_end_time-total_start_time)
        return output_dictionary    

    def stack_sequence(self, sequence, sequence_length):

        keys = sequence.keys()
        output = {}
        for key in keys:
            if key =='video_path' or key=='length':
                output[key]=sequence[key]
                continue
            # print(key)
            data = sequence[key]
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

            output[key] = data
        output['padding_mask'] = padding_mask
        return output


def relative_deltas(sequence):
    deltas = np.diff(sequence, axis=0, prepend=sequence[[0]])
    return deltas
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    features = ["hp", "gaze"]
    ds = MavosFeatureDataset("/home/biodeep/alin/datasets/MAVOS-DD",
                             "/home/biodeep/alin/datasets/mavosdd_features_final",
                             "/home/biodeep/alin/datasets/restructured_dataset_audio",
                             "train",
                             features,
                             100,
                             10, 512,
                             {"hp": relative_deltas, "gaze": relative_deltas},
                             True)
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)
    # print(len(dataloader))
    result = {0: [], 1:[]}
    for batch in tqdm(dataloader):
        # continue
        # print(list(batch.keys()))
        # for feature in features:
            # print(feature, batch[feature].shape)
        batch['length'] = batch['length'].cpu().numpy()
        batch['label'] = batch['label'].cpu().numpy()
        for i, label in enumerate(batch['label']):
            result[int(label)].append(int(batch['length'][i]))
        if len(result[0])>5000 and len(result[1])>5000:
            break
    bin_edges = np.arange(0, 101, 10) # 0 to 100, in steps of 10


    for label in result:
        counts, bins = np.histogram(result[label], bins=bin_edges)


        bin_labels = [f"[{bins[i]}-{bins[i+1]})" for i in range(len(counts))]

        plt.figure(figsize=(12, 6)) 
        plt.bar(
            bin_labels,      
            counts,          
            width=0.9,       
            edgecolor='black'
        )


        plt.xlabel('Value Bins')
        plt.ylabel('Frequency (Count)')
        plt.title('Frequency Bar Plot from Binned Integer Array')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7) 
        plt.tight_layout() 

        plt.savefig(f'binned_barplot_{label}.png')

 
        print("Histogram counts:", counts)
        print("Bin edges:", bins)
        print("Bin labels:", bin_labels)
        print("Plot saved as binned_barplot.png") 