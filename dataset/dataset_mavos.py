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

class MavosFeatureDataset:

    def __init__(self, data_video_path: str,
                 data_features_path: str,
                 split: str,
                 features_to_keep: list[str],
                 sequence_length: int,
                 hop_length: int,
                 frame_resolution: int,
                 transforms_dictionary: list[Callable] = None,
                 synchronize_audio_features: bool = False,
                 video_level=False,
                 cache_dir:str = None):

        self.video_data = load_from_disk(data_video_path).filter(lambda sample: sample['split'] == split)
        # print(len(self.video_data))
        self.sequence_data = []
        self.data_features_path = data_features_path
        self.data_video_path = data_video_path
        self.features_to_keep = set(features_to_keep)
        self.sequence_length = sequence_length
        self.frame_resolution = frame_resolution
        self.transforms_dictionary = transforms_dictionary
        self.synchronize_audio_features = synchronize_audio_features
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

        if cache_dir is None or not os.path.exists(cache_dir) or len(os.listdir(cache_dir))==0:
            for i, video_path in enumerate(tqdm(self.video_data['video_path'], desc="Building dataset")):
                sequences = [] # we have a sequence for each video and for each person
                identity_to_idx = {} # each identity will be mapped to an index in the above list
                video_dir = os.path.join(data_features_path, video_path[:-4])

                multimodal_features_path = os.path.join(video_dir, "data.pkl")
                audio_features_path = os.path.join(video_dir, "audio_features.pt")
                frames_path = os.path.join(video_dir, "frames")
                try:
                    with open(multimodal_features_path, "rb") as f:
                        multimodal_features = pickle.load(f)
                except:
                    print(f"Skipping {multimodal_features_path}. Failed reading it.")
                    continue
                fps = get_fps_opencv(os.path.join(self.data_video_path, video_path))
                index_identity = 0
                sorted_keys = sorted(list(multimodal_features.keys())) # !!! CHECK this sorting, might not be what we want since we do not explicitly
                if video_level and len(sorted_keys)>0:
                   sequences.append({'video_path':video_path[:-4],
                                          'identity': [],
                                          "sequence_keys":[],
                                          "frame_idx": [],
                                          "fps":fps,
                                          "audio_label": 0 if self.video_data[i]['audio_fake'] else 1,
                                          "video_label": 0 if self.video_data[i]['video_fake'] else 1})
                for key_frame in sorted_keys:
                    components = key_frame.split("_")
                    identity, frame_index = components[0], components[1]
                    if video_level:
                        sequences[-1]["sequence_keys"].append(key_frame)
                        sequences[-1]['frame_idx'].append(int(frame_index))
                        sequences[-1]['identity'].append(int(identity)-1)
                    else:
                        if identity in identity_to_idx:
                            # existing identity => update existing sequence
                            sequences[identity_to_idx[identity]]['sequence_keys'].append(key_frame)
                            sequences[identity_to_idx[identity]]['frame_idx'].append(int(frame_index))
                            sequences[identity_to_idx[identity]]['identity'].append(int(identity)-1)
                        else:
                            # new identity => create a new sequence
                            identity_to_idx[identity] = index_identity
                            index_identity +=1
                            sequences.append({'video_path':video_path[:-4],
                                            'identity': [int(identity)-1],
                                            "sequence_keys":[key_frame],
                                            "frame_idx": [int(frame_index)],
                                            "fps":fps,
                                            "audio_label": 0 if self.video_data[i]['audio_fake'] else 1,
                                            "video_label": 0 if self.video_data[i]['video_fake'] else 1})
                self.sequence_data.extend(sequences)
            final_sequence_data = []
            for sequence in self.sequence_data:
                final_sequence_data.append(sequence)
                if hop_length < len(sequence['sequence_keys']):
                    for i in range(hop_length, len(sequence['sequence_keys']), hop_length):
                        final_sequence_data.append(copy.deepcopy(sequence))
                        final_sequence_data[-1]['sequence_keys'] = final_sequence_data[-1]['sequence_keys'][i:]
                        final_sequence_data[-1]['frame_idx'] = final_sequence_data[-1]['frame_idx'][i:]
                        final_sequence_data[-1]['identity'] = final_sequence_data[-1]['identity'][i:]
            self.sequence_data = final_sequence_data
            self.dataset = Dataset.from_list(self.sequence_data, features=Features({
                'video_path': Value('string'),
                'identity': List(Value('int16')),
                "sequence_keys": List(Value('string')),
                "frame_idx": List(Value('int16')),
                "audio_label": Value('int8'),
                "fps": Value('int16'),
                "video_label":  Value('int8')
            }))
            self.dataset.save_to_disk(cache_dir)
        else:
            self.dataset = load_from_disk(cache_dir)

    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        start_time = time.time()
        entry = self.dataset[idx]
        # print(entry)
        sequence_keys = entry['sequence_keys']
        start_read_time = time.time()
        multimodal_features_path = os.path.join(self.data_features_path, entry['video_path'], "data.pkl")
        with open(multimodal_features_path, "rb") as f:
            multimodal_features = pickle.load(f)
        end_read_time = time.time()
        list_feature_dictionaries = []
        for key in sequence_keys:
            list_feature_dictionaries.append(multimodal_features[key])
        available_features = set(list_feature_dictionaries[0].keys())
        features_to_keep = self.features_to_keep.intersection(available_features)
        output_dictionary = {'identites': entry['identity']}
        for feature_type in features_to_keep:
            for feature_dictionary in list_feature_dictionaries:
                if feature_type == 'rPPG' and len(np.array(feature_dictionary[feature_type]).shape)<3:
                    feature_dictionary[feature_type] = np.zeros((1, 5, 224))
                if feature_type not in output_dictionary:
                    if feature_type == 'emotion_video':
                        output_dictionary[feature_type] = [self.emotion_to_idx[feature_dictionary[feature_type][0]]]
                    else:
                        output_dictionary[feature_type] = [np.array(feature_dictionary[feature_type]).squeeze()]
                else:
                    if feature_type == 'emotion_video':
                        output_dictionary[feature_type].append(self.emotion_to_idx[feature_dictionary[feature_type][0]])
                    else:
                        output_dictionary[feature_type].append(np.array(feature_dictionary[feature_type]).squeeze())
        # print(output_dictionary['emotion_video'])
        if "frames" in self.features_to_keep:
            output_dictionary['frames'] = []
            for key in sequence_keys:
                frame = Image.open(os .path.join(self.data_features_path, entry['video_path'], "frames", key+".jpg")).resize((self.frame_resolution, self.frame_resolution  ))
                output_dictionary['frames'].append(np.array(frame)/255.)
                # frame.save(f'visualization/'+key+'.png')

        output_dictionary = self.stack_sequence(output_dictionary)
        if "audio_features" in self.features_to_keep:
            audio_features = torch.load(os.path.join(self.data_features_path, entry['video_path'], "audio_features.pt")).numpy().squeeze()
            if self.synchronize_audio_features:
                audio_features = align_audio_to_frames_by_fps(audio_features, entry['frame_idx'], entry['fps'])
                # print(audio_features.shape)
                T = audio_features.shape[0]
                feat_shape = audio_features.shape[1:]
                if T > self.sequence_length:
                    audio_features = audio_features[:self.sequence_length]
                elif T < self.sequence_length:
                    pad_shape = (self.sequence_length - T,) + feat_shape
                    pad = np.zeros(pad_shape, dtype=audio_features.dtype)
                    audio_features = np.concatenate([audio_features, pad], axis=0)
            output_dictionary['audio_features'] = audio_features
            label = max(entry['audio_label'], entry['video_label'])
        else:
            label = entry['video_label']
       
        if self.transforms_dictionary is not None:
            for feature_type in output_dictionary:
                if feature_type in self.transforms_dictionary:
                    transformed_features = self.transforms_dictionary[feature_type](output_dictionary[feature_type])
                    output_dictionary[feature_type] = np.concatenate((transformed_features, output_dictionary[feature_type]), axis=-1)
        output_dictionary['label'] = label
        end_time = time.time()
        print('total time', end_time-start_time)
        # print("Processing time", end_time - start_time, "Reading time", end_read_time -start_read_time)
        return output_dictionary    

    def stack_sequence(self, sequence):

        keys = sequence.keys()
        output = {}

        for key in keys:
            # print(key)
            data = sequence[key]
            data = np.stack(data, axis=0)
            # try:
                # print(len(data))
            T = data.shape[0]
            # except:
            #     print(key, data.shape, data)
            #     exit()
            feat_shape = data.shape[1:]

            if T > self.sequence_length:
                data = data[:self.sequence_length]
            elif T < self.sequence_length:
                pad_shape = (self.sequence_length - T,) + feat_shape
                pad = np.zeros(pad_shape, dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)

            output[key] = data
        
        return output


def relative_deltas(sequence):
    deltas = np.diff(sequence, axis=0, prepend=sequence[[0]])
    return deltas
## OBS: The dataloader is slower when we read the frames and the audio features, from the two the frames have the highest impact on time. 
if __name__ == "__main__":
    features = ["hp", "gaze", "emotion_video", "rPPG", "face_parse", "audio_features"]
    ds = MavosFeatureDataset("/home/biodeep/alin/datasets/MAVOS-DD",
                             "/home/biodeep/alin/datasets/output",
                             "train",
                             features,
                             50, 5,
                             256,
                             {"hp": relative_deltas, "gaze": relative_deltas},
                             True, True,
                             "video_cache_dir")
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)
    # print(len(dataloader))
    for batch in dataloader:
        # continue
        # print(list(batch.keys()))
        for feature in features:
            print(feature, batch[feature].shape)


            
