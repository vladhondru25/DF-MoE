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

from torchcodec.decoders import VideoDecoder, AudioDecoder
from transformers import AutoImageProcessor, DINOv3ViTImageProcessorFast
import torchvision.transforms as T
import torchaudio
from einops import rearrange
from decord import VideoReader
from torchcodec import AudioSamples
import random
class DeepfakeEvalFeatureDataset:
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
                 return_video_path: bool = False,
                avff_like_labels: bool = False):
        self.avff_like_labels = avff_like_labels
        self.video_dir = data_video_path
        self.num_frames = 16
        self.video_data = load_dataset("csv", data_files=data_video_path)['train']
        self.video_data = self.video_data.filter(lambda sample: sample['Finetuning Set']==split)# and sample['Filename']=="0DOGSzwaUcyt9m0E4thYW8DCvRs.mp4")
        self.features_data = load_from_disk(data_features_path, keep_in_memory=False)#.select(range(10000))
        print(self.video_data.column_names)
        self.audio_features_data = load_from_disk(audio_features_path, keep_in_memory=False)
        self.video_labels = {video_path:0 if self.video_data['Video Ground Truth'][i] =='Fake' else 1 for i, video_path in enumerate(self.video_data['Filename'])}
        self.audio_labels = {video_path:0 if self.video_data['Audio Ground Truth'][i] == 'Fake' else 1 for i, video_path in enumerate(self.video_data['Filename'])}
        self.audio_indices = {}
        for i, video_path in enumerate(self.audio_features_data['video_path']):
            self.audio_indices[video_path] = i
        self.audio_present_videos = []
        for video_path in self.audio_indices:
            if video_path in self.video_labels:
                self.audio_present_videos.append(video_path)
        print(f"split={split}: ", np.unique(list(self.video_labels.values()), return_counts=True))
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
                    if len(start_pos)==0 and current_sequence_length>2:
                        start_pos = np.array([0])

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
        self.dino_image_preprocessor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16plus-pretrain-lvd1689m")
        self.preprocess_full_frames = T.Compose([
            Crop(),
            T.Resize(size=(224, 224)),
            T.v2.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )
        ])
        self.melbins = 128
        self.target_length=1024
        self.norm_mean = -5.081
        self.norm_std = 4.4849
    def __len__(self):
        return len(self.indices['start'])
    
    def __getitem__(self, idx):
        total_start_time = time.time()
        ds_position, start, length = int(self.indices['ds_position'][idx]), int(self.indices['start'][idx]), self.sequence_length
        # print(ds_position, start)
        video_path = self.features_data['video_path'][ds_position]
        missing_audio = False
        if video_path not in self.audio_indices:
            missing_audio = True
        rollback_video = random.choice(self.audio_present_videos)
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
                output_dictionary[feature_type] = [self.emotion_to_idx[emotion[0]] for emotion in entry[feature_type]][start:start+length]
            elif 'emotion_audio' == feature_type:
                if entry[feature_type]is not None:
                    output_dictionary[feature_type] = [self.emotion_audio_to_idx[emotion if emotion is not None else 'neutral'] for emotion in entry[feature_type]][start:start+length]
                else:
                    output_dictionary[feature_type] = [self.emotion_audio_to_idx['neutral']]*length
            else:
                output_dictionary['length'] = len(entry[feature_type])
                output_dictionary[feature_type] = entry[feature_type][start:start+length]
        end_time = time.time()
        
        if "bbox_mouth" in features_to_keep:
            output_dictionary["bbox_mouth"] = self.interpolate_boxes_np(output_dictionary["bbox_mouth"])
            # output_dictionary["bbox_mouth"] = [bbox if bbox != None else [0, 0, 0, 0] for bbox in output_dictionary["bbox_mouth"]]
        
        # print("output dict", end_time-start_time)
        output_dictionary = self.stack_sequence(output_dictionary, length)
        fps = None
        if "raw_audio_features" in self.features_to_keep:
            start_time = time.time()
            output_dictionary['full_frames'], fps = self.read_frames(video_path = self.features_data['video_path'][ds_position], indices = self.features_data['frame_idx_in_original_video'][ds_position][start:start+length])
            end_time = time.time()
            # print(f"Read time full frames {end_time - start_time}")
            # start_time = time.time()
            if not missing_audio:
                output_dictionary['raw_audio_features'] = self._wav2fbank(filename = self.features_data['video_path'][ds_position], indices = self.features_data['frame_idx_in_original_video'][ds_position][start:start+length], fps = fps)
            else:
                output_dictionary['raw_audio_features'] = self._wav2fbank(filename = rollback_video, indices = None, fps = fps)
            # end_time = time.time()
            # print(f"Read time raw audio features {end_time - start_time}")
        if 'audio_features' in self.features_to_keep:
            if not missing_audio:
                audio_features = np.array(self.audio_features_data[self.audio_indices[video_path]]['audio_features'])
                # print(audio_features.mean(), audio_features.std())
            else:
                audio_features = np.array(self.audio_features_data[self.audio_indices[rollback_video]]['audio_features'])
            if self.synchronize_audio_features:
                audio_features = align_audio_to_frames_by_fps(audio_features, self.features_data['frame_idx_in_original_video'][ds_position][start:], self.features_data['fps'][ds_position] if fps is None else fps)
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
        if self.avff_like_labels:
            label = np.array([0, 1]) if label == 1 else np.array([1, 0])
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
        
        if "frames" in features_to_keep:
            # output_dictionary["frames"] = mask_outside_bbox(
            #     torch.from_numpy(output_dictionary["frames"]),
            #     torch.from_numpy(output_dictionary["bbox_mouth"])
            # )
            # output_dictionary["frames"] = mask_outside_bbox(torch.from_numpy(output_dictionary["frames"]), torch.from_numpy(output_dictionary["bbox_mouth"]))

            output_dictionary["frames"] = crop_and_resize(torch.from_numpy(output_dictionary["frames"]), torch.from_numpy(output_dictionary["bbox_mouth"]))
            output_dictionary["frames"] = self.dino_image_preprocessor(output_dictionary["frames"], return_tensors="pt").pixel_values
            # for i, frame in enumerate(output_dictionary["frames"]):
            #     frame = rearrange(frame, 'c h w->h w c')
            #     Image.fromarray(frame.astype(np.uint8)).save(f"qualitative_videos/{i}.png")
            # exit()
        identity = self.features_data['identity'][ds_position]
        output_dictionary['identity'] = identity
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

            if key=='rPPG':
                if np.array(data[0]).shape[1]==224:
                    data[0] = np.zeros((5, 512))
                if len(data)>1 and np.array(data[1]).shape[1]==224:
                    data[1] = np.zeros((5, 512))
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
    
    def interpolate_boxes_np(self, boxes):
        # Convert list of lists into a float array with NaNs instead of None
        arr = np.array(
            [b if b is not None else [np.nan]*4 for b in boxes],
            dtype=float
        )
        
        # If everything is None/NaN: return zeros
        if np.isnan(arr).all():
            return [[0.0, 0.0, 0.0, 0.0] for _ in boxes]

        # Indices 0..N-1
        x = np.arange(len(arr))

        for col in range(arr.shape[1]):
            y = arr[:, col]
            mask = ~np.isnan(y)
            arr[:, col] = np.interp(x, x[mask], y[mask])

        return arr.tolist()

    def read_frames(self, video_path, indices):
        try:
            # vr = VideoReader(os.path.join(self.video_dir, video_path))
            # total_frames = len(vr)
            for retry in range(3):
                try:
                    input_video_path = os.path.join(os.path.dirname(self.video_dir), "video-data", video_path)
                    vr = VideoDecoder(input_video_path, device="cpu", num_ffmpeg_threads=0)
                except:
                    print(f"Failed reading {input_video_path} {retry+1} times. Retrying...")
            total_frames = vr.metadata.num_frames
            fps = vr.metadata.average_fps
            # Calculate the indices to sample uniformly
            frame_indices = np.linspace(indices[0], indices[-1], self.num_frames).astype(int)
            # print(frame_indices)
            # start_time =time.time()
            # indices = np.array(indices)
            frames = vr.get_frames_at(frame_indices).data#torch.from_numpy(vr.get_batch(indices=frame_indices).asnumpy())#.data
            # frames = torch.from_numpy(vr.get_batch(indices=frame_indices).asnumpy())
            # print(f"Reading time: {time.time()-start_time}")
            # print(frames.shape)
            # frames = rearrange(frames, "s h w c-> s c h w")
            # print(frames.shape)
            # for i, frame in enumerate(frames):
            #     Image.fromarray(frame.numpy()).save(f"qualitative_videos/{i}.png")
            # exit()
            # frames = rearrange(frames, "s h w c-> s c h w")
            frames = self.preprocess_full_frames(frames) 
        except:
            frames = torch.zeros(self.num_frames, 3, 224, 224)
            print(f"Failed to read files {os.path.join(os.path.dirname(self.video_dir), "video-data", video_path)}")
                
        return frames, fps

    def _wav2fbank(self, filename, indices, fps):
               
        # total_frames = len(decoder)
        # frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        # indices = np.array(indices)[frame_indices]
        if indices is not None:
            start_time = indices[0]/fps
            end_time = max(start_time+1, indices[-1]/fps)
        else:
            start_time = 0
            end_time = 2
        # print(start_time, end_time)
        try:
            decoder = AudioDecoder(os.path.join(os.path.dirname(self.video_dir), "video-data", filename))
            audio_sample = decoder.get_samples_played_in_range(start_time, end_time)
            # print(audio_sample.data.shape, audio_sample.data.mean(axis=1), audio_sample.data.std(axis=1), audio_sample.sample_rate)
            # exit()
        except Exception as e:
            print(f"Failed file {filename}, {fps}, {end_time}, {start_time}")
            sample_rate = 44100 
            duration = end_time - start_time
            num_samples = int(duration * sample_rate)
            audio_sample = AudioSamples(
                data=torch.randn(size=(2, 220500)),
                pts_seconds=duration,
                duration_seconds=duration,
                sample_rate=sample_rate,
            )
        # print(audio_sample.data.shape)
        audio_sample.data = audio_sample.data - audio_sample.data.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(audio_sample.data, htk_compat=True, sample_frequency=audio_sample.sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except Exception as e:
            fbank = torch.zeros([self.target_length, 128]) + 0.01
            print(f'there is a loading error {filename} , {fps}, {end_time}, {start_time}, {indices}')
            # raise e

        target_length = self.target_length

        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length, ), mode='linear', align_corners=False).transpose(1,2).squeeze(0)
        fbank = (fbank - self.norm_mean) / (self.norm_std)
        return fbank

def relative_deltas(sequence):
    deltas = np.diff(sequence, axis=0, prepend=sequence[[0]])
    return deltas


def mask_outside_bbox_batched(frames, bboxes):
    B, T, C, H, W = frames.shape

    # Expand for broadcasting: (B, T, 1, 1, 1)
    x1 = bboxes[..., 0].view(B, T, 1, 1, 1)
    y1 = bboxes[..., 1].view(B, T, 1, 1, 1)
    x2 = bboxes[..., 2].view(B, T, 1, 1, 1)
    y2 = bboxes[..., 3].view(B, T, 1, 1, 1)

    # Create coordinate grids: shape (1, 1, H, W)
    yy = torch.arange(H, device=frames.device).view(1, 1, H, 1)
    xx = torch.arange(W, device=frames.device).view(1, 1, 1, W)

    # Broadcasting to (B, T, H, W)
    inside_x = (xx >= x1) & (xx < x2)
    inside_y = (yy >= y1) & (yy < y2)
    mask = inside_x & inside_y  # (B, T, H, W)

    # Ignore zero boxes [0,0,0,0]
    ignore = (bboxes.sum(dim=-1) == 0).view(B, T, 1, 1, 1)
    mask = mask & (~ignore)

    # Apply mask: keep pixels inside box, set others to black
    masked_frames = frames * mask

    return masked_frames

def mask_outside_bbox(frames, bboxes):
    T, C, H, W = frames.shape

    # Expand for broadcasting: (B, T, 1, 1, 1)
    x1 = bboxes[..., 0].view(T, 1, 1, 1)
    y1 = bboxes[..., 1].view(T, 1, 1, 1)
    x2 = bboxes[..., 2].view(T, 1, 1, 1)
    y2 = bboxes[..., 3].view(T, 1, 1, 1)

    # Create coordinate grids: shape (1, 1, H, W)
    yy = torch.arange(H, device=frames.device).view(1, H, 1)
    xx = torch.arange(W, device=frames.device).view(1, 1, W)

    # Broadcasting to (B, T, H, W)
    inside_x = (xx >= x1) & (xx < x2)
    inside_y = (yy >= y1) & (yy < y2)
    mask = inside_x & inside_y  # (B, T, H, W)

    # Ignore zero boxes [0,0,0,0]
    ignore = (bboxes.sum(dim=-1) == 0).view(T, 1, 1, 1)
    mask = mask & (~ignore)

    # Apply mask: keep pixels inside box, set others to black
    masked_frames = frames * mask

    return masked_frames

def crop_and_resize(images, boxes, size=224):
    """
    images: (T, C, H, W)
    boxes:  (T, 4) absolute coords (x1, y1, x2, y2)
    size: output size (int or (h, w))
    """
    T = images.shape[0]

    # roi_align expects a list of boxes with batch indices:
    # [batch_idx, x1, y1, x2, y2]
    batched_boxes = []
    for i in range(T):
        batched_boxes.append(
            torch.cat([torch.tensor([i], dtype=boxes.dtype, device=boxes.device), boxes[i]])
        )
    batched_boxes = torch.stack(batched_boxes, dim=0)

    # ROI Align produces (T, C, size, size)
    crops = ops.roi_align(
        images.float(),                 # (T, C, H, W)
        batched_boxes.float(),          # (T, 5)
        output_size=size,       # int or (H, W)
        spatial_scale=1.0,      # boxes are in absolute pixel coords
        aligned=True
    )

    return crops

from PIL import Image, ImageOps
class Crop(torch.nn.Module):
    def forward(self, img):
        # print(img.shape)
        height = img.shape[-2]
        width = img.shape[-1]
        if height>width:
            difference = height - width
            side_difference = int(difference//2)
            new_img = img[:,:,side_difference:-side_difference, :]
        elif height<width:
            difference = width - height
            side_difference = int(difference//2)
            new_img = img[:,:,:, side_difference:-side_difference]
        else:
            new_img = img
        # viz_img = rearrange(new_img, "b c h w-> b h w c")[0]
        # Image.fromarray(viz_img.numpy().astype(np.uint8)).show()
        return new_img
if __name__ == "__main__":

    features = ["hp", "gaze", "emotion_video", "rPPG", "face_parse", "audio_features"]
    features = ["frames", "bbox_mouth", "audio_features"]
    ds = MavosFeatureDataset(
        data_video_path="/home/eivor/data/MAVOS-DD",
        data_features_path="/home/eivor/biodeep/Detection/BiodeepDetection/interim_outputs2/mavosdd_features_final",
        audio_features_path="/home/eivor/biodeep/Detection/BiodeepDetection/interim_outputs2/restructured_dataset_audio",
        split="validation",
        features_to_keep=features,
        sequence_length=50, hop_length=5,
        frame_resolution=512,
        transforms_dictionary={"hp": relative_deltas, "gaze": relative_deltas},
        synchronize_audio_features=True
    )
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    
    # print(len(dataloader))
    for batch in dataloader:
        # continue
        # print(list(batch.keys()))
        if "frames" in features and "bbox_mouth" in features:
            batch["frames"] = mask_outside_bbox(batch["frames"], batch["bbox_mouth"])
        for feature in features:
            print(feature, batch[feature].shape)
