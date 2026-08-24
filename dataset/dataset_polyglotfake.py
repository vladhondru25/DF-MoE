from torch.utils.data import DataLoader
import torchvision.ops as ops
from collections.abc import Callable
from datasets import load_from_disk, Dataset, Features, Value, List, concatenate_datasets
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
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor
import librosa
import cv2
class PolyglotFeatureDataset:
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

        root_dir = Path(self.video_dir)


        mp4_files = list(root_dir.rglob("*.mp4"))
        self.video_data = []
        for file_path in mp4_files:
            str_path = str(file_path).split("/")
            self.video_data.append(("/".join([str_path[-2], str_path[-3], str_path[-1]]), str(file_path).split("/")[-3]=='fake'))
        # print(self.video_data)
        self.features_data = load_from_disk(data_features_path, keep_in_memory=False)#.select(range(1000, 2000))
        self.features_data = self.features_data.add_column("source", ["uncompresssed"] * len(self.features_data))
        self.audio_features_data = load_from_disk(audio_features_path, keep_in_memory=False)
        self.video_labels = {video_path[0]:0 if self.video_data[i][1] else 1 for i, video_path in enumerate(self.video_data)}
        self.audio_labels = {video_path[0]:0 if self.video_data[i][1] else 1 for i, video_path in enumerate(self.video_data)}
        self.audio_indices = {}
        for i, video_path in enumerate(self.audio_features_data['video_path']):
            self.audio_indices[video_path] = i

        print(f"split={split}: ", np.unique(list(self.video_labels.values()), return_counts=True))

        feat_video_paths = self.features_data['video_path']
        # print(feat_video_paths)
        def get_len(batch):
            # print(batch)
            return {'seq_len': [len(x) for x in batch]}

        valid_paths_set = set(self.video_labels.keys())
        
        is_valid_split = [vp in valid_paths_set for vp in feat_video_paths]
        valid_indices = np.where(is_valid_split)[0]
        
        valid_features = self.features_data.select(valid_indices)
        
        lengths_dataset = valid_features.map(get_len, input_columns=['emotion_video'], batched=True, batch_size=1000)
        seq_lengths = np.array(lengths_dataset['seq_len'])
        
        start_indices_list = []
        ds_positions_list = []
        
        if hop_length == 'sequence_length':
            starts = [np.arange(0, L, L, dtype=np.int32) for L in seq_lengths]
        else:
            starts = [np.arange(0, max(0, L - hop_length), hop_length, dtype=np.int32) for L in seq_lengths]

        counts = [len(s) for s in starts]
        
        self.indices = {}
        self.indices['ds_position'] = np.repeat(valid_indices, counts).astype(np.int32)
        self.indices['start'] = np.concatenate(starts).astype(np.int32)

        del lengths_dataset, seq_lengths, starts, counts, valid_features

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
        model_id = "facebook/wav2vec2-xls-r-300m"
        self.wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)


    def __len__(self):
        return len(self.indices['start'])
    
    def __getitem__(self, idx):
        total_start_time = time.time()
        ds_position, start, length = int(self.indices['ds_position'][idx]), int(self.indices['start'][idx]), self.sequence_length
        # print(ds_position, start)
        video_path = self.features_data['video_path'][ds_position]
        identity = self.features_data['identity'][ds_position]
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
            # start_time = time.time()
            if 'emotion_video' == feature_type:
                output_dictionary[feature_type] = [self.emotion_to_idx[emotion[0]] for emotion in entry[feature_type]][start:start+length]
            elif 'emotion_audio' == feature_type:
                output_dictionary[feature_type] = [self.emotion_audio_to_idx[emotion] for emotion in entry[feature_type]][start:start+length]
            elif 'frames' == feature_type:
                output_dictionary[feature_type] = entry[feature_type].get_frames_in_range(start, start+length).data
            else:
                output_dictionary['length'] = len(entry[feature_type])
                output_dictionary[feature_type] = entry[feature_type][start:start+length]
        
            # end_time = time.time()
            # print(f"Reading time {feature_type}: {end_time-start_time}")
        end_time = time.time()
        
        if "bbox_mouth" in features_to_keep:
            segmentations = rearrange(output_dictionary['face_parse'], 'b c h w->b h w c').numpy()[:, :, :, 0]
            bboxes = []
            for i, segmentation in enumerate(segmentations):
                
                bbox_low_lips = self.get_bounding_box(segmentation, target_label=12)
                bbox_upper_lips = self.get_bounding_box(segmentation, target_label=11)
                # bbox_teeths = self.get_bounding_box(parsing, target_label=11)
                valid_bboxes = [b for b in [bbox_low_lips, bbox_upper_lips] if b is not None]
                if valid_bboxes:
                    x_min = min(b[0] for b in valid_bboxes)
                    y_min = min(b[1] for b in valid_bboxes)
                    x_max = max(b[2] for b in valid_bboxes)
                    y_max = max(b[3] for b in valid_bboxes)
                    bbox_mouth = [x_min, y_min, x_max, y_max]
                    # Image.fromarray((segmentation[y_min:y_max, x_min:x_max]*10).astype(np.uint8)).save("extracted_frames/seg.png")
                    # exit()
                else:
                    bbox_mouth = None
                
                bboxes.append(bbox_mouth)
            
            output_dictionary["bbox_mouth"] = self.interpolate_boxes_np(bboxes)
        
        # print("output dict", end_time-start_time)
        output_dictionary = self.stack_sequence(output_dictionary, length)
        fps = None
        if "full_frames" in self.features_to_keep and "raw_audio_features" in self.features_to_keep:
            start_time = time.time()
            output_dictionary['full_frames'], fps = self.read_frames(video_path = self.features_data['video_path'][ds_position], indices = self.features_data['frame_idx_in_original_video'][ds_position][start:start+length])
            end_time = time.time()
            # print(f"Read time full frames {end_time - start_time}")
            # start_time = time.time()
            output_dictionary['raw_audio_features'], output_dictionary['wav2vec'], output_dictionary['wav2vec_mask'] = self._wav2fbank(filename = self.features_data['video_path'][ds_position], indices = self.features_data['frame_idx_in_original_video'][ds_position][start:start+length], fps = fps)
            # end_time = time.time()
            # print(f"Read time raw audio features {end_time - start_time}")
        if 'audio_features' in self.features_to_keep:
            audio_features = np.array(self.audio_features_data[self.audio_indices[video_path]]['audio_features'])
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
        transform_start_time = time.time()
        if self.transforms_dictionary is not None:
            for feature_type in output_dictionary:
                if feature_type in self.transforms_dictionary:
                    transformed_features = self.transforms_dictionary[feature_type](output_dictionary[feature_type])
                    output_dictionary[feature_type] = np.concatenate((transformed_features, output_dictionary[feature_type]), axis=-1)
        
        if self.avff_like_labels:
            label = np.array([0, 1]) if label == 1 else np.array([1, 0])
        output_dictionary['label'] = label
        transform_end_time = time.time()
        # print("transform time", transform_end_time-transform_start_time)
        total_end_time = time.time()
        # print("total time", total_end_time-total_start_time)
        
        if "frames" in features_to_keep:
            output_dictionary["frames"] = crop_and_resize(torch.from_numpy(output_dictionary["frames"]), torch.from_numpy(output_dictionary["bbox_mouth"]))
            output_dictionary["frames"] = self.dino_image_preprocessor(output_dictionary["frames"], return_tensors="pt").pixel_values

        identity = self.features_data['identity'][ds_position]
        output_dictionary['identity'] = identity
        return output_dictionary  

    def get_bounding_box(self, segmentation_map, target_label=11, margin=5):
        segmentation_map = (segmentation_map == target_label).astype(np.uint8)
        segmentation_map = cv2.erode(segmentation_map, np.ones((3, 3)))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmentation_map, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            segmentation_map = np.zeros_like(segmentation_map)
            segmentation_map[labels == largest_label] = 255
        # Image.fromarray(((segmentation_map)).astype(np.uint8)).save(f"extracted_frames/seg_full_{target_label}.png")
        ys, xs = np.where(segmentation_map != 0)
        if len(xs) == 0 or len(ys) == 0:
            return None  # Mouth not found
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        # Add margin, clip to image bounds
        h, w = segmentation_map.shape
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, w - 1)
        y_max = min(y_max + margin, h - 1)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
        
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
                if np.array(data[1]).shape[1]==224:
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
        # try:
            # vr = VideoReader(os.path.join(self.video_dir, video_path))
            # total_frames = len(vr)
            video_path = video_path.split("/")
            video_path = "/".join([video_path[-2], video_path[-3], video_path[-1]])
            vr = VideoDecoder(os.path.join(self.video_dir, video_path), device="cpu", num_ffmpeg_threads=0)
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
            # print(frames.shape)
            # 
        # except:
            # frames = torch.zeros(self.num_frames, 3, 224, 224)
            # print("Failed to read files")
                
            return frames, fps

    def _wav2fbank(self, filename, indices, fps):
        filename = filename.split("/")
        filename = "/".join([filename[-2], filename[-3], filename[-1]])
        decoder = AudioDecoder(os.path.join(self.video_dir, filename))
        # total_frames = len(decoder)
        # frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        # indices = np.array(indices)[frame_indices]
        start_time = indices[0]/fps
        end_time = max(start_time+1, indices[-1]/fps)
        # print(start_time, end_time)
        try:
            audio_sample = decoder.get_samples_played_in_range(start_time, end_time)
        except Exception as e:
            print(f"Failed file {filename}", {fps})
            raise e

        max_samples = 16000 * 10

        # Convert to tensor
        audio_wav2vec = audio_sample.data#librosa.load(os.path.join(video_dir, filename), sr=16000)#
        # print(audio_wav2vec.shape)
        # exit()
        if audio_wav2vec.shape[0]==2:
            audio_wav2vec = audio_wav2vec.mean(dim=0, keepdim=True)
        audio_wav2vec = audio_wav2vec.squeeze()
        # print(filename, audio_wav2vec.mean(), audio_wav2vec.amax(), audio_wav2vec.amin(), audio_sample.sample_rate, end_time-start_time)
        audio_wav2vec = librosa.resample(y = audio_wav2vec.numpy(), orig_sr = audio_sample.sample_rate, target_sr=16000)
        # print(filename, audio_wav2vec.mean(), audio_wav2vec.max(), audio_wav2vec.min(), end_time-start_time)
        # exit()
        # Slice to 10 seconds if longer
        if audio_wav2vec.shape[-1] > max_samples:
            audio_wav2vec = audio_wav2vec[:max_samples]
        
        # Prepare input features (log-Mel)
        inputs = self.wav2vec_processor(
            audio_wav2vec, padding='max_length', max_length=max_samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        # input_features = inputs.input_features
        input_features = inputs.input_values
        # print(filename, input_features.amax(), input_features.amin(), input_features.shape)
        mask = inputs.attention_mask

        audio_sample.data = audio_sample.data - audio_sample.data.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(audio_sample.data, htk_compat=True, sample_frequency=audio_sample.sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([self.target_length, 128]) + 0.01
            print('there is a loading error')

        target_length = self.target_length

        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length, ), mode='linear', align_corners=False).transpose(1,2).squeeze(0)
        fbank = (fbank - self.norm_mean) / (self.norm_std)
        return fbank, input_features.squeeze(0).clone(), mask.squeeze(0).clone()
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

if __name__ == "__main__":

    features = ["hp", "gaze", "emotion_video", "rPPG", "face_parse", "audio_features"]
    features = ["full_frames", "raw_audio_features", "rPPG", "hp", "gaze", "emotion_video", "rPPG", "face_parse", 'bbox_mouth', "audio_features", "frames"]
    ds = MavosFeatureDataset(
        data_video_path="../datasets/MAVOS-DD",
        data_features_path="../datasets/features_mavos_complete/mavosdd_features_final",
        audio_features_path="../datasets/features_mavos_complete/restructured_dataset_audio",
        split="validation",
        features_to_keep=features,
        sequence_length=100, hop_length=20,
        frame_resolution=512,
        transforms_dictionary={"hp": relative_deltas, "gaze": relative_deltas},
        synchronize_audio_features=True
    )
    dataloader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    
    # print(len(dataloader))
    for batch in dataloader:
        # continue
        # print(list(batch.keys()))
        # if "frames" in features and "bbox_mouth" in features:
            # batch["frames"] = mask_outside_bbox(batch["frames"], batch["bbox_mouth"])
        for feature in features:
            print(feature, batch[feature].shape)
