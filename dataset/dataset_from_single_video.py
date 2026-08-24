import numpy as np
import torch
from torchcodec.decoders import VideoDecoder, AudioDecoder
from dataset.dataset_mavosv2 import VIDEO_EMOTIONS, AUDIO_EMOTIONS, MavosFeatureDataset, crop_and_resize, align_audio_to_frames_by_fps
from transformers import AutoImageProcessor, DINOv3ViTImageProcessorFast
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
import torchaudio
from torchcodec import AudioSamples
import torchvision.transforms.functional as F


class CenterCropToSquare:

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:] 
        else:
            w, h = img.size
        if h > w:
            min_dim = min(h, w)
            return F.center_crop(img, output_size=[min_dim, min_dim])
        else:
            return img
class MavosFeatureSingleVideo:

    def __init__(self, audio_features, video_features, hop_length, sequence_length, features_to_keep, synchronize_audio_features, video_path):
        self.audio_features = audio_features
        self.video_features = video_features
        self.sequence_length = sequence_length
        self.features_to_keep = set(features_to_keep)
        self.synchronize_audio_features = synchronize_audio_features
        emotions = self.video_features['emotion_video']
        self.video_path = self.video_features['video_path']
        device = "cpu" 
        decoder = VideoDecoder(self.video_features['frames'], device=device)
        current_sequence_length = len(emotions)
        self.video_features['frames'] = decoder
        self.video_features['face_parse'] = VideoDecoder(self.video_features['face_parse'], device=device)
        if hop_length >= current_sequence_length:
            start_pos = np.arange(0, current_sequence_length, current_sequence_length, dtype=int) 
        else:
            start_pos = np.arange(0, max(0, max(1, current_sequence_length - hop_length)), hop_length, dtype=int)
        if len(start_pos) == 0 and current_sequence_length>2:
            start_pos = [0]
        self.indices = start_pos
        self.dino_image_preprocessor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16plus-pretrain-lvd1689m")
        self.emotion_to_idx = VIDEO_EMOTIONS
        self.emotion_audio_to_idx = AUDIO_EMOTIONS
        self.video_path = video_path
        self.num_frames = 16
        self.preprocess_full_frames = T.Compose([
            CenterCropToSquare(),
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
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        sequence_length = min(self.sequence_length, len(self.video_features['emotion_video'])-start)
        available_features = set(list(self.video_features.keys()))
        output_dictionary = {}
        features_to_keep = self.features_to_keep.intersection(available_features)
        for feature_type in features_to_keep:
            if 'emotion_video' == feature_type:
                output_dictionary[feature_type] = [self.emotion_to_idx[emotion[0]] for emotion in self.video_features[feature_type]][start:start + sequence_length]
            elif 'emotion_audio' == feature_type:
                output_dictionary[feature_type] = [self.emotion_audio_to_idx[emotion] for emotion in self.video_features[feature_type]][start:start + sequence_length]
            elif 'rPPG' == feature_type:
                output_dictionary['rPPG'] = np.array(self.video_features[feature_type][start:start+sequence_length])
            else:
                output_dictionary['length'] = len(self.video_features[feature_type])
                output_dictionary[feature_type] = self.video_features[feature_type][start:start+sequence_length]
        if "bbox_mouth" in self.features_to_keep:
            segmentations = rearrange(output_dictionary['face_parse'], 'b c h w->b h w c').numpy()[:, :, :, 0]
            bboxes = []
            for segmentation in segmentations:
                bbox_low_lips = MavosFeatureDataset.get_bounding_box(None, segmentation, target_label=12)
                bbox_upper_lips = MavosFeatureDataset.get_bounding_box(None, segmentation, target_label=11)
                valid_bboxes = [b for b in [bbox_low_lips, bbox_upper_lips] if b is not None]
                if valid_bboxes:
                    x_min = min(b[0] for b in valid_bboxes)
                    y_min = min(b[1] for b in valid_bboxes)
                    x_max = max(b[2] for b in valid_bboxes)
                    y_max = max(b[3] for b in valid_bboxes)
                    bbox_mouth = [x_min, y_min, x_max, y_max]
                else:
                    bbox_mouth = None
                bboxes.append(bbox_mouth)
            output_dictionary["bbox_mouth"] = MavosFeatureDataset.interpolate_boxes_np(bboxes)
        output_dictionary = MavosFeatureDataset.stack_sequence(output_dictionary, sequence_length)
        if "full_frames" in self.features_to_keep and "raw_audio_features" in self.features_to_keep:
            output_dictionary['full_frames'], fps = self.read_frames(indices = self.video_features['frame_idx_in_original_video'][start:])
            output_dictionary['raw_audio_features'] = self._wav2fbank(indices = self.video_features['frame_idx_in_original_video'][start:], fps = fps)

        audio_features = self.audio_features
        # print(audio_features.max(), audio_features.min())
        if self.synchronize_audio_features:
            audio_features = align_audio_to_frames_by_fps(audio_features, self.video_features['frame_idx_in_original_video'][start:], self.video_features['fps'])
            # print(audio_features.shape)
            T = audio_features.shape[0]
            feat_shape = audio_features.shape[1:]
            if T > sequence_length:
                audio_features = audio_features[: sequence_length]
            elif T < sequence_length:
                pad_shape = ( sequence_length - T,) + feat_shape
                pad = np.zeros(pad_shape, dtype=audio_features.dtype)
                audio_features = np.concatenate([audio_features, pad], axis=0)
        output_dictionary['audio_features'] = audio_features
        if "frames" in features_to_keep:
            output_dictionary["frames"] = crop_and_resize(torch.from_numpy(output_dictionary["frames"]), torch.from_numpy(output_dictionary['bbox_mouth']))
            output_dictionary["frames"] = self.dino_image_preprocessor(output_dictionary["frames"], return_tensors="pt").pixel_values
        output_dictionary['video_path'] = self.video_path
        return output_dictionary


    def read_frames(self, indices):

            vr = VideoDecoder(self.video_path, device="cpu", num_ffmpeg_threads=0)
            total_frames = vr.metadata.num_frames
            fps = vr.metadata.average_fps
            frame_indices = np.linspace(indices[0], indices[-1], self.num_frames).astype(int)
            frames = vr.get_frames_at(frame_indices).data
            frames = self.preprocess_full_frames(frames)    
            return frames, fps

    def _wav2fbank(self, indices, fps):
        start_time = indices[0]/fps
        end_time = max(start_time+1, indices[-1]/fps)
        try:
            decoder = AudioDecoder(self.video_path)
            audio_sample = decoder.get_samples_played_in_range(start_time, end_time)
        except Exception as e:
            decoder = AudioDecoder("assets/real/9rjQ5sfeUTg_out_151_2.mp4")
            audio_sample = decoder.get_samples_played_in_range(0, 8)
        audio_sample.data = audio_sample.data - audio_sample.data.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(audio_sample.data, htk_compat=True, sample_frequency=audio_sample.sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([self.target_length, 128]) + 0.01
            print('Failed to extract audio features')

        target_length = self.target_length

        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length, ), mode='linear', align_corners=False).transpose(1,2).squeeze(0)
        fbank = (fbank - self.norm_mean) / (self.norm_std)
        return fbank


        