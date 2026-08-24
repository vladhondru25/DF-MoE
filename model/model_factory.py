
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.moe import MultimodalMoEDetector, BehaviorEncoder, SpatialEncoder, SemanticEncoder, rPPGEncoder
from model.moe import MultimodalMoEDetector as MultimodalMoEDetectorLast
# TODO: Move some of the values in a config file

def create_model(checkpoint_path):
    device="cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_name = os.path.basename(checkpoint_path)
    if 'last_moe' in checkpoint_name:
        model = MultimodalMoEDetectorLast(
            embed_dim=128, 
            num_experts=6, 
            k=2
        ).to(device)
        feature_types=["hp", "gaze", "emotion_video", "emotion_audio", "rPPG", "face_parse", "audio_features", "frames", "bbox_mouth","full_frames", "raw_audio_features"]
        model_type='moe'
    elif 'moe' in checkpoint_name:
        model = MultimodalMoEDetector(
            embed_dim=128, 
            num_experts=6, 
            k=2
        ).to(device)
        feature_types=["hp", "gaze", "emotion_video", "emotion_audio", "rPPG", "face_parse", "audio_features", "frames", "bbox_mouth","full_frames", "raw_audio_features"]
        model_type='moe'
    elif 'gaze' in checkpoint_name:
        model = BehaviorEncoder(output_dim = 128, use_classifier=True).to(device)
        feature_types=["hp", "gaze"]
        model_type='head_pose_gaze'
    elif 'parse' in  checkpoint_name:
        model = SpatialEncoder(input_channels=1, output_dim=128, use_classifier=True).to(device)
        feature_types=[ "face_parse"]
        model_type='face_segmaps'
    elif 'emotion' in checkpoint_name:
        model = SemanticEncoder(vid_features=64, aud_features=64, 
                                output_dim=128, use_classifier=True).to(device)
        feature_types=[ "emotion_video", "emotion_audio"]
        model_type='emotion'
    elif 'av_enc' in checkpoint_name:
        model = MultimodalMoEDetector(
            embed_dim=128, 
            num_experts=6, 
            k=2
        ).to(device)
        model = model.audio_video_encoder
        model.return_features = False
        model_type="audio_video_transformer"
        feature_types=["frames", "audio_features"]
    elif 'rPPG' in checkpoint_name:
        model = rPPGEncoder(input_features=512, output_dim=128, use_classifier=True).to(device)
        feature_types=["rPPG"]
        model_type='rPPG'
    else:
        raise ValueError("Unrecognized model weights")
    return model, feature_types, model_type