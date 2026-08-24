import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_audio_classifier import WhisperEmotionClassifier, CREMA_EMO_MAP, DEF_LABELS


def load_audio_emotion_model() -> WhisperEmotionClassifier:
    model = WhisperEmotionClassifier(whisper_model_name="openai/whisper-tiny", num_classes=len(CREMA_EMO_MAP))
    
    weights = torch.load(
        "model_checkpoints/best_model_audio_emotion.pt",
        map_location=torch.device("cpu")
    )["model_state"]
    model.load_state_dict(weights)
    
    model.eval()
    
    return model

def extract_centered_whisper_segments(encoder_outputs, frame_index, fps=30, sr=16000):
    """
    Extract 2s (≈200 frames) Whisper feature windows centered on given video frames.
    
    Args:
        encoder_outputs: np.ndarray of shape (N, d)  # last_hidden_state[0]
        frame_index: int — video frame index
        fps: video frames per second
        sr: sampling rate (default 16000)
    
    Returns:
        List of np.ndarray — each of shape (80, 200)
    """
    feature_rate = (sr / 160) / 2   # 100 / 2 = 50 encoder frames/s
    total_frames = encoder_outputs.shape[0]
    half_window = int(feature_rate)  # 1 s = 50 encoder frames

    center = int((frame_index / fps) * feature_rate)
    start = max(0, center - half_window)
    end = min(total_frames, center + half_window)
    segment = encoder_outputs[start:end]

    # pad if necessary (use zeros)
    if segment.shape[0] < 2 * half_window:
        pad_left = max(0, half_window - center)
        pad_right = max(0, (center + half_window) - total_frames)
        segment = torch.nn.functional.pad(segment, (0, 0, pad_left, pad_right))
        
    return segment
def find_audio_feature_paths(directory_path):
    """Recursively find audio_features.pt files under directory_path, returning absolute paths."""
    return sorted(str(path) for path in Path(directory_path).rglob("audio_features.pt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run audio emotion inference on extracted whisper audio features and update the corresponding data.pkl files.")
    parser.add_argument("--input_path", type=str, default="interim_outputs",
                         help="Root directory to recursively search for audio_features.pt files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_audio_emotion_model()

    files_with_loading_errors = []
    files_with_no_audio = []

    audio_feature_paths = find_audio_feature_paths(args.input_path)

    for audio_features_path in tqdm(audio_feature_paths):
        # audio_features.pt and data.pkl are always written side by side, regardless of
        # how deep/shallow the surrounding directory structure is.
        video_dir = os.path.dirname(audio_features_path)
        data_pkl_path = os.path.join(video_dir, "data.pkl")

        try:
            with open(data_pkl_path, "rb") as f:
                data = pickle.load(f)
        except:
            files_with_loading_errors.append(data_pkl_path)
            continue

        try:
            audio_features = torch.load(audio_features_path)[0]
        except:
            files_with_no_audio.append(data_pkl_path)
            continue

        for frame_name, _ in data.items():
            frame_index = int(frame_name.split("_")[1])

            last_hidden_state = extract_centered_whisper_segments(audio_features, frame_index)

            with torch.no_grad():
                emotion_logits = model.forward_precomputed(last_hidden_state.unsqueeze(0))
                emotion_logits = emotion_logits.numpy().astype(np.float16)
                emotion_pred = DEF_LABELS[np.argmax(emotion_logits)]
            data[frame_name]["emotion_audio"] = emotion_pred

        with open(data_pkl_path, "wb") as f:
            pickle.dump(data, f)

    print(files_with_loading_errors)


if __name__ == "__main__":
    main()
