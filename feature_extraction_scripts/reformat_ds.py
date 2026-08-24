import datasets
import os
import sys
from PIL import Image
import pickle
import glob
import torch
from datasets import Features, Array2D, Value, Dataset, List, Video
from moviepy import ImageSequenceClip
import io
import numpy as np
import tempfile
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import get_fps_opencv


def frames_to_mp4_bytes(frames, fps=25):

    clip = ImageSequenceClip(frames, fps=fps)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    # write_videofile expects a filename or file-like object with a .write() method
    clip.write_videofile(tmp_path, codec="libx264", audio=False, logger=None)
    clip.close()

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    os.remove(tmp_path)
    return video_bytes


def find_data_pkl_paths(directory_path):
    """Recursively find data.pkl files under directory_path, returning absolute paths."""
    return sorted(
        os.path.join(root, file_name)
        for root, _, files in os.walk(directory_path)
        for file_name in files
        if file_name == "data.pkl"
    )


def generator_features(root_dir, video_data_dir):
    for data_pkl_path in find_data_pkl_paths(root_dir):
        video_path = os.path.dirname(data_pkl_path)

        try:
            with open(data_pkl_path, "rb") as file:
                signals = pickle.load(file)
        except:
            continue

        # The original video mirrors the same relative path (minus the trailing
        # video-name directory becoming a .mp4 file) under video_data_dir.
        video_path_mp4 = os.path.relpath(video_path, root_dir) + ".mp4"
        fps = get_fps_opencv(os.path.join(video_data_dir, video_path_mp4))
        keys = list(signals.keys())
        keys.sort(key=lambda x: int(x.split("_")[-1]))

        sequences = {}
        for key in keys:
            segmentation_maps = np.repeat(np.expand_dims(signals[key]['face_parse'], axis=-1), 3, axis=-1)

            rPPG = np.array(signals[key]['rPPG']).squeeze().astype(np.float16)
            if len(rPPG.shape)<2:
                rPPG=np.zeros((5, 224)).astype(np.float16)
            identity, frame_idx = key.split("_")
            if identity in sequences:
                sequences[identity]['hp'].append(np.array(signals[key]['hp']).astype(np.float16).tolist())
                sequences[identity]['bbox_mouth'].append(signals[key]['bbox_mouth'])
                sequences[identity]['gaze'].append(np.array(signals[key]['gaze']).astype(np.float16).tolist())
                sequences[identity]['emotion_video'].append(signals[key]['emotion_video'])
                sequences[identity]['emotion_audio'].append(signals[key]['emotion_audio'])
                sequences[identity]['rPPG'].append(rPPG)
                sequences[identity]['face_parse'].append(segmentation_maps)
                sequences[identity]['frame_idx_in_original_video'].append(int(frame_idx))
                sequences[identity]['frame_idx_in_sequence'].append(len(sequences[identity]['frame_idx_in_sequence']))
                try:
                    sequences[identity]['frames'].append(np.array(Image.open(os.path.join(video_path, "frames", key+'.jpg')).resize((512, 512))))
                except:
                    sequences[identity]['frames'].append(np.zeros((512, 512, 3)).astype(np.uint8))
                    continue
            else:
                sequences[identity]= {
                    'bbox_mouth':[signals[key]['bbox_mouth']], 'hp':[signals[key]['hp']],
                    'gaze':[signals[key]['gaze']], 'emotion_video':[signals[key]['emotion_video']],
                    'emotion_audio':[signals[key]['emotion_audio']],
                    'rPPG':[rPPG],
                    'face_parse':[segmentation_maps],
                    'frame_idx_in_original_video': [int(frame_idx)],
                    'frame_idx_in_sequence': [0],

                }
                try:
                    sequences[identity]['frames'] = [np.array(Image.open(os.path.join(video_path, "frames", key+'.jpg')).resize((512, 512)))]
                except:
                    sequences[identity]['frames'] = [np.zeros((512, 512, 3)).astype(np.uint8)]
                    continue
        for identity in sequences:
            sequences[identity]['face_parse'] = {"bytes": frames_to_mp4_bytes(sequences[identity]['face_parse'])}
            sequences[identity]['frames'] = {"bytes": frames_to_mp4_bytes(sequences[identity]['frames'])}
            sequences[identity]['video_path'] = video_path_mp4
            sequences[identity]['identity'] = int(identity)
            sequences[identity]['fps'] = fps
        identities = list(sequences.keys())

        for identity in identities:
            yield sequences[identity]

features_sequence = Features({
    "bbox_mouth": List(List(Value('int16'))),
    "hp": List(List(Value('float16'))),
    "gaze":List(List(Value('float16'))),
    "emotion_video": List(List(Value("string"))),
    'emotion_audio': List(Value("string")),
    "rPPG": List(List(List(Value('float16')))),
    "face_parse": Video(),
    "frame_idx_in_original_video": List(Value('int16')),
    "frame_idx_in_sequence": List(Value('int16')),
    "fps": Value('uint8'),
    "frames": Video(),
    "video_path": Value("string"),
    'identity': Value('int16')
})
def parse_args():
    parser = argparse.ArgumentParser(prog='reformat_ds')
    parser.add_argument("--input_path", type=str, default="interim_outputs2",
                         help="Root directory of extracted per-frame features (data.pkl files), searched recursively.")
    parser.add_argument("--video_data_dir", type=str, default="/home/eivor/data/MAVOS-DD",
                         help="Root directory of the original videos, mirroring --input_path's structure.")
    parser.add_argument("--output_dir", type=str, default="/mnt/e/biodeep_output/restructured_dataset",
                         help="Directory to write the reformatted Hugging Face dataset to.")
    parser.add_argument("--num_proc", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ds = Dataset.from_generator(
        generator_features,
        features=features_sequence,
        num_proc=args.num_proc,
        gen_kwargs={"root_dir": args.input_path, "video_data_dir": args.video_data_dir},
    )
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir, num_proc=args.num_proc)