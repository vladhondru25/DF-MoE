import argparse
import datasets
import os
from pathlib import Path
from PIL import Image
import pickle
import glob
import torch
from datasets import Features, Array2D, Value, Dataset, List, Video
from moviepy import ImageSequenceClip
import io
import numpy as np
import tempfile


def find_audio_feature_paths(directory_path):
    """Recursively find audio_features.pt files under directory_path, returning absolute paths."""
    return sorted(str(path) for path in Path(directory_path).rglob("audio_features.pt"))


def generator_audio_features(root_dir):
    for audio_features_path in find_audio_feature_paths(root_dir):
        video_path = os.path.dirname(audio_features_path)

        try:
            audio_features = torch.load(audio_features_path).cpu().numpy().squeeze()
        except:
            continue

        video_path_mp4 = os.path.relpath(video_path, root_dir) + ".mp4"
        yield {
            "video_path": video_path_mp4,
            "audio_features": audio_features
        }


features_audio_features = Features({
    "video_path": Value("string"),
    "audio_features": Array2D(shape=(1500, 384), dtype='float16')
})


def parse_args():
    parser = argparse.ArgumentParser(prog='reformat_ds_audio')
    parser.add_argument("--input_path", type=str, default="interim_outputs2",
                         help="Root directory of extracted audio features (audio_features.pt files), searched recursively.")
    parser.add_argument("--output_dir", type=str, default="/mnt/e/biodeep_output/restructured_dataset_audio",
                         help="Directory to write the reformatted Hugging Face dataset to.")
    parser.add_argument("--num_proc", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ds_audio = Dataset.from_generator(
        generator_audio_features,
        features=features_audio_features,
        gen_kwargs={"root_dir": args.input_path},
    )
    os.makedirs(args.output_dir, exist_ok=True)
    ds_audio.save_to_disk(args.output_dir, num_proc=args.num_proc)
# n_shards = 30
# for i in range(n_shards):
#     shard = ds_audio.shard(num_shards=n_shards, index=i, contiguous=True)
#     shard.to_parquet(f"test_dataset_audio/shard_{i:04d}.parquet", compression="zstd")