import argparse
import os

import datasets
from datasets import load_from_disk, concatenate_datasets
import torch
import numpy as np
import wandb
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image

from utils_stats import calculate_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-split deepfake detection stats against the MAVOS-DD dataset.")
    parser.add_argument("--predictions_paths", type=str, required=True,
                         help="Comma-separated list of Hugging Face predictions dataset paths to concatenate.")
    parser.add_argument("--mavos_path", type=str, default="/mnt/data/datasets/MAVOS-DD",
                         help="Path to the MAVOS-DD dataset (loaded via load_from_disk) used to build the evaluation splits.")
    parser.add_argument("--splits", type=str, default="closed-set,open-model,open-language,open-set",
                         help="Comma-separated MAVOS-DD evaluation splits to compute stats for. Choices: closed-set, open-model, open-language, open-set, validation.")
    parser.add_argument("--wandb_project", type=str, default="MOE deepfake stats")
    parser.add_argument("--wandb_name", type=str, default=None,
                         help="Defaults to the basename of the first predictions path.")
    parser.add_argument("--output_dir", type=str, default="confusion_matrices")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prediction_paths = [path.strip() for path in args.predictions_paths.split(",") if path.strip()]
    splits_to_evaluate = [split.strip() for split in args.splits.split(",") if split.strip()]
    wandb_name = args.wandb_name or os.path.basename(prediction_paths[0])
    wandb.init(project=args.wandb_project, name=wandb_name)

    predictions = concatenate_datasets([load_from_disk(path) for path in prediction_paths])

    video_level_predictions = {}
    for entry in predictions:
        if entry['video_path'] not in video_level_predictions:
            video_level_predictions[entry['video_path']] = {'pred': [entry['prediction']], "true": [entry['gt']]}
        else:
            video_level_predictions[entry['video_path']]['true'].append(entry['gt'])
            video_level_predictions[entry['video_path']]['pred'].append(entry['prediction'])

    mavos_dd = load_from_disk(args.mavos_path, keep_in_memory=False)

    output_dir = os.path.join(args.output_dir, wandb_name)
    os.makedirs(output_dir, exist_ok=True)

    entries_table = []
    for split_to_evaluate in splits_to_evaluate:
        if split_to_evaluate == 'validation':
            curr_split = mavos_dd.filter(lambda sample: sample['split'] == "validation")
        else:
            split_closed_set = mavos_dd.filter(lambda sample: sample['split'] == "test" and sample['open_set_model'] == False and sample["open_set_language"] == False)
        if split_to_evaluate == "closed-set":
            # Test closed-set
            curr_split = split_closed_set
        elif split_to_evaluate == "open-model":
            # Open model
            curr_split = datasets.concatenate_datasets([
                split_closed_set,
                mavos_dd.filter(lambda sample: sample['split'] == "test" and sample['open_set_model'] == True and sample["open_set_language"] == False)
            ])
        elif split_to_evaluate == "open-language":
            # Open language
            curr_split = datasets.concatenate_datasets([
                split_closed_set,
                mavos_dd.filter(lambda sample: sample['split'] == "test" and sample['open_set_model'] == False and sample["open_set_language"] == True)
            ])
        elif split_to_evaluate == "open-set":
            # Open set
            curr_split = mavos_dd.filter(lambda sample: sample['split'] == "test")

        y_pred, y_true = [], []
        for sample in curr_split:
            if sample['video_path'] in video_level_predictions:
                entry = video_level_predictions[sample["video_path"]]
                # Movie-level
                y_pred.append(np.mean(entry["pred"]))
                y_true.append(np.min(entry["true"]))
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_true_labels = np.array(['Fake' if l == 0 else 'Real' for l in y_true])
        y_pred_labels = np.array(['Fake' if l < 0.5 else 'Real' for l in y_pred])

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true_labels, y_pred_labels, normalize='true',
            display_labels=['Fake', "Real"], cmap='Blues')
        disp.ax_.set_title(split_to_evaluate)
        confusion_matrix_path = os.path.join(output_dir, f"{split_to_evaluate}.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        wandb.log({split_to_evaluate: wandb.Image(Image.open(confusion_matrix_path))})

        stats = calculate_stats(torch.from_numpy(y_pred).float(), torch.from_numpy(y_true).float())
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']  # accuracy is identical across per-class entries here, not class-wise accuracy

        entries_table.append((split_to_evaluate, mAP, mAUC, acc))

    table = wandb.Table(columns=['split', "mAP", "AUC", "acc"], data=entries_table)
    wandb.log({'stats': table})
