import argparse
import os

from datasets import load_from_disk, concatenate_datasets
import torch
import numpy as np
import wandb
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image

from utils_stats import calculate_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Compute video-level deepfake detection stats for a non-MAVOS-DD predictions dataset.")
    parser.add_argument("--predictions_paths", type=str, required=True,
                         help="Comma-separated list of Hugging Face predictions dataset paths to concatenate.")
    parser.add_argument("--split_name", type=str, default="Deepfake eval",
                         help="Label used for the confusion matrix title and the wandb stats row.")
    parser.add_argument("--wandb_project", type=str, default="MOE deepfake stats")
    parser.add_argument("--wandb_name", type=str, default=None,
                         help="Defaults to the basename of the first predictions path.")
    parser.add_argument("--output_dir", type=str, default="confusion_matrices")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prediction_paths = [path.strip() for path in args.predictions_paths.split(",") if path.strip()]
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

    y_pred, y_true = [], []
    y_pred_seq, y_true_seq = [], []
    for video_path, entry in video_level_predictions.items():
        y_pred.append(np.mean(entry["pred"]))
        y_true.append(np.max(entry["true"]))
        y_pred_seq.extend(entry['pred'])
        y_true_seq.extend(entry['true'])

    print(np.unique(list(np.array(y_true)), return_counts=True))
    print(np.unique(list(np.array(y_pred) > 0.5), return_counts=True))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred_seq = np.array(y_pred_seq)
    y_true_seq = np.array(y_true_seq)
    y_true_labels = np.array(['Fake' if l == 0 else 'Real' for l in y_true])
    y_pred_labels = np.array(['Fake' if l < 0.5  else 'Real' for l in y_pred])

    y_real = y_true_seq[y_true_seq == 1]
    y_pred_real = y_pred_seq[y_true_seq == 1] > 0.5
    # print("Acc real: ", np.mean(y_real == y_pred_real), len(y_real))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true_labels, y_pred_labels, normalize='true',
        display_labels=['Fake', "Real"], cmap='Blues')
    disp.ax_.set_title(args.split_name)

    output_dir = os.path.join(args.output_dir, wandb_name)
    os.makedirs(output_dir, exist_ok=True)
    confusion_matrix_path = os.path.join(output_dir, f"{args.split_name}.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    wandb.log({args.split_name: wandb.Image(Image.open(confusion_matrix_path))})

    stats = calculate_stats(torch.from_numpy(y_pred).float(), torch.from_numpy(y_true).float())
    print([stat['AP'] for stat in stats])
    print([stat['auc'] for stat in stats])
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc']  # accuracy is identical across per-class entries here, not class-wise accuracy

    table = wandb.Table(columns=['split', "mAP", "AUC", "acc"], data=[(args.split_name, mAP, mAUC, acc)])
    wandb.log({'stats': table})
