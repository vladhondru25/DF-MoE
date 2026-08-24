import argparse
import os
import random
import sys
import time
from PIL import Image

from datasets import load_from_disk, Value, Dataset, Features
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_deepfake_eval import DeepfakeEvalFeatureDataset
from dataset.dataset_mavosv2 import MavosFeatureDataset, relative_deltas, mask_outside_bbox
from model.video_audio_transformer import VideoAudioTransformer
from model.video_audio_transformer_self_attn import VideoAudioTransformer as VideoAudioTransformerSA
from dataset.dataset_fakeavceleb import FakeAVCelebFeatureDataset
from dataset.dataset_avlips import AVLipsFeatureDataset
from dataset.dataset_celebdf import CelebDFFeatureDataset
import torch
from torch.utils.data import ConcatDataset, Subset
from dataset.dataset_social_media import SocialMediaFeatureDataset
import wandb
from stats.utils_stats import calculate_stats

FEATURES_TO_KEEP = ["full_frames", "raw_audio_features", "face_parse", "frames", "bbox_mouth", "audio_features"]


def build_dataset(name, split, args):
    common = dict(
        features_to_keep=FEATURES_TO_KEEP,
        sequence_length=args.sequence_length,
        hop_length=args.hop_length,
        frame_resolution=args.frame_resolution,
        transforms_dictionary={},  # {"hp": relative_deltas, "gaze": relative_deltas},
        synchronize_audio_features=True,
        return_video_path=True,
    )

    if name == "avlips":
        return AVLipsFeatureDataset(
            "/mnt/data/datasets/AVLips", "/mnt/data/datasets/features_avlips/video",
            "/mnt/data/datasets/features_avlips/audio", split, **common)
    elif name == "celebdf":
        return CelebDFFeatureDataset(
            "/mnt/data/datasets/celebdf_v2", "/mnt/data/datasets/finalds_celebdf/video",
            None, split, **common)
    elif name == "social_media":
        # This dataset only has a "test" split; the original training pipeline always draws from it.
        return SocialMediaFeatureDataset(
            "/mnt/data/datasets/social_media", "/mnt/data/datasets/features_social_media/video",
            "/mnt/data/datasets/features_social_media/audio", "test", **common)
    elif name == "mavos":
        return MavosFeatureDataset(
            "/mnt/data/datasets/MAVOS-DD", None,
            "/mnt/data/datasets/features_mavos_complete/video", None,
            "/mnt/data/datasets/features_mavos_complete/audio", split, **common)
    elif name == "fakeavceleb":
        return FakeAVCelebFeatureDataset(
            "../datasets/FakeAVCeleb", "/mnt/data/datasets/finalds_fakeavceleb_2/video",
            "/mnt/data/datasets/finalds_fakeavceleb_2/audio", split, **common)
    else:
        raise ValueError(f"Unknown dataset name '{name}'. Available: {', '.join(DATASET_NAMES)}")


DATASET_NAMES = ["avlips", "celebdf", "social_media", "mavos", "fakeavceleb"]


def create_random_balanced_dataset(datasets):
    min_len = min(len(ds) for ds in datasets)

    subsets = [Subset(ds, torch.randperm(len(ds))[:min_len].tolist()) for ds in datasets]
    combined_dataset = ConcatDataset(subsets)

    print(f"Combined Dataset Size: {len(combined_dataset)} ({min_len} samples from each of {len(datasets)} datasets)")
    return combined_dataset


def main(args) -> None:
    wandb.init(project=args.wandb_project)
    pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    my_model = VideoAudioTransformerSA(args.device, pretrained_model_name, d_video=384, d_audio=384)
    my_model = my_model.to(device=args.device)
    # my_model.load_state_dict(torch.load("/home/galadriel/projects/BiodeepDetection/model_checkpoints/cross_transformer/only_mavos/epoch_7.pth"))
    # Suppose your model has a submodule `visual_encoder`
    for param in my_model.visual_backbone.parameters():
        param.requires_grad = False

    optim = AdamW(filter(lambda p: p.requires_grad, my_model.parameters()), lr=args.lr)
    # Reduce learning rate by factor of 0.1 every 2 epochs
    scheduler = StepLR(optim, step_size=2, gamma=0.5)
    criterion = BCEWithLogitsLoss()

    # Initialize best accuracy tracking
    best_accuracy = 0.0

    train_dataset_names = [name.strip() for name in args.train_datasets.split(",") if name.strip()]
    unknown_names = [name for name in train_dataset_names if name not in DATASET_NAMES]
    if unknown_names:
        raise ValueError(f"Unknown dataset name(s) {unknown_names}. Available: {', '.join(DATASET_NAMES)}")

    train_datasets = [build_dataset(name, "train", args) for name in train_dataset_names]

    val_dataset = build_dataset(args.val_dataset, "validation", args)
    val_datalaoder = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for epoch in range(args.epochs):
        my_model.train()
        train_dataset = create_random_balanced_dataset(train_datasets)
        train_datalaoder = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        loop = tqdm(train_datalaoder, total=len(train_datalaoder), leave=True, desc=f"Epoch {epoch+1} / {args.epochs}")
        total_loss=0
        total_correct=0
        total_count=0
        for batch_idx, batch in enumerate(loop):
            images = batch["frames"].to(device=args.device)
            images_masks = batch["padding_mask"].to(device=args.device)
            audios = batch["audio_features"].float().to(device=args.device)
            labels = batch["label"].float().to(device=args.device)

            logits = my_model(images, audios, images_masks).squeeze(1)

            loss = criterion(logits, labels)

            optim.zero_grad() # Reset gradients

            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), args.max_grad_norm)

            optim.step()

            # update tqdm bar
            # loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct = (preds.cpu() == labels.cpu().long()).sum().item()

            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_count += labels.size(0)
            if batch_idx % 100==0:
                train_acc = total_correct/total_count
                train_loss = total_loss/ total_count
                wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
            })

        my_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_datalaoder, total=len(val_datalaoder), leave=False, desc=f"Validating epoch {epoch+1} / {args.epochs}"):
                images =batch["frames"].to(device=args.device)
                images_masks = batch["padding_mask"].to(device=args.device)
                audios = batch["audio_features"].float().to(device=args.device)
                labels = batch["label"].float().to(device=args.device)

                logits = my_model(images, audios, images_masks).squeeze(1)

                loss = criterion(logits, labels)

                # Predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                correct = (preds == labels.long()).sum().item()

                # Accumulate
                total_loss += loss.item() * labels.size(0)
                total_correct += correct
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        current_lr = scheduler.get_last_lr()[0]
        print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Learning Rate: {current_lr:.2e}")

        # Save model if validation accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pth")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(my_model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")

        # Step the scheduler at the end of each epoch
        scheduler.step()

def inference(args):
    test_dataset = build_dataset(args.test_dataset, "test", args)
    test_datalaoder = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    my_model = VideoAudioTransformerSA(args.device, pretrained_model_name, d_video=384, d_audio=384)
    my_model.load_state_dict(torch.load(args.model_checkpoint_path))

    my_model = my_model.to(device=args.device)
    my_model.eval()
    wandb.init(project=args.wandb_project)
    predictions = []
    gt = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_datalaoder, total=len(test_datalaoder), leave=False, desc="Inferencing")):
            images = batch["frames"].to(device=args.device)
            images_masks = batch["padding_mask"].to(device=args.device)
            audios = batch["audio_features"].float().to(device=args.device)
            labels = batch["label"].float().to(device=args.device)

            logits = my_model(images, audios, images_masks).squeeze(1)

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            predictions.extend(probs.cpu())
            gt.extend(labels.cpu())
            if batch_idx % 100 == 0:
                stats = calculate_stats(np.array(predictions), np.array(gt))
                mAP = np.mean([stat['AP'] for stat in stats])
                mAUC = np.mean([stat['auc'] for stat in stats])
                acc = np.mean([stat['acc'] for stat in stats])
                wandb.log({"mAP": mAP, "AUC": mAUC, "Accuracy": acc})

            for i, video_path in enumerate(batch['video_path']):
                yield {"video_path": video_path, "prediction": probs[i], "gt": labels[i].long()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train/run inference/score the audio-video cross transformer model.")
    parser.add_argument("--action", type=str, choices=["train", "inference", "test"], default="train",
                         help="train: fit the model. inference: run a checkpoint over --test_dataset and save predictions. test: score an existing predictions dataset.")

    parser.add_argument("--train_datasets", type=str, default="mavos,avlips,social_media",
                         help=f"Comma-separated dataset names combined (equally, randomly resampled each epoch) for training. Available: {', '.join(DATASET_NAMES)}")
    parser.add_argument("--val_dataset", type=str, default="mavos", choices=DATASET_NAMES,
                         help="Dataset used for validation during training.")
    parser.add_argument("--test_dataset", type=str, default="fakeavceleb", choices=DATASET_NAMES,
                         help="Dataset used for the inference action.")

    parser.add_argument("--checkpoint_dir", type=str, default="model_checkpoints/cross_transformer/avlips_mavos_social_media_equalized",
                         help="Directory new best checkpoints are saved to during training.")
    parser.add_argument("--model_checkpoint_path", type=str, default=None,
                         help="Checkpoint to load for the inference action.")
    parser.add_argument("--predictions_output_dir", type=str, default="predictions/av_transformer_favc_original_trained_on_mavos_avlips",
                         help="Where the inference action saves its predictions dataset.")
    parser.add_argument("--test_predictions_path", type=str, default="results/predictions2",
                         help="Predictions dataset scored by the test action.")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--hop_length", type=int, default=40)
    parser.add_argument("--frame_resolution", type=int, default=512)
    parser.add_argument("--wandb_project", type=str, default="AV Transformer training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.action == "train":
        main(args)
    elif args.action == "inference":
        if not args.model_checkpoint_path:
            raise ValueError("--model_checkpoint_path is required for --action inference")

        features = Features({
            "video_path": Value("string"),
            "prediction": Value("float32"),
            "gt": Value("int32")
        })

        predictions_ds = Dataset.from_generator(lambda: inference(args), features=features)
        predictions_ds.save_to_disk(args.predictions_output_dir)
    elif args.action == "test":
        predictions_ds = load_from_disk(args.test_predictions_path)
        y_true = predictions_ds['gt']
        y_pred = [1 if p > 0.5 else 0 for p in predictions_ds['prediction']]

        print(classification_report(y_true, y_pred, digits=4))
        print(confusion_matrix(y_true, y_pred))
