#!/usr/bin/env python3
"""
Train an emotion classifier on CREMA-D using Hugging Face Whisper (tiny) as backbone.

Example:
    python train_whisper_cremad_hf.py \
        --data_dir /path/to/CREMA-D/AudioWAV \
        --output_dir ./runs/whisper_tiny_cremad \
        --epochs 10 --batch_size 8
"""

import os
import argparse
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from transformers import WhisperFeatureExtractor, WhisperModel, get_linear_schedule_with_warmup

# =============================
#  Dataset & utilities
# =============================
NUM_WORKERS = 0
CREMA_EMO_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
DEF_LABELS = list(CREMA_EMO_MAP.values())


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CREMADataset(Dataset):
    """Loads audio files from CREMA-D and extracts emotion labels from filenames."""

    def __init__(self, filepaths: List[Path], labels: List[int], feature_extractor = None, max_duration: float = 6.0, sample_rate: int = 16000):
        self.feature_extractor = feature_extractor
        
        self.filepaths = filepaths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_len = int(max_duration * sample_rate)
        self.frame_step = int(max_duration * 1000 / 10)

    @classmethod
    def from_dir(cls, data_dir: str, allowed_emotions=CREMA_EMO_MAP, sample_rate=16000):
        data_dir = Path(data_dir)
        wavs = sorted(list(data_dir.rglob("*.wav")))
        filepaths, labels = [], []
        for p in wavs:
            parts = p.name.split("_")
            if len(parts) < 3:
                continue
            emo_code = parts[2].upper()
            if emo_code not in allowed_emotions:
                continue
            emo = allowed_emotions[emo_code]
            filepaths.append(p)
            labels.append(DEF_LABELS.index(emo))
        return cls(filepaths, labels, sample_rate=sample_rate)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Extract features
        input_features = self.feature_extractor(
            waveform, 
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        # Calculate which frame corresponds to your 20ms window
        # Each frame represents ~10ms (hop length of 160 samples)
        # start_time_ms = 1000  # Start at 1 second, for example
        # frame_index = int(start_time_ms / 10)  # 10ms per frame
        frame_index = random.randint(0, input_features.shape[-1]-self.frame_step-1)

        # Extract 2 frames (~20ms worth)
        features_20ms = input_features[:, :, frame_index:frame_index+self.frame_step]
        pad = 3000 - self.frame_step
        features_20ms = F.pad(features_20ms, pad=(0, pad), mode="constant", value=0)

        label = self.labels[idx]
        
        return input_features.squeeze(0), label, str(path)
        return features_20ms.squeeze(0), label, str(path)


def collate_fn(batch):
    waveforms = torch.stack([x[0] for x in batch])
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    paths = [x[2] for x in batch]
    return waveforms, labels, paths


# =============================
#  Model
# =============================

class WhisperEmotionClassifier(nn.Module):
    """Wraps WhisperModel encoder + classifier head."""

    def __init__(self, whisper_model_name: str, num_classes: int, freeze_encoder: bool = False, dropout: float = 0.2):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        hidden_size = self.whisper.config.d_model

        if freeze_encoder:
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2 // 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 // 3, num_classes),
        )
        
    def forward_precomputed(self, last_hidden_state):
        x = last_hidden_state.permute(0, 2, 1)  # (B, D, T)
        x = self.pool(x).squeeze(-1)            # (B, D)
        
        logits = self.classifier(x)
        
        return F.softmax(logits)

    def forward(self, input_features):
        # input_features: (B, n_mels, T)
        enc_out = self.whisper.encoder(input_features)
        x = enc_out.last_hidden_state  # (B, T, D)
        x = x.permute(0, 2, 1)         # (B, D, T)
        x = self.pool(x).squeeze(-1)   # (B, D)
        logits = self.classifier(x)
        return logits


# =============================
#  Training & Validation
# =============================

def make_dataloaders(feature_extractor, data_dir, batch_size, val_split=0.15, sample_rate=16000, max_duration=0.02):
    ds = CREMADataset.from_dir(data_dir, sample_rate=sample_rate)
    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    n_val = int(val_split * len(ds))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    def subset(idxs):
        return [ds.filepaths[i] for i in idxs], [ds.labels[i] for i in idxs]

    train_files, train_labels = subset(train_idx)
    val_files, val_labels = subset(val_idx)

    train_ds = CREMADataset(train_files, train_labels, feature_extractor=feature_extractor, sample_rate=sample_rate, max_duration=max_duration)
    val_ds = CREMADataset(val_files, val_labels, feature_extractor=feature_extractor, sample_rate=sample_rate, max_duration=max_duration)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS // 2, collate_fn=collate_fn)
    return train_dl, val_dl


def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels, _ in tqdm(loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Feature extractor (for Whisper input)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)

    # Datasets
    train_dl, val_dl = make_dataloaders(feature_extractor, args.data_dir, args.batch_size, args.val_split, feature_extractor.sampling_rate, args.max_duration)
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # Model
    model = WhisperEmotionClassifier(args.model_name, num_classes=len(DEF_LABELS), freeze_encoder=args.freeze_encoder, dropout=args.dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", leave=False)
        running_loss = 0.0
        step = 0

        for inputs, labels, _ in pbar:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step = step + 1
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        val_acc, val_f1 = validate(model, val_dl, device)
        print(f"Epoch {epoch}: TrainLoss={running_loss/len(train_dl):.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_f1": val_f1,
            "args": vars(args),
        }
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pt")
        # torch.save(ckpt, ckpt_path)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(ckpt, best_path)
            print(f"New best model saved at {best_path} (F1={val_f1:.4f})")

    print("Training complete!")


# =============================
#  CLI
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="Train Whisper Tiny on CREMA-D for emotion classification")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./runs/whisper_cremad_hf")
    p.add_argument("--model_name", type=str, default="openai/whisper-tiny")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=7.5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # p.add_argument("--max_duration", type=float, default=0.2)
    p.add_argument("--max_duration", type=float, default=2)
    return p.parse_args()

# 50, 7.5e-4, 1e-4, 0.3  
# 100, 7.5e-4, 1e-4, 0.3 -> 68.6  


if __name__ == "__main__":
    args = parse_args()
    train(args)
