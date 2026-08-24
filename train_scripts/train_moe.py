import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from dataset.dataset_mavosv2 import MavosFeatureDataset
from dataset.datasetv2_video_level import MavosFeatureDataset as MavosFeatureDatasetVideoLevel
from dataset.dataset_avlips import AVLipsFeatureDataset
from dataset.dataset_celebdf import CelebDFFeatureDataset
from dataset.dataset_social_media import SocialMediaFeatureDataset
import argparse
import wandb

from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import copy
from einops import rearrange
from model.video_audio_transformer import VideoAudioTransformer
from model.avff.video_cav_mae import VideoCAVMAEFT
from model.moe import MultimodalMoEDetector, BehaviorEncoder, SpatialEncoder, SemanticEncoder, rPPGEncoder
import random
from dataset.dataset_deepfake_eval import DeepfakeEvalFeatureDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 32})
import os
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model import models_vit
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from model.wav2vec_aasist import W2V_AASIST
EMBED_DIM = 128         
SEQ_LEN_RPPG = 150      
SEQ_LEN_BEHAVIOR = 150  
SEQ_LEN_EMOTION = 50    
SEQ_LEN_SPATIAL = 30    
NUM_EXPERTS = 6         
TOP_K = 2               
import torch
from torch.utils.data import ConcatDataset, Subset
from model.avff.effort_detector import apply_svd_residual_to_self_attn
from model.wav2vec_aasist import W2V_AASIST
from model.cam_loss import ClassAnchorMarginLoss
from model.video_audio_transformer_self_attn import VideoAudioTransformer as VideoAudioTransformerSA
DATASET_NAMES = ["avlips", "celebdf", "social_media", "mavos"]


def build_dataset(name, split, args, feature_types):
    common = dict(
        features_to_keep=feature_types,
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
            "/mnt/data/datasets/celebdf_v2", "/mnt/data/datasets/final_ds_celebdf_2/video",
            None, split, **common)
    elif name == "social_media":
        # This dataset only has a "test" split; the training pipeline always draws from it.
        return SocialMediaFeatureDataset(
            "/mnt/data/datasets/social_media", "/mnt/data/datasets/features_social_media/video",
            "/mnt/data/datasets/features_social_media/audio", "test", **common)
    elif name == "mavos":
        dataset_class = MavosFeatureDatasetVideoLevel if args.video_level else MavosFeatureDataset
        return dataset_class(
            "/mnt/data/datasets/MAVOS-DD", None,
            "/mnt/data/datasets/features_mavos_complete/video", None,
            "/mnt/data/datasets/features_mavos_complete/audio", split, **common)
    else:
        raise ValueError(f"Unknown dataset name '{name}'. Available: {', '.join(DATASET_NAMES)}")


def create_random_balanced_dataset(datasets):
    min_len = min(len(ds) for ds in datasets)

    subsets = [Subset(ds, torch.randperm(len(ds))[:min_len].tolist()) for ds in datasets]
    combined_dataset = ConcatDataset(subsets)

    print(f"Combined Dataset Size: {len(combined_dataset)} ({min_len} samples from each of {len(datasets)} datasets)")
    return combined_dataset


def create_class_balanced_subset(ds):
    """Randomly subsample the majority class so real/fake counts match. Training only."""
    if isinstance(ds.indices, dict) and 'ds_position' in ds.indices:
        return ds
    else:
        # video-level dataset: self.indices is {video_path: [ds_positions]}, __getitem__ indexes self.video_paths
        labels = np.array([ds.video_labels[vp] for vp in ds.video_paths])

    real_idx = np.where(labels == 1)[0]
    fake_idx = np.where(labels == 0)[0]
    n = min(len(real_idx), len(fake_idx))

    balanced_idx = np.concatenate([
        np.random.choice(real_idx, n, replace=False),
        np.random.choice(fake_idx, n, replace=False),
    ])
    np.random.shuffle(balanced_idx)

    print(f"Class-balanced subset: {n} real + {n} fake (from {len(real_idx)} real, {len(fake_idx)} fake)")
    return Subset(ds, balanced_idx.tolist())


def train_model(train_ds_list, valid_ds, args, feature_types, type_of_encoder = None):
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = rank
    if type_of_encoder is None:
        model = MultimodalMoEDetector(
            embed_dim=args.embed_dim, 
            num_experts=args.num_experts, 
            k=args.top_k
        ).to(device)
        if args.resume_checkpoint is not None:
            checkpoint = torch.load(args.resume_checkpoint, weights_only=False)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if args.use_cam_loss:
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=True)
            state_dict, checkpoint = None, None
            for param in model.avff.parameters():
                param.requires_grad = False
            for param in model.audio_video_encoder.parameters():
                param.requires_grad = False
            # for param in model.face_encoder.parameters():
            #     param.requires_grad = False
        if args.ckpt_moe is not None:
            state_dict = torch.load(args.ckpt_moe)
            state_dict.pop("classification_head.0.weight")
            model.load_state_dict(state_dict, strict=False)
        if args.ckpt_av_tf is not None:
            # ckpt = torch.load(args.ckpt_av_tf)
            # new_dict = {}
            # for key in ckpt:
            #     new_dict[key[20:]] = ckpt[key]
            # model.audio_video_encoder.load_state_dict(new_dict, strict=True)
            # for param in model.audio_video_encoder.parameters():
            #     param.requires_grad = False
            ckpt = torch.load(args.ckpt_av_tf)
            model.audio_video_encoder.load_state_dict(ckpt)
            for param in model.audio_video_encoder.parameters():
                param.requires_grad = False
        if args.ckpt_spatial is not None:
            ckpt = torch.load(args.ckpt_spatial)
            model.spatial_encoder.load_state_dict(ckpt)
            for param in model.spatial_encoder.parameters():
                param.requires_grad = False
        if args.ckpt_semantic is not None:
            ckpt = torch.load(args.ckpt_semantic)
            model.semantic_encoder.load_state_dict(ckpt)
            for param in model.semantic_encoder.parameters():
                param.requires_grad = False
        if args.ckpt_behavior is not None:
            ckpt = torch.load(args.ckpt_behavior)
            model.behavior_encoder.load_state_dict(ckpt)
            for param in model.behavior_encoder.parameters():
                param.requires_grad = False
        if args.ckpt_rppg is not None:
            ckpt = torch.load(args.ckpt_rppg)
            model.rppg_encoder.load_state_dict(ckpt)
            for param in model.rppg_encoder.parameters():
                param.requires_grad = False
        if args.ckpt_audio_encoder is not None:
            model.audio_feature_extractor.load_model(args.ckpt_audio_encoder)
            for param in model.audio_feature_extractor.parameters():
                param.requires_grad = False
        if args.ckpt_avff is not None:
            # ckpt = torch.load(args.ckpt_avff)
            # new_dict = {}
            # for key in ckpt:
            #     new_dict[key[5:]] = ckpt[key]
            # model.avff.load_state_dict(new_dict, strict=True)
            # for param in model.avff.parameters():
            #     param.requires_grad = False
            ckpt = torch.load(args.ckpt_avff)
            new_dict = {}
            for key in ckpt:
                new_dict[key[7:]] = ckpt[key]
            model.avff.load_state_dict(new_dict)
            for param in model.avff.parameters():
                param.requires_grad = False
        if args.ckpt_face is not None:
            checkpoint = torch.load(args.ckpt_face, weights_only=False)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.face_encoder.load_state_dict(state_dict, strict=True)
            # for param in model.face_encoder.parameters():
            #     param.requires_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        
    task_loss_fn = nn.BCEWithLogitsLoss()
    mse_recon_loss = nn.MSELoss(reduction = 'none')
    cls_recon_loss = nn.CrossEntropyLoss()
    aux_loss_alpha = 0.01 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    validation_sampler = DistributedSampler(valid_ds, shuffle=False)
    valid_dl = DataLoader(valid_ds, sampler=validation_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    best_val_acc = 0.
    best_state = None
    best_model_path = f"model_checkpoints/best_{args.wandb_run_name}.pt"
    epoch_path = f"model_checkpoints/{args.wandb_run_name}/"
    os.makedirs(epoch_path, exist_ok=True)
    for epoch in range(args.epochs):
        train_ds = create_random_balanced_dataset(train_ds_list)
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_dl = DataLoader(train_ds, sampler =train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
        model.train()
        model.module.audio_feature_extractor.eval()
        total_loss, total_correct, total_count = 0.0, 0, 0
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(tqdm(train_dl)):
            # print(batch['video_path'])
            batch['rPPG'] = torch.mean(batch['rPPG'], dim=2)
            if "full_frames" in batch:
                batch['full_frames'] = rearrange(batch['full_frames'], 'b s c h w->b c s h w')
            if 'face_parse' in batch:
                batch['face_parse'] = torch.mean(batch['face_parse'].float(), dim=2).unsqueeze(dim=2)
                batch['face_parse'] = rearrange(batch['face_parse'], 'b s c h w->b c s h w')
                batch['face_parse'] = F.interpolate(batch['face_parse'], (batch['face_parse'].shape[2], 224, 224))
            if 'padding_mask' in batch:
                batch['padding_mask'] = batch['padding_mask'].to(device)
            # for entry in batch:
            #     print(entry, batch[entry].shape)
            input_dict = {tof: batch[tof].float().to(device) for tof in feature_types if tof != 'padding_mask'}
            y = batch['label']
            yf = y.float().to(device)
            input_dict['padding_mask'] = batch['padding_mask']
            input_dict['label'] = y
            # print(y)
            input_dict['dropout_modalities'] = args.dropout_modalities
            if type_of_encoder is None:
                if args.use_cam_loss:
                    logits, aux_loss, cam_l = model(input_dict, return_embeddings=args.use_cam_loss)
                else:
                    logits, aux_loss = model(input_dict, return_embeddings=args.use_cam_loss)

                
                task_loss = task_loss_fn(logits, yf)
                loss = task_loss
                if aux_loss is not None:
                    loss = task_loss+ aux_loss_alpha * aux_loss
                if args.use_cam_loss:
                    # print(cam_l.shape, loss.shape)
                    loss = loss + cam_l
            elif type_of_encoder == "behavior":
                logits = model(input_dict['hp'], input_dict['gaze']).squeeze(dim=-1)
                loss = task_loss_fn(logits, yf)
            elif type_of_encoder == 'spatial':
                logits = model(input_dict['face_parse']).squeeze(dim=-1)
                loss = task_loss_fn(logits, yf)
            elif type_of_encoder == 'semantic':
                logits = model( input_dict['emotion_video'],  input_dict['emotion_audio']).squeeze(dim=-1)
                loss = task_loss_fn(logits, yf)
            elif type_of_encoder == 'rPPG':
                logits = model( input_dict['rPPG']).squeeze(dim=-1)
                loss = task_loss_fn(logits, yf)
            
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct = (preds.cpu() == y.cpu().long()).sum().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += correct
            total_count += y.size(0)

            
            if i % args.log_step==0 and i > 0 and rank==0:
                train_loss = total_loss / total_count
                train_acc = total_correct / total_count
                print(f"Train epoch {epoch:02d} | loss {train_loss:.4f} | acc {train_acc:.4f}")
                wandb.log({"train_loss": train_loss, "train_acc": train_acc})
        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        # scheduler.step()
        if valid_dl is None:
            print(f"Train epoch {epoch:02d} | loss {train_loss:.4f} | acc {train_acc:.4f}")
            wandb.log({"train_loss": train_loss, "train_acc": train_acc})
            continue
        else:
            print(f"Train epoch {epoch:02d} | loss {train_loss:.4f} | acc {train_acc:.4f}")
        if rank ==0:
            current_state = model.module.state_dict() 
            torch.save(current_state, epoch_path + f"{epoch}.pth")
        model.eval()
        v_loss, v_correct, v_count = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(valid_dl):
                batch['rPPG'] = torch.mean(batch['rPPG'], dim=2)
                if "full_frames" in batch:
                    # batch['full_frames'] = batch['frames']
                    batch['full_frames'] = rearrange(batch['full_frames'], 'b s c h w->b c s h w')
                if 'face_parse' in batch:
                    batch['face_parse'] = torch.mean(batch['face_parse'].float(), dim=2).unsqueeze(dim=2)
                    batch['face_parse'] = rearrange(batch['face_parse'], 'b s c h w->b c s h w')
                    batch['face_parse'] = F.interpolate(batch['face_parse'], (batch['face_parse'].shape[2], 224, 224))
                if 'padding_mask' in batch:
                    batch['padding_mask'] = batch['padding_mask'].to(device)
                # for entry in batch:
                #     print(entry, batch[entry].shape)
                input_dict = {tof: batch[tof].float().to(device) for tof in feature_types if tof != 'padding_mask'}
                input_dict['padding_mask'] = batch['padding_mask']
                y = batch['label'].to('cuda')
                yf = y.float().to(device)
                # print(y)
                if type_of_encoder is None:
                    logits, aux_loss = model(input_dict)
                    task_loss = task_loss_fn(logits, yf)
                    loss = task_loss
                elif type_of_encoder == "behavior":
                    logits = model(input_dict['hp'], input_dict['gaze']).squeeze(dim=-1)
                    loss = task_loss_fn(logits, yf)
                elif type_of_encoder == 'spatial':
                    logits = model(input_dict['face_parse']).squeeze(dim=-1)
                    loss = task_loss_fn(logits, yf)
                elif type_of_encoder == 'semantic':
                    logits = model( input_dict['emotion_video'],  input_dict['emotion_audio']).squeeze(dim=-1)
                    loss = task_loss_fn(logits, yf)
                elif type_of_encoder == 'rPPG':
                    logits = model( input_dict['rPPG']).squeeze(dim=-1)
                    loss = task_loss_fn(logits, yf)

                preds = (torch.sigmoid(logits) > 0.5).long()
                correct = (preds.cpu() == y.cpu().long()).sum().item()

                v_loss += loss.item() * y.size(0)
                v_correct += correct
                v_count += y.size(0)

        val_loss = v_loss / v_count
        val_acc  = v_correct / v_count
        if rank ==0:
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "validation_loss": val_loss, "validation_accuracy": val_acc
            })

        print(f"Epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")
        if rank==0:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.module.state_dict())
                if best_model_path:
                    torch.save(best_state, best_model_path)
                wandb.summary["best_val_acc"] = best_val_acc
                wandb.summary["best_epoch"] = epoch
    if args.log_model:
        wandb.log_model(path=best_model_path, name=f"best_model_{'-'.join(feature_types)}")
    dist.destroy_process_group()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MOE Deepfake detection")
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--hop_length", type=int, default=40)
    parser.add_argument("--frame_resolution", type=int, default=512)
    parser.add_argument("--train_datasets", type=str, default="mavos,avlips,celebdf,social_media",
                         help=f"Comma-separated dataset names combined (equally, randomly resampled each epoch) for training. Available: {', '.join(DATASET_NAMES)}")
    parser.add_argument("--val_dataset", type=str, default="mavos", choices=DATASET_NAMES,
                         help="Dataset used for validation during training.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument("--type_of_encoder", type=str, default=None)
    parser.add_argument("--log_model", action='store_true')
    parser.add_argument("--video_level", action='store_true')
    parser.add_argument("--wandb_run_name", type=str, default="moe")
    parser.add_argument("--ckpt_av_tf", type=str, default=None)
    parser.add_argument("--ckpt_moe", type=str, default=None)
    parser.add_argument("--ckpt_spatial", type=str, default=None)
    parser.add_argument("--ckpt_behavior", type=str, default=None)
    parser.add_argument("--ckpt_rppg", type=str, default=None)
    parser.add_argument("--ckpt_semantic", type=str, default=None)
    parser.add_argument("--ckpt_avff", type=str, default=None)
    parser.add_argument("--ckpt_audio_encoder", type=str, default=None)
    parser.add_argument("--ckpt_face", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--dropout_modalities", type=float, default=0.)
    parser.add_argument("--use_cam_loss", action='store_true')
    rank = int(os.environ.get("LOCAL_RANK", 0))
    args = parser.parse_args()
    if rank==0:
        wandb.init(project="MOE deepfake detection", 
               config=vars(args), name=args.wandb_run_name)
    if args.type_of_encoder is None:
        # feature_types=["hp", "gaze", "emotion_video", "emotion_audio", "rPPG", "face_parse", "audio_features", "frames", "bbox_mouth", "full_frames", "raw_audio_features"]
        feature_types=["hp", "gaze", "emotion_video", "emotion_audio", "rPPG", "face_parse", "audio_features", "frames", "bbox_mouth", "full_frames", "raw_audio_features"]
    elif args.type_of_encoder == 'spatial':
        feature_types = ["rPPG", "face_parse"]
    elif args.type_of_encoder == 'behavior':
        feature_types = ["rPPG", "hp", 'gaze']
    elif args.type_of_encoder == 'semantic':
        feature_types = ["rPPG", "emotion_video", 'emotion_audio']
    elif args.type_of_encoder == 'rPPG':
        feature_types = ['rPPG']
    train_dataset_names = [name.strip() for name in args.train_datasets.split(",") if name.strip()]
    unknown_names = [name for name in train_dataset_names if name not in DATASET_NAMES]
    if unknown_names:
        raise ValueError(f"Unknown dataset name(s) {unknown_names}. Available: {', '.join(DATASET_NAMES)}")

    train_datasets = [build_dataset(name, "train", args, feature_types) for name in train_dataset_names]
    valid_ds = build_dataset(args.val_dataset, "validation", args, feature_types)

    print(len(valid_ds), *(len(ds) for ds in train_datasets))
    train_model(train_datasets, valid_ds, args, feature_types, type_of_encoder = args.type_of_encoder)
