import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from dataset.dataset_mavosv2 import MavosFeatureDataset
import argparse
import wandb
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import copy
from datasets import Value, Dataset, Features
from model.moe import MultimodalMoEDetector, BehaviorEncoder, SpatialEncoder, SemanticEncoder, rPPGEncoder
from torch.utils.data.dataloader import default_collate
from dataset.datasetv2_video_level import MavosFeatureDataset as MavosFeatureDatasetVideoLevel
from dataset.dataset_deepfake_eval import DeepfakeEvalFeatureDataset
from dataset.dataset_polyglotfake import PolyglotFeatureDataset
from dataset.dataset_fakeavceleb import FakeAVCelebFeatureDataset
from dataset.dataset_biodeepav import BiodeepAVFeatureDataset
from dataset.dataset_voxceleb import VoxCelebFeatureDataset
from dataset.dataset_social_media import SocialMediaFeatureDataset
from model import models_vit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
from dataset.dataset_avlips import AVLipsFeatureDataset
from dataset.dataset_celebdf import CelebDFFeatureDataset
@torch.no_grad
def test_model(test_ds, args, feature_types, type_of_encoder = None):
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = rank
    
    task_loss_fn = nn.BCEWithLogitsLoss()
    validation_sampler = DistributedSampler(test_ds, shuffle=False)
    test_dl = DataLoader(test_ds, sampler = validation_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dict_collate_fn)
    # print("before load")

    # for name, param in model.named_parameters():
    #     if "decoder" in name:
    #         continue
    # # for name, param in state_dict.items()
    #     print(name, param.amax(), param.amin())
    # exit()
    # result = []
    def inference():
        
        if type_of_encoder is None:
            model = MultimodalMoEDetector(
                embed_dim=args.embed_dim, 
                num_experts=args.num_experts, 
                k=args.top_k
            ).to(device)
        

        checkpoint = torch.load(args.checkpoint_path, weights_only=False)

        # print(list(checkpoint.keys()))
        # exit()
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        new_dict = {}
        for name in state_dict:
            # if "audio_feature_extractor" in name:
                # continue
            new_dict[name]= state_dict[name]
        
        model.load_state_dict(new_dict, strict=True)

        if args.ckpt_audio_encoder is not None:
            model.audio_feature_extractor.load_model(args.ckpt_audio_encoder)
            for param in model.audio_feature_extractor.parameters():
                param.requires_grad = False
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model.eval()
        v_loss, v_correct, v_count = 0.0, 0, 0
        # print(len(test_dl))
        for batch in tqdm(test_dl):
            # print(list(batch.keys()), feature_types)
            batch['rPPG'] = torch.mean(batch['rPPG'], dim=2)
            # print(batch['video_path'])
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
            input_dict = {tof: batch[tof].float().to(device) for tof in feature_types}
            # for tof in ['raw_audio_features']:
            #     input_dict[tof] = torch.zeros_like(input_dict[tof]) 
            # input_dict['emotion_video'] = torch.zeros_like(input_dict['emotion_video'])
            input_dict['padding_mask'] = batch['padding_mask']
            y = batch['label'].to(device)
            yf = y.float().to(device)
            # print(y)
            if type_of_encoder is None or type_of_encoder=='mlp':
                # for feature in input_dict:
                #     print(feature, input_dict[feature].amax(), input_dict[feature].amin(), input_dict[feature].shape)
                input_dict['use_features'] = args.use_features
                logits, aux_loss = model(input_dict)
                # result.append(logits)
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
            elif type_of_encoder == 'face':
                b, _, s, _, _ = input_dict['full_frames'].shape
                
                frames = rearrange(input_dict['full_frames'], 'b c s h w-> (b s) c h w')
                print(frames.shape)
                logits = model(frames)
                logits = rearrange(logits, '(b s) p-> b s p', s=s, b=b)
                loss = task_loss_fn(torch.mean(logits, dim=1)[:, 0], yf)
            preds = (torch.sigmoid(logits) > 0.5).long()
            # preds = (torch.mean(torch.softmax(logits, dim=-1), dim = 1)[:, 0]>0.5).long()
            correct = (preds.cpu() == y.cpu().long()).sum().item()
            v_loss += loss.item() * y.size(0)
            v_correct += correct
            v_count += y.size(0)

            val_loss = v_loss / v_count
            val_acc  = v_correct / v_count
            if not args.ignore_wandb:
                wandb.log({
                    "loss": val_loss, "accuracy": val_acc
                })
            else:
                print({
                    "loss": val_loss, "accuracy": val_acc
                })
            # predictions = (torch.mean(torch.softmax(logits, dim=-1), dim = 1))
            predictions = torch.sigmoid(logits)
            for i, video_path in enumerate(batch['video_path']):
                prediction =  int(predictions[i]>0.5)
                # print(batch['identity'][i])
                out_dir = {"video_path": video_path, "prediction": predictions[i].item(), "gt": y[i].item(), 'identity': batch['identity'][i]}
                # if prediction != y[i].item():
                #     print(out_dir)
                #     with open("preds_deepfakeeval.txt", "a") as f:
                #         f.write(str(out_dir))
                #         f.write("\n")
                    # exit()
                
                print(out_dir)
                yield out_dir
        # print(result)
    features = Features({
        "video_path":Value("string"),
        "prediction":Value("float32"),
        "gt":Value("int32"),
        "identity":Value("int32")
    })
    predictions_ds =  Dataset.from_generator(inference, features=features)
    predictions_ds.save_to_disk(args.save_path + f"{rank}")

def dict_collate_fn(batch):

    paths = [item.pop('video_path') for item in batch]
    # print(paths)
    batched_data = default_collate(batch)
    batched_data['video_path'] = paths
    return batched_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MOE Deepfake detection")
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--hop_length", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument("--type_of_encoder", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="model_checkpoints/best_moe.pt")
    parser.add_argument("--save_path", type=str, default="predictions/moe")
    parser.add_argument("--video_level", action='store_true')
    parser.add_argument("--wandb_run_name", type=str, default="moe")
    parser.add_argument("--ignore_wandb", action="store_true")
    parser.add_argument("--path_features_video", type=str, default=None)
    parser.add_argument("--path_features_audio", type=str, default=None)
    parser.add_argument("--ckpt_audio_encoder", type=str, default=None)
    parser.add_argument("--use_features", type=str, default=None)
    args = parser.parse_args()
    if not args.ignore_wandb:
        wandb.init(project="MOE deepfake detection test", 
                config=vars(args))
    feature_types=["hp", "gaze", "emotion_video", "emotion_audio", "rPPG", "face_parse", "audio_features", "frames", "bbox_mouth", "full_frames", "raw_audio_features"]
    if args.type_of_encoder is None:
        feature_types=["hp", "gaze", "emotion_video", "emotion_audio", "rPPG", "face_parse", "audio_features", "frames", "bbox_mouth", "full_frames", "raw_audio_features"]
    elif args.type_of_encoder == 'spatial':
        feature_types = ["rPPG", "face_parse"]
    elif args.type_of_encoder == 'behavior':
        feature_types = ["rPPG", "hp", 'gaze']
    elif args.type_of_encoder == 'semantic':
        feature_types = ["rPPG", "emotion_video", 'emotion_audio']
    elif args.type_of_encoder == 'rPPG':
        feature_types = ['rPPG']
    elif args.type_of_encoder == "face":
        feature_types = ["rPPG", "full_frames", "frames", "bbox_mouth"]
    dataset_class = MavosFeatureDatasetVideoLevel if args.video_level else MavosFeatureDataset
    if "deepfakeeval" in args.path_features_video:
        test_ds = DeepfakeEvalFeatureDataset("../datasets/Deepfake-Eval-2024/video-metadata-publish-with-links.csv",
                                args.path_features_video,
                                args.path_features_audio,
                                "test",
                                feature_types,
                                args.sequence_length,
                                args.hop_length, 512,
                                {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                True, True)
    elif "mavos" in args.path_features_video:
        test_ds = dataset_class("/mnt/data/datasets/MAVOS-DD","../datasets/MAVOS_compressed",
                                 args.path_features_video,"../datasets/features_mavos_compressed/video",
                                 args.path_features_audio,
                                 "test",
                                 feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    elif "polyglot" in args.path_features_video:
        test_ds = PolyglotFeatureDataset("../../datasets/PolyGlotFake",
                                 args.path_features_video,
                                 args.path_features_audio,
                                 "test",
                                 feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    elif "fakeavceleb" in args.path_features_video:
        test_ds = FakeAVCelebFeatureDataset("../../datasets/FakeAVCeleb",
                                args.path_features_video,
                                args.path_features_audio,
                                "test",
                                feature_types,
                                args.sequence_length,
                                args.hop_length, 512,
                                {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                True, True)
    elif "biodeepav" in args.path_features_video:
        test_ds = BiodeepAVFeatureDataset("../../datasets/BioDeepAV",
                                 args.path_features_video,
                                 args.path_features_audio,
                                 "test",
                                 feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    elif 'vox' in args.path_features_video:
        test_ds = VoxCelebFeatureDataset("../../datasets/vox2_mp4_2/dev/mp4",
                                 args.path_features_video,
                                 args.path_features_audio,
                                 "test",
                                 feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    elif 'avlips' in args.path_features_video:
        test_ds = AVLipsFeatureDataset(
        "/mnt/data/datasets/AVLips", args.path_features_video,
        args.path_features_audio, "test", feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    elif 'social' in args.path_features_video:
        test_ds = SocialMediaFeatureDataset(
        "/mnt/data/datasets/social_media_test", args.path_features_video,
        args.path_features_audio, "test", feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    # train_ds_2[0]
    elif 'test_videos' in args.path_features_video:
        test_ds = SocialMediaFeatureDataset(
        "/mnt/data/datasets/test_videos", args.path_features_video,
        args.path_features_audio, "test", feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    elif 'celebdf' in args.path_features_video:
        test_ds = CelebDFFeatureDataset(
        "/mnt/data/datasets/celebdf_v2", args.path_features_video,
        None, "test", feature_types,
                                 args.sequence_length,
                                 args.hop_length, 512,
                                 {},#{"hp": relative_deltas, "gaze": relative_deltas},
                                 True, True)
    # print(len(test_ds))
    test_model(test_ds, args, feature_types, type_of_encoder = args.type_of_encoder)