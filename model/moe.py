import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import wandb
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import copy
from einops import rearrange
from model.video_audio_transformer import VideoAudioTransformer
from model.avff.video_cav_mae import VideoCAVMAEFT
import random
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
from model.avff.effort_detector import apply_svd_residual_to_self_attn
from model.wav2vec_aasist import W2V_AASIST
from model.cam_loss import ClassAnchorMarginLoss
from model.video_audio_transformer_self_attn import VideoAudioTransformer as VideoAudioTransformerSA
EMBED_DIM = 128         
SEQ_LEN_RPPG = 150      
SEQ_LEN_BEHAVIOR = 150  
SEQ_LEN_EMOTION = 50    
SEQ_LEN_SPATIAL = 30    
NUM_EXPERTS = 6         
TOP_K = 2               

class rPPGEncoder(nn.Module):
    def __init__(self, input_features=512, output_dim=EMBED_DIM, use_classifier=False):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        conv_output_features = 128
        
        self.gru = nn.GRU(conv_output_features, output_dim, num_layers=2, 
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(output_dim * 2, output_dim)
        self.use_classifier=use_classifier
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, x):
        
        x = x.permute(0, 2, 1)  
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)  
        _, h_n = self.gru(x)
        
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        if not self.use_classifier:
            return self.fc(h_n)
        else:
            h = self.fc(h_n)
            return self.classifier(self.relu(h))

class BehaviorEncoder(nn.Module):
    def __init__(self, pose_features=3, gaze_features=2, output_dim=EMBED_DIM, use_classifier=False):
        super().__init__()
        self.pose_lstm = nn.LSTM(pose_features, 64, num_layers=2, 
                                 batch_first=True, bidirectional=True)
        self.gaze_lstm = nn.LSTM(gaze_features, 64, num_layers=2, 
                                 batch_first=True, bidirectional=True)
        
        lstm_out_dim = 128
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=lstm_out_dim, num_heads=4, 
                                                batch_first=True)
        
        self.fc = nn.Linear(lstm_out_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.use_classifier=use_classifier
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, pose, gaze, padding_mask=None):

        pose_out, _ = self.pose_lstm(pose) 
        gaze_out, _ = self.gaze_lstm(gaze) 
        
        if padding_mask is None:
            attn_out, _ = self.cross_attn(query=gaze_out, key=pose_out, value=pose_out)
        else:
            attn_out, _ = self.cross_attn(query=gaze_out, key=pose_out, value=pose_out, key_padding_mask=padding_mask)
        
        gaze_pooled = gaze_out.mean(dim=1)
        attn_pooled = attn_out.mean(dim=1)
        
        combined = torch.cat((gaze_pooled, attn_pooled), dim=1)
        if not self.use_classifier:
            return self.fc(combined)
        else:
            h = self.fc(combined)
            return self.classifier(self.relu(h))

class SpatialEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=EMBED_DIM, use_classifier=False):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, SEQ_LEN_SPATIAL, 64, 64)
            dummy_out = self.pool(self.conv_stack(dummy_input))
            flattened_size = self.flatten(dummy_out).shape[1]
            
        self.fc = nn.Linear(flattened_size, output_dim)

        self.use_classifier=use_classifier
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, x):
        
        x = self.conv_stack(x)
        x = self.pool(x)
        x = self.flatten(x)
        if not self.use_classifier:
            return self.fc(x)
        else:
            h = self.fc(x)
            return self.classifier(self.relu(h))

class SemanticEncoder(nn.Module):
    def __init__(self, vid_features=64, aud_features=64, output_dim=EMBED_DIM, use_classifier=False):
        super().__init__()
        self.video_gru = nn.GRU(vid_features, 64, batch_first=True, bidirectional=True)
        self.audio_gru = nn.GRU(aud_features, 64, batch_first=True, bidirectional=True)
        self.emotion_embedding  = nn.Embedding(num_embeddings=8, embedding_dim=vid_features)
        gru_out_dim = 128
        self.relu = nn.ReLU()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=gru_out_dim, 
            num_heads=4, 
            batch_first=True
        )
        self.fc = nn.Linear(gru_out_dim * 2, output_dim)

        self.use_classifier=use_classifier
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, video_emotion, audio_emotion, padding_mask=None):
        video_emotion = self.emotion_embedding(video_emotion.int())
        audio_emotion = self.emotion_embedding(audio_emotion.int())
        vid_out, _ = self.video_gru(video_emotion)
        aud_out, _ = self.audio_gru(audio_emotion)
        if padding_mask is None:
            attn_out, _ = self.cross_attn(query=vid_out, key=aud_out, value=aud_out)
        else:
            attn_out, _ = self.cross_attn(query=vid_out, key=aud_out, value=aud_out, key_padding_mask=padding_mask)
        
        vid_pooled = vid_out.mean(dim=1)
        attn_pooled = attn_out.mean(dim=1)
    
        combined_features = torch.cat((vid_pooled, attn_pooled), dim=1)
        
        if not self.use_classifier:
            return self.fc(combined_features)
        else:
            h = self.fc(combined_features)
            return self.classifier(self.relu(h))

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
def create_expert_mask(gating_logits_or_probs: torch.Tensor, dropout_rate: float) -> torch.Tensor:
    batch_size, num_experts = gating_logits_or_probs.shape
    
    keep_prob = 1.0 - dropout_rate
    
    random_tensor = torch.rand(
        batch_size, 
        num_experts, 
        dtype=gating_logits_or_probs.dtype, 
        device=gating_logits_or_probs.device
    )
    mask = (random_tensor < keep_prob).float()
    return mask
class SparseMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.output_dim = output_dim
        
        self.gating_net = nn.Linear(input_dim, num_experts)
        
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) 
                                      for _ in range(num_experts)])
        # self.attention_pool = MultiHeadAttentionPool(dim = output_dim, num_heads=8)
        

    def forward(self, x):
        # print(x.shape)
        bs, no_encoders, feature_size = x.shape
        x = rearrange(x, 'b s f -> (b s) f', b=bs, s=no_encoders)
        batch_size, _ = x.shape
        
        logits = self.gating_net(x)
        # print(self.training)
        if self.training:
            mask = create_expert_mask(logits, dropout_rate=0.2)
            logits = logits * mask
        # print(x.shape)
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=1) 
        top_k_weights = F.softmax(top_k_logits, dim=1)
        router_probs = F.softmax(logits, dim=1)
        tokens_per_expert_prob = router_probs.mean(dim=0)
        one_hot_indices = F.one_hot(top_k_indices, 
                                    num_classes=self.num_experts).sum(dim=1)
        fraction_tokens_routed = one_hot_indices.float().mean(dim=0)
        
        aux_loss = self.num_experts * (fraction_tokens_routed * 
                                       tokens_per_expert_prob).sum()
        
        output = torch.zeros(batch_size, self.output_dim).to(x.device)
        
        for i in range(self.k):
            expert_indices = top_k_indices[:, i]
            weights = top_k_weights[:, i].unsqueeze(1)
            # print(expert_indices.shape)
            for j in range(self.num_experts):
                token_indices = (expert_indices == j).nonzero(as_tuple=True)[0]
                # print(j, token_indices)
                if token_indices.shape[0] > 0:
                    selected_inputs = x[token_indices]
                    # print(selected_inputs.shape)
                    selected_weights = weights[token_indices]
                    
                    expert_output = self.experts[j](selected_inputs)
                    
                    output.index_add_(0, token_indices, 
                                      expert_output * selected_weights)
        output = rearrange(output,'(b s) f-> b s f', b=bs, s=no_encoders, f=feature_size)
        output_cls = output.mean(dim=1)
        # output = self.attention_pool(output)
        return output_cls, output, aux_loss

class SequenceDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, seq_len=100, hidden_dim = 256):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.gru(x)
        x = self.head(x)
        return x

class MultimodalMoEDetector(nn.Module):


    def __init__(self, embed_dim=EMBED_DIM, num_experts=NUM_EXPERTS, k=TOP_K, device='cuda'):
        super().__init__()
        self.rppg_encoder = rPPGEncoder(input_features=512, output_dim=embed_dim)
        self.behavior_encoder = BehaviorEncoder(pose_features=3, gaze_features=2, 
                                                output_dim=embed_dim)
        self.spatial_encoder = SpatialEncoder(input_channels=1, output_dim=embed_dim)
        self.semantic_encoder = SemanticEncoder(vid_features=64, aud_features=64, 
                                                output_dim=embed_dim)
        self.audio_video_encoder = VideoAudioTransformerSA(device, "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                                                         d_video=384, d_audio=384, return_features=True)
        self.audio_projection_layer = nn.Linear(160, embed_dim)
        self.audio_feature_extractor = W2V_AASIST(device)
        self.avff = VideoCAVMAEFT()
        self.avff = apply_svd_residual_to_self_attn(self.avff, r=523)
        self.avff_projection_layer = nn.Linear(2048, embed_dim)

        # self.face_encoder = models_vit.__dict__["vit_base_patch16"](num_classes=4, drop_path_rate=0.1, global_pool=True)
        # self.face_proj_layer = nn.Linear(768, embed_dim)
        total_embed_dim = embed_dim
        self.av_projection_layer = nn.Linear(512, embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=total_embed_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        self.moe_layer = SparseMoE(
            input_dim=total_embed_dim,
            output_dim=embed_dim,
            num_experts=num_experts,
            k=k
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.cam_loss = ClassAnchorMarginLoss(num_classes=2, embedding_dim=embed_dim)
        self.anchors = nn.Parameter(torch.randn(2, embed_dim))
        nn.init.kaiming_uniform_(self.anchors, a=1)
        self.decoder_rppg = SequenceDecoder(embed_dim, 512)
        self.decoder_pose = SequenceDecoder(embed_dim, 3)
        self.decoder_gaze = SequenceDecoder(embed_dim, 2)
        self.decoder_aud_emotion = SequenceDecoder(embed_dim, 6)
        self.decoder_vid_emotion = SequenceDecoder(embed_dim, 8)

    def forward(self, input_dict, keep_modalities=None, return_embeddings=False):
        rppg = input_dict['rPPG']
        pose = input_dict['hp']
        gaze = input_dict['gaze']
        seg_maps = input_dict['face_parse']
        vid_emotion = input_dict['emotion_video']
        aud_emotion = input_dict['emotion_audio']
        frames = input_dict['frames']
        audio_features = input_dict['audio_features']
        full_frames = input_dict['full_frames']
        raw_audio_features = input_dict['raw_audio_features']
        e_avff = self.avff(raw_audio_features, full_frames)
        e_avff = self.avff_projection_layer(e_avff)

        e_rppg = self.rppg_encoder(rppg)
        if 'padding_mask' in input_dict:
            e_behavior = self.behavior_encoder(pose, gaze, padding_mask=input_dict['padding_mask'])
        else:
            e_behavior = self.behavior_encoder(pose, gaze)
        e_spatial = self.spatial_encoder(seg_maps)
        if 'padding_mask' in input_dict:
            e_semantic = self.semantic_encoder(vid_emotion, aud_emotion, padding_mask=input_dict['padding_mask'])
        else:
            e_semantic = self.semantic_encoder(vid_emotion, aud_emotion)
        e_audio_video = self.audio_video_encoder(frames, audio_features, images_masks=input_dict.get("padding_mask", None))
        e_audio_video = self.av_projection_layer(e_audio_video)
        all_modalities = []
        keep_modalities = input_dict['use_features'].split(",") if input_dict.get('use_features', None) is not None else None
        if keep_modalities is None:
            all_modalities = [e_spatial[:, None, :], e_audio_video[:, None, :], e_semantic[:, None, :], e_rppg[:,None, :], e_behavior[:, None, :], e_avff[:, None, :]]
        else:
            if "spatial" in keep_modalities:
                all_modalities.append(e_spatial[:, None, :])
            if "audio_video" in keep_modalities:
                all_modalities.append(e_audio_video[:, None, :])
            if "semantic" in keep_modalities:
                all_modalities.append(e_semantic[:, None, :])
            if "rppg" in keep_modalities:
                all_modalities.append(e_rppg[:, None, :])
            if "behavior" in keep_modalities:
                all_modalities.append(e_behavior[:, None, :])
            if "avff" in keep_modalities:
                all_modalities.append(e_avff[:, None, :])
        
        concatenated_embeddings = torch.cat(
            all_modalities,
            dim=1
        )
        attn_out, attn_weights = self.self_attn(query=concatenated_embeddings, key=concatenated_embeddings, value=concatenated_embeddings)
        moe_output_cls, moe_output_seq, aux_loss = self.moe_layer(attn_out)
        logits = self.classification_head(moe_output_cls).squeeze(1)

        if return_embeddings:
            cam_l = self.cam_loss(moe_output_cls, input_dict['label'], self.anchors)
            return logits, aux_loss, cam_l
        else:
            return logits, aux_loss