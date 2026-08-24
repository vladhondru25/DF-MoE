import torch
import torch.nn as nn

from einops import rearrange
from transformers import AutoModel


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_in, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        
        

        self.proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, T, d_in)
        """
        x = self.proj(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)  # self-attention
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, video_feats, audio_feats, key_padding_mask=None):
        """
        video_feats: (B, T_v, d_model)   [queries]
        audio_feats: (B, T_a, d_model)   [keys, values]
        """
        attn_out, _ = self.cross_attn(
            query=video_feats, 
            key=audio_feats, 
            value=audio_feats,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(video_feats + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class VideoAudioTransformer(nn.Module):
    def __init__(self, device, visual_pretrained_model_name, d_video, d_audio, d_model=512, n_heads=8, num_layers=4, return_features=False):
        super().__init__()
        
        self.visual_backbone = AutoModel.from_pretrained(
            visual_pretrained_model_name, 
            device_map=device,
            # torch_dtype=torch.bfloat16
        )
        
        # Encoders for each modality
        self.video_encoder = nn.ModuleList([
            SelfAttentionBlock(d_video if i == 0 else d_model, d_model, n_heads) 
            for i in range(num_layers)
        ])
        self.audio_encoder = nn.ModuleList([
            SelfAttentionBlock(d_audio if i == 0 else d_model, d_model, n_heads) 
            for i in range(num_layers)
        ])
        # Cross-modal fusion layers
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads) 
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Linear(d_model, 1)  # binary output
        self.return_features = return_features
        
    def train(self, mode: bool = True):
        super().train(mode)
        
        # always keep visual_backbone in eval
        self.visual_backbone.eval()
        return self

    def forward(self, images_inputs, audio_feats, images_masks=None):
        """
        video_feats: (B, T_v, d_video)
        audio_feats: (B, T_a, d_audio)
        """
        b, n, _, _, _ = images_inputs.shape
        
        images_inputs = rearrange(images_inputs, "b n c h w -> (b n) c h w")
        # video_feats = self.visual_backbone(**images_inputs).last_hidden_state
        with torch.no_grad():
            output_visual_backbone = self.visual_backbone(images_inputs)
            video_feats = output_visual_backbone.last_hidden_state[:, 0, :]
        
            video_feats = rearrange(video_feats, "(b n) t -> b n t", b=b, n=n)
        
        # Encode video
        for layer in self.video_encoder:
            if images_masks is None:
                video_feats = layer(video_feats)
            else:
                video_feats = layer(video_feats, images_masks)
        # Encode audio
        for layer in self.audio_encoder:
            audio_feats = layer(audio_feats)
        # Cross attention (video attends to audio)
        for layer in self.cross_layers:
            if images_masks is None:
                video_feats = layer(video_feats, audio_feats)
            else:
                video_feats = layer(video_feats, audio_feats, images_masks)
        # Global pooling
        x = video_feats.transpose(1, 2)  # (B, d_model, T_v)
        feats = self.pool(x).squeeze(-1)
        
        # Classify from the features obtained
        x = self.classifier(feats)
        if self.return_features:
            return feats
        else:
            return x
