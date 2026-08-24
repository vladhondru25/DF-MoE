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

        x = self.proj(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)  
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class VideoAudioTransformer(nn.Module):
    def __init__(self, device, visual_pretrained_model_name, d_video, d_audio, d_model=512, n_heads=8, num_layers=4, return_features=False):
        super().__init__()
        
        self.visual_backbone = AutoModel.from_pretrained(
            visual_pretrained_model_name, 
            device_map=device,
        )
        
        self.fusion_proj = nn.Linear(d_video + d_audio, d_model)
        

        self.joint_encoder = nn.ModuleList([
            SelfAttentionBlock(d_model, d_model, n_heads) 
            for _ in range(num_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, 1)  
        self.return_features = return_features
        
    def train(self, mode: bool = True):
        super().train(mode)
        
        self.visual_backbone.eval()
        return self

    def forward(self, images_inputs, audio_feats, images_masks=None):

        b, n, _, _, _ = images_inputs.shape
        
        images_inputs = rearrange(images_inputs, "b n c h w -> (b n) c h w")
        
        with torch.no_grad():
            output_visual_backbone = self.visual_backbone(images_inputs)
            video_feats = output_visual_backbone.last_hidden_state[:, 0, :]
            video_feats = rearrange(video_feats, "(b n) d -> b n d", b=b, n=n)

        fused_feats = torch.cat([video_feats, audio_feats], dim=-1) 

        x = self.fusion_proj(fused_feats) 

        for layer in self.joint_encoder:
            if images_masks is None:
                x = layer(x)
            else:
                x = layer(x, images_masks)
                
        x = x.transpose(1, 2)  
        feats = self.pool(x).squeeze(-1) 

        out = self.classifier(feats)
        
        if self.return_features:
            return feats
        else:
            return out