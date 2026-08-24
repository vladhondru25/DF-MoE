import torch.nn as nn
import torch
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, num_tokens):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim),
        )
        self.learnable_tokens = nn.parameter.Parameter(torch.zeros((1, num_tokens, input_dim)), requires_grad=True)
        nn.init.trunc_normal_(self.learnable_tokens, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, 
                                                batch_first=True)
    def forward(self, x):
        return self.net(x)