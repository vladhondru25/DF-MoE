import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class LSTMDeepfakeDetection(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 bidirectional=True, num_classes=1, dropout=0.1,
                 emotion_embedding_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attn = nn.Linear(hidden_size * 2, 1)
        self.layernorm = nn.LayerNorm(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.embedding = nn.Linear(hidden_size * 2, 64)
        self.mlp = nn.Linear(64, num_classes)
        self.dropout2 = nn.Dropout(dropout)
        self.emotion_embedding  = nn.Embedding(num_embeddings=8, embedding_dim=emotion_embedding_dim)
        self.identity_embedding = nn.Embedding(num_embeddings=10, embedding_dim=emotion_embedding_dim)
    def forward(self, x):
        
        if 'emotion_video' in x:
            x['emotion_video'] = self.emotion_embedding(x['emotion_video'].int())
        # for key in x:
        #     print(x[key].shape)
        # print(list(x.keys()))
        list_features = []
        for feature_type in x:
            list_features.append(x[feature_type])
        x = torch.concat(list_features, dim=-1)
        # print(x.shape)
        output, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attn(output).squeeze(-1), dim=1)
        # print(output.shape, attn_weights.shape)
        pooled = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)
        # normed = self.layernorm(pooled)
        # dropped = self.dropout1(normed)

        embedding = self.embedding(pooled)
        embedding = F.relu(embedding)
       
        logits = self.mlp(embedding)
        return logits