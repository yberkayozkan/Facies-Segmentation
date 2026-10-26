from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FaciesQueryHead(nn.Module):
    def __init__(self, dense_channels: int, token_channels: int, query_channels: int, num_classes: int = 6):
        super().__init__()
        self.query_channels = query_channels
        self.queries = nn.Parameter(torch.randn(num_classes, query_channels) * 0.02)
        self.token_proj = nn.Linear(token_channels, query_channels)
        self.dense_proj = nn.Conv2d(dense_channels, query_channels, 1)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, dense_features: torch.Tensor, high_tokens: torch.Tensor) -> torch.Tensor:
        b, _, h, w = dense_features.shape
        tokens = high_tokens.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        queries = self.queries.unsqueeze(0).expand(b, -1, -1)
        attn = torch.softmax(torch.matmul(queries, tokens.transpose(1, 2)) / (self.query_channels**0.5), dim=-1)
        prototypes = torch.matmul(attn, tokens)
        dense = self.dense_proj(dense_features).flatten(2).transpose(1, 2)
        dense = F.normalize(dense, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        logits = torch.matmul(dense, prototypes.transpose(1, 2)).transpose(1, 2)
        return logits.reshape(b, prototypes.shape[1], h, w) * self.scale
