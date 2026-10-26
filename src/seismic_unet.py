from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.layers(x) + self.skip(x))


class SeismicUNet(nn.Module):
    """Four-level residual U-Net trained from scratch on three seismic features."""

    def __init__(self, in_channels: int = 3, num_classes: int = 6, base_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(5)]
        self.encoder = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else channels[i - 1], channels[i], dropout if i >= 2 else 0.0)
            for i in range(5)
        ])
        self.decoder = nn.ModuleList([
            ResidualBlock(channels[i + 1] + channels[i], channels[i], dropout if i >= 2 else 0.0)
            for i in range(3, -1, -1)
        ])
        self.classifier = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.encoder):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            if i < len(self.encoder) - 1:
                skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.classifier(x)
