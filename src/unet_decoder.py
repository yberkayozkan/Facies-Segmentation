from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class FidelityAwareProjection(nn.Module):
    """Project high-dimensional DINO features while preserving detail fidelity.

    This follows the FAPM idea: decompose input into shared context and
    scale-specific detail, use context to modulate detail, then refine with
    depthwise separable convolution and SE recalibration.
    """

    def __init__(self, in_ch: int, out_ch: int, context_ch: int | None = None):
        super().__init__()
        context_ch = int(context_ch or max(32, out_ch // 2))
        self.context = nn.Sequential(
            nn.Conv2d(in_ch, context_ch, 1, bias=False),
            nn.GroupNorm(min(8, context_ch), context_ch),
            nn.GELU(),
        )
        self.specific = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.gamma_beta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(context_ch, out_ch * 2, 1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            SqueezeExcite(out_ch),
        )
        self.shortcut = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = self.context(x)
        sp = self.specific(x)
        gamma, beta = self.gamma_beta(ctx).chunk(2, dim=1)
        mod = (1.0 + torch.tanh(gamma)) * sp + beta
        return self.refine(mod) + self.shortcut(mod)


class DINOUnetDecoder(nn.Module):
    def __init__(self, feature_channels: list[int], decoder_channels: int, num_classes: int, use_fapm: bool = False, fapm_context_channels: int | None = None):
        super().__init__()
        if use_fapm:
            self.proj = nn.ModuleList([FidelityAwareProjection(ch, decoder_channels, fapm_context_channels) for ch in feature_channels])
        else:
            self.proj = nn.ModuleList([nn.Conv2d(ch, decoder_channels, 1) for ch in feature_channels])
        self.fuse = nn.ModuleList([ConvBlock(decoder_channels * 2, decoder_channels) for _ in range(len(feature_channels) - 1)])
        self.refine = ConvBlock(decoder_channels, decoder_channels)
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward_features(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        projected = [proj(feat) for proj, feat in zip(self.proj, features)]
        x = projected[-1]
        for skip, block in zip(reversed(projected[:-1]), self.fuse):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = self.refine(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        return self.classifier(self.forward_features(features, output_size))


class PlainUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6, base: int = 48):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.b = ConvBlock(base * 8, base * 8)
        self.d3 = ConvBlock(base * 16, base * 4)
        self.d2 = ConvBlock(base * 8, base * 2)
        self.d1 = ConvBlock(base * 4, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        d3 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e4], dim=1))
        d2 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e3], dim=1))
        d1 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e2], dim=1))
        d0 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(d0)
