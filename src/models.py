from __future__ import annotations

import torch
from torch import nn

from .dino_encoder import DINOEncoder
from .facies_query import FaciesQueryHead
from .seismic_unet import SeismicUNet
from .unet_decoder import DINOUnetDecoder, PlainUNet


class DINOUnet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        self.encoder = DINOEncoder(
            model_cfg["backbone"],
            model_cfg["dino_layers"],
            bool(model_cfg.get("gradient_checkpointing", False)),
        )
        ch = self.encoder.hidden_size
        self.decoder = DINOUnetDecoder(
            [ch] * len(model_cfg["dino_layers"]),
            int(model_cfg["decoder_channels"]),
            int(model_cfg["num_classes"]),
            use_fapm=bool(model_cfg.get("use_fapm", False)),
            fapm_context_channels=model_cfg.get("fapm_context_channels"),
        )
        self.use_query = bool(model_cfg.get("use_facies_query", False))
        self.query_beta = float(model_cfg.get("query_beta", 0.25))
        decoder_channels = int(model_cfg["decoder_channels"])
        query_channels = int(model_cfg.get("query_channels", decoder_channels))
        self.query_head = (
            FaciesQueryHead(decoder_channels, ch, query_channels, int(model_cfg["num_classes"]))
            if self.use_query
            else None
        )

    def set_encoder_schedule(self, epoch: int, freeze_epochs: int, trainable_blocks: int) -> None:
        n_blocks = 0 if epoch < freeze_epochs else trainable_blocks
        self.encoder.set_trainable_last_blocks(n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        dense = self.decoder.forward_features(features, x.shape[-2:])
        logits = self.decoder.classifier(dense)
        if self.query_head is not None:
            logits = logits + self.query_beta * self.query_head(dense, features[-1])
        return logits


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]["name"].lower()
    if name == "seismic_unet":
        return SeismicUNet(
            in_channels=3,
            num_classes=int(cfg["model"]["num_classes"]),
            base_channels=int(cfg["model"].get("base_channels", 32)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
    if name in {"unet", "plain_unet", "e0_unet"}:
        return PlainUNet(in_channels=3, num_classes=int(cfg["model"]["num_classes"]))
    if name in {"dino_unet", "dinov2_unet", "dinov3_fapm_unet"}:
        return DINOUnet(cfg)
    raise ValueError(f"Unknown model name: {name}")


def parameter_groups(model: nn.Module, cfg: dict) -> list[dict]:
    if cfg["model"]["name"].lower() == "seismic_unet":
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": float(cfg["training"]["decoder_lr"])}]
    encoder_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": float(cfg["training"]["decoder_lr"])})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(cfg["training"]["encoder_lr"])})
    return groups
