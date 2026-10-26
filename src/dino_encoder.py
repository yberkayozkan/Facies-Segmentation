from __future__ import annotations

import os

import torch
from torch import nn


class DINOEncoder(nn.Module):
    def __init__(self, model_name: str, layers: list[int], gradient_checkpointing: bool = False):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for DINO encoders. Install requirements_colab.txt.") from exc
        hf_token = os.environ.get("HF_TOKEN")
        load_kwargs = {"output_hidden_states": True}
        if hf_token:
            load_kwargs["token"] = hf_token
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.layers = layers
        self.hidden_size = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "embed_dim", 0)))
        if self.hidden_size <= 0:
            raise ValueError(f"Could not infer hidden size from config for {model_name}.")
        self.patch_size = int(getattr(self.model.config, "patch_size", getattr(self.model.config, "patch_size", 16)))
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def set_trainable_last_blocks(self, n_blocks: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        if n_blocks <= 0:
            return
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            for p in self.model.parameters():
                p.requires_grad = True
            return
        for block in layers[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for name, p in self.model.named_parameters():
            if "layernorm" in name.lower() or "norm" in name.lower():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        b, _, h, w = x.shape
        gh = h // self.patch_size
        gw = w // self.patch_size
        features = []
        for layer_idx in self.layers:
            state = hidden_states[layer_idx]
            if state.ndim == 4:
                feat = state
            else:
                expected_tokens = gh * gw
                token_count = state.shape[1]
                if token_count == expected_tokens:
                    patch_tokens = state
                elif token_count > expected_tokens:
                    # DINOv2/DINOv3 variants may prepend cls/register tokens.
                    # Patch tokens are kept at the end in HF ViT-style outputs.
                    patch_tokens = state[:, -expected_tokens:, :]
                else:
                    raise RuntimeError(
                        f"Cannot reshape DINO tokens to patch grid: got {token_count} tokens, "
                        f"expected at least {expected_tokens} for input {(h, w)} and patch_size={self.patch_size}."
                    )
                feat = patch_tokens.transpose(1, 2).reshape(b, self.hidden_size, gh, gw)
            features.append(feat)
        return features


DINOv2Encoder = DINOEncoder
