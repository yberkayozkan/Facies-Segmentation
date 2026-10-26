from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .features import build_section_features, make_triplet


def _starts(size: int, tile: int, overlap: float) -> list[int]:
    if size <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def _weight(tile: int, device) -> torch.Tensor:
    one = torch.hann_window(tile, periodic=False, device=device).clamp_min(1e-3)
    w = torch.outer(one, one)
    return w[None, None]


@torch.no_grad()
def predict_section(model, image_3hw: np.ndarray, normalization: dict, device: torch.device, cfg: dict) -> np.ndarray:
    model.eval()
    x_np = build_section_features(image_3hw, normalization, cfg.get("feature_engineering", {}).get("mode", "neighbor_triplet"))
    _, h, w = x_np.shape
    tile = int(cfg["inference"]["tile_size"])
    overlap = float(cfg["inference"]["tile_overlap"])
    num_classes = int(cfg["model"]["num_classes"])
    out = torch.zeros((1, num_classes, h, w), device=device)
    norm = torch.zeros((1, 1, h, w), device=device)
    weight = _weight(tile, device)
    for y in _starts(h, tile, overlap):
        for x0 in _starts(w, tile, overlap):
            patch = x_np[:, y : y + tile, x0 : x0 + tile]
            ph, pw = patch.shape[-2:]
            if ph < tile or pw < tile:
                patch = np.pad(patch, ((0, 0), (0, tile - ph), (0, tile - pw)), mode="edge")
            inp = torch.from_numpy(patch[None]).to(device=device, dtype=torch.float32)
            logits = model(inp)
            probs = logits.softmax(dim=1)
            if cfg["inference"].get("tta_horizontal_flip", False):
                flip_logits = model(torch.flip(inp, dims=[3]))
                probs = 0.5 * (probs + torch.flip(flip_logits.softmax(dim=1), dims=[3]))
            probs = probs[:, :, :ph, :pw]
            wy = weight[:, :, :ph, :pw]
            out[:, :, y : y + ph, x0 : x0 + pw] += probs * wy
            norm[:, :, y : y + ph, x0 : x0 + pw] += wy
    local_probs = out / norm.clamp_min(1e-6)
    global_weight = float(cfg["inference"].get("global_context_weight", 0.0))
    if not 0.0 <= global_weight <= 1.0:
        raise ValueError("global_context_weight must be between 0 and 1")
    if global_weight:
        # Match the full-section examples used during training, then restore the
        # coarse scene prediction to the original grid before blending with tiles.
        full_section = torch.from_numpy(x_np[None]).to(device=device, dtype=torch.float32)
        overview = F.interpolate(full_section, size=(tile, tile), mode="bilinear", align_corners=False)
        global_probs = model(overview).softmax(dim=1)
        global_probs = F.interpolate(global_probs, size=(h, w), mode="bilinear", align_corners=False)
        local_probs = (1.0 - global_weight) * local_probs + global_weight * global_probs
    return local_probs.squeeze(0).cpu().numpy()

@torch.no_grad()
def predict_volume_probabilities(model, volume: np.ndarray, normalization: dict, device: torch.device, cfg: dict, orientation: str) -> np.ndarray:
    num_classes = int(cfg["model"]["num_classes"])
    probs = np.memmap(
        Path(cfg["outputs"]["predictions"]) / f"prob_{orientation}.dat",
        dtype="float32",
        mode="w+",
        shape=(num_classes,) + tuple(volume.shape),
    )
    n = volume.shape[0] if orientation == "inline" else volume.shape[1]
    for idx in tqdm(range(n), desc=f"inference-{orientation}"):
        section_prob = predict_section(model, make_triplet(volume, orientation, idx), normalization, device, cfg)
        if orientation == "inline":
            probs[:, idx, :, :] = section_prob
        else:
            probs[:, :, idx, :] = section_prob
        probs.flush()
    return probs


def fuse_probabilities(p_inline: np.ndarray, p_crossline: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    pred = np.argmax(float(alpha) * p_inline + (1.0 - float(alpha)) * p_crossline, axis=0).astype(np.uint8) + 1
    unique = set(np.unique(pred).astype(int).tolist())
    if not unique.issubset(set(range(1, 7))):
        raise ValueError(f"Unexpected prediction labels: {sorted(unique)}")
    return pred
