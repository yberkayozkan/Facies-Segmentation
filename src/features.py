from __future__ import annotations

import numpy as np

from .transforms import normalize_volume_or_slice


def make_triplet(volume: np.ndarray, orientation: str, idx: int, bounds: tuple[int, int] | None = None) -> np.ndarray:
    if orientation == "inline":
        n = volume.shape[0]
        lo, hi = bounds if bounds is not None else (0, n)
        if not 0 <= lo <= idx < hi <= n:
            raise ValueError(f"Invalid inline index/bounds: {idx}, {bounds}")
        ids = [max(lo, idx - 1), idx, min(hi - 1, idx + 1)]
        return np.stack([volume[i, :, :] for i in ids], axis=0)
    if orientation == "crossline":
        n = volume.shape[1]
        lo, hi = bounds if bounds is not None else (0, n)
        if not 0 <= lo <= idx < hi <= n:
            raise ValueError(f"Invalid crossline index/bounds: {idx}, {bounds}")
        ids = [max(lo, idx - 1), idx, min(hi - 1, idx + 1)]
        return np.stack([volume[:, i, :] for i in ids], axis=0)
    raise ValueError(f"Unknown orientation: {orientation}")


def depth_coordinate(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    depth = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    return np.broadcast_to(depth[None, :], (h, w)).copy()


def vertical_gradient(center_section: np.ndarray) -> np.ndarray:
    grad = np.gradient(center_section.astype(np.float32), axis=1)
    scale = np.percentile(np.abs(grad), 99.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(grad / scale, -1.0, 1.0).astype(np.float32)


def build_section_features(
    triplet_raw: np.ndarray,
    normalization: dict[str, float],
    mode: str = "neighbor_triplet",
) -> np.ndarray:
    """Build a 3-channel section tensor for DINO/UNet input.

    Modes:
    - neighbor_triplet: previous, center, next normalized seismic sections.
    - amplitude_gradient_depth: center amplitude, vertical gradient, absolute depth coordinate.
    """
    norm_triplet = normalize_volume_or_slice(triplet_raw, normalization)
    mode = mode.lower()
    if mode == "neighbor_triplet":
        return norm_triplet
    if mode in {"amplitude_gradient_depth", "agd"}:
        center = norm_triplet[1]
        grad = vertical_gradient(center)
        depth = depth_coordinate(center.shape)
        return np.stack([center, grad, depth], axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown feature mode: {mode}")
