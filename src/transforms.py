from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import save_json


def _stats_from_sample(sample: np.ndarray, lower_percentile: float, upper_percentile: float) -> dict[str, float]:
    """Robust clipping bounds plus normalized mean/std for a finite sample."""
    lo, hi = np.percentile(sample, [lower_percentile, upper_percentile])
    clipped = np.clip(sample, lo, hi)
    std = float(np.std(clipped))
    return {
        "lower": float(lo),
        "upper": float(hi),
        "mean": float(np.mean(clipped)),
        "std": std if std > 0 else 1.0,
    }


def compute_normalization_stats(
    volume: np.ndarray, lower: float = 0.5, upper: float = 99.5, max_samples: int = 2_000_000
) -> dict:
    """Estimate robust statistics from a bounded, spatially distributed grid."""
    if volume.ndim != 3 or max_samples < 1:
        raise ValueError("Expected a 3D volume and a positive max_samples")
    stride = max(1, math.ceil((volume.size / max_samples) ** (1 / 3)))
    sample = np.asarray(volume[::stride, ::stride, ::stride]).ravel()
    if sample.size > max_samples:
        sample = sample[::math.ceil(sample.size / max_samples)]
    sample = sample[np.isfinite(sample)].astype(np.float64, copy=False)
    if sample.size == 0:
        raise ValueError("No finite seismic samples available for normalization")
    stats = _stats_from_sample(sample, lower, upper)
    return {"sample_count": int(sample.size), "sample_stride": stride, **stats}


def load_or_create_normalization_stats(
    volume: np.ndarray,
    split: dict,
    source_path: str | Path,
    cfg: dict,
    path: str | Path,
) -> dict:
    source = Path(source_path).resolve()
    stat = source.stat()
    provenance = {
        "version": 2,
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "shape": list(volume.shape),
        "split_axis": split["axis"],
        "train_range": split["train"],
        "train_ranges": split.get("train_ranges", [split["train"]]),
        "lower_percentile": float(cfg["lower_percentile"]),
        "upper_percentile": float(cfg["upper_percentile"]),
        "max_samples": int(cfg.get("max_samples", 2_000_000)),
    }
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("provenance") == provenance:
            return saved
    ranges = provenance["train_ranges"]
    if len(ranges) == 1:
        start, end = ranges[0]
        slicer = [slice(None)] * volume.ndim
        slicer[int(split["axis_index"])] = slice(start, end)
        stats = compute_normalization_stats(
            volume[tuple(slicer)], provenance["lower_percentile"],
            provenance["upper_percentile"], provenance["max_samples"],
        )
    else:
        axis = int(split["axis_index"])
        total = sum(end - start for start, end in ranges) * (volume.size // volume.shape[axis])
        stride = max(1, math.ceil((total / provenance["max_samples"]) ** (1 / 3)))
        samples = []
        for start, end in ranges:
            slicer = [slice(None)] * volume.ndim
            slicer[axis] = slice(start, end, stride)
            for dim in range(volume.ndim):
                if dim != axis:
                    slicer[dim] = slice(None, None, stride)
            samples.append(np.asarray(volume[tuple(slicer)]).ravel())
        sample = np.concatenate(samples)
        if sample.size > provenance["max_samples"]:
            sample = sample[::math.ceil(sample.size / provenance["max_samples"])]
        sample = sample[np.isfinite(sample)].astype(np.float64, copy=False)
        if not sample.size:
            raise ValueError("No finite training samples available for normalization")
        stats = dict(
            _stats_from_sample(sample, provenance["lower_percentile"], provenance["upper_percentile"]),
            sample_count=int(sample.size), sample_stride=stride,
        )
    stats["provenance"] = provenance
    save_normalization_stats(stats, path)
    return stats


def save_normalization_stats(stats: dict[str, float], path: str | Path) -> None:
    save_json(stats, path)


def normalize_volume_or_slice(array: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    x = np.clip(array, stats["lower"], stats["upper"])
    x = (x - stats["mean"]) / stats["std"]
    return x.astype(np.float32, copy=False)


def augment_image_label(image: np.ndarray, label: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rotation = float(cfg.get("rotation_degrees", 0.0))
    if rotation > 0:
        try:
            from scipy.ndimage import rotate
        except ImportError as exc:
            raise ImportError("scipy is required for rotation augmentation.") from exc
        angle = float(rng.uniform(-rotation, rotation))
        image = rotate(image, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
        label = rotate(label, angle, axes=(0, 1), reshape=False, order=0, mode="nearest")
    if cfg.get("horizontal_flip", False) and rng.random() < 0.5:
        image = image[:, ::-1, :].copy()
        label = label[::-1, :].copy()
    scale = float(cfg.get("amplitude_scale", 0.0))
    if scale > 0:
        image = image * float(rng.uniform(1.0 - scale, 1.0 + scale))
    shift = float(cfg.get("intensity_shift", 0.0))
    if shift > 0:
        image = image + float(rng.uniform(-shift, shift))
    noise = float(cfg.get("gaussian_noise", 0.0))
    if noise > 0:
        image = image + rng.normal(0.0, noise, size=image.shape).astype(np.float32)
    return image.astype(np.float32, copy=False), label
