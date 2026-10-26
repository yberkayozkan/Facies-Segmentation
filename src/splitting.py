"""Physical spatial holdout math: inline/crossline range assignment and label counting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


AXIS_INDEX = {"inline": 0, "crossline": 1}


def subset_ranges(split: dict, subset: str) -> list[list[int]]:
    return split.get(f"{subset}_ranges", [split[subset]])


def range_for_index(split: dict, subset: str, idx: int) -> tuple[int, int]:
    for start, end in subset_ranges(split, subset):
        if start <= idx < end:
            return int(start), int(end)
    raise ValueError(f"Inline {idx} is outside {subset} ranges")


def make_spatial_split(shape: Iterable[int], cfg: dict) -> dict:
    shape = list(shape)
    axis_name = cfg.get("axis", "inline")
    axis = AXIS_INDEX[axis_name]
    n = shape[axis]
    if "ranges" in cfg:
        if axis_name != "inline":
            raise ValueError("Distributed comparison ranges require inline axis")
        ranges = {name: [list(map(int, pair)) for pair in cfg["ranges"].get(name, [])]
                  for name in ("train", "val", "test", "buffer")}
        assignments = np.full(n, -1, dtype=np.int8)
        for value, name in enumerate(ranges):
            for start, end in ranges[name]:
                if not 0 <= start < end <= n or np.any(assignments[start:end] != -1):
                    raise ValueError(f"Invalid or overlapping {name} range: {[start, end]}")
                assignments[start:end] = value
        if np.any(assignments == -1) or any(not ranges[name] for name in ("train", "val", "test")):
            raise ValueError("Distributed ranges must cover every inline and include train, val, and test")
        result = {"axis": axis_name, "axis_index": axis, "shape": shape, "parameters": cfg["ranges"]}
        for name, blocks in ranges.items():
            result[f"{name}_ranges"] = blocks
            result[name] = max(blocks, key=lambda pair: pair[1] - pair[0])
        return result
    if float(cfg.get("test_fraction", 0.0)) > 0:
        fractions = {key: float(cfg.get(key, 0.0)) for key in (
            "train_fraction", "buffer_fraction", "val_fraction", "test_buffer_fraction", "test_fraction"
        )}
        if any(value < 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
            raise ValueError(f"Train/buffer/val/test fractions must be nonnegative and sum to 1: {fractions}")
        train_end = int(round(n * fractions["train_fraction"]))
        val_start = train_end + max(1, int(round(n * fractions["buffer_fraction"])))
        val_end = val_start + int(round(n * fractions["val_fraction"]))
        test_start = val_end + max(1, int(round(n * fractions["test_buffer_fraction"])))
        if train_end < 3 or val_end - val_start < 3 or n - test_start < 3:
            raise ValueError(f"Split slabs need at least three sections: n={n}, fractions={fractions}")
        if abs((n - test_start) - round(n * fractions["test_fraction"])) > 2:
            raise ValueError(f"Test slab differs from requested fraction: n={n}, fractions={fractions}")
        return {
            "axis": axis_name, "axis_index": axis, "shape": shape,
            "train": [0, train_end], "buffer": [train_end, val_start],
            "val": [val_start, val_end], "test_buffer": [val_end, test_start],
            "test": [test_start, n], "parameters": fractions,
        }
    train_end = int(round(n * float(cfg.get("train_fraction", 0.72))))
    buffer = max(1, int(round(n * float(cfg.get("buffer_fraction", 0.04)))))
    val_len = max(1, int(round(n * float(cfg.get("val_fraction", 0.20)))))
    val_start = min(n - val_len, train_end + buffer)
    train_end = max(1, min(train_end, val_start - buffer))
    split = {
        "axis": axis_name,
        "axis_index": axis,
        "shape": shape,
        "train": [0, train_end],
        "buffer": [train_end, val_start],
        "val": [val_start, val_start + val_len],
        "parameters": {
            "train_fraction": float(cfg.get("train_fraction", 0.72)),
            "buffer_fraction": float(cfg.get("buffer_fraction", 0.04)),
            "val_fraction": float(cfg.get("val_fraction", 0.20)),
        },
    }
    return split


def save_split(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)


def load_or_create_split(shape: Iterable[int], cfg: dict, path: str | Path) -> dict:
    path = Path(path)
    expected = make_spatial_split(shape, cfg)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        comparable_keys = ("axis", "axis_index", "shape", "train", "buffer", "val", "test_buffer", "test",
                           "train_ranges", "buffer_ranges", "val_ranges", "test_ranges")
        if all(saved.get(key) == expected.get(key) for key in comparable_keys):
            return saved
    save_split(expected, path)
    return expected


def split_volume(volume: np.ndarray, split: dict, subset: str = "train") -> np.ndarray:
    """Return the requested physical split blocks joined along the holdout axis."""
    if subset not in {"train", "val", "buffer", "test_buffer", "test"} or subset not in split:
        raise ValueError(f"Unknown split subset: {subset}")
    parts = []
    for start, end in subset_ranges(split, subset):
        slicer = [slice(None)] * volume.ndim
        slicer[int(split["axis_index"])] = slice(int(start), int(end))
        parts.append(np.asarray(volume[tuple(slicer)]))
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=int(split["axis_index"]))


def split_class_summary(labels_1_6: np.ndarray, split: dict) -> dict[str, dict[str, object]]:
    """Summarize class support in train and validation physical slabs."""
    summary: dict[str, dict[str, object]] = {}
    for subset in ("train", "val"):
        counts = count_labels_in_split(labels_1_6, split, subset)
        count_map = {label: int(count) for label, count in enumerate(counts, 1) if count}
        total = int(counts.sum())
        summary[subset] = {
            "present": sorted(count_map),
            "missing": sorted(set(range(1, 7)) - set(count_map)),
            "counts": count_map,
            "percent": {key: 100.0 * value / max(1, total) for key, value in count_map.items()},
        }
    return summary


def count_labels_in_split(labels_1_6: np.ndarray, split: dict, subset: str, chunk_size: int = 8) -> np.ndarray:
    """Count 1..6 labels without materializing the full split as int64."""
    if subset not in {"train", "val", "buffer", "test_buffer", "test"} or subset not in split:
        raise ValueError(f"Unknown split subset: {subset}")
    axis = int(split["axis_index"])
    counts = np.zeros(7, dtype=np.int64)
    for start, end in subset_ranges(split, subset):
        for position in range(start, end, chunk_size):
            slicer = [slice(None)] * labels_1_6.ndim
            slicer[axis] = slice(position, min(end, position + chunk_size))
            chunk = np.asarray(labels_1_6[tuple(slicer)])
            chunk_counts = np.bincount(chunk.astype(np.int64, copy=False).ravel(), minlength=7)
            if len(chunk_counts) != 7 or chunk_counts[0]:
                raise ValueError(f"Invalid label values in {subset} split at {position}")
            counts += chunk_counts
    return counts[1:7]
