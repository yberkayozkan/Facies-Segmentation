from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

CACHE_VERSION = 2


def _source_signature(path: str | Path) -> dict[str, object]:
    p = Path(path)
    stat = p.stat()
    return {"path": str(p), "mtime": stat.st_mtime, "size": stat.st_size}


def load_or_create_volume_cache(
    source_path: str | Path,
    cache_dir: str | Path,
    name: str,
    reader: Callable[[str | Path], tuple[np.ndarray, dict]],
    *,
    mmap: bool = True,
) -> tuple[np.ndarray, dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{name}.npy"
    meta_path = cache_dir / f"{name}.json"
    signature = _source_signature(source_path)
    if npy_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("source") == signature and meta.get("cache_version") == CACHE_VERSION:
            return np.load(npy_path, mmap_mode="r" if mmap else None), meta
    volume, read_meta = reader(source_path)
    np.save(npy_path, volume)
    meta = {
        "source": signature,
        "cache_version": CACHE_VERSION,
        "cache_path": str(npy_path),
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "axis_order": "inline_crossline_depth",
        "reader": read_meta,
    }
    tmp = str(meta_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp, meta_path)
    return np.load(npy_path, mmap_mode="r" if mmap else None), meta
