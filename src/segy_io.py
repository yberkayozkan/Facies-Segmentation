from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _require_segyio():
    try:
        import segyio
    except ImportError as exc:
        raise ImportError("segyio is required for SEG-Y IO. Install requirements_colab.txt in Colab.") from exc
    return segyio


def read_segy_volume(path: str | Path, *, strict: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a SEG-Y file as canonical [inline, crossline, depth].

    `segyio.tools.cube` is used when geometry is available. If geometry is not
    inferable and `strict=False`, traces are returned as [trace, 1, sample].
    """
    segyio = _require_segyio()
    path = Path(path)
    try:
        with segyio.open(str(path), "r", ignore_geometry=False) as f:
            cube = segyio.tools.cube(f)
            arr = np.asarray(cube)
            meta = {
                "path": str(path),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "axis_order": "inline_crossline_depth",
                "read_method": "segyio.tools.cube",
                "trace_count": int(f.tracecount),
                "sample_count": int(len(f.samples)),
                "iline_count": int(len(f.ilines)),
                "xline_count": int(len(f.xlines)),
            }
            return arr, meta
    except Exception:
        if strict:
            raise

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        traces = np.asarray([np.array(trace, copy=True) for trace in f.trace], dtype=np.float32)
        arr = traces[:, None, :]
        meta = {
            "path": str(path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "axis_order": "trace_single_crossline_depth",
            "read_method": "trace_fallback",
            "trace_count": int(f.tracecount),
            "sample_count": int(len(f.samples)),
        }
    return arr, meta


def validate_label_volume(labels: np.ndarray) -> None:
    for start in range(0, labels.shape[0], 8):
        chunk = np.asarray(labels[start:start + 8])
        invalid = ~np.isfinite(chunk) | (chunk < 1) | (chunk > 6)
        if not np.issubdtype(chunk.dtype, np.integer):
            invalid |= chunk != np.rint(chunk)
        if np.any(invalid):
            value = chunk[tuple(np.argwhere(invalid)[0])]
            raise ValueError(f"Label volume must contain only integer values 1..6; found {value} near inline {start}")


def write_metadata(meta: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def export_prediction_segy(template_path: str | Path, prediction: np.ndarray, output_path: str | Path) -> dict[str, Any]:
    """Copy template SEG-Y and replace trace samples with 1..6 prediction labels."""
    segyio = _require_segyio()
    template_path = Path(template_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(prediction)
    if prediction.dtype != np.uint8:
        prediction = prediction.astype(np.uint8)
    unique = set(np.unique(prediction).astype(int).tolist())
    if not unique.issubset(set(range(1, 7))):
        raise ValueError(f"Prediction labels must be in 1..6, got {sorted(unique)}")
    shutil.copyfile(template_path, output_path)
    with segyio.open(str(output_path), "r+", ignore_geometry=True) as f:
        flat = prediction.reshape(-1, prediction.shape[-1])
        if flat.shape[0] != f.tracecount:
            raise ValueError(f"Trace count mismatch: prediction has {flat.shape[0]}, SEG-Y has {f.tracecount}")
        if flat.shape[1] != len(f.samples):
            raise ValueError(f"Sample count mismatch: prediction has {flat.shape[1]}, SEG-Y has {len(f.samples)}")
        for idx in range(f.tracecount):
            f.trace[idx] = flat[idx].astype(np.float32)
    with segyio.open(str(output_path), "r", ignore_geometry=True) as f:
        meta = {
            "output": str(output_path),
            "template": str(template_path),
            "trace_count": int(f.tracecount),
            "sample_count": int(len(f.samples)),
            "labels": sorted(unique),
        }
    return meta
