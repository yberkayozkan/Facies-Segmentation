from __future__ import annotations

from pathlib import Path

import numpy as np

from .segy_io import export_prediction_segy


def save_prediction_outputs(prediction: np.ndarray, npy_path: str | Path, template_segy: str | Path | None = None, output_segy: str | Path | None = None) -> dict:
    prediction = np.asarray(prediction, dtype=np.uint8)
    if not set(np.unique(prediction).astype(int).tolist()).issubset(set(range(1, 7))):
        raise ValueError("Prediction must contain only labels 1..6.")
    npy_path = Path(npy_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, prediction)
    meta = {"npy": str(npy_path), "shape": list(prediction.shape), "dtype": str(prediction.dtype)}
    if template_segy and output_segy:
        meta["segy"] = export_prediction_segy(template_segy, prediction, output_segy)
    return meta
