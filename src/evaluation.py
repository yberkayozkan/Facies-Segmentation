from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .features import make_triplet
from .inference import predict_section
from .metrics import classification_metrics_from_confusion
from .splitting import range_for_index, subset_ranges


@torch.no_grad()
def evaluate_labeled_inlines(model, dataset, normalization: dict, device: torch.device, cfg: dict) -> dict:
    """Score a fixed labeled inline holdout without changing model or checkpoints."""
    if not getattr(dataset, "full_inline", False) or dataset.subset != "test":
        raise ValueError("Expected a labeled full-inline test dataset")
    model.eval()
    classes = int(cfg["model"]["num_classes"])
    cm = np.zeros((classes, classes), dtype=np.int64)
    nll_by_class = np.zeros(classes, dtype=np.float64)
    support = np.zeros(classes, dtype=np.int64)
    for _, idx in tqdm(dataset.indices, desc="test labeled inlines"):
        probabilities = predict_section(
            model, make_triplet(dataset.seismic, "inline", idx, bounds=range_for_index(dataset.split, "test", idx)),
            normalization, device, cfg
        )
        target = np.asarray(dataset.labels_1_6[idx], dtype=np.int64) - 1
        if probabilities.shape != (classes,) + target.shape:
            raise ValueError(f"Prediction/label shape mismatch at inline {idx}")
        if target.min() < 0 or target.max() >= classes:
            raise ValueError(f"Invalid test labels at inline {idx}")
        pred = probabilities.argmax(axis=0)
        cm += np.bincount((target * classes + pred).ravel(), minlength=classes * classes).reshape(classes, classes)
        true_prob = np.take_along_axis(probabilities, target[None], axis=0)[0]
        pixel_nll = -np.log(np.clip(true_prob, 1e-8, 1.0))
        support += np.bincount(target.ravel(), minlength=classes)
        nll_by_class += np.bincount(target.ravel(), weights=pixel_nll.ravel(), minlength=classes)
    metrics = classification_metrics_from_confusion(cm)
    metrics["test_nll"] = float(nll_by_class.sum() / max(1, int(support.sum())))
    present = support > 0
    metrics["miou_present_classes"] = float(np.mean(np.asarray(metrics["class_iou"])[present])) if present.any() else 0.0
    metrics["missing_test_classes"] = (np.flatnonzero(~present) + 1).tolist()
    metrics["class_nll"] = np.divide(
        nll_by_class, support, out=np.zeros_like(nll_by_class), where=support > 0
    ).tolist()
    metrics["evaluation_mode"] = "labeled_full_inline_test"
    metrics["test_inlines"] = [idx for _, idx in dataset.indices]
    metrics["test_ranges"] = subset_ranges(dataset.split, "test")
    return metrics
