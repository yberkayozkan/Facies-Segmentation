from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    pred = pred.detach().view(-1).to(torch.int64)
    target = target.detach().view(-1).to(torch.int64)
    mask = (target >= 0) & (target < num_classes)
    idx = target[mask] * num_classes + pred[mask]
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def iou_from_confusion(cm: torch.Tensor | np.ndarray) -> tuple[np.ndarray, float]:
    if isinstance(cm, torch.Tensor):
        cm_np = cm.detach().cpu().numpy()
    else:
        cm_np = np.asarray(cm)
    tp = np.diag(cm_np).astype(np.float64)
    fp = cm_np.sum(axis=0) - tp
    fn = cm_np.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return iou, float(np.mean(iou))


def classification_metrics_from_confusion(cm: torch.Tensor | np.ndarray) -> dict[str, object]:
    if isinstance(cm, torch.Tensor):
        cm_np = cm.detach().cpu().numpy()
    else:
        cm_np = np.asarray(cm)
    class_iou, miou = iou_from_confusion(cm_np)
    tp = np.diag(cm_np).astype(np.float64)
    fp = cm_np.sum(axis=0).astype(np.float64) - tp
    fn = cm_np.sum(axis=1).astype(np.float64) - tp
    gt = cm_np.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    class_f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    total = float(cm_np.sum())
    class_acc = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    pixel_accuracy = float(tp.sum() / total) if total > 0 else 0.0
    mean_class_accuracy = float(np.mean(class_acc))
    freq = np.divide(gt, total, out=np.zeros_like(gt), where=total > 0)
    frequency_weighted_iou = float(np.sum(freq * class_iou))
    macro_f1 = float(np.mean(class_f1))
    weighted_f1 = float(np.sum(freq * class_f1))
    return {
        "confusion_matrix": cm_np.tolist(),
        "class_support": gt.astype(np.int64).tolist(),
        "class_predicted": cm_np.sum(axis=0).astype(np.int64).tolist(),
        "class_iou": class_iou.tolist(),
        "miou": miou,
        "class_accuracy": class_acc.tolist(),
        "class_precision": precision.tolist(),
        "class_recall": recall.tolist(),
        "class_f1": class_f1.tolist(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "pixel_accuracy": pixel_accuracy,
        "mean_class_accuracy": mean_class_accuracy,
        "frequency_weighted_iou": frequency_weighted_iou,
    }


class ConfusionMeter:
    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        self.cm += confusion_matrix(pred.cpu(), target.cpu(), self.num_classes)

    def compute(self) -> dict[str, object]:
        return classification_metrics_from_confusion(self.cm)
