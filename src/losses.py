from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 6, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(target.clamp_min(0), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    losses = []
    for c in range(num_classes):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    probs = logits.softmax(dim=1).permute(0, 2, 3, 1).reshape(-1, num_classes)
    labels = target.reshape(-1)
    return lovasz_softmax_flat(probs, labels, num_classes)


def build_loss(name: str, class_weights: torch.Tensor | None = None, num_classes: int = 6):
    name = name.lower()

    def ce(logits, target):
        return F.cross_entropy(logits, target, weight=class_weights)

    if name == "ce":
        return ce
    if name == "ce_dice":
        return lambda logits, target: ce(logits, target) + dice_loss(logits, target, num_classes)
    if name == "ce_lovasz":
        return lambda logits, target: ce(logits, target) + lovasz_softmax_loss(logits, target, num_classes)
    raise ValueError(f"Unknown loss: {name}")


def inverse_sqrt_class_weights(labels_0_5, num_classes: int = 6) -> torch.Tensor:
    counts = torch.bincount(torch.as_tensor(labels_0_5, dtype=torch.long).view(-1), minlength=num_classes).float()
    return inverse_sqrt_class_weights_from_counts(counts)


def inverse_sqrt_class_weights_from_counts(counts) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=torch.float32)
    weights = 1.0 / torch.sqrt(counts.clamp_min(1.0))
    weights = weights / weights.mean()
    return weights
