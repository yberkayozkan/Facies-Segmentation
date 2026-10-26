from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

FACIES_LABELS = [f"Class {idx}" for idx in range(1, 7)]
FACIES_COLORS = plt.get_cmap("tab10").colors[:6]
FACIES_CMAP = ListedColormap(FACIES_COLORS, name="facies_tab10")
FACIES_NORM = BoundaryNorm(np.arange(0.5, 7.5, 1.0), FACIES_CMAP.N)


def add_facies_legend(ax, loc: str = "upper right", ncol: int = 1, fontsize: int = 8):
    handles = [Patch(facecolor=FACIES_COLORS[idx], edgecolor="none", label=label) for idx, label in enumerate(FACIES_LABELS)]
    return ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize, framealpha=0.9, borderpad=0.4)


def save_training_curves(history: list[dict], path: str | Path) -> None:
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, (loss_ax, metric_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    loss_ax.plot(epochs, [h.get("train_loss", np.nan) for h in history], label="train loss")
    loss_ax.plot(epochs, [h.get("val_loss", np.nan) for h in history], label="validation loss")
    loss_ax.set_ylabel("loss")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend(loc="best")

    metric_ax.plot(epochs, [h.get("miou", np.nan) for h in history], color="tab:green", label="mIoU")
    metric_ax.plot(epochs, [h.get("macro_f1", np.nan) for h in history], color="tab:red", label="macro F1")
    metric_ax.plot(epochs, [h.get("pixel_accuracy", np.nan) for h in history], color="tab:purple", label="pixel accuracy")
    if any("inline_miou" in h for h in history):
        metric_ax.plot(epochs, [h.get("inline_miou", np.nan) for h in history], color="tab:blue", linestyle="--", label="inline mIoU")
    if any("crossline_miou" in h for h in history):
        metric_ax.plot(epochs, [h.get("crossline_miou", np.nan) for h in history], color="tab:orange", linestyle="--", label="crossline mIoU")
    metric_ax.set_xlabel("epoch")
    metric_ax.set_ylabel("score")
    metric_ax.set_ylim(0.0, 1.0)
    metric_ax.grid(alpha=0.25)
    metric_ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def display_section(section: np.ndarray) -> np.ndarray:
    """Return [depth, lateral] view for plotting section arrays stored as [lateral, depth]."""
    return np.asarray(section).T


def save_qualitative_slice(seismic: np.ndarray, label: np.ndarray | None, pred: np.ndarray | None, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = 1 + int(label is not None) + int(pred is not None)
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(display_section(seismic), cmap="gray", aspect="auto")
    axes[0].set_title("seismic")
    idx = 1
    if label is not None:
        axes[idx].imshow(display_section(label), cmap=FACIES_CMAP, norm=FACIES_NORM, aspect="auto")
        axes[idx].set_title("label")
        add_facies_legend(axes[idx])
        idx += 1
    if pred is not None:
        axes[idx].imshow(display_section(pred), cmap=FACIES_CMAP, norm=FACIES_NORM, aspect="auto")
        axes[idx].set_title("prediction")
        add_facies_legend(axes[idx])
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, path: str | Path | None = None):
    cm = np.asarray(cm, dtype=np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(cm.shape[1]), labels=[str(i) for i in range(1, cm.shape[1] + 1)])
    ax.set_yticks(range(cm.shape[0]), labels=[str(i) for i in range(1, cm.shape[0] + 1)])
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            value = cm_norm[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    return fig
