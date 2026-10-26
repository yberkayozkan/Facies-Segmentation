from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .losses import build_loss, inverse_sqrt_class_weights, inverse_sqrt_class_weights_from_counts
from .metrics import ConfusionMeter
from .mlflow_tracking import MLflowTracker
from .models import parameter_groups
from .features import make_triplet
from .inference import predict_section


def dataloader_kwargs(cfg: dict) -> dict:
    num_workers = int(cfg["training"]["num_workers"])
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": bool(cfg["training"].get("persistent_workers", True)) and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg["training"].get("prefetch_factor", 2))
    return kwargs


def amp_dtype(precision: str) -> torch.dtype | None:
    precision = precision.lower()
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision in {"bf16", "fp16", "float16"} and torch.cuda.is_available():
        return torch.float16
    return None


def validate_batch(batch: dict, num_classes: int, phase: str, step: int) -> None:
    image = batch["image"]
    mask = batch["mask"]
    if image.ndim != 4:
        raise ValueError(f"{phase} step {step}: expected image [B,C,H,W], got {tuple(image.shape)}")
    if mask.ndim != 3:
        raise ValueError(f"{phase} step {step}: expected mask [B,H,W], got {tuple(mask.shape)}")
    if image.shape[0] != mask.shape[0] or image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(f"{phase} step {step}: image/mask shape mismatch {tuple(image.shape)} vs {tuple(mask.shape)}")
    if not torch.isfinite(image).all():
        raise ValueError(f"{phase} step {step}: image contains NaN or Inf")
    mask_min = int(mask.min().item())
    mask_max = int(mask.max().item())
    if mask_min < 0 or mask_max >= num_classes:
        raise ValueError(f"{phase} step {step}: mask values must be in 0..{num_classes - 1}, got min={mask_min}, max={mask_max}")


def build_optimizer_scheduler(model: torch.nn.Module, cfg: dict, steps_per_epoch: int):
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg), weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = int(cfg["training"]["epochs"])
    warmup_steps = max(1, int(cfg["training"]["warmup_epochs"]) * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)
    min_lr_ratio = float(cfg["training"].get("min_lr_ratio", 0.05))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict,
        train_dataset,
        val_dataset,
        normalization: dict,
        split: dict,
        device: torch.device | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.normalization = normalization
        self.split = split
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs(cfg),
        )
        self.val_loader = None if getattr(val_dataset, "full_inline", False) else DataLoader(
            val_dataset,
            batch_size=int(cfg["training"].get("val_batch_size", cfg["training"]["batch_size"])),
            shuffle=False,
            **dataloader_kwargs(cfg),
        )
        class_weights = None
        if cfg["training"].get("class_weights") == "inverse_sqrt":
            if hasattr(train_dataset, "class_counts_for_weights"):
                class_weights = inverse_sqrt_class_weights_from_counts(train_dataset.class_counts_for_weights()).to(self.device)
            else:
                class_weights = inverse_sqrt_class_weights(train_dataset.labels).to(self.device)
        self.criterion = build_loss(cfg["training"]["loss"], class_weights, int(cfg["model"]["num_classes"]))
        accumulation = int(cfg["training"]["gradient_accumulation"])
        if accumulation < 1:
            raise ValueError("gradient_accumulation must be at least 1")
        if len(self.train_loader) == 0:
            raise ValueError("Training dataset is smaller than one batch with drop_last=True")
        optimizer_steps_per_epoch = math.ceil(len(self.train_loader) / accumulation)
        self.optimizer, self.scheduler = build_optimizer_scheduler(self.model, cfg, optimizer_steps_per_epoch)
        self.autocast_dtype = amp_dtype(cfg["training"]["precision"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.autocast_dtype == torch.float16)
        self.best_miou = -1.0
        self.start_epoch = 0
        self.history: list[dict] = []
        self.checkpoint_dir = Path(cfg["outputs"]["checkpoints"])
        self.best_path = self.checkpoint_dir / "best_model.pth"
        self.last_path = self.checkpoint_dir / "last_model.pth"
        self.log_path = Path(cfg["outputs"]["logs"]) / "experiment_results.csv"
        self.batch_log_path = Path(cfg["outputs"]["logs"]) / "batch_losses.csv"
        self.metrics_json = Path(cfg["outputs"]["logs"]) / "last_validation_metrics.json"
        self.best_metrics_json = Path(cfg["outputs"]["logs"]) / "best_validation_metrics.json"
        self.status_path = Path(cfg["outputs"]["logs"]) / "run_status.json"
        self.tracker = MLflowTracker(cfg)
        if not self.cfg["training"].get("resume", True):
            for path in (self.log_path, self.batch_log_path):
                if path.exists():
                    path.unlink()

    def maybe_resume(self) -> None:
        if not self.cfg["training"].get("resume", True) or not self.last_path.exists():
            return
        ckpt = load_checkpoint(self.last_path, self.model, self.optimizer, self.scheduler, self.scaler, map_location=self.device)
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_miou = float(ckpt.get("best_miou", -1.0))

    def _write_status(self, phase: str, **details) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_path, "w", encoding="utf-8") as f:
            json.dump({"phase": phase, **details}, f, indent=2)

    def _track(self, method: str, *args) -> None:
        if not self.tracker.enabled:
            return
        try:
            getattr(self.tracker, method)(*args)
        except Exception as exc:
            print(f"MLflow {method} failed: {type(exc).__name__}: {exc}. Continuing with local outputs.")
            self.tracker.enabled = False
            self.tracker.active = False

    def _set_encoder_schedule(self, epoch: int) -> None:
        if hasattr(self.model, "set_encoder_schedule"):
            self.model.set_encoder_schedule(
                epoch,
                int(self.cfg["model"].get("freeze_encoder_epochs", 0)),
                int(self.cfg["model"].get("trainable_dino_blocks", 0)),
            )

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._set_encoder_schedule(epoch)
        total = 0.0
        steps = 0
        accum = int(self.cfg["training"]["gradient_accumulation"])
        batch_log_interval = max(1, int(self.cfg["training"].get("batch_log_interval", 1)))
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"train {epoch}")
        for step, batch in enumerate(pbar):
            validate_batch(batch, int(self.cfg["model"]["num_classes"]), "train", step)
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)
            group_start = (step // accum) * accum
            group_size = min(accum, len(self.train_loader) - group_start)
            with ctx:
                logits = self.model(image)
                loss = self.criterion(logits, mask) / group_size
            self.scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["training"]["max_grad_norm"]))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu()) * group_size
            steps += 1
            batch_loss = float(loss.detach().cpu()) * group_size
            running_loss = total / max(1, steps)
            if step % batch_log_interval == 0:
                batch_row = {
                    "epoch": epoch,
                    "step": step,
                    "batch_loss": batch_loss,
                    "running_loss": running_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                self._append_csv(self.batch_log_path, batch_row)
                self._track("log_batch", epoch, step, {"loss": batch_loss, "running_loss": running_loss, "lr": self.optimizer.param_groups[0]["lr"]})
            pbar.set_postfix(loss=running_loss, lr=self.optimizer.param_groups[0]["lr"])
        return total / max(1, steps)

    @torch.no_grad()
    def validate(self) -> tuple[float, dict]:
        if getattr(self.val_dataset, "full_inline", False):
            return self._validate_full_inline()
        self.model.eval()
        num_classes = int(self.cfg["model"]["num_classes"])
        meter = ConfusionMeter(num_classes)
        orientation_meters: dict[str, ConfusionMeter] = {}
        total = 0.0
        steps = 0
        nll_sum = 0.0
        pixel_count = 0
        for step, batch in enumerate(tqdm(self.val_loader, desc="validate")):
            validate_batch(batch, int(self.cfg["model"]["num_classes"]), "validate", step)
            image = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)
            with ctx:
                logits = self.model(image)
                loss = self.criterion(logits, mask)
            meter.update(logits, mask)
            for orientation in sorted(set(batch["orientation"])):
                batch_indices = [idx for idx, value in enumerate(batch["orientation"]) if value == orientation]
                selected = torch.tensor(batch_indices, device=logits.device, dtype=torch.long)
                orientation_meters.setdefault(orientation, ConfusionMeter(num_classes)).update(
                    logits.index_select(0, selected), mask.index_select(0, selected)
                )
            total += float(loss.detach().cpu())
            steps += 1
            nll_sum += float(F.cross_entropy(logits.float(), mask, reduction="sum").detach().cpu())
            pixel_count += mask.numel()
        metrics = meter.compute()
        metrics["orientation_metrics"] = {
            orientation: orientation_meter.compute()
            for orientation, orientation_meter in sorted(orientation_meters.items())
        }
        metrics["val_loss"] = total / max(1, steps)
        metrics["val_nll"] = nll_sum / max(1, pixel_count)
        self.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics["val_loss"], metrics

    @torch.no_grad()
    def _validate_full_inline(self) -> tuple[float, dict]:
        self.model.eval()
        meter = ConfusionMeter(int(self.cfg["model"]["num_classes"]))
        total_loss = 0.0
        nll_sum = 0.0
        nll_by_class = torch.zeros(int(self.cfg["model"]["num_classes"]), dtype=torch.float64)
        count_by_class = torch.zeros_like(nll_by_class)
        for _, idx in tqdm(self.val_dataset.indices, desc="validate full inline"):
            triplet = make_triplet(
                self.val_dataset.seismic, "inline", idx,
                bounds=tuple(self.val_dataset.split[self.val_dataset.subset]),
            )
            ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)
            with ctx:
                probabilities = predict_section(
                    self.model, triplet, self.normalization, self.device, self.cfg
                )
            log_probs = torch.from_numpy(probabilities.copy()).to(self.device).clamp_min(1e-8).log().unsqueeze(0)
            mask = torch.as_tensor(
                np.asarray(self.val_dataset.labels_1_6[idx], dtype=np.int64) - 1,
                device=self.device,
            ).unsqueeze(0)
            total_loss += float(self.criterion(log_probs, mask).detach().cpu())
            pixel_nll = F.nll_loss(log_probs, mask, reduction="none")
            nll_sum += float(pixel_nll.sum().detach().cpu())
            nll_by_class += torch.bincount(mask.reshape(-1), weights=pixel_nll.reshape(-1).double(), minlength=len(nll_by_class)).cpu()
            count_by_class += torch.bincount(mask.reshape(-1), minlength=len(nll_by_class)).cpu()
            meter.update(log_probs, mask)
        metrics = meter.compute()
        metrics["orientation_metrics"] = {"inline": meter.compute()}
        metrics["val_loss"] = total_loss / len(self.val_dataset)
        metrics["val_nll"] = nll_sum / max(1, int(count_by_class.sum()))
        metrics["class_nll"] = torch.where(count_by_class > 0, nll_by_class / count_by_class.clamp_min(1), 0).tolist()
        metrics["validation_mode"] = "full_inline"
        metrics["val_loss_source"] = "training_criterion_on_blended_tile_probabilities"
        metrics["validation_inlines"] = [idx for _, idx in self.val_dataset.indices]
        self.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return metrics["val_loss"], metrics

    def _append_csv(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def fit(self) -> list[dict]:
        self._write_status("starting")
        try:
            self._track("start")
            self.maybe_resume()
            epochs = int(self.cfg["training"]["epochs"])
            for epoch in range(self.start_epoch, epochs):
                self._write_status("training", epoch=epoch)
                start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                train_loss = self.train_one_epoch(epoch)
                val_loss, metrics = self.validate()
                miou = float(metrics["miou"])
                orientation_metrics = metrics.get("orientation_metrics", {})
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_nll": metrics.get("val_nll"),
                    "miou": miou,
                    "class_iou": json.dumps(metrics["class_iou"]),
                    "class_f1": json.dumps(metrics["class_f1"]),
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "mean_class_accuracy": metrics["mean_class_accuracy"],
                    "frequency_weighted_iou": metrics["frequency_weighted_iou"],
                    "inline_miou": orientation_metrics.get("inline", {}).get("miou"),
                    "crossline_miou": orientation_metrics.get("crossline", {}).get("miou"),
                    "inline_macro_f1": orientation_metrics.get("inline", {}).get("macro_f1"),
                    "crossline_macro_f1": orientation_metrics.get("crossline", {}).get("macro_f1"),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "lr_groups": json.dumps([group["lr"] for group in self.optimizer.param_groups]),
                    "epoch_time_sec": time.time() - start,
                    "peak_gpu_gb": peak_mem,
                }
                self.history.append(row)
                self._append_csv(self.log_path, row)
                self._track("log_epoch", epoch, row)
                is_best = miou > self.best_miou
                if is_best:
                    self.best_miou = miou
                    save_checkpoint(self.best_path, self.model, self.optimizer, self.scheduler, self.scaler, epoch, self.best_miou, self.cfg, self.normalization, self.split)
                    with open(self.best_metrics_json, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2)
                save_checkpoint(self.last_path, self.model, self.optimizer, self.scheduler, self.scaler, epoch, self.best_miou, self.cfg, self.normalization, self.split)
                self._write_status("epoch_complete", epoch=epoch, miou=miou, best_miou=self.best_miou)
            self._track("log_artifacts")
            self._write_status("complete", last_epoch=epochs - 1, best_miou=self.best_miou)
            return self.history
        except Exception as exc:
            self._write_status("failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            self._track("end")
