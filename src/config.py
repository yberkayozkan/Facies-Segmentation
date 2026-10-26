from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "project_root": "/content/drive/MyDrive/parihaka_dino_unet",
    "data": {
        "train_seismic": "data/TrainingData_Image.segy",
        "train_labels": "data/TrainingData_Labels.segy",
        "test1_seismic": "data/TestData_Image1.segy",
        "test2_seismic": "data/TestData_Image2.segy",
        "axis_order": "inline_crossline_depth",
    },
    "cache": {"dir": "cache", "mmap": True},
    "outputs": {
        "dir": "outputs",
        "checkpoints": "outputs/checkpoints",
        "predictions": "outputs/predictions",
        "logs": "outputs/logs",
        "figures": "outputs/figures",
    },
    "tracking": {
        "mlflow": {
            "enabled": False,
            "tracking_uri": "https://dagshub.com/dragnellstr/sismoML.mlflow",
            "experiment_name": "parihaka-dino-unet",
            "run_name": None,
            "log_system_metrics": True,
            "log_checkpoints": False,
        }
    },
    "split": {
        "axis": "inline",
        "train_fraction": 0.72,
        "buffer_fraction": 0.04,
        "val_fraction": 0.20,
        "path": "outputs/split.json",
    },
    "normalization": {
        "lower_percentile": 0.5,
        "upper_percentile": 99.5,
        "max_samples": 2_000_000,
        "path": "outputs/normalization_stats.json",
    },
    "model": {
        "name": "dino_unet",
        "num_classes": 6,
        "input_size": 448,
        "backbone": "facebook/dinov2-small",
        "dino_layers": [3, 6, 9, 12],
        "decoder_channels": 256,
        "freeze_encoder_epochs": 3,
        "trainable_dino_blocks": 4,
        "use_facies_query": False,
        "query_beta": 0.25,
        "gradient_checkpointing": False,
    },
    "training": {
        "epochs": 20,
        "batch_size": 2,
        "val_batch_size": 1,
        "num_workers": 2,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "gradient_accumulation": 2,
        "precision": "bf16",
        "decoder_lr": 1.0e-4,
        "encoder_lr": 1.0e-5,
        "min_lr_ratio": 0.05,
        "batch_log_interval": 1,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 1,
        "max_grad_norm": 1.0,
        "resume": True,
        "loss": "ce_dice",
        "class_weights": "inverse_sqrt",
    },
    "dataset": {
        "orientations": ["inline", "crossline"],
        "patch_size": 448,
        "random_crop": True,
        "crop_strategy": "random_crop",
        "class_aware_sampling": False,
        "class_aware_fraction": 0.5,
        "global_context_fraction": 0.0,
        "samples_per_epoch": None,
        "val_samples_per_epoch": None,
        "random_sample_when_limited": True,
        "require_all_orientations": False,
        "balance_orientations": False,
    },
    "feature_engineering": {
        "mode": "neighbor_triplet",
    },
    "augmentation": {
        "horizontal_flip": True,
        "rotation_degrees": 0.0,
        "amplitude_scale": 0.05,
        "intensity_shift": 0.03,
        "gaussian_noise": 0.01,
    },
    "inference": {
        "tile_size": 448,
        "tile_overlap": 0.35,
        "global_context_weight": 0.0,
        "alpha": 0.5,
        "tta_horizontal_flip": False,
        "save_probabilities": False,
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, loaded)
    if overrides:
        cfg = deep_update(cfg, overrides)
    if cfg["model"].get("name") == "seismic_unet":
        model_keys = ("name", "num_classes", "input_size", "backbone", "base_channels", "dropout")
        cfg["model"] = {key: cfg["model"][key] for key in model_keys if key in cfg["model"]}
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(cfg["project_root"]) / p


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for value in cfg["outputs"].values():
        Path(resolve_path(cfg, value)).mkdir(parents=True, exist_ok=True)
    Path(resolve_path(cfg, cfg["cache"]["dir"])).mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
