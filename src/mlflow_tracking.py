from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
        else:
            flat[name] = json.dumps(value)
    return flat


class MLflowTracker:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("tracking", {}).get("mlflow", {}).get("enabled", False))
        self.mlflow = None
        self.active = False
        self.run_id = None
        if not self.enabled:
            return
        try:
            import mlflow
        except ImportError:
            print("MLflow tracking requested but mlflow is not installed. Continuing without remote tracking.")
            self.enabled = False
            return
        self.mlflow = mlflow

    def start(self) -> None:
        if not self.enabled or self.mlflow is None or self.active:
            return
        tracking_cfg = self.cfg["tracking"]["mlflow"]
        tracking_uri = tracking_cfg.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        experiment_name = tracking_cfg.get("experiment_name") or "parihaka-dino-unet"
        self.mlflow.set_experiment(experiment_name)
        run_name = tracking_cfg.get("run_name")
        run = self.mlflow.start_run(run_name=run_name, log_system_metrics=bool(tracking_cfg.get("log_system_metrics", True)))
        self.run_id = run.info.run_id
        self.active = True
        params = flatten_dict({k: v for k, v in self.cfg.items() if k not in {"tracking"}})
        # MLflow parameter values have practical length limits; keep logged params compact.
        compact_params = {k: (str(v)[:240] if v is not None else "None") for k, v in params.items()}
        self.mlflow.log_params(compact_params)
        self.mlflow.set_tags(
            {
                "project": "sismoML",
                "model": self.cfg["model"]["name"],
                "backbone": self.cfg["model"].get("backbone", ""),
                "feature_mode": self.cfg.get("feature_engineering", {}).get("mode", ""),
            }
        )

    def log_batch(self, epoch: int, step: int, metrics: dict[str, float]) -> None:
        if not self.active or self.mlflow is None:
            return
        global_step = epoch * int(self.cfg["dataset"].get("samples_per_epoch") or 0) + step
        self.mlflow.log_metrics({f"batch/{k}": float(v) for k, v in metrics.items()}, step=global_step)

    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        if not self.active or self.mlflow is None:
            return
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self.mlflow.log_metrics(numeric, step=int(epoch))
        class_iou = metrics.get("class_iou")
        if isinstance(class_iou, str):
            try:
                class_iou = json.loads(class_iou)
            except json.JSONDecodeError:
                class_iou = None
        if isinstance(class_iou, list):
            self.mlflow.log_metrics({f"class_iou/{idx + 1}": float(value) for idx, value in enumerate(class_iou)}, step=int(epoch))

    def log_artifacts(self) -> None:
        if not self.active or self.mlflow is None:
            return
        tracking_cfg = self.cfg["tracking"]["mlflow"]
        for key in ("logs", "figures"):
            path = Path(self.cfg["outputs"][key])
            if path.exists():
                self.mlflow.log_artifacts(str(path), artifact_path=key)
        if tracking_cfg.get("log_checkpoints", False):
            ckpt_dir = Path(self.cfg["outputs"]["checkpoints"])
            if ckpt_dir.exists():
                self.mlflow.log_artifacts(str(ckpt_dir), artifact_path="checkpoints")

    def end(self) -> None:
        if not self.active or self.mlflow is None:
            return
        self.mlflow.end_run()
        self.active = False
