"""Train a YOLOv8 detector for VisDrone person/car (Task-02).

Loads a hyperparameter YAML (`configs/task02_train.yaml`), forwards every
key as a kwarg to `ultralytics.YOLO.train`, and writes a JSON summary of the
final validation metrics so that downstream tasks can consume them.
"""

from __future__ import annotations

import argparse
import builtins
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


_ORIGINAL_OPEN = builtins.open


def _retry_open(*args, **kwargs):
    """Drop-in replacement for `open` that retries on transient PermissionError.

    Windows antivirus / search-indexer occasionally locks small files
    (`results.csv`, JSON summaries) for a few hundred ms while a writer
    process tries to append. This wrapper retries up to ~3 seconds total
    before giving up so a long training run is not killed by a transient
    lock.
    """
    last_exc: PermissionError | None = None
    for delay in (0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.0):
        if delay:
            time.sleep(delay)
        try:
            return _ORIGINAL_OPEN(*args, **kwargs)
        except PermissionError as exc:  # transient on Windows
            last_exc = exc
    raise last_exc  # type: ignore[misc]


builtins.open = _retry_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for Task-02.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/task02_train.yaml"),
        help="Path to the training hyperparameter YAML.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override the data YAML path (default: value from train YAML).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model checkpoint (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--name", type=str, default=None, help="Override Ultralytics run name.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. '0' for first CUDA GPU or 'cpu'.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to last.pt to resume from. Reuses the original run's hyperparameters.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/metrics/task02_training_summary.json"),
        help="Where to save the training/validation summary JSON.",
    )
    return parser.parse_args()


def load_train_kwargs(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def select_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def normalize_metrics(metrics_obj) -> dict[str, float]:
    """Pull a flat dict of validation metrics from an Ultralytics results object."""
    if metrics_obj is None:
        return {}
    raw = getattr(metrics_obj, "results_dict", None) or {}
    return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def write_summary(
    summary_path: Path,
    train_kwargs: dict,
    metrics: dict[str, float],
    save_dir: Path,
    device: str,
    duration_s: float,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "duration_seconds": round(duration_s, 2),
        "train_kwargs": train_kwargs,
        "save_dir": str(save_dir),
        "best_weights": str(Path(save_dir) / "weights" / "best.pt"),
        "last_weights": str(Path(save_dir) / "weights" / "last.pt"),
        "val_metrics": metrics,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved training summary to: {summary_path}")


def main() -> None:
    args = parse_args()
    train_kwargs = load_train_kwargs(args.config)

    if args.data is not None:
        train_kwargs["data"] = str(args.data)
    if args.model is not None:
        train_kwargs["model"] = args.model
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.imgsz is not None:
        train_kwargs["imgsz"] = args.imgsz
    if args.batch is not None:
        train_kwargs["batch"] = args.batch
    if args.name is not None:
        train_kwargs["name"] = args.name

    train_kwargs["device"] = select_device(args.device)
    if "project" in train_kwargs:
        train_kwargs["project"] = str(Path(train_kwargs["project"]).resolve())

    model_arg = train_kwargs.pop("model")
    if args.resume:
        resume_weights = Path(args.resume)
        if not resume_weights.exists():
            raise FileNotFoundError(f"--resume weights not found: {resume_weights}")
        print(f"Resuming training from: {resume_weights}")
        model = YOLO(str(resume_weights))
        train_kwargs = {"resume": True, "device": train_kwargs["device"]}
    else:
        print(f"Loading base model: {model_arg}")
        model = YOLO(model_arg)

    print(f"Starting training on device {train_kwargs['device']} with {train_kwargs.get('epochs')} epochs")
    started_at = datetime.now()
    results = model.train(**train_kwargs)
    duration_s = (datetime.now() - started_at).total_seconds()

    save_dir = Path(getattr(results, "save_dir", train_kwargs.get("project", "runs")))
    print(f"Training finished in {duration_s/60:.1f} min. Save dir: {save_dir}")

    print("Running final validation pass on best.pt ...")
    best_weights = save_dir / "weights" / "best.pt"
    val_model = YOLO(str(best_weights)) if best_weights.exists() else model
    val_results = val_model.val(
        data=train_kwargs["data"],
        imgsz=train_kwargs.get("imgsz", 640),
        batch=train_kwargs.get("batch", 8),
        device=train_kwargs["device"],
        plots=True,
    )
    metrics = normalize_metrics(val_results)

    write_summary(
        args.summary_path,
        {**train_kwargs, "model": model_arg},
        metrics,
        save_dir,
        train_kwargs["device"],
        duration_s,
    )


if __name__ == "__main__":
    main()
