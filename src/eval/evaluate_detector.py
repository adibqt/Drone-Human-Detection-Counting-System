"""Re-run validation on a trained YOLO checkpoint and save a metrics summary.

Intended local-side companion to the Colab training notebook: after you
download `best.pt` into `outputs/weights/best.pt`, run this script to
reproduce the validation metrics on your own machine and write
`outputs/metrics/task02_eval_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO detector (Task-02).")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/weights/best.pt"),
        help="Path to trained .pt weights.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/task02_data.yaml"),
        help="Ultralytics data YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=800, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate against.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/metrics/task02_eval_summary.json"),
        help="JSON file to write the summary into.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. '0' for first CUDA GPU or 'cpu'.",
    )
    return parser.parse_args()


def select_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def normalize_metrics(metrics_obj) -> dict[str, float]:
    raw = getattr(metrics_obj, "results_dict", None) or {}
    return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def per_class_metrics(metrics_obj, names: dict[int, str]) -> dict[str, dict[str, float]]:
    """Extract per-class P / R / mAP50 / mAP50-95 if available."""
    out: dict[str, dict[str, float]] = {}
    box = getattr(metrics_obj, "box", None)
    if box is None:
        return out
    try:
        for idx, cls_id in enumerate(box.ap_class_index.tolist()):
            cls_name = names.get(int(cls_id), str(cls_id))
            out[cls_name] = {
                "precision": float(box.p[idx]) if idx < len(box.p) else float("nan"),
                "recall": float(box.r[idx]) if idx < len(box.r) else float("nan"),
                "mAP50": float(box.ap50[idx]) if idx < len(box.ap50) else float("nan"),
                "mAP50_95": float(box.ap[idx]) if idx < len(box.ap) else float("nan"),
            }
    except Exception:
        return out
    return out


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data}")

    device = select_device(args.device)
    print(f"Loading {args.weights} on {device}")
    model = YOLO(str(args.weights))
    names = {int(k): v for k, v in model.names.items()}

    t0 = time.perf_counter()
    metrics_obj = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=device,
        plots=True,
        verbose=True,
    )
    elapsed_s = time.perf_counter() - t0

    overall = normalize_metrics(metrics_obj)
    per_cls = per_class_metrics(metrics_obj, names)

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(args.weights),
        "data_yaml": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "host": platform.node(),
        "duration_seconds": round(elapsed_s, 2),
        "overall": overall,
        "per_class": per_cls,
        "class_names": names,
    }
    args.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved evaluation summary to: {args.summary_path}")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
