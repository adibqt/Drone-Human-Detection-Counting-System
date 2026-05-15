"""Task-03 — Human & Car detection with human counting.

Loads a trained YOLOv8 checkpoint and produces annotated outputs plus a
per-input record of `person_count`, `car_count`, and inference time.

Supports two input modes:

* `--image <path>`     — single image (.jpg / .png / ...).
* `--image-dir <dir>`  — every image in the directory (sorted, or
                         randomly sampled if `--num-samples` is given).

The counting logic is intentionally simple, exactly as the assignment
permits: after class-specific confidence filtering and a minimum-bbox-
area filter, `human_count = number of remaining 'person' detections`.

All thresholds and drawing config live in `configs/task03_count.yaml`
so they are version-controlled and easy to tweak. Video / streaming
inputs are out of scope for Task-03 and live with the Task-04
tracking work.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


CLASS_COLORS = {
    "person": (0, 200, 0),
    "car": (0, 165, 255),
}
DEFAULT_COLOR = (200, 200, 200)
BANNER_BG = (0, 0, 0)
BANNER_FG_HUMAN = (255, 255, 0)    # cyan-ish yellow for the primary metric
BANNER_FG_CAR = (255, 255, 255)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class CountingConfig:
    weights: Path
    imgsz: int
    iou: float
    class_thresholds: dict[str, float]
    min_bbox_area_px: float
    draw: dict[str, object] = field(default_factory=dict)
    primary_metric: str = "human_count"

    @classmethod
    def from_yaml(cls, path: Path) -> "CountingConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        draw = raw.get("draw") or {}
        return cls(
            weights=Path(raw.get("weights", "outputs/weights/best.pt")),
            imgsz=int(raw.get("imgsz", 800)),
            iou=float(raw.get("iou", 0.5)),
            class_thresholds={k: float(v) for k, v in (raw.get("class_thresholds") or {}).items()},
            min_bbox_area_px=float(raw.get("min_bbox_area_px", 0)),
            draw=draw,
            primary_metric=str(draw.get("primary_metric", "human_count")),
        )


@dataclass
class Detection:
    cls_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-03 detection + human counting pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/task03_count.yaml"),
        help="Counting / threshold / drawing config YAML.",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Run on a single image file.")
    src.add_argument(
        "--image-dir",
        type=Path,
        help="Run on every image inside the directory (or a random sample of `--num-samples`).",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="If >0 and --image-dir is given, randomly sample this many images instead of processing all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for --num-samples.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/task03/predictions"),
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=Path("outputs/metrics/task03_counts.csv"),
        help="CSV path for per-image counts.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/metrics/task03_summary.json"),
        help="JSON path for the aggregated summary (totals, mean ms/image, etc.).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device, e.g. '0' for first CUDA GPU or 'cpu'.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Override the weights path from the config YAML.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override the inference image size from the config YAML.",
    )
    return parser.parse_args()


def select_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def lowest_threshold(cfg: CountingConfig) -> float:
    """Use the minimum across classes for YOLO's raw `conf` arg so we can
    apply per-class thresholds ourselves afterwards."""
    if not cfg.class_thresholds:
        return 0.25
    return float(min(cfg.class_thresholds.values()))


def filter_detections(result, cfg: CountingConfig, names: dict[int, str]) -> list[Detection]:
    """Apply class-specific thresholds + min-area filter to a YOLO result."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    detections: list[Detection] = []
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
        cls_name = names.get(int(cls_id), str(cls_id))
        threshold = cfg.class_thresholds.get(cls_name, lowest_threshold(cfg))
        if float(conf) < threshold:
            continue
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        if area < cfg.min_bbox_area_px:
            continue
        detections.append(
            Detection(
                cls_name=cls_name,
                confidence=float(conf),
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
        )
    return detections


def count_classes(detections: Iterable[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.cls_name] = counts.get(det.cls_name, 0) + 1
    return counts


def draw_overlay(image: np.ndarray, detections: list[Detection], cfg: CountingConfig) -> np.ndarray:
    """Draw all bounding boxes + the Task-03 counting banner."""
    canvas = image.copy()

    box_thickness = int(cfg.draw.get("box_thickness", 2))
    label_scale = float(cfg.draw.get("label_font_scale", 0.45))
    label_thickness = int(cfg.draw.get("label_font_thickness", 1))
    banner_scale = float(cfg.draw.get("banner_font_scale", 0.9))
    banner_thickness = int(cfg.draw.get("banner_font_thickness", 2))

    for det in detections:
        color = CLASS_COLORS.get(det.cls_name, DEFAULT_COLOR)
        cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), color, box_thickness)

        text = f"{det.cls_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
        ty1 = max(0, det.y1 - th - baseline - 2)
        cv2.rectangle(canvas, (det.x1, ty1), (det.x1 + tw + 4, ty1 + th + baseline + 2), color, -1)
        cv2.putText(
            canvas,
            text,
            (det.x1 + 2, ty1 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (0, 0, 0),
            label_thickness,
            cv2.LINE_AA,
        )

    counts = count_classes(detections)
    human_count = counts.get("person", 0)
    car_count = counts.get("car", 0)
    primary_text = f"Human Count: {human_count}"
    secondary_text = f"Car Count: {car_count}"

    (pw, ph), pbl = cv2.getTextSize(primary_text, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thickness)
    (sw, sh), sbl = cv2.getTextSize(secondary_text, cv2.FONT_HERSHEY_SIMPLEX, banner_scale * 0.7, banner_thickness)

    banner_w = max(pw, sw) + 24
    banner_h = ph + sh + pbl + sbl + 24
    cv2.rectangle(canvas, (10, 10), (10 + banner_w, 10 + banner_h), BANNER_BG, -1)

    cv2.putText(
        canvas,
        primary_text,
        (22, 10 + ph + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        banner_scale,
        BANNER_FG_HUMAN,
        banner_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        secondary_text,
        (22, 10 + ph + 12 + sh + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        banner_scale * 0.7,
        BANNER_FG_CAR,
        banner_thickness,
        cv2.LINE_AA,
    )

    return canvas


def predict_one(
    model: YOLO,
    image: np.ndarray,
    cfg: CountingConfig,
    device: str,
    base_conf: float,
) -> tuple[list[Detection], float]:
    t0 = time.perf_counter()
    results = model.predict(
        source=image,
        imgsz=cfg.imgsz,
        conf=base_conf,
        iou=cfg.iou,
        device=device,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    names = {int(k): v for k, v in model.names.items()}
    detections = filter_detections(results[0], cfg, names)
    return detections, elapsed_ms


def run_image_inputs(
    image_paths: list[Path],
    model: YOLO,
    cfg: CountingConfig,
    device: str,
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_conf = lowest_threshold(cfg)
    records: list[dict] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        detections, elapsed_ms = predict_one(model, image, cfg, device, base_conf)
        counts = count_classes(detections)
        annotated = draw_overlay(image, detections, cfg)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), annotated)
        records.append(
            {
                "source": image_path.name,
                "person_count": int(counts.get("person", 0)),
                "car_count": int(counts.get("car", 0)),
                "inference_time_ms": round(elapsed_ms, 2),
                "annotated_path": str(out_path).replace("\\", "/"),
            }
        )

    return records


def write_csv(records: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "person_count", "car_count", "inference_time_ms", "annotated_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_summary(records: list[dict], cfg: CountingConfig, summary_path: Path, mode: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        total_persons = sum(r["person_count"] for r in records)
        total_cars = sum(r["car_count"] for r in records)
        mean_ms = sum(r["inference_time_ms"] for r in records) / len(records)
        max_persons = max(r["person_count"] for r in records)
        max_cars = max(r["car_count"] for r in records)
    else:
        total_persons = total_cars = max_persons = max_cars = 0
        mean_ms = 0.0

    payload = {
        "mode": mode,
        "weights": str(cfg.weights),
        "imgsz": cfg.imgsz,
        "iou": cfg.iou,
        "class_thresholds": cfg.class_thresholds,
        "min_bbox_area_px": cfg.min_bbox_area_px,
        "num_inputs": len(records),
        "total_persons_detected": total_persons,
        "total_cars_detected": total_cars,
        "max_persons_in_single_input": max_persons,
        "max_cars_in_single_input": max_cars,
        "mean_inference_time_ms": round(mean_ms, 2),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    cfg = CountingConfig.from_yaml(args.config)
    if args.weights is not None:
        cfg.weights = args.weights
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz

    if not cfg.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {cfg.weights}. Train on Colab and copy best.pt to outputs/weights/best.pt first."
        )

    device = select_device(args.device)
    print(f"Loading {cfg.weights} on {device}")
    model = YOLO(str(cfg.weights))
    print(
        f"Inference imgsz={cfg.imgsz} iou={cfg.iou} thresholds={cfg.class_thresholds} "
        f"min_area={cfg.min_bbox_area_px}px^2"
    )

    if args.image is not None:
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        image_paths = [args.image]
        mode = "single_image"
    else:
        source_dir = args.image_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {source_dir}")
        image_paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.num_samples and args.num_samples < len(image_paths):
            random.seed(args.seed)
            image_paths = random.sample(image_paths, k=args.num_samples)
        mode = "image_dir"
    records = run_image_inputs(image_paths, model, cfg, device, args.output_dir)

    write_csv(records, args.counts_csv)
    write_summary(records, cfg, args.summary_json, mode)

    if records:
        total_h = sum(r["person_count"] for r in records)
        total_c = sum(r["car_count"] for r in records)
        mean_ms = sum(r["inference_time_ms"] for r in records) / len(records)
        print(
            f"Processed {len(records)} inputs | total humans={total_h} | total cars={total_c} | "
            f"mean inference={mean_ms:.1f} ms"
        )
    print(f"Counts CSV : {args.counts_csv}")
    print(f"Summary    : {args.summary_json}")


if __name__ == "__main__":
    main()
