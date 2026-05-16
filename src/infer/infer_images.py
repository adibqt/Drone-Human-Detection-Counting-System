

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


CLASS_COLORS = {
    "person": (0, 200, 0),
    "car": (0, 165, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on sample VisDrone images.")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained .pt weights (e.g. outputs/runs/task02/yolov8s_visdrone/weights/best.pt).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/visdrone/val/images"),
        help="Directory of images to sample from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/task02/predictions"),
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/metrics/task02_sample_predictions.json"),
        help="Path for the per-image summary JSON.",
    )
    parser.add_argument("--num-samples", type=int, default=12, help="Number of images to sample.")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image size (match Colab training).")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
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


def sample_images(source_dir: Path, num_samples: int, seed: int) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in source_dir.glob("*") if p.suffix.lower() in extensions]
    if not image_paths:
        raise RuntimeError(f"No images found in {source_dir}")
    random.seed(seed)
    return random.sample(image_paths, k=min(num_samples, len(image_paths)))


def draw_predictions(image: np.ndarray, result, names: dict[int, str]) -> tuple[np.ndarray, dict[str, int]]:
    canvas = image.copy()
    counts = {name: 0 for name in names.values()}
    if result.boxes is None or result.boxes.xyxy is None:
        return canvas, counts

    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
        label = names.get(int(cls_id), str(cls_id))
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty1 = max(0, y1 - th - baseline - 2)
        cv2.rectangle(canvas, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 2), color, -1)
        cv2.putText(
            canvas,
            text,
            (x1 + 2, ty1 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        counts[label] = counts.get(label, 0) + 1

    person_count = counts.get("person", 0)
    car_count = counts.get("car", 0)
    banner = f"persons: {person_count}   cars: {car_count}"
    (bw, bh), bl = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(canvas, (8, 8), (8 + bw + 12, 8 + bh + bl + 10), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (14, 8 + bh + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas, counts


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights {args.weights} on device {device}")
    model = YOLO(str(args.weights))
    names = {int(k): v for k, v in model.names.items()}

    sample_paths = sample_images(args.source_dir, args.num_samples, args.seed)
    print(f"Running inference on {len(sample_paths)} images ...")

    summary: list[dict] = []
    for image_path in sample_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        t0 = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result = results[0]
        annotated, counts = draw_predictions(image, result, names)
        out_path = args.output_dir / image_path.name
        cv2.imwrite(str(out_path), annotated)
        summary.append(
            {
                "image": image_path.name,
                "person_count": int(counts.get("person", 0)),
                "car_count": int(counts.get("car", 0)),
                "inference_time_ms": round(elapsed_ms, 2),
                "annotated_path": str(out_path).replace("\\", "/"),
            }
        )

    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} annotated images to: {args.output_dir}")
    print(f"Saved per-image summary to: {args.summary_path}")


if __name__ == "__main__":
    main()
