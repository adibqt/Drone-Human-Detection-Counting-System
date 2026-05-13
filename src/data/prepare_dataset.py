from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2


VISDRONE_TO_TARGET = {
    1: 0,  # pedestrian -> person
    2: 0,  # people -> person
    4: 1,  # car -> car
}
YOLO_VISDRONE_TO_TARGET = {
    0: 0,  # pedestrian -> person
    1: 0,  # people -> person
    3: 1,  # car -> car
}
TARGET_NAMES = {0: "person", 1: "car"}


@dataclass
class CleaningStats:
    total_annotations: int = 0
    kept_annotations: int = 0
    dropped_invalid_size: int = 0
    dropped_non_target_class: int = 0
    fixed_out_of_bound_boxes: int = 0
    corrupted_images: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and convert VisDrone labels for Task-01.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path returned by kagglehub.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/visdrone"),
        help="Directory where cleaned labels and reports are written.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/metrics/task01_cleaning_report.json"),
        help="Path for cleaning summary JSON report.",
    )
    return parser.parse_args()


def discover_splits(dataset_root: Path) -> dict[str, tuple[Path, Path, str]]:
    split_keywords = {"train", "val", "test"}
    candidates: dict[str, tuple[Path, Path, str]] = {}

    for image_dir in dataset_root.rglob("images"):
        parent_name = image_dir.parent.name.lower()
        split = next((k for k in split_keywords if k in parent_name), None)
        if split is None:
            continue
        labels_dir = image_dir.parent / "labels"
        ann_dir = image_dir.parent / "annotations"
        if labels_dir.exists():
            candidates[split] = (image_dir, labels_dir, "yolo")
        elif ann_dir.exists():
            candidates[split] = (image_dir, ann_dir, "visdrone")
    return candidates


def normalize_dataset_root(dataset_root: Path) -> Path:
    nested = dataset_root / "VisDrone_Dataset"
    if nested.exists():
        return nested
    return dataset_root


def read_annotation_rows(annotation_file: Path, label_format: str) -> list[list[float]]:
    if not annotation_file.exists():
        return []
    raw = annotation_file.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    rows: list[list[float]] = []
    for line in raw.splitlines():
        if label_format == "visdrone":
            parts = line.split(",")
            if len(parts) < 8:
                continue
            rows.append([float(value) for value in parts[:8]])
        else:
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append([float(value) for value in parts[:5]])
    return rows


def to_yolo_row(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def process_split(
    split: str,
    image_dir: Path,
    ann_dir: Path,
    label_format: str,
    output_root: Path,
    stats: CleaningStats,
) -> dict[str, int]:
    split_counts = defaultdict(int)
    out_label_dir = output_root / split / "labels_yolo"
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            stats.corrupted_images += 1
            continue

        img_h, img_w = img.shape[:2]
        rows = read_annotation_rows(ann_dir / f"{image_path.stem}.txt", label_format)
        yolo_lines: list[str] = []

        for row in rows:
            stats.total_annotations += 1
            if label_format == "visdrone":
                x, y, w, h, _score, category, _trunc, _occ = row
                category = int(category)
                class_mapping = VISDRONE_TO_TARGET
            else:
                category, cx, cy, nw, nh = row
                category = int(category)
                class_mapping = YOLO_VISDRONE_TO_TARGET
                w = nw * img_w
                h = nh * img_h
                x = (cx * img_w) - (w / 2.0)
                y = (cy * img_h) - (h / 2.0)

            if category not in class_mapping:
                stats.dropped_non_target_class += 1
                continue

            if w <= 0 or h <= 0:
                stats.dropped_invalid_size += 1
                continue

            original = (x, y, w, h)
            x = max(0.0, min(x, img_w - 1.0))
            y = max(0.0, min(y, img_h - 1.0))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if (x, y, w, h) != original:
                stats.fixed_out_of_bound_boxes += 1

            if w <= 0 or h <= 0:
                stats.dropped_invalid_size += 1
                continue

            target_cls = class_mapping[category]
            cx, cy, nw, nh = to_yolo_row(x, y, w, h, img_w, img_h)
            yolo_lines.append(f"{target_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats.kept_annotations += 1
            split_counts[TARGET_NAMES[target_cls]] += 1

        label_file = out_label_dir / f"{image_path.stem}.txt"
        label_file.write_text("\n".join(yolo_lines), encoding="utf-8")
        split_counts["images"] += 1
        split_counts["images_with_targets"] += int(len(yolo_lines) > 0)

    return dict(split_counts)


def save_report(report_path: Path, stats: CleaningStats, split_stats: dict[str, dict[str, int]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cleaning_stats": asdict(stats),
        "split_stats": split_stats,
        "class_mapping": {"person": 0, "car": 1},
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required path(s): {missing}")


def main() -> None:
    args = parse_args()
    dataset_root = normalize_dataset_root(args.dataset_root)
    ensure_paths_exist([dataset_root])

    split_dirs = discover_splits(dataset_root)
    if not split_dirs:
        raise RuntimeError(
            "Could not discover split directories. Expected folders like '*train*/images' and '*train*/annotations'."
        )

    stats = CleaningStats()
    split_stats: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        if split not in split_dirs:
            continue
        image_dir, ann_dir, label_format = split_dirs[split]
        split_stats[split] = process_split(split, image_dir, ann_dir, label_format, args.output_root, stats)

    save_report(args.report_path, stats, split_stats)
    print(f"Saved cleaning report to: {args.report_path}")


if __name__ == "__main__":
    main()
