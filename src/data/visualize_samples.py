from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


VISDRONE_TO_TARGET = {
    1: ("person", (0, 255, 0)),
    2: ("person", (0, 180, 0)),
    4: ("car", (0, 165, 255)),
}
YOLO_VISDRONE_TO_TARGET = {
    0: ("person", (0, 255, 0)),
    1: ("person", (0, 180, 0)),
    3: ("car", (0, 165, 255)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Task-01 before/after augmentation visualizations.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path returned by kagglehub.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to sample from.")
    parser.add_argument("--num-samples", type=int, default=10, help="How many sample images to save for each set.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/task01/samples"),
        help="Output directory for visualizations.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def discover_split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path, str]:
    for image_dir in dataset_root.rglob("images"):
        parent_name = image_dir.parent.name.lower()
        if split not in parent_name:
            continue
        labels_dir = image_dir.parent / "labels"
        if labels_dir.exists():
            return image_dir, labels_dir, "yolo"
        ann_dir = image_dir.parent / "annotations"
        if ann_dir.exists():
            return image_dir, ann_dir, "visdrone"
    raise RuntimeError(f"Could not find '{split}' images/annotations under {dataset_root}")


def normalize_dataset_root(dataset_root: Path) -> Path:
    nested = dataset_root / "VisDrone_Dataset"
    if nested.exists():
        return nested
    return dataset_root


def read_annotations(path: Path, label_format: str, image_shape: tuple[int, int, int]) -> list[tuple[float, float, float, float, int]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    out = []
    img_h, img_w = image_shape[:2]
    for line in raw.splitlines():
        if label_format == "visdrone":
            parts = line.split(",")
            if len(parts) < 8:
                continue
            x, y, w, h = map(float, parts[:4])
            cls = int(float(parts[5]))
        else:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, nw, nh = map(float, parts[:5])
            cls = int(cls)
            w = nw * img_w
            h = nh * img_h
            x = (cx * img_w) - (w / 2.0)
            y = (cy * img_h) - (h / 2.0)
        out.append((x, y, w, h, cls))
    return out


def draw_boxes(img, boxes, label_format: str):
    canvas = img.copy()
    h, w = canvas.shape[:2]
    for x, y, bw, bh, cls in boxes:
        class_map = VISDRONE_TO_TARGET if label_format == "visdrone" else YOLO_VISDRONE_TO_TARGET
        if cls not in class_map or bw <= 0 or bh <= 0:
            continue
        label, color = class_map[cls]
        x1 = int(max(0, min(w - 1, x)))
        y1 = int(max(0, min(h - 1, y)))
        x2 = int(max(0, min(w - 1, x + bw)))
        y2 = int(max(0, min(h - 1, y + bh)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def augment_image(img):
    out = img.copy()
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    alpha = random.uniform(0.9, 1.1)  # contrast
    beta = random.uniform(-12, 12)  # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if random.random() < 0.4:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    noise_std = random.uniform(0.0, 8.0)
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, out.shape).astype("float32")
        out = np.clip(out.astype("float32") + noise, 0, 255).astype("uint8")
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    dataset_root = normalize_dataset_root(args.dataset_root)
    image_dir, ann_dir, label_format = discover_split_dirs(dataset_root, args.split)

    image_paths = [p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    sample_paths = random.sample(image_paths, k=min(args.num_samples, len(image_paths)))
    before_dir = args.output_dir / "before_preprocessing"
    after_dir = args.output_dir / "after_augmentation"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sample_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        ann_path = ann_dir / f"{image_path.stem}.txt"
        boxes = read_annotations(ann_path, label_format, img.shape)
        drawn_before = draw_boxes(img, boxes, label_format)
        drawn_after = draw_boxes(augment_image(img), boxes, label_format)

        cv2.imwrite(str(before_dir / image_path.name), drawn_before)
        cv2.imwrite(str(after_dir / image_path.name), drawn_after)

    print(f"Saved sample visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
