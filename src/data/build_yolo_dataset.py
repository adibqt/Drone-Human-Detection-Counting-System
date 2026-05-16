

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

import yaml


SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO-ready dataset layout for Task-02.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path returned by kagglehub.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/visdrone"),
        help="Where Task-01 wrote labels_yolo/ for each split.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/task02_data.yaml"),
        help="Output Ultralytics data YAML path.",
    )
    return parser.parse_args()


def normalize_dataset_root(dataset_root: Path) -> Path:
    nested = dataset_root / "VisDrone_Dataset"
    return nested if nested.exists() else dataset_root


def discover_image_dirs(dataset_root: Path) -> dict[str, Path]:
    """Return {split: image_dir} for splits found under the kagglehub cache.

    For the test split, VisDrone ships *test-dev* (with annotations) and
    *test-challenge* (no annotations). We must select test-dev because we
    only have labels for that one. We also require a sibling
    `annotations/` or `labels/` directory so we never pick up an
    annotation-less split.
    """
    found: dict[str, Path] = {}
    for image_dir in sorted(dataset_root.rglob("images")):
        parent = image_dir.parent
        parent_name = parent.name.lower()
        has_annotations = (parent / "annotations").exists() or (parent / "labels").exists()
        if not has_annotations:
            continue
        for split in SPLIT_NAMES:
            if split in parent_name:
                if split == "test" and "challenge" in parent_name:
                    continue
                if split not in found:
                    found[split] = image_dir.resolve()
                break
    if not found:
        raise RuntimeError(f"No '*/<split>/images' directories with annotations found under {dataset_root}")
    return found


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _same_volume(p1: Path, p2: Path) -> bool:
    try:
        return Path(p1).resolve().drive.lower() == Path(p2).resolve().drive.lower()
    except Exception:
        return False


def materialize_images(source_dir: Path, dest_dir: Path) -> tuple[str, int, int]:
    """Copy/hardlink images from source_dir -> dest_dir. Returns (mode, copied, skipped)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    use_hardlink = _same_volume(source_dir, dest_dir)
    copied = 0
    skipped = 0
    for src in source_dir.iterdir():
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        dst = dest_dir / src.name
        if dst.exists():
            skipped += 1
            continue
        if use_hardlink:
            try:
                os.link(src, dst)
                copied += 1
                continue
            except OSError:
                use_hardlink = False
        shutil.copy2(src, dst)
        copied += 1
    return ("hardlink" if use_hardlink else "copy", copied, skipped)


def materialize_labels(labels_yolo: Path, labels_dir: Path) -> tuple[str, int, int]:
    """Mirror labels_yolo/*.txt -> labels/*.txt using hardlinks when possible."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    use_hardlink = _same_volume(labels_yolo, labels_dir)
    copied = 0
    skipped = 0
    for src in labels_yolo.iterdir():
        if src.suffix.lower() != ".txt":
            continue
        dst = labels_dir / src.name
        if dst.exists():
            skipped += 1
            continue
        if use_hardlink:
            try:
                os.link(src, dst)
                copied += 1
                continue
            except OSError:
                use_hardlink = False
        shutil.copy2(src, dst)
        copied += 1
    return ("hardlink" if use_hardlink else "copy", copied, skipped)


def materialize_split(split: str, source_image_dir: Path, processed_root: Path) -> dict[str, object]:
    split_dir = processed_root / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    labels_yolo = split_dir / "labels_yolo"

    if not labels_yolo.exists():
        raise RuntimeError(
            f"Missing cleaned labels for split '{split}'. Run Task-01 prepare_dataset.py first."
        )

    img_mode, img_copied, img_skipped = materialize_images(source_image_dir, images_dir)
    lab_mode, lab_copied, lab_skipped = materialize_labels(labels_yolo, labels_dir)
    return {
        "images": {"mode": img_mode, "copied": img_copied, "already_present": img_skipped},
        "labels": {"mode": lab_mode, "copied": lab_copied, "already_present": lab_skipped},
    }


def write_data_yaml(config_path: Path, processed_root: Path, splits: dict[str, Path]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(processed_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person", 1: "car"},
        "nc": 2,
    }
    if "test" in splits:
        payload["test"] = "test/images"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = normalize_dataset_root(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    splits = discover_image_dirs(dataset_root)
    actions: dict[str, dict[str, object]] = {}
    for split, image_dir in splits.items():
        actions[split] = materialize_split(split, image_dir, args.processed_root)
        img = actions[split]["images"]
        lab = actions[split]["labels"]
        print(
            f"[{split}] images: {img['mode']} (+{img['copied']} new, "
            f"{img['already_present']} present); labels: {lab['mode']} "
            f"(+{lab['copied']} new, {lab['already_present']} present)"
        )

    write_data_yaml(args.config_path, args.processed_root, splits)
    print(f"Wrote Ultralytics data YAML: {args.config_path}")


if __name__ == "__main__":
    main()
