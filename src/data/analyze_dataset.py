from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


VISDRONE_TO_NAME = {
    1: "person",
    2: "person",
    4: "car",
}
YOLO_VISDRONE_TO_NAME = {
    0: "person",
    1: "person",
    3: "car",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Task-01 VisDrone analysis figures and tables.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path returned by kagglehub.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/task01"),
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory to save tabular outputs.",
    )
    return parser.parse_args()


def discover_splits(dataset_root: Path) -> dict[str, tuple[Path, Path, str]]:
    candidates: dict[str, tuple[Path, Path, str]] = {}
    for image_dir in dataset_root.rglob("images"):
        parent_name = image_dir.parent.name.lower()
        split = next((k for k in ("train", "val", "test") if k in parent_name), None)
        if not split:
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


def read_rows(annotation_path: Path, label_format: str) -> list[list[float]]:
    raw = annotation_path.read_text(encoding="utf-8").strip() if annotation_path.exists() else ""
    if not raw:
        return []
    rows = []
    for line in raw.splitlines():
        if label_format == "visdrone":
            parts = line.split(",")
            if len(parts) < 8:
                continue
            rows.append([float(v) for v in parts[:8]])
        else:
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append([float(v) for v in parts[:5]])
    return rows


def area_bucket(area_px: float) -> str:
    if area_px < 32**2:
        return "tiny"
    if area_px < 96**2:
        return "small"
    if area_px < 224**2:
        return "medium"
    return "large"


def analyze(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, int], Counter, list[dict[str, object]]]:
    split_dirs = discover_splits(dataset_root)
    if not split_dirs:
        raise RuntimeError("Could not discover VisDrone split folders.")

    rows = []
    images_per_split: dict[str, int] = {}
    class_counter: Counter = Counter()
    hard_case_scores: list[dict[str, object]] = []

    for split, (image_dir, ann_dir, label_format) in split_dirs.items():
        image_paths = sorted(
            [p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        images_per_split[split] = len(image_paths)

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_h, image_w = image.shape[:2]
            ann_path = ann_dir / f"{image_path.stem}.txt"
            ann_rows = read_rows(ann_path, label_format)
            target_count = 0
            hard_score = 0

            for ann in ann_rows:
                truncation = 0
                occlusion = 0
                if label_format == "visdrone":
                    x, y, w, h, _score, category, truncation, occlusion = ann
                    category = int(category)
                    mapping = VISDRONE_TO_NAME
                else:
                    category, cx, cy, nw, nh = ann
                    category = int(category)
                    mapping = YOLO_VISDRONE_TO_NAME
                    w = nw * image_w
                    h = nh * image_h
                    x = (cx * image_w) - (w / 2.0)
                    y = (cy * image_h) - (h / 2.0)

                if category not in mapping:
                    continue
                if w <= 0 or h <= 0:
                    continue

                w = min(w, image_w - max(0.0, x))
                h = min(h, image_h - max(0.0, y))
                if w <= 0 or h <= 0:
                    continue

                cls_name = mapping[category]
                class_counter[cls_name] += 1
                target_count += 1

                area_px = w * h
                rows.append(
                    {
                        "split": split,
                        "image_name": image_path.name,
                        "class_name": cls_name,
                        "bbox_area_px": area_px,
                        "area_bucket": area_bucket(area_px),
                    }
                )

                if int(occlusion) >= 2:
                    hard_score += 2
                if int(truncation) >= 1:
                    hard_score += 1
                if area_px < 32**2:
                    hard_score += 1

            if hard_score > 0:
                hard_case_scores.append(
                    {
                        "split": split,
                        "image_name": image_path.name,
                        "hard_score": hard_score,
                        "target_count": target_count,
                    }
                )

    return pd.DataFrame(rows), images_per_split, class_counter, hard_case_scores


def save_outputs(
    df: pd.DataFrame,
    images_per_split: dict[str, int],
    class_counter: Counter,
    hard_case_scores: list[dict[str, object]],
    output_dir: Path,
    metrics_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    split_df = pd.DataFrame({"split": list(images_per_split.keys()), "images": list(images_per_split.values())})
    split_df.to_csv(metrics_dir / "task01_images_per_split.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=split_df, x="split", y="images", hue="split", palette="viridis", legend=False)
    plt.title("Images per Split")
    plt.tight_layout()
    plt.savefig(output_dir / "images_per_split.png", dpi=150)
    plt.close()

    cls_df = pd.DataFrame({"class_name": list(class_counter.keys()), "instances": list(class_counter.values())})
    cls_df.to_csv(metrics_dir / "task01_instances_per_class.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=cls_df, x="class_name", y="instances", hue="class_name", palette="magma", legend=False)
    plt.title("Instances per Class")
    plt.tight_layout()
    plt.savefig(output_dir / "instances_per_class.png", dpi=150)
    plt.close()

    area_df = df["area_bucket"].value_counts().reset_index()
    area_df.columns = ["area_bucket", "count"]
    area_df.to_csv(metrics_dir / "task01_bbox_area_distribution.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=area_df, x="area_bucket", y="count", order=["tiny", "small", "medium", "large"])
    plt.title("Bounding Box Area Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "bbox_area_distribution.png", dpi=150)
    plt.close()

    object_count_df = df.groupby(["split", "image_name"]).size().reset_index(name="objects")
    object_count_df.to_csv(metrics_dir / "task01_objects_per_image.csv", index=False)
    plt.figure(figsize=(8, 4))
    sns.histplot(data=object_count_df, x="objects", bins=30, kde=False)
    plt.title("Objects per Image Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "objects_per_image_distribution.png", dpi=150)
    plt.close()

    hard_cases_df = (
        pd.DataFrame(hard_case_scores).sort_values(["hard_score", "target_count"], ascending=False).head(20)
    )
    hard_cases_df.to_csv(metrics_dir / "task01_hard_cases.csv", index=False)
    summary_path = metrics_dir / "task01_hard_cases_summary.json"
    summary_path.write_text(json.dumps(hard_cases_df.to_dict(orient="records"), indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = normalize_dataset_root(args.dataset_root)
    df, images_per_split, class_counter, hard_cases = analyze(dataset_root)
    if df.empty:
        raise RuntimeError("No person/car annotations found. Verify dataset-root points to VisDrone data.")
    save_outputs(df, images_per_split, class_counter, hard_cases, args.output_dir, args.metrics_dir)
    print(f"Saved Task-01 analysis artifacts to: {args.output_dir} and {args.metrics_dir}")


if __name__ == "__main__":
    main()
