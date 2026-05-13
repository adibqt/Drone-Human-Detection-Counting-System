from __future__ import annotations

import argparse
import json
from pathlib import Path

import kagglehub


def download_dataset(dataset_ref: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(kagglehub.dataset_download(dataset_ref))

    metadata = {
        "dataset_ref": dataset_ref,
        "downloaded_path": str(downloaded_path),
    }
    metadata_path = output_dir / "dataset_download_info.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return downloaded_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VisDrone dataset from Kaggle.")
    parser.add_argument(
        "--dataset-ref",
        default="banuprasadb/visdrone-dataset",
        help="Kaggle dataset reference in the format owner/dataset-name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory where download metadata will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = download_dataset(args.dataset_ref, args.output_dir)
    print(f"Path to dataset files: {path}")


if __name__ == "__main__":
    main()
