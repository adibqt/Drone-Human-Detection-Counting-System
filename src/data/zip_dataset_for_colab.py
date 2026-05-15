"""Package the prepared YOLO dataset into a single zip for Colab upload.

Produces `visdrone_yolo.zip` at the repo root by default with this
layout:

```text
train/images/*.jpg
train/labels/*.txt
val/images/*.jpg
val/labels/*.txt
test/images/*.jpg
test/labels/*.txt
```

Upload the resulting zip to your Google Drive at
`MyDrive/drone-detection/visdrone_yolo.zip` and run
`notebooks/task02_train_colab.ipynb`.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SPLITS = ("train", "val", "test")
SUBDIRS = ("images", "labels")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zip the YOLO-ready dataset for Colab upload.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/visdrone"),
        help="Directory containing train/val/test/{images,labels}.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("visdrone_yolo.zip"),
        help="Output zip path.",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=zipfile.ZIP_STORED,
        choices=[zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED],
        help="0=STORED (fast, large), 8=DEFLATED (slow, smaller).",
    )
    return parser.parse_args()


def gather_files(processed_root: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for split in SPLITS:
        for sub in SUBDIRS:
            src_dir = processed_root / split / sub
            if not src_dir.exists():
                continue
            for fp in src_dir.iterdir():
                if not fp.is_file():
                    continue
                if sub == "images" and fp.suffix.lower() not in IMAGE_EXTS:
                    continue
                if sub == "labels" and fp.suffix.lower() != ".txt":
                    continue
                entries.append((fp, f"{split}/{sub}/{fp.name}"))
    return entries


def main() -> None:
    args = parse_args()
    if not args.processed_root.exists():
        raise FileNotFoundError(f"Processed root not found: {args.processed_root}")

    entries = gather_files(args.processed_root)
    if not entries:
        raise RuntimeError("No files to zip. Run build_yolo_dataset.py first.")

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(entries)} files to {args.output_zip} ...")
    with zipfile.ZipFile(args.output_zip, "w", compression=args.compression) as zf:
        for src, arcname in entries:
            zf.write(src, arcname)

    size_mb = args.output_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output_zip} ({size_mb:.1f} MB)")
    print("Upload it to MyDrive/drone-detection/visdrone_yolo.zip and run notebooks/task02_train_colab.ipynb")


if __name__ == "__main__":
    main()
