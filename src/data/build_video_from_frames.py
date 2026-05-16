

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch a folder of frames into an .mp4 video.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the frames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output .mp4 file.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix filter (e.g. '0000026' to pick only that VisDrone sequence).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap total frames written. 0 means use all matching frames.",
    )
    parser.add_argument(
        "--natural-sort",
        action="store_true",
        help="Sort filenames numerically rather than lexicographically.",
    )
    return parser.parse_args()


def _natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def gather_frames(input_dir: Path, prefix: str | None, natural_sort: bool) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    frames = [
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and (prefix is None or p.name.startswith(prefix))
    ]
    if not frames:
        raise RuntimeError(f"No matching frames in {input_dir} (prefix={prefix})")

    if natural_sort:
        frames.sort(key=lambda p: _natural_key(p.name))
    else:
        frames.sort()
    return frames


def encode_video(frames: list[Path], output: Path, fps: int, max_frames: int) -> tuple[int, int, int]:
    if max_frames and max_frames < len(frames):
        frames = frames[:max_frames]

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")
    height, width = first.shape[:2]

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {output}")

    written = 0
    skipped = 0
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                skipped += 1
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    return written, skipped, height * width


def main() -> None:
    args = parse_args()
    frames = gather_frames(args.input_dir, args.prefix, args.natural_sort)
    print(f"Found {len(frames)} matching frames in {args.input_dir}")
    written, skipped, _ = encode_video(frames, args.output, args.fps, args.max_frames)
    duration_s = written / max(1, args.fps)
    print(
        f"Wrote {written} frames ({duration_s:.1f}s @ {args.fps} FPS) to {args.output}"
        + (f"; skipped {skipped} unreadable frames" if skipped else "")
    )


if __name__ == "__main__":
    main()
