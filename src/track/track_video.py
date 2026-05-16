

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
BANNER_BG = (0, 0, 0)
BANNER_FG_PRIMARY = (255, 255, 0)     # "Unique Humans" — primary deliverable
BANNER_FG_SECONDARY = (255, 255, 255)


@dataclass
class TrackConfig:
    weights: Path
    imgsz: int
    iou: float
    class_thresholds: dict[str, float]
    min_bbox_area_px: float
    tracker: str
    draw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrackConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            weights=Path(raw.get("weights", "outputs/weights/best.pt")),
            imgsz=int(raw.get("imgsz", 800)),
            iou=float(raw.get("iou", 0.5)),
            class_thresholds={k: float(v) for k, v in (raw.get("class_thresholds") or {}).items()},
            min_bbox_area_px=float(raw.get("min_bbox_area_px", 0)),
            tracker=str(raw.get("tracker", "bytetrack.yaml")),
            draw=raw.get("draw") or {},
        )


@dataclass
class TrackedDetection:
    cls_name: str
    track_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-04 detection + tracking + unique-ID counting.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/task04_track.yaml"),
        help="Task-04 config YAML.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Either a video file (.mp4 / .mov / ...) or a folder of frame images.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("outputs/videos/task04_demo.mp4"),
        help="Where to write the annotated output video.",
    )
    parser.add_argument(
        "--counts-csv",
        type=Path,
        default=Path("outputs/metrics/task04_track_counts.csv"),
        help="CSV path for per-frame counts.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/metrics/task04_track_summary.json"),
        help="JSON path for the aggregate summary.",
    )
    parser.add_argument(
        "--out-fps",
        type=int,
        default=15,
        help="FPS for the output video. If --source is a video file, the input FPS is used instead.",
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
    return parser.parse_args()


def select_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def lowest_threshold(cfg: TrackConfig) -> float:
    if not cfg.class_thresholds:
        return 0.25
    return float(min(cfg.class_thresholds.values()))


def color_for_track(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread color per track ID (BGR)."""
    rng = np.random.default_rng(seed=int(track_id) * 9301 + 49297)
    color = rng.integers(64, 256, size=3, dtype=np.int32)
    return (int(color[0]), int(color[1]), int(color[2]))


def class_color(cls_name: str) -> tuple[int, int, int]:
    if cls_name == "person":
        return (0, 200, 0)
    if cls_name == "car":
        return (0, 165, 255)
    return (200, 200, 200)


def filter_tracked(result, cfg: TrackConfig, names: dict[int, str]) -> list[TrackedDetection]:
    """Apply class thresholds + min area to a YOLO tracking result and
    drop any detection whose tracker did not assign an ID yet."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.boxes.id is None:  # tracker still warming up on this frame
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = result.boxes.id.cpu().numpy().astype(int)

    out: list[TrackedDetection] = []
    for (x1, y1, x2, y2), conf, cls_id, tid in zip(xyxy, confs, cls_ids, track_ids):
        cls_name = names.get(int(cls_id), str(cls_id))
        threshold = cfg.class_thresholds.get(cls_name, lowest_threshold(cfg))
        if float(conf) < threshold:
            continue
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        if area < cfg.min_bbox_area_px:
            continue
        out.append(
            TrackedDetection(
                cls_name=cls_name,
                track_id=int(tid),
                confidence=float(conf),
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
        )
    return out


def draw_frame(
    frame: np.ndarray,
    detections: list[TrackedDetection],
    visible_counts: dict[str, int],
    unique_counts: dict[str, int],
    cfg: TrackConfig,
) -> np.ndarray:
    canvas = frame.copy()
    color_by_track = bool(cfg.draw.get("color_by_track_id", True))
    box_thickness = int(cfg.draw.get("box_thickness", 2))
    label_scale = float(cfg.draw.get("label_font_scale", 0.45))
    label_thickness = int(cfg.draw.get("label_font_thickness", 1))
    banner_scale = float(cfg.draw.get("banner_font_scale", 0.9))
    banner_thickness = int(cfg.draw.get("banner_font_thickness", 2))

    for det in detections:
        color = color_for_track(det.track_id) if color_by_track else class_color(det.cls_name)
        cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2), color, box_thickness)
        text = f"{det.cls_name} ID:{det.track_id}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
        ty1 = max(0, det.y1 - th - baseline - 2)
        cv2.rectangle(canvas, (det.x1, ty1), (det.x1 + tw + 4, ty1 + th + baseline + 2), color, -1)
        cv2.putText(
            canvas, text, (det.x1 + 2, ty1 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 0), label_thickness, cv2.LINE_AA,
        )

    primary = f"Unique Humans: {unique_counts.get('person', 0)}"
    secondary_1 = f"Unique Cars : {unique_counts.get('car', 0)}"
    secondary_2 = (
        f"Visible Humans: {visible_counts.get('person', 0)}   "
        f"Visible Cars: {visible_counts.get('car', 0)}"
    )

    (pw, ph), pbl = cv2.getTextSize(primary, cv2.FONT_HERSHEY_SIMPLEX, banner_scale, banner_thickness)
    (s1w, s1h), s1bl = cv2.getTextSize(secondary_1, cv2.FONT_HERSHEY_SIMPLEX, banner_scale * 0.7, banner_thickness)
    (s2w, s2h), s2bl = cv2.getTextSize(secondary_2, cv2.FONT_HERSHEY_SIMPLEX, banner_scale * 0.55, banner_thickness)

    banner_w = max(pw, s1w, s2w) + 24
    banner_h = ph + s1h + s2h + pbl + s1bl + s2bl + 30
    cv2.rectangle(canvas, (10, 10), (10 + banner_w, 10 + banner_h), BANNER_BG, -1)

    y = 10 + ph + 12
    cv2.putText(canvas, primary, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                banner_scale, BANNER_FG_PRIMARY, banner_thickness, cv2.LINE_AA)
    y += s1h + 8
    cv2.putText(canvas, secondary_1, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                banner_scale * 0.7, BANNER_FG_SECONDARY, banner_thickness, cv2.LINE_AA)
    y += s2h + 6
    cv2.putText(canvas, secondary_2, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                banner_scale * 0.55, BANNER_FG_SECONDARY, banner_thickness, cv2.LINE_AA)
    return canvas


def iter_frames(source: Path) -> "tuple[float, int, int, iter]":
    """Yield (fps, width, height, frame_iterator) for either a video file
    or a folder of images.

    The frame_iterator yields (frame_index, BGR ndarray) tuples.
    """
    if source.is_file():
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        def gen():
            idx = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield idx, frame
                    idx += 1
            finally:
                cap.release()

        return fps, width, height, gen()

    if source.is_dir():
        frame_paths = sorted(
            p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not frame_paths:
            raise RuntimeError(f"No image frames in {source}")
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
        height, width = first.shape[:2]

        def gen():
            for idx, fp in enumerate(frame_paths):
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                yield idx, frame

        return 0.0, width, height, gen()

    raise FileNotFoundError(f"Source not found: {source}")


def main() -> None:
    args = parse_args()

    cfg = TrackConfig.from_yaml(args.config)
    if args.weights is not None:
        cfg.weights = args.weights
    if not cfg.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {cfg.weights}. Copy Colab-trained best.pt to outputs/weights/."
        )

    device = select_device(args.device)
    print(f"Loading {cfg.weights} on {device}")
    model = YOLO(str(cfg.weights))
    names = {int(k): v for k, v in model.names.items()}
    print(
        f"Tracker={cfg.tracker} imgsz={cfg.imgsz} iou={cfg.iou} thresholds={cfg.class_thresholds} "
        f"min_area={cfg.min_bbox_area_px}px^2"
    )

    src_fps, width, height, frame_iter = iter_frames(args.source)
    out_fps = float(src_fps) if src_fps > 0 else float(args.out_fps)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output_video), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {args.output_video}")

    base_conf = lowest_threshold(cfg)
    unique_ids: dict[str, set[int]] = {}
    records: list[dict] = []
    total_inference_ms = 0.0
    max_visible: dict[str, int] = {}

    try:
        for frame_idx, frame in frame_iter:
            t0 = time.perf_counter()
            results = model.track(
                source=frame,
                imgsz=cfg.imgsz,
                conf=base_conf,
                iou=cfg.iou,
                tracker=cfg.tracker,
                persist=True,
                device=device,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_inference_ms += elapsed_ms

            detections = filter_tracked(results[0], cfg, names)

            visible: dict[str, int] = {}
            for det in detections:
                visible[det.cls_name] = visible.get(det.cls_name, 0) + 1
                unique_ids.setdefault(det.cls_name, set()).add(det.track_id)
            for cls_name, count in visible.items():
                max_visible[cls_name] = max(max_visible.get(cls_name, 0), count)

            unique_counts = {k: len(v) for k, v in unique_ids.items()}
            annotated = draw_frame(frame, detections, visible, unique_counts, cfg)
            writer.write(annotated)

            records.append(
                {
                    "frame": frame_idx,
                    "visible_persons": int(visible.get("person", 0)),
                    "visible_cars": int(visible.get("car", 0)),
                    "unique_persons_so_far": int(unique_counts.get("person", 0)),
                    "unique_cars_so_far": int(unique_counts.get("car", 0)),
                    "inference_time_ms": round(elapsed_ms, 2),
                }
            )
    finally:
        writer.release()

    args.counts_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.counts_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "visible_persons", "visible_cars",
                "unique_persons_so_far", "unique_cars_so_far", "inference_time_ms",
            ],
        )
        w.writeheader()
        w.writerows(records)

    summary = {
        "source": str(args.source),
        "output_video": str(args.output_video),
        "weights": str(cfg.weights),
        "tracker": cfg.tracker,
        "imgsz": cfg.imgsz,
        "iou": cfg.iou,
        "class_thresholds": cfg.class_thresholds,
        "min_bbox_area_px": cfg.min_bbox_area_px,
        "frames_processed": len(records),
        "output_fps": out_fps,
        "unique_persons": len(unique_ids.get("person", set())),
        "unique_cars": len(unique_ids.get("car", set())),
        "max_simultaneous_persons": max_visible.get("person", 0),
        "max_simultaneous_cars": max_visible.get("car", 0),
        "mean_inference_time_ms": round(total_inference_ms / max(1, len(records)), 2),
        "wall_clock_seconds": round(total_inference_ms / 1000.0, 2),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Processed {len(records)} frames | unique humans={summary['unique_persons']} "
        f"unique cars={summary['unique_cars']} | mean {summary['mean_inference_time_ms']} ms/frame"
    )
    print(f"Annotated video: {args.output_video}")
    print(f"Counts CSV     : {args.counts_csv}")
    print(f"Summary        : {args.summary_json}")


if __name__ == "__main__":
    main()
