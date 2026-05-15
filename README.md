# Drone Human Detection and Counting System

A drone/aerial computer-vision pipeline for detecting **persons** and **cars**
in VisDrone imagery, counting humans, and visualizing the results.

## Repository layout

```text
configs/                YAMLs for dataset cleaning + model training
data/processed/         Cleaned VisDrone splits in YOLO format
outputs/                Figures, metrics, and trained run artifacts
scripts/                Convenience .bat entry points (Windows)
src/data/               Task-01 dataset prep / analysis / visualization
src/train/              Task-02 training driver
src/infer/              Task-02 sample-image inference / counting overlays
```

## Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# CUDA build of PyTorch (install separately so the right wheel is selected):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Task-01: Dataset Understanding and Preprocessing

### Dataset structure

The downloaded VisDrone dataset is organized under:

- `VisDrone2019-DET-train/images` and `VisDrone2019-DET-train/labels`
- `VisDrone2019-DET-val/images` and `VisDrone2019-DET-val/labels`
- `VisDrone2019-DET-test-dev/images` and `VisDrone2019-DET-test-dev/labels`

For this assignment, only `person` and `car` classes are used:

- `person` <- VisDrone classes `pedestrian` and `people`
- `car` <- VisDrone class `car`

### Preprocessing and augmentation steps

The Task-01 pipeline performs the following:

1. **Label filtering and class mapping**
   - Kept only labels mapped to `person` and `car`
   - Dropped all non-target classes
2. **Bounding box cleaning**
   - Removed invalid boxes (`w <= 0` or `h <= 0`)
   - Clamped out-of-bound coordinates to image boundaries
3. **Image integrity check**
   - Skipped corrupted/unreadable images
4. **Export cleaned labels**
   - Saved converted YOLO labels for the selected classes
5. **Task-01 augmentations (visualization set)**
   - Random horizontal flip
   - Brightness/contrast jitter
   - Mild blur
   - Mild Gaussian noise

### Challenges noticed in the dataset

The dataset contains several practical aerial-scene challenges:

- **Small object scale**: Many people are tiny and difficult to separate from background
- **Crowded scenes**: Dense traffic/crowd regions increase overlap and missed detections
- **Partial occlusions and truncations**: Objects are frequently partly hidden
- **Out-of-bound boxes**: Some labels require coordinate clamping
- **Class imbalance tendency**: Car instances are generally more frequent than persons in multiple splits

Detailed cleaning statistics are logged in:

- `outputs/metrics/task01_cleaning_report.json`

### Where to find sample visualizations

Task-01 visual outputs are available here:

- **Before preprocessing overlays**: `outputs/figures/task01/samples/before_preprocessing`
- **After augmentation overlays**: `outputs/figures/task01/samples/after_augmentation`
- **Analysis plots**:
  - `outputs/figures/task01/images_per_split.png`
  - `outputs/figures/task01/instances_per_class.png`
  - `outputs/figures/task01/bbox_area_distribution.png`
  - `outputs/figures/task01/objects_per_image_distribution.png`

---

## Task-02: Model Training

### Goal

Train an object detector that, given a VisDrone aerial image, predicts
bounding boxes for the two consolidated classes used downstream:

| ID  | Class    | Source VisDrone classes        |
| --- | -------- | ------------------------------ |
| 0   | `person` | `pedestrian` (1), `people` (2) |
| 1   | `car`    | `car` (4)                      |

### Model choice — and why YOLOv8s

I picked **YOLOv8s (Ultralytics)** as the Task-02 detector. The decision
was driven by the dataset characteristics from Task-01 (lots of tiny,
crowded targets in 1080p+ aerial frames) and by an honest read of the
available compute (free Colab T4 / L4 with ~15 GB VRAM for training, a
local 6 GB GTX 1660 SUPER for inference and counting).

- **Aerial small-object friendly.** YOLOv8 uses a multi-scale
  PANet-style neck with detection heads at strides 8 / 16 / 32. The
  stride-8 head matters a lot for VisDrone since most pedestrians fall
  in the *tiny* (<32²) and *small* (<96²) buckets according to
  Task-01's `bbox_area_distribution.png`.
- **Anchor-free + DFL.** YOLOv8 drops legacy anchors and uses
  Distribution Focal Loss for box regression. This generally helps with
  dense, overlapping aerial targets compared to anchor-based YOLOv5 /
  SSD / Faster R-CNN.
- **Fits both training and inference budgets.** YOLOv8s is ~11.1 M
  parameters / ~28.6 GFLOPs. On a Colab T4 it trains at `imgsz=800,
  batch=16` in AMP comfortably, and the same `best.pt` runs at
  interactive speed on the local 6 GB GPU for inference. Going bigger
  (YOLOv8m / -l, or RT-DETR-L) would have forced micro-batches in
  training and/or shrunk the local inference imgsz.
- **Mature tooling and reproducibility.** Ultralytics ships training,
  resume, validation, ONNX export, and the same pre/post-processing in
  one CLI/SDK — much less scaffolding than a custom Faster R-CNN /
  Detectron2 trainer.
- **Why not the alternatives I considered:**
  - *Faster R-CNN / Detectron2*: best-in-class accuracy on COCO but
    2-3× slower per epoch and overkill for a 2-class fine-tune.
  - *SSD300 / SSD512*: weakest of the listed options for tiny aerial
    objects (limited multi-scale heads, anchor coverage struggles
    below 30² boxes).
  - *RT-DETR*: very strong on COCO, but the official Ultralytics
    RT-DETR-L is ~32 M params + transformer attention — much heavier
    per iteration and harder to converge inside a single Colab
    training session.
  - *YOLOv11*: also a fine choice, but YOLOv8 has more public
    reference numbers on VisDrone, which makes the experiment easier
    to sanity-check.

### Where training actually runs: Google Colab

The local machine used here is a Windows + GTX 1660 SUPER setup. Two
things made it a poor fit for the long training run:

1. **AMP is auto-disabled** by Ultralytics on GTX 16xx (TU116) because
   of a known FP16 NaN issue. Training in pure FP32 roughly halves
   throughput and pushes 6 GB VRAM uncomfortably tight.
2. **Windows file-locking** (Defender + indexing) intermittently locks
   `results.csv` between epochs, killing long Ultralytics runs with
   `PermissionError`. We hit this twice in practice.

So the pipeline is **Colab-first**:

```text
┌────────────────────────────┐        ┌─────────────────────────────┐
│ Local (Windows + GTX 1660) │        │ Google Colab (T4 / L4 / A100)│
├────────────────────────────┤        ├─────────────────────────────┤
│ • Task-01 cleaning         │        │ • notebooks/task02_train_   │
│ • build_yolo_dataset.py    │        │   colab.ipynb               │
│ • zip_dataset_for_colab.py │  zip → │ • YOLOv8s + imgsz=800,       │
│                            │        │   batch=16, epochs=40, AMP   │
│            best.pt ◄─────────────────│ • Saves best.pt + plots to   │
│                            │        │   MyDrive/drone-detection/   │
│ • src/eval/evaluate_       │        │                              │
│   detector.py              │        └─────────────────────────────┘
│ • src/infer/infer_images.py│
└────────────────────────────┘
```

The notebook is self-contained: install deps → mount Drive → unzip
dataset → write data YAML → train → final val → copy run dir back to
Drive. Local validation against the resulting `best.pt` reproduces the
metrics on your own machine and rules out "it only works in Colab"
problems.

### Training hyperparameters (Colab canonical run)

Defined inside `notebooks/task02_train_colab.ipynb` and mirrored in
`configs/task02_train.yaml` for the local-fallback path. Key choices:

| Setting              | Colab value                                                                                  | Why                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Backbone weights     | `yolov8s.pt` (COCO)                                                                          | Transfer learning — COCO already covers `person` and `car`.                                      |
| Image size           | **800**                                                                                      | Bigger than YOLOv8's default 640 to better catch tiny aerial pedestrians; T4/L4 can fit it at batch=16. |
| Batch size           | **16**                                                                                       | Largest stable batch in AMP on a 15 GB T4 at imgsz=800.                                          |
| Epochs               | **40** (with `patience=20`)                                                                  | Long enough to see clear convergence and to benefit from `close_mosaic=10` over the last 10 epochs. |
| Optimizer            | SGD, momentum=0.937, weight_decay=5e-4                                                       | Default YOLOv8 recipe; more stable mAP than AdamW at this batch size.                            |
| LR schedule          | cosine, `lr0=0.01`, `lrf=0.01`, warmup=3 epochs                                              | Standard YOLOv8 recipe.                                                                          |
| Augmentation         | mosaic 1.0 (closed last 10 epochs), HSV-S/V jitter, `fliplr=0.5`, scale jitter 0.5, translate 0.1 | Conservative aerial-friendly recipe — `flipud` and rotation off (drone footage has a clear up direction), mixup / copy-paste off (preserves small-object semantics). |
| Mixed precision      | **on** (AMP)                                                                                 | T4 supports stable FP16; gives ~2× throughput and halves VRAM.                                   |
| Determinism          | `seed=42, deterministic=True`                                                                | Reproducible across runs given the same data.                                                    |

A local-fallback config tuned for 6 GB GPUs (`imgsz=640, batch=6,
amp=false`) lives in `configs/task02_train.yaml` and is consumed by
`src/train/train_detector.py` if you ever need to retrain without
Colab. It is documented as a fallback, not the canonical recipe.

The val/test splits used are the official VisDrone
`VisDrone2019-DET-val` and `VisDrone2019-DET-test-dev` directories.
`test-challenge` is intentionally excluded because it ships without
public labels.

### How to reproduce end-to-end

```powershell
# 1) LOCAL: build the YOLO-ready dataset layout from the kagglehub cache
python src\data\build_yolo_dataset.py ^
    --dataset-root "C:\Users\<you>\.cache\kagglehub\datasets\banuprasadb\visdrone-dataset\versions\1"

# 2) LOCAL: bundle the prepared dataset for Colab
python src\data\zip_dataset_for_colab.py --output-zip visdrone_yolo.zip
#   then upload visdrone_yolo.zip to:
#   MyDrive/drone-detection/visdrone_yolo.zip

# 3) COLAB: open notebooks/task02_train_colab.ipynb in Colab and run all cells.
#    When it finishes, download these from
#      MyDrive/drone-detection/runs/yolov8s_visdrone/
#    into the local repo:
#      weights/best.pt                   -> outputs/weights/best.pt
#      results.png / .csv / matrices     -> outputs/figures/task02/ + outputs/metrics/
#      task02_training_summary.json      -> outputs/metrics/

# 4) LOCAL: re-validate the downloaded best.pt on this machine
python src\eval\evaluate_detector.py --weights outputs\weights\best.pt --split val

# 5) LOCAL: render qualitative predictions on 12 random val images
python src\infer\infer_images.py ^
    --weights outputs\weights\best.pt ^
    --source-dir data\processed\visdrone\val\images ^
    --output-dir outputs\figures\task02\predictions
```

`scripts\run_task02.bat <KAGGLE_CACHE_PATH>` chains steps 1, 4, and 5
(it skips training and politely no-ops if `outputs\weights\best.pt`
doesn't exist yet).

### Training infrastructure

| Stage                    | Hardware                                | Notes                                  |
| ------------------------ | --------------------------------------- | -------------------------------------- |
| Dataset prep             | Local — Windows 11 + Python 3.12        | One-time per machine.                  |
| Training (canonical)     | Google Colab — NVIDIA T4 16 GB / L4 24 GB | `notebooks/task02_train_colab.ipynb`.  |
| Local validation         | Local — GTX 1660 SUPER 6 GB             | `src/eval/evaluate_detector.py`.       |
| Sample inference / Task-03 | Local — GTX 1660 SUPER 6 GB           | `src/infer/infer_images.py`.           |

- Train images: **6 471** (VisDrone2019-DET-train, cleaned to 2 classes).
- Val images: **548** (VisDrone2019-DET-val).
- Test images: **1 610** (VisDrone2019-DET-test-dev).

### Validation metrics

After the Colab run completes, metrics are saved to
`outputs/metrics/task02_training_summary.json` (overall + per-class mAP)
and `outputs/metrics/task02_eval_summary.json` (the local re-validation
pass on `best.pt`). Ultralytics' own `results.csv`, `results.png`,
`confusion_matrix.png`, and per-class PR curves are downloaded from
Drive into `outputs/figures/task02/` and `outputs/metrics/`.

<!-- METRICS_TABLE_START -->

**Final 40-epoch YOLOv8s on Colab T4 (`imgsz=800, batch=16, AMP`)**, reproduced
locally on the GTX 1660 SUPER via `src/eval/evaluate_detector.py`:

| Split (images / instances) | Class    | Precision | Recall | mAP@50 | mAP@50-95 |
| -------------------------- | -------- | --------: | -----: | -----: | --------: |
| **val** (548 / 28 033)     | all      | **0.778** | 0.644  | **0.702** | 0.415  |
|                            | person   | 0.729     | 0.508  | 0.578  | 0.251     |
|                            | car      | 0.827     | 0.779  | 0.827  | 0.579     |
| **test-dev** (1 610 / 55 456) | all   | 0.689     | 0.536  | 0.559  | 0.313     |
|                            | person   | 0.612     | 0.329  | 0.352  | 0.139     |
|                            | car      | 0.766     | 0.742  | 0.767  | 0.487     |

Speed on the local GTX 1660 SUPER at `imgsz=800`:
**~9 ms / image inference** (1 ms preprocess + 2 ms postprocess), i.e.
~83 FPS end-to-end for a single-image stream — comfortably real-time
even on this 6 GB card.

Observations:

- **Cars are easy, persons are hard.** Cars have ~1.6× the bbox area
  of persons on average in VisDrone, are usually less occluded, and
  have rigid silhouettes — that shows up as mAP50 0.83 vs 0.58 (val)
  and 0.77 vs 0.35 (test). Most missed persons are the *tiny* (<32²)
  pedestrians flagged in Task-01's `bbox_area_distribution.png`.
- **Val vs test gap.** The drop from val mAP50 0.70 → test mAP50 0.56
  is mostly absorbed by person recall (0.51 → 0.33). Test-dev contains
  more dense crowd scenes which compound the small-object failure mode.
- **Counting reliability.** Per-image car counts in the sample sweep
  (`outputs/metrics/task02_sample_predictions.json`) match human ground
  truth within ±1-2 for clear scenes. Crowded pedestrian frames
  (e.g. `0000155_00801_d_0000001.jpg`, 43 detected persons) will need
  the tracking-aware counting from Task-04 to debounce flicker.

All numeric outputs live in:

- `outputs/metrics/task02_training_summary.json` (Colab T4 metrics)
- `outputs/metrics/task02_eval_summary.json` (local val re-validation)
- `outputs/metrics/task02_eval_test_summary.json` (local test-dev re-validation)
- `outputs/metrics/task02_results.csv` (per-epoch training curves)
- `outputs/metrics/task02_sample_predictions.json` (per-image counts + ms)

Plot artifacts:

- `outputs/figures/task02/results.png` — loss / mAP curves over 40 epochs.
- `outputs/figures/task02/confusion_matrix.png` (+ normalized variant).
- `outputs/figures/task02/BoxPR_curve.png` etc. (from Colab).
- `outputs/figures/task02/local_val/` and `local_test/` — Ultralytics'
  PR / P / R / F1 curves, confusion matrices, and `val_batch*_pred.jpg`
  vs `val_batch*_labels.jpg` side-by-side previews from the local
  re-validation run.

<!-- METRICS_TABLE_END -->

### Sample predictions

`src/infer/infer_images.py` runs the trained `best.pt` on a random
sample of validation images and renders annotated overlays with
per-class colors plus a top-left `persons: X   cars: Y` banner. Outputs
land in `outputs/figures/task02/predictions/` and a per-image summary
(class counts + inference time in ms) is written to
`outputs/metrics/task02_sample_predictions.json`.

---

## Task-03: Human & Car Detection with Human Counting

### Goal

> Build a system that, given a drone image (or a folder of them), runs
> the Task-02 detector, draws bounding boxes for the two classes, and
> prominently displays the **total human count** for that image. The
> counting logic is intentionally simple.

### Pipeline

`src/infer/detect_and_count.py` is the Task-03 entry point. It loads
the Colab-trained `best.pt`, reads a small config YAML, and supports
two input modes from one CLI:

| Mode             | Flag                  | What it produces                                                     |
| ---------------- | --------------------- | -------------------------------------------------------------------- |
| Single image     | `--image <path>`      | One annotated image in `--output-dir` + 1-row CSV                    |
| Image directory  | `--image-dir <dir>`   | An annotated image per input in `--output-dir`, full counts CSV      |

Video / streaming input is intentionally out of scope here — that
belongs to the Task-04 tracking work, where the count becomes
"unique IDs in window" instead of "detections this frame".

The actual counting logic is one line:

```python
human_count = sum(1 for d in detections if d.cls_name == "person")
car_count   = sum(1 for d in detections if d.cls_name == "car")
```

What is *not* one line is the **filtering** that happens before counting,
because raw YOLO output on aerial imagery has two characteristic noise
patterns we want to suppress.

### Counting robustness — three knobs

All three are configured in [`configs/task03_count.yaml`](configs/task03_count.yaml)
and are wired into the pipeline through `CountingConfig.from_yaml`:

1. **Class-specific confidence thresholds.**
   Persons in VisDrone are small and often partially occluded, so
   raising the threshold above ~0.25 starts eating recall fast. Cars
   are larger and rigid, so a slightly higher threshold (0.30) removes
   most low-confidence false positives without hurting recall. Both
   thresholds are applied *after* YOLO's own NMS, with YOLO itself
   called at the lower of the two so neither class is starved.

   ```yaml
   class_thresholds:
     person: 0.25
     car: 0.30
   ```

2. **Minimum bbox area filter (`min_bbox_area_px: 32`).**
   The detector occasionally emits 1-2 pixel "speck" boxes on dense
   foliage / sun glare. Their per-instance impact on mAP is small but
   they directly inflate the human count, which is the deliverable we
   actually care about, so we drop any detection with area < 32 px².

3. **NMS IoU threshold (`iou: 0.5`).**
   YOLO's built-in NMS de-duplicates overlapping boxes of the same
   class. Loosening this below 0.5 starts merging close-by-but-distinct
   pedestrians in crowds; tightening it past 0.5 starts admitting
   duplicates for elongated cars at oblique angles. 0.5 is the standard
   COCO setting and works well here.

### Visual output

Each annotated frame gets:

- A per-class colored bounding box (green for `person`, orange for
  `car`) with a `class conf` label rendered on a filled bar above each
  box for readability against busy backgrounds.
- A black banner in the top-left containing:

  ```text
  Human Count: <X>
  Car Count: <Y>
  ```

  `Human Count` is rendered in larger cyan-yellow text to satisfy the
  Task-03 brief ("display total human count"); `Car Count` is
  secondary, smaller, and white. Both are drawn with anti-aliased text
  on top of a filled rectangle so they remain readable on any
  background.

Example output on `0000155_00801_d_0000001.jpg` — a dense urban scene
with 43 humans and 21 cars detected:

```7:7:outputs/figures/task03/predictions/0000155_00801_d_0000001.jpg
(see file)
```

### How to run

```powershell
# Default: 20 random val images, defaults to outputs/weights/best.pt
scripts\run_task03.bat

# Or point at any image dir
scripts\run_task03.bat data\processed\visdrone\test\images

# Single image
python src\infer\detect_and_count.py ^
    --config configs\task03_count.yaml ^
    --image data\processed\visdrone\val\images\0000155_00801_d_0000001.jpg
```

### Sample run (20 val images)

The default `scripts\run_task03.bat` pass produced:

| Metric                                    | Value                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------- |
| Inputs                                    | 20 images sampled from `data/processed/visdrone/val/images/`          |
| Total **humans** detected                 | **206**                                                               |
| Total **cars** detected                   | 330                                                                   |
| Max humans in a single image              | 43 (`0000155_00801_d_0000001.jpg`)                                    |
| Max cars in a single image                | 54 (`0000330_00201_d_0000801.jpg`)                                    |
| Steady-state inference                    | **~17 ms / image** on the local GTX 1660 SUPER (≈ 58 FPS, batch=1)    |
| First-call warm-up                        | ~551 ms (one-time CUDA kernel compile + weight upload)                |
| Mean inference reported                   | 44.4 ms (skewed by the warm-up; ignore the first row of the CSV)      |

Outputs:

- Annotated overlays — `outputs/figures/task03/predictions/*.jpg`.
- Per-image counts + timing — `outputs/metrics/task03_counts.csv`.
- Aggregate summary — `outputs/metrics/task03_summary.json`.

### Strengths and known failure modes

Strengths:

- The class-specific thresholds + min-area filter make the **human
  count rock-stable on clear scenes** (typo errors ≤ 1 vs ground truth
  in samples like `0000026_00500_d_0000025.jpg`).
- Car counts are reliable even in heavy traffic — saw 54 cars correctly
  separated in `0000330_00201_d_0000801.jpg` with no visible duplicates.
- Counting is **deterministic per input** (same seed → same numbers,
  no temporal accumulation), which makes it easy to validate.

Failure modes worth knowing about for Task-05 / Task-04:

- **Dense crowds → undercount.** Person recall is 0.51 on val; in
  scenes like `0000193_01876_d_0000113.jpg` (36 detected vs noticeably
  more in the ground truth) the model fuses pairs of overlapping
  pedestrians or drops the ones smaller than ~16 px.
- **Static cars on shoulders / parking lots → overcount.** A few
  partially-occluded cars in `0000330_00201_d_0000801.jpg` show up as
  two close-by boxes if the NMS IoU is loosened from 0.5.
- **Counting is per-image, with no temporal awareness.** If the same
  scene is later observed as a video, every frame is counted
  independently and the same pedestrian reappears across many frames.
  Turning that into a stable per-scene human count requires per-object
  identity tracking, which is the Task-04 bonus (ByteTrack / BoT-SORT
  on top of the same detector).


