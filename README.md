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
_Metrics populate here after the Colab training run finishes and
`outputs/weights/best.pt` is copied back. Re-run
`python src\eval\evaluate_detector.py --weights outputs\weights\best.pt`
on this machine to refresh the JSON summary._
<!-- METRICS_TABLE_END -->

### Sample predictions

`src/infer/infer_images.py` runs the trained `best.pt` on a random
sample of validation images and renders annotated overlays with
per-class colors plus a top-left `persons: X   cars: Y` banner. Outputs
land in `outputs/figures/task02/predictions/` and a per-image summary
(class counts + inference time in ms) is written to
`outputs/metrics/task02_sample_predictions.json`.


