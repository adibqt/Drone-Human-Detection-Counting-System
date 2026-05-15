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
crowded targets in 1080p+ aerial frames) and by the practical compute
budget (a single GTX 1660 SUPER with 6 GB VRAM):

- **Aerial small-object friendly.** YOLOv8 uses a multi-scale
  PANet-style neck with detection heads at strides 8 / 16 / 32. The
  stride-8 head matters a lot for VisDrone since most pedestrians fall
  in the *tiny* (<32²) and *small* (<96²) buckets according to
  Task-01's `bbox_area_distribution.png`.
- **Anchor-free + DFL.** YOLOv8 dropped the legacy anchors and uses
  Distribution Focal Loss for box regression. This generally helps with
  dense, overlapping aerial targets compared to anchor-based YOLOv5 /
  SSD / Faster R-CNN.
- **Fits a 6 GB GPU comfortably.** YOLOv8s is ~11.1 M parameters /
  ~28.6 GFLOPs. At `imgsz=640, batch=6` it peaks around ~4.6 GB VRAM
  in FP32 on the GTX 1660 SUPER, leaving headroom for AdamW state and
  cuDNN workspace. YOLOv8m / YOLOv8l would have been pushed to OOM or
  micro-batches on this card, which materially hurts BatchNorm
  statistics.
- **Mature tooling and reproducibility.** Ultralytics ships training,
  resume, validation, ONNX export, hyperparameter search, and the same
  pre/post-processing in one CLI/SDK. That removed weeks of
  scaffolding work compared to writing a custom Faster R-CNN /
  Detectron2 trainer.
- **Why not the alternatives I considered:**
  - *Faster R-CNN / Detectron2*: best-in-class accuracy on COCO but
    2-3× slower per epoch and harder to fit at high resolution on this
    GPU; overkill for a 2-class fine-tune.
  - *SSD300 / SSD512*: weakest of the listed options for tiny aerial
    objects (limited multi-scale heads, anchor coverage struggles
    below 30² boxes).
  - *RT-DETR*: very strong on COCO, but the official Ultralytics
    RT-DETR-L weights are ~32 M params and the matching feature is
    transformer attention — heavier per-iter cost and harder
    convergence on a 6 GB GPU within a 1-day training budget.
  - *YOLOv11* (the next Ultralytics generation): would also have been
    a fine choice, but YOLOv8 has more public reference numbers on
    VisDrone, which makes the experiment easier to sanity-check.

### Training approach

Hyperparameters are version-controlled in
[`configs/task02_train.yaml`](configs/task02_train.yaml). The training
driver `src/train/train_detector.py` loads the YAML, forwards it as
kwargs to `ultralytics.YOLO.train`, then runs a final `.val()` pass on
`best.pt` and writes a JSON summary to
`outputs/metrics/task02_training_summary.json`.

Key choices and the reasoning behind them:

| Setting              | Value             | Why                                                                                        |
| -------------------- | ----------------- | ------------------------------------------------------------------------------------------ |
| Backbone weights     | `yolov8s.pt` COCO | Transfer learning — COCO already covers `person` and `car`, so the head warm-starts well.  |
| Image size           | 640               | Default YOLOv8; balances tiny-object recall with VRAM. (Higher imgsz=960+ would be ideal but doesn't fit at batch=6 in FP32 on a 1660 SUPER.) |
| Batch size           | 6                 | Largest stable batch in FP32 on 6 GB; AMP is auto-disabled by Ultralytics on GTX 16xx because of known FP16 NaN issues. |
| Epochs               | 20                | Enough to clearly see convergence on the cleaned 6 471-image train split within a single-GPU day; with `patience=15` and `cos_lr=true` we stop early if validation mAP plateaus. |
| Optimizer            | SGD (0.937 mom.)  | Default YOLOv8 recipe; gives more stable mAP than AdamW for this batch size in our smoke tests. |
| LR schedule          | cosine, lr0=0.01  | Standard YOLOv8 recipe.                                                                    |
| Augmentation         | mosaic 1.0 (closed last 8 epochs), HSV-S/V jitter, fliplr 0.5, scale jitter 0.5, translate 0.1 | Conservative aerial-friendly recipe. We disabled flipud (drone footage has a clear up direction) and disabled mixup / copy-paste to keep small-object semantics intact. |
| Mixed precision      | off               | Forced off because the GTX 1660 SUPER fails Ultralytics' AMP sanity check (known FP16 issue on TU116). |
| Early stopping       | `patience=15`     | Conservative; lets the model finish the cosine schedule.                                    |

The val/test splits used for training are the official VisDrone
`VisDrone2019-DET-val` and `VisDrone2019-DET-test-dev` directories.
`test-challenge` is intentionally excluded because it ships without
public labels.

### How to reproduce

```powershell
# 1. (Once) build the YOLO-ready dataset layout from the kagglehub cache
python src\data\build_yolo_dataset.py --dataset-root "C:\Users\<you>\.cache\kagglehub\datasets\banuprasadb\visdrone-dataset\versions\1"

# 2. Train (CUDA auto-detected)
python src\train\train_detector.py --config configs\task02_train.yaml

# 3. Render qualitative predictions on 12 random val images
python src\infer\infer_images.py ^
    --weights outputs\runs\task02\yolov8s_visdrone\weights\best.pt ^
    --source-dir data\processed\visdrone\val\images ^
    --output-dir outputs\figures\task02\predictions
```

The convenience wrapper `scripts\run_task02.bat <KAGGLE_CACHE_PATH>`
runs all three steps in order.

### Training infrastructure used for this run

- GPU: NVIDIA GeForce GTX 1660 SUPER, 6 GB VRAM
- PyTorch 2.5.1+cu121, Ultralytics 8.4.50, Python 3.12
- Train images: **6 471** (VisDrone2019-DET-train, cleaned to 2 classes)
- Val images: **548** (VisDrone2019-DET-val)
- Wall-clock: ~5 min per epoch + final validation

### Validation metrics

After the run completes, metrics are summarized in
`outputs/metrics/task02_training_summary.json` and Ultralytics also
writes its own `results.csv`, `results.png`, `confusion_matrix.png`,
and per-class PR curves under
`outputs/runs/task02/yolov8s_visdrone/`.

A snapshot of the final `best.pt` validation pass is reproduced below
(filled in automatically at the end of training).

<!-- METRICS_TABLE_START -->
_Metrics will populate after the training run finishes._
<!-- METRICS_TABLE_END -->

### Sample predictions

`src/infer/infer_images.py` runs the trained `best.pt` on a random
sample of validation images and renders annotated overlays with
per-class colors plus a top-left `persons: X   cars: Y` banner. Outputs
land in `outputs/figures/task02/predictions/` and a per-image summary
(class counts + inference time in ms) is written to
`outputs/metrics/task02_sample_predictions.json`.


