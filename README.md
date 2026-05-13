# Drone Human Detection and Counting System

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
