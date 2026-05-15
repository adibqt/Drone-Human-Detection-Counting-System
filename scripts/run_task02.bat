@echo off
setlocal

if "%~1"=="" (
  echo Usage: scripts\run_task02.bat ^<DATASET_ROOT_PATH^>
  exit /b 1
)

set DATASET_ROOT=%~1

python src\data\build_yolo_dataset.py --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1

python src\train\train_detector.py --config configs\task02_train.yaml
if errorlevel 1 exit /b 1

python src\infer\infer_images.py ^
  --weights outputs\runs\task02\yolov8s_visdrone\weights\best.pt ^
  --source-dir data\processed\visdrone\val\images ^
  --output-dir outputs\figures\task02\predictions ^
  --num-samples 12
if errorlevel 1 exit /b 1

echo Task-02 artifacts generated successfully.
