@echo off
setlocal

REM ---------------------------------------------------------------------
REM Task-02 LOCAL pipeline. Training itself runs on Google Colab
REM (notebooks/task02_train_colab.ipynb). This script handles:
REM
REM   1. Materializing the YOLO-ready dataset layout from the kagglehub
REM      cache (one-time).
REM   2. Re-running validation on the Colab-trained best.pt to confirm
REM      the metrics on this machine.
REM   3. Rendering sample qualitative predictions for the deliverable.
REM
REM Usage:
REM   scripts\run_task02.bat ^<DATASET_ROOT_PATH^>
REM
REM Expects outputs\weights\best.pt to exist (downloaded from Colab).
REM ---------------------------------------------------------------------

if "%~1"=="" (
  echo Usage: scripts\run_task02.bat ^<DATASET_ROOT_PATH^>
  exit /b 1
)

set DATASET_ROOT=%~1
set WEIGHTS=outputs\weights\best.pt

python src\data\build_yolo_dataset.py --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1

if not exist "%WEIGHTS%" (
  echo.
  echo [task02] outputs\weights\best.pt not found.
  echo         Train the model on Colab via notebooks\task02_train_colab.ipynb
  echo         then copy best.pt into outputs\weights\ before re-running.
  exit /b 0
)

python src\eval\evaluate_detector.py --weights "%WEIGHTS%" --split val
if errorlevel 1 exit /b 1

python src\infer\infer_images.py ^
  --weights "%WEIGHTS%" ^
  --source-dir data\processed\visdrone\val\images ^
  --output-dir outputs\figures\task02\predictions ^
  --num-samples 12
if errorlevel 1 exit /b 1

echo Task-02 local artifacts generated successfully.
