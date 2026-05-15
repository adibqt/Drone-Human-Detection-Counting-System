@echo off
setlocal

REM ---------------------------------------------------------------------
REM Task-03 LOCAL pipeline. Runs detection + human counting on the
REM validation split (random 20 image sample by default), saves
REM annotated overlays, a per-image counts CSV, and an aggregate summary.
REM
REM Optional first arg: image directory to override the default val split.
REM ---------------------------------------------------------------------

set IMAGE_DIR=%~1
if "%IMAGE_DIR%"=="" set IMAGE_DIR=data\processed\visdrone\val\images

if not exist outputs\weights\best.pt (
  echo [task03] outputs\weights\best.pt not found.
  echo         Train the model via notebooks\task02_train_colab.ipynb and
  echo         copy best.pt into outputs\weights\ first.
  exit /b 1
)

python src\infer\detect_and_count.py ^
  --config configs\task03_count.yaml ^
  --image-dir "%IMAGE_DIR%" ^
  --num-samples 20 ^
  --output-dir outputs\figures\task03\predictions ^
  --counts-csv outputs\metrics\task03_counts.csv ^
  --summary-json outputs\metrics\task03_summary.json

if errorlevel 1 exit /b 1

echo Task-03 artifacts generated successfully.
