@echo off
setlocal

if "%~1"=="" (
  echo Usage: scripts\run_task01.bat ^<DATASET_ROOT_PATH^>
  exit /b 1
)

set DATASET_ROOT=%~1

python src\data\prepare_dataset.py --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1

python src\data\analyze_dataset.py --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1

python src\data\visualize_samples.py --dataset-root "%DATASET_ROOT%" --split train --num-samples 10
if errorlevel 1 exit /b 1

echo Task-01 artifacts generated successfully.
