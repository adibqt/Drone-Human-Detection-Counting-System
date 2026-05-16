@echo off
setlocal

REM ---------------------------------------------------------------------
REM Task-04 LOCAL pipeline. Takes either:
REM   - a .mp4 video file, OR
REM   - a folder of frames (e.g. a VisDrone-VID sequence)
REM and runs detection + ByteTrack + unique-ID human counting on it.
REM
REM Usage:
REM   scripts\run_task04.bat <SOURCE>
REM     where SOURCE is either a video file or a folder of images.
REM
REM Example:
REM   scripts\run_task04.bat outputs\videos\smoke_test.mp4
REM   scripts\run_task04.bat path\to\VisDrone2019-VID-val\sequences\uav0000086_00000_v
REM ---------------------------------------------------------------------

if "%~1"=="" (
  echo Usage: scripts\run_task04.bat ^<SOURCE^>
  echo   SOURCE: an .mp4 file or a folder of frames.
  exit /b 1
)

set SOURCE=%~1

if not exist outputs\weights\best.pt (
  echo [task04] outputs\weights\best.pt not found.
  echo         Train via notebooks\task02_train_colab.ipynb and copy
  echo         best.pt into outputs\weights\ first.
  exit /b 1
)

python src\track\track_video.py ^
  --config configs\task04_track.yaml ^
  --source "%SOURCE%" ^
  --output-video outputs\videos\task04_demo.mp4 ^
  --counts-csv outputs\metrics\task04_track_counts.csv ^
  --summary-json outputs\metrics\task04_track_summary.json

if errorlevel 1 exit /b 1

echo Task-04 artifacts generated successfully.
