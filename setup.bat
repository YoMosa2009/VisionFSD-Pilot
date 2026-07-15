@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo VisionFSD Pilot — environment setup
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  echo Install Python 3.11+ from https://www.python.org/downloads/
  echo and re-run this script.
  goto end
)

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Creating virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv.
    goto end
  )
)

echo Upgrading pip...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
echo Installing requirements...
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo pip install failed.
  goto end
)

if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0models\ultralytics" mkdir "%~dp0models\ultralytics"

echo.
echo Setup complete.
echo   Webcam:   run.bat
echo   YouTube:  run_youtube_test.bat  (needs Edge signed into YouTube, Node.js, FFmpeg)
echo.
echo See README.md for full instructions.

:end
echo.
pause
endlocal
