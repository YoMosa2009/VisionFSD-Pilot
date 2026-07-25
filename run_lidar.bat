@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo VisionFSD environment is missing. Run setup.bat first.
  pause
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0pi3b\lidar_visualizer.py" %*
pause
