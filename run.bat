@echo off
setlocal
cd /d "%~dp0"
set YOLO_CONFIG_DIR=%CD%\models\ultralytics
set PYTHONPATH=%CD%\src;%PYTHONPATH%
REM Camera indices on this laptop (DirectShow):
REM   0 = Surface Camera Front (built-in)
REM   1 = icspring camera (USB webcam)  <-- use this for the Pilot
REM LiveWebcamCapture opens only after models load; drop-to-newest grab thread.
"%~dp0.venv\Scripts\python.exe" "%~dp0src\visionfsd_3d.py" --camera 1 --width 1280 --height 720 --fps 30 --monitor 0 --view split --window-width 1152 --window-height 648 --model yolo11n_openvino_model --model-task detect --device intel:gpu --imgsz 512 --detect-interval 3 --learned-road --road-model models/yolopv2/openvino_fp16/yolopv2_road.xml --road-device GPU --road-interval 5 --depth --depth-device GPU --depth-interval 10 --ufld --ufld-device CPU --ufld-interval 22 --cpu-threads 4
pause
