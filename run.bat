@echo off
setlocal
cd /d "%~dp0"
set YOLO_CONFIG_DIR=%CD%\models\ultralytics
"%~dp0.venv\Scripts\python.exe" "%~dp0src\visionfsd_3d.py" --camera 0 --width 1280 --height 720 --fps 30 --monitor 0 --view split --window-width 1152 --window-height 648 --model yolo11n_openvino_model --model-task detect --device intel:gpu --imgsz 512 --detect-interval 3 --learned-road --road-model models/yolopv2/openvino_fp16/yolopv2_road.xml --road-device GPU --road-interval 3 --cpu-threads 4
pause
