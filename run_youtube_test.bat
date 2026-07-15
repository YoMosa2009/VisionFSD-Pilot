@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Node + FFmpeg on PATH (YouTube JS challenges + section downloads)
set "PATH=%ProgramFiles%\nodejs;%ProgramFiles%\ffmpeg\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

set "YOLO_CONFIG_DIR=%CD%\models\ultralytics"
if not exist "%~dp0logs" mkdir "%~dp0logs"
if exist "%~dp0logs\crash.log" del /f /q "%~dp0logs\crash.log"

echo.
echo VisionFSD YouTube test
echo   Source: https://www.youtube.com/watch?v=JS5HvyvhhxM
echo   Start:  01:01:30  - stream only; video is NOT downloaded
echo   UFLD:   ON     Depth: OFF by default
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo ERROR: Python venv not found:
  echo   %~dp0.venv\Scripts\python.exe
  goto end
)

rem Auto-export Edge YouTube cookies if missing (avoids DPAPI browser-cookie issues)
if exist "%~dp0logs\youtube-cookies.txt" goto cookies_ready
echo Exporting YouTube cookies from Edge automatically...
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\export_edge_youtube_cookies.py"
if errorlevel 1 (
  echo.
  echo Auto cookie export failed.
  echo Open Edge, sign into YouTube, visit youtube.com, then re-run this bat.
  echo See logs\YOUTUBE_COOKIES.md
  goto end
)

:cookies_ready
set "YT_COOKIE_ARGS=--cookies logs\youtube-cookies.txt"
echo Using cookies file: logs\youtube-cookies.txt
echo.
echo NOTE: A loading window opens immediately (not black / not hung).
echo First run at a new start time may download a short clip to logs\youtube_cache
echo OpenVINO GPU compile can take 20-40 seconds after the clip is ready.
echo Press Q or Esc to cancel during load or quit once ready.
echo.

"%~dp0.venv\Scripts\python.exe" "%~dp0src\visionfsd_3d.py" --source "https://www.youtube.com/watch?v=JS5HvyvhhxM" --start-seconds 3690 %YT_COOKIE_ARGS% --realtime-video --view split --window-width 1024 --window-height 576 --model yolo11n_openvino_model --model-task detect --device intel:gpu --imgsz 512 --detect-interval 3 --learned-road --road-model models/yolopv2/openvino_fp16/yolopv2_road.xml --road-device GPU --road-interval 6 --no-depth --ufld --ufld-model models/ufld/openvino_fp16/ufld_tusimple_18.xml --ufld-device CPU --ufld-interval 18 --cpu-threads 2 --monitor 0 --test-report logs\youtube-test-report.json
set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" goto failed
echo Process exited cleanly.
goto end

:failed
echo Process exited with code %EXITCODE%.
if exist "%~dp0logs\crash.log" (
  echo --- logs\crash.log ---
  type "%~dp0logs\crash.log"
) else (
  echo No new crash.log was written.
)
echo.
echo If YouTube failed:
echo   - Re-export cookies:  .venv\Scripts\python tools\export_edge_youtube_cookies.py
echo   - Ensure Node.js is installed for yt-dlp JS challenges
goto end

:end
echo.
pause
endlocal
