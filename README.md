# VisionFSD Pilot

**Read-only** forward-camera driving-scene visualizer with dual-pane **3D world + camera** display.

It has **no** CAN-bus, steering, braking, throttle, actuator, or vehicle-control code.  
Do **not** use it to make or automate driving decisions.

![status](https://img.shields.io/badge/status-prototype-blue)
![python](https://img.shields.io/badge/python-3.11%2B-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

| Area | What it does |
|------|----------------|
| Objects | YOLO11n (OpenVINO FP16) + ByteTrack multi-object tracking |
| Road | YOLOPv2 lane + drivable-area segmentation (OpenVINO) |
| Ego path | Ultra-Fast Lane Detection (UFLD) for painted ego lanes |
| 3D view | OpenGL low-poly meshes, sticky LEAD, road-locked lane slots |
| Modes | World / camera / split; webcam or YouTube highway clip |

Designed for **Intel iGPU** (OpenVINO `intel:gpu` / `GPU`) with a target of ~25+ FPS split view on mid-range laptops. Performance depends on scene complexity, power mode, and drivers.

## Requirements

- **Windows 10/11** (primary; Linux may work with path tweaks)
- **Python 3.11+**
- **Intel GPU drivers** recommended for OpenVINO GPU inference  
  (CPU fallback works but is slower)
- Optional for YouTube tests:
  - [Node.js](https://nodejs.org/) (yt-dlp JS challenges)
  - [FFmpeg](https://ffmpeg.org/) on `PATH`
  - Microsoft Edge signed into YouTube (cookie export)

## Quick start

```bat
git clone https://github.com/YoMosa2009/VisionFSD-Pilot.git
cd VisionFSD-Pilot
setup.bat
run.bat
```

Or manually:

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
run.bat
```

First startup can take **20–40 seconds** while OpenVINO compiles graphs. Later runs use the local model cache under `models/**/cache/` (gitignored).

## Raspberry Pi 3B installation (separate runtime)

The Raspberry Pi version is a **separate, low-resource installation** in
[`pi3b/`](pi3b/). Do not use the Windows `setup.bat` / `run.bat` flow on a Pi:
the desktop application depends on Intel OpenVINO GPU and several perception
models that are intentionally not part of the Pi runtime.

The Pi runtime uses a small quantized TFLite detector, keeps only the newest
camera frame, and displays **one sticky selected car** in its camera and
low-cost world views. It is still read-only visualization software and never
controls a vehicle.

On a networked Raspberry Pi 3B running Raspberry Pi OS, install everything with:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/main/pi3b/install.sh | bash
```

The installer creates `~/visionfsd-pi`, installs the Pi-only dependencies,
downloads the verified TFLite model, and verifies its SHA-256. Then start it:

```bash
~/visionfsd-pi/pi3b/run.sh --camera 0 --fps 25
```

Use `--fps 30` only after the Pi's sustained benchmark proves it can maintain
that rate without thermal throttling. The HUD reports display FPS and detector
FPS separately. See [`pi3b/README.md`](pi3b/README.md) for camera, model, and
benchmark details.

### Controls

| Key | Action |
|-----|--------|
| `1` / `M` | 3D world |
| `2` | Annotated camera |
| `3` | Split world + camera |
| `V` | Cycle views |
| `L` | Toggle lane/path overlay |
| `S` | Screenshot → `logs\screenshots` |
| `F` | Fullscreen |
| `Q` / `Esc` | Quit |

If the webcam is not index `0`, edit `--camera` in `run.bat`.

## YouTube highway test

```bat
run_youtube_test.bat
```

- Starts at **01:01:30** of the bundled demo URL (edit `--start-seconds` to change).
- First run at a new start time downloads a **short local clip** to `logs\youtube_cache\` (not the full video).
- A loading window appears immediately; Esc cancels during load.
- Cookies: auto-exported from Edge into `logs\youtube-cookies.txt` (never commit this file).  
  See [`logs/YOUTUBE_COOKIES.md`](logs/YOUTUBE_COOKIES.md).

## Repository layout

```
VisionFSD-Pilot/
  src/                 # Application (visionfsd_3d.py entrypoint)
  tools/               # Model export + cookie helpers
  models/              # OpenVINO IRs used at runtime
  yolo11n_openvino_model/
  config/              # ByteTrack YAML
  run.bat              # Webcam launcher
  run_youtube_test.bat # YouTube regression launcher
  setup.bat            # Create venv + install deps
  requirements.txt
```

### Included models (runtime)

| Model | Path | Notes |
|-------|------|--------|
| YOLO11n detect | `yolo11n_openvino_model/` | Default object detector |
| YOLOPv2 road | `models/yolopv2/openvino_fp16/` | Lanes + drivable area |
| UFLD TuSimple | `models/ufld/openvino_fp16/` | Ego-path lanes (Git LFS) |
| Depth Anything V2 S | `models/depth_anything_v2_small/openvino_fp16/` | Optional; off by default |

Large training checkpoints (`.pt` / TensorFlow saved models) are **not** in the repo. Re-export with:

```bat
.venv\Scripts\python tools\export_yolopv2_openvino.py
.venv\Scripts\python tools\export_ufld_openvino.py
.venv\Scripts\python tools\export_depth_anything_openvino.py
```

> **Git LFS:** the UFLD `.bin` exceeds GitHub’s 100 MB limit and is stored with [Git LFS](https://git-lfs.com/).  
> After clone: `git lfs install` then `git lfs pull` if weights are missing.

## Camera calibration

After mounting the camera, set in `run.bat` / CLI:

- `--camera-height` — optical centre height above road (metres)
- `--horizon-ratio` — vanishing-row fraction (default `0.52`)
- `--fov` — horizontal FOV if known

Mis-calibration strongly affects distance and 3D placement.

## CLI overview

```bat
.venv\Scripts\python src\visionfsd_3d.py --help
```

Useful flags:

- `--source <path|youtube-url>` — file or YouTube instead of webcam  
- `--start-seconds N` — seek / YouTube window start  
- `--device intel:gpu` / `--road-device GPU` / `--ufld-device CPU`  
- `--no-depth` / `--depth` — monocular only vs Depth Anything  
- `--test-seconds N` — automated self-test then exit  

## Safety & limits

- Prototype visualizer only — **not** an ADAS or autonomous driving stack.
- Monocular RGB cannot measure true depth, cover blind spots, or guarantee lanes in all weather/lighting.
- Tracks and ranges are estimates; sticky LEAD and lane slots are display heuristics.

## Model sources & licenses

- Project code: **MIT** (see [`LICENSE`](LICENSE))
- [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) — MIT (see `models/yolopv2/LICENSE.txt`)
- [YOLO11 / Ultralytics](https://github.com/ultralytics/ultralytics) — review Ultralytics terms before commercial use
- UFLD TuSimple weights — follow upstream project terms; OpenVINO IR is a local export

## Performance notes (reference hardware)

On an **i5-1035G7 + Intel Iris Plus** class machine, split-view highway playback has been measured around **~25–30 FPS** display with adaptive detect/road/UFLD intervals. Numbers are not guarantees.

## Contributing

Issues and PRs welcome. Please do **not** open PRs that include:

- `logs/youtube-cookies.txt` or any personal cookies
- `logs/youtube_cache/` media
- Compiled OpenVINO `cache/` blobs
- Secrets or private video URLs
