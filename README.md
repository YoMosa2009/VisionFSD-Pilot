# VisionFSD Pilot

Forward-camera driving-scene visualizer with dual-pane **3D world + camera**
display, a separate Raspberry Pi 3B runtime, an LD19 2D LiDAR inspection tool,
and a low-speed OSOYOO indoor-robot integration.

It has **no** CAN-bus, steering, braking, throttle, actuator, or real-vehicle
control code. Do **not** use it to make or automate driving decisions.

## About

VisionFSD Pilot is a prototype perception and visualization project with two
deliberately separate paths:

- **Desktop Pilot:** Windows/OpenVINO scene visualization using a forward USB
  camera, road/lane models, and a low-poly world view.
- **Pi 3B runtime:** a lightweight LiteRT visualizer designed for Raspberry Pi
  3B hardware, with one sticky lead vehicle maximum and its own installer.
- **LD19 LiDAR visualizer:** a read-only 360-degree horizontal point-cloud
  viewer for an LD19 through its USB-UART adapter. It renders only fresh range
  returns, suppresses weak near-sensor noise, and groups adjacent returns into
  geometric obstacle clusters. It does not classify or control anything.
- **Pi indoor robot mode:** an optional, supervised Pi + LD19 + webcam + Uno
  runtime for the OSOYOO robot kit. The Uno keeps the final ultrasonic stop and
  motor dead-man timeout; the Pi supplies conservative high-level planning.

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

The Pi runtime uses a hash-verified **EfficientDet-Lite0 INT8** TFLite detector
with automatic SSD-MobileNetV1 fallback, keeps only the newest camera frame,
and displays **one confirmed, sticky lead vehicle maximum** in
both its camera and low-cost world views. It prefers a visibly near ego-lane
vehicle, then falls back to one near adjacent-lane vehicle. Tiny horizon boxes,
duplicate vehicle-class hypotheses, and unstable ID/class changes are filtered
with lightweight temporal evidence. Front/left/right placement uses sticky
lane voting. Its world view can also show confirmed pedestrians, traffic
lights, and stop signs; pedestrians must pass stricter confidence, shape,
vehicle-overlap, and four-frame confirmation gates. These never appear in the
camera view. It is still read-only visualization software and never controls
a vehicle.

On a networked Raspberry Pi 3B running **64-bit** Raspberry Pi OS, install
everything with:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/main/pi3b/install.sh | bash
```

The installer creates `~/visionfsd-pi`, installs the Pi-only dependencies,
downloads the verified neural models, and verifies their SHA-256 values. Then
start it:

```bash
~/visionfsd-pi/pi3b/run.sh --camera 0 --fps 25 --threads 3
```

To update an existing Pi installation (it remembers the Pi release branch):

```bash
cd ~/visionfsd-pi && bash ./pi3b/update.sh
```

If an older updater aborts because local changes would be overwritten, recover
it once without deleting those edits:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/codex/pi3b-runtime/pi3b/recover-update.sh | bash
```

After recovery, the normal update command above works for later releases.

The Pi 3B preset targets a 25 FPS display using newest-frame asynchronous
inference, one OpenCV worker, and three LiteRT threads. The HUD reports display
FPS and detector FPS separately; 25 FPS inference is not claimed without a
sustained physical-Pi benchmark. The HUD also shows the installed Pi runtime
version and active detector.
See [`pi3b/README.md`](pi3b/README.md) for camera, model, and benchmark details.

### Pi OSOYOO robot integration

The Pi robot runtime is separate from the read-only desktop visualizer. It
uses the LD19 as 360-degree measured range, the front static ultrasonic sensor
as an independent near-field stop, and the webcam as both a live-frame safety
gate and a confirmed-person veto. The Pi sends bounded differential motor
commands/status over the Uno's normal USB cable. It starts with a 25-second
no-motion standby, makes gentle LiDAR-guided arcs around obstacles, and uses a
short LiDAR-cleared pivot only for close escape manoeuvres. Its local LiDAR map
uses commanded-motion dead reckoning and is explicitly approximate.

It is not vehicle autonomy and is not robust room-scale SLAM: the kit has no
wheel encoders or IMU. Do not run it unsupervised, near stairs, pets, people,
or property that can be damaged. Details, firmware location, boot behaviour,
and the one-command Pi update are in [`pi3b/README.md`](pi3b/README.md#osoyoo-robot-mode-pi--ld19--camera--uno).

### LD19 LiDAR visualizer

The LD19 visualizer is separate from the camera runtime and sends no commands
to any vehicle or robot. It shows the LiDAR's current horizontal scan and
geometric obstacle clusters; a 2D scan cannot identify a cluster as a specific
object type.

On the Pi, after the standard update/install:

```bash
cd ~/visionfsd-pi/pi3b
bash ./run_lidar.sh
```

On Windows, run `run_lidar.bat`. The tool automatically selects a single USB
serial adapter; use `--port COMx` or `--port /dev/ttyUSB0` if more than one is
connected. Returns expire after 300 ms by default, so changes clear promptly
instead of leaving a multi-second history trail.

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
  pi3b/                # Separate Pi 3B runtime + LD19 visualizer
  run.bat              # Webcam launcher
  run_lidar.bat        # Windows LD19 point-cloud launcher
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
- The LD19 viewer is a single horizontal 2D scan. It does not provide height,
  semantic object identity, or a safety guarantee.

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
