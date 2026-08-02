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
as an independent near-field stop, and the webcam as a live-frame gate plus
low-cost optical-flow pose aid. Robot mode does not run camera object/person
detection. The Pi sends bounded differential motor
commands/status over the Uno's normal USB cable. It starts with a 25-second
no-motion standby, uses hysteresis and direction locking for stable LiDAR-guided
arcs, and uses a bounded reverse-turn-commit recovery sequence when a close
obstacle blocks forward progress. It then resumes live corridor planning instead
of latching a terminal stop. A USB LSM6DS3 through an MCP2221A, or the existing
GPIO MPU-6050 fallback, bounds recovery turns by measured short-term yaw and
supplies yaw to an 8 m occupancy map. The runtime
marks observed free space, selects reachable unexplored frontiers, plans a
collision-inflated grid route, and uses its next waypoint as long-horizon
guidance. Current LD19 geometry still authorizes every motor direction. LD19
scan matching supplies cautious yaw and translation correction, but this remains
estimated navigation rather than true metric SLAM because the kit has no wheel
encoders, loop closure, or absolute position reference.

If both IMUs are missing or stale, the same occupancy, frontier, A*, waypoint,
patrol, and live-corridor stack remains active. Turn prediction uses differential
motor commands and is corrected by successive LD19 scans; recovery turns use
that corrected map heading instead of relying only on elapsed time. A fail-safe
leased heartbeat refreshes the selected motor output independently of
camera/display/planner scheduling, rear clearance uses a body-width LiDAR
corridor, and the controller checks all forward body corridors before entering
recovery. Camera neural inference was removed from robot mode; optical flow
continues to refine non-IMU pose without spending CPU on person detection. The
normal updater performs the one-time MCP2221 Linux setup and installs
its Python transport. Robot startup probes LSM6DS3 addresses `0x6A` and `0x6B`,
accepts the LSM6DS3TR-C identity `0x6A`, verifies the programmed registers,
consumes only fresh complete samples, and retains GPIO/non-IMU fallbacks. If
the Uno USB serial node changes, the runtime holds STOP, rediscovers the exact
Uno USB identity, and repeats the capability handshake instead of terminating.
Its calibrated gyro bias continues adapting only during confirmed stationary
periods to reduce temperature-related yaw drift. In v1.9.5, unfinished IMU
calibration pauses instead of resetting when the chassis moves or another USB
device is handled. Motor commands survive bounded half-second scheduling stalls,
a camera reset receives at most one second of last-frame grace, and sustained
camera loss still stops the robot. The local planner selects an 11-degree-wide
opening instead of trusting one long LiDAR ray, pivots away before a straight
obstacle reaches 40 cm, and uses clearance-weighted frontier routes to avoid
unnecessary wall-hugging while retaining reachable narrow passages. In v1.9.6,
robot boot calibrates a connected USB IMU before starting webcam streaming,
avoiding Pi 3B USB/CPU contention during the stationary calibration window.
Stationary frames no longer run optical flow, and each reopened webcam receives
its own startup timeout instead of being rejected against the previous camera's
stale timestamp. In v1.9.7, IMU sampling runs independently of camera, display,
and planner work. Once an IMU is detected, motor authority remains locked until
that IMU reaches `LIVE`; only a genuinely absent IMU enters non-IMU mode.
Calibration uses a rolling still-sample window and retains partial progress
through a temporary MCP2221 USB reset instead of falling back to a displayed 0%.
In v1.9.8, the USB LSM6DS3 calibration window is 40 valid samples. Calibration
uses total acceleration and robust trimmed gyro variance instead of requiring a
perfectly level board or rejecting the stationary bias it needs to measure. The
dashboard reports the reason whenever sample collection is intentionally held.
In v1.9.9, a blocked reverse path no longer leaves recovery stopped when a
complete LiDAR-cleared turn sweep exists: it uses a bounded low-PWM centre
pivot and then resumes planning. The map display follows the estimated robot
pose, route selection penalizes unnecessary detours while retaining obstacle
clearance, and the mapper consumes the IMU's asynchronously integrated yaw
delta instead of estimating every turn only from the latest rate sample. The
dashboard also reports acceleration deviation as a motion/vibration diagnostic;
it is not treated as position.
In v1.9.10, the occupancy grid recenters around the robot instead of clamping
its dead-reckoned position at a fixed buffer edge, which previously froze the
pose and stopped the map from updating on a long one-direction traverse. A new
stuck detector combines LD19 approach-progress, camera optical flow, IMU yaw
rate, and the Uno's `blocked` flag into independent motion evidence; when a
commanded drive keeps running with no corroborating evidence of real motion,
it tries a different LiDAR-checked maneuver instead of repeating one that
is not working, and reports a clear `STOP:STUCK_*_NEEDS_RESET` rather than
grinding the motors after a few failed attempts. It resumes automatically once
any source reports real motion again, including a manual reposition. The
escape state machine's turn-side scoring now uses the same body-width windowed
minimum as forward path selection instead of the single farthest ray in a
sweep, so a gap narrower than the chassis can no longer look like a viable
escape direction. The USB LSM6DS3's stationary-calibration acceptance bar is
moderately tighter than the GPIO MPU-6050's, reflecting its lower datasheet
noise. The on-screen robot dashboard and window title now show the running
version, not only the startup log line. This version's physical driving
behavior has not yet been confirmed on the robot; software-only verification
(compileall, pyflakes, targeted unit tests) is not a substitute for that.
The Pi launcher uses the available XWayland display and reapplies fullscreen
after the first dashboard frames so the LiDAR UI fills the connected screen.

It is not vehicle autonomy and is not robust room-scale SLAM. Do not run it
unsupervised, near stairs, pets, people, or property that can be damaged.
Details, firmware location, boot behaviour, and the one-command Pi update are
in [`pi3b/README.md`](pi3b/README.md#osoyoo-robot-mode-pi--ld19--camera--uno).

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
