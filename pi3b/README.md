# VisionFSD Pi 3B runtime

This is a **separate** Raspberry Pi 3B runtime derived from the design of the
desktop VisionFSD Pilot. It is deliberately not a direct port.

It preserves newest-frame capture, bounded asynchronous inference, a sticky
lane-aware lead target, and a target shown in both camera and world panels.
It intentionally removes OpenVINO GPU, neural road/lane/depth models,
ByteTrack, PyTorch, OpenGL, YouTube, and desktop-scale multi-object rendering.

Use **64-bit** Raspberry Pi OS (`aarch64`). Current LiteRT has an ARM64 wheel
for modern Pi OS/Python 3.13; the obsolete `tflite-runtime` package does not.

## Runtime contract

The primary neural detector is the official 4.6 MB quantized
**EfficientDet-Lite0 INT8** COCO model. The installer/updater downloads it from
TensorFlow Hub, verifies its pinned SHA-256, and keeps the included 4.2 MB
SSD-MobileNetV1 INT8 model as an automatic startup fallback. The active model
is named in the camera HUD and benchmark report.

The shared detector pass handles vehicles, pedestrians, traffic lights, and
stop signs without adding a second inference model. It keeps only one sticky
**lead vehicle**
(car, motorcycle, bus, or truck) in both the camera and world views. Lead
selection prefers a visibly near vehicle inside the detected ego lane; only
when no such vehicle exists can one near left/right-lane vehicle become the
target. Tiny horizon vehicles are rejected before tracking. Semantic vehicle
NMS, scale-aware ID association, and temporal class evidence reduce duplicate
cars, ID jumps, and car/bus/truck classification flicker. Temporal lane voting
requires sustained evidence before an existing target moves between front,
left, and right. A vehicle requires two consecutive detections before it
becomes the lead. The world view can also show confirmed pedestrians, traffic
lights, and stop signs. Pedestrians require four consecutive high-confidence,
human-shaped boxes; boxes mostly contained inside vehicles are rejected. Scene
objects disappear after two misses. Those extras never clutter the camera view.
This immediately makes the Pi
folder runnable. The desktop repository's OpenVINO/PyTorch artifacts are not
Pi runtime artifacts. `tools/export_pi_tflite.py` remains available for a
future, Pi-specific YOLO model once a Linux x86/macOS export environment is
available.

The bundled COCO model identifies **stop signs**, not arbitrary traffic-sign
types such as speed-limit signs. Traffic-light colour is not inferred.

The one-command deployment shape is:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/main/pi3b/install.sh | bash
```

The installer verifies both model downloads by SHA-256. The optional
`--model-url`/`--model-sha256` pair customizes only the SSD fallback; the
EfficientDet primary stays pinned unless its dedicated environment variables
are explicitly changed.
It intentionally does not require the optional `libatlas-base-dev` package,
which is unavailable on some current Raspberry Pi OS package sources.

The installer uses Git sparse checkout: it keeps only `pi3b/` and the exact
Uno firmware sketch needed by the robot runtime. Desktop OpenVINO models,
Windows scripts, CAD assets, and unrelated source files are removed from an
existing Pi checkout during installation/update.

To update an existing installation, preserving the release branch it was
installed from:

```bash
cd ~/visionfsd-pi && bash ./pi3b/update.sh
```

If an older updater aborts with `commit your changes or stash them`, use the
checkout-independent recovery command once:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/codex/pi3b-runtime/pi3b/recover-update.sh | bash
```

It saves tracked local edits in a named Git stash, installs the current Pi
release, and leaves the model and virtual environment in place. Normal future
updates can then use `bash ~/visionfsd-pi/pi3b/update.sh`.

## OSOYOO robot mode: Pi + LD19 + camera + Uno

The optional indoor robot runtime connects all four sensor/control parts:

```text
Pi USB webcam ──────────────> Pi (semantic person veto + display)
LD19 USB-UART ──────────────> Pi (360-degree range obstacles/local map)
Arduino Uno USB ────────────> Pi (serial commands/status)
front static ultrasonic ────> Uno (independent final forward-stop guard)
Uno motor shield ───────────> motors
```

**Yes:** connect the Uno's normal USB port directly to a Pi USB port. It
provides the serial link and, when the Pi is powered adequately, can power the
Uno's logic. The motors must remain on their own correctly rated battery pack;
do not try to power the motors from the Pi or its USB power bank.

Flash this separate sketch to the Uno first:

```text
~/visionfsd-pi/robot/firmware/visionfsd_pi_autonomy/visionfsd_pi_autonomy.ino
```

It is intentionally different from the earlier manual-drive sketch: the
ultrasonic sensor is static and front-facing, the servo is unused/detached to
avoid its continuous battery draw, and every motion command expires after
350 ms. The Pi sends bounded differential motor commands, allowing gentle
forward arcs instead of only straight/pivot motion. The Uno blocks forward
travel below 18 cm even if the Pi crashes or sends a bad command.
**Re-flash this sketch after each robot-firmware update.**

### Motor power and why PWM is not capped low

The shield's L298N bridge drops roughly 2 V, so a 7.9 V pack puts at most about
5.6 V across a motor at full duty. An earlier 105/255 cap therefore delivered
around 2.3 V. That is enough to spin a free wheel with the robot on blocks and
**not** enough to move the loaded chassis on a floor: the motor sits energised
and buzzing, the robot creeps, pauses and stutters, and the battery sags for no
useful work. Full 8-bit range is available, but the Pi uses a cautious direct
PWM ceiling of **118** in clear space and slows from there.

For the same reason the practical deadband is measured loaded, not in the air.
`MIN_MOVE_PWM` is 105: any commanded wheel value is either zero or above it,
because in between the motor only buzzes. A wheel deliberately dropped to zero
is how a tight arc is made.

The Uno applies a direct 20 ms PWM ramp from that floor to the commanded value.
It does not use a high-power kickstart or an on/off pulse train, because either
would make a small robot lurch or audibly stop-start.

### Speed is governed by measured clearance

`--speed` (default 118) is the ceiling used in **clear space only**. The planner
scales down from it in proportion to how far the robot's own body can actually
travel along the heading it has chosen, so it slows approaching an obstacle
rather than running flat out until a last-moment stop.

The usable band is narrow in PWM terms but wide in speed terms, because only
the voltage *above* the stall threshold does any work: roughly 105 PWM is a
slow crawl and 118 is a cautious cruise. Below about 105 the loaded chassis
stops moving entirely, so that is the floor.

Tune with two knobs. `--speed` sets the ceiling. `--min-move-pwm` (default 105)
is the lowest PWM that turns a loaded wheel; raise it if the robot buzzes
without moving, lower it if even the crawl is too quick. `MIN_MOVE_PWM` is an
estimate for this drivetrain, not a measurement of yours.

### Corridor profile: how it steers around things

Rather than five fixed sectors, the LD19 returns are swept into a **body-inflated
corridor profile**: for each of 73 candidate headings the planner computes how
far a rectangle of the robot's own width can travel before anything enters its
path. That is what lets it say "there is a 0.5 m gap 20 degrees to the right,
3 m deep" — something a five-sector summary structurally cannot express, and
the reason it can now curve around an object instead of treating a whole side
as blocked.

The chosen heading comes from that profile, preferring straight ahead when it
has at least 1.1 m of clear travel, with a switch margin so scan noise cannot
make it weave between two near-tied options. The selected corridor receives
arc-lead compensation because a differential-drive chassis cannot instantly
assume a new straight heading. Avoidance uses a forward arc with both wheels
powered, never a fast counter-rotating pivot. The outer wheel receives bounded
steering headroom while the inside wheel remains at or above its loaded floor.

The runtime writes one `NAV` telemetry line per second to
`pi3b/logs/robot.log`. If commanded and Uno-reported PWM drop to zero during a
hiccup, the same line identifies the safety gate that stopped it. If both stay
nonzero while the wheels cut out, the fault is in motor power or wiring. A 9 V
PP3-style battery is not a usable sustained motor supply here; its voltage can
look normal at idle and collapse under motor current.

Run a supervised first test on blocks, wheels free, then on an empty floor:

```bash
cd ~/visionfsd-pi && bash ./pi3b/run_robot.sh
```

At boot the robot runtime starts in a **25-second STOP standby**. It will not
move during that interval. Afterwards its authority order is fixed:

1. A stale Uno, LD19, or webcam stops the robot; it will not drive blind.
2. A confirmed person in the camera's forward path stops it. Camera inference
   remains a safety gate, but camera frames are not rendered in the robot
   display to reduce Pi 3B display work.
3. The LD19 begins a direction-locked forward arc when the straight inflated
   corridor drops below 1.1 m. Steering can use up to a 28-PWM wheel split,
   both motors stay above the loaded-wheel stall region, and steering changes
   are slew-limited. A high-confidence close return is retained even when a
   thin obstacle occupies only one angular bin; distant weak speckle is ignored.
4. At 52 cm of body-path clearance (or an ultrasonic return below 30 cm), it
   runs a finite recovery sequence: a short LiDAR-cleared reverse curve with
   both wheels driven, a slow one-wheel reverse turn toward the best full
   body-width LiDAR corridor, then a short forward commit and immediate return
   to live corridor planning. A calibrated MPU-6050 releases the turn after
   measured yaw reaches the clear corridor and hard-limits each turn to
   88 degrees; a 2.4-second bound applies if IMU yaw is unavailable. It may try
   the opposite side once. It stops as `STOP:BOXED_IN` when neither bounded
   attempt has a safe side/rear path, and automatically rechecks materially
   changed geometry.
5. Uno ultrasonic hard-stop (under 18 cm) always wins and blocks forward
   motor commands even if the Pi fails.
6. A live, calibrated MPU-6050 supplies measured yaw rate to the local mapper
   and recovery controller, and progressively removes steering split above
   38 deg/s, reaching zero additional split at 55 deg/s. If the IMU is absent
   or stale, navigation continues with bounded time/command-yaw fallback instead
   of refusing to move.

This keeps roles separate: LD19 geometry chooses an open direction, the camera
prevents movement toward confirmed people, and the Uno enforces the final
close-range stop. A camera classification never overrides measured range data.

The motor battery in the described 9 V-style holder is not adequate for
autonomous testing: voltage sag can still cause buzzing, stuttering, or a
controller reset regardless of this software. Replace it with the kit's rated
7.4 V 2x18650 pack or a 6xAA NiMH pack before testing this runtime on the floor.

The runtime no longer translates steering into legacy `L`/`R` commands while
waiting for `CAPS DRIVE`; those commands are fast counter-rotating pivots. It
holds STOP unless the dashboard reports `UNO DIFFERENTIAL`. The Pi retries the
capability request every 0.5 seconds until it receives that exact response.
The LiDAR-only dashboard shows commanded PWM, Uno-reported actual PWM, the Uno
ultrasonic `blocked` flag, and `MPU-6050 CALIBRATING/LIVE/STALE` so a software
STOP is distinguishable from a motor-power or sensor problem.

Camera startup defaults to `auto`. The runtime tries stable V4L by-id paths and
camera indexes 0 through 7. If no webcam currently delivers frames, the LiDAR
dashboard remains open in `CAMERA STALE` safe-STOP mode and retries instead of
terminating the complete robot runtime.

`run_robot.sh` holds a process lock before opening USB devices. This prevents
XDG and compositor autostart entries from launching two robot processes that
compete for the same webcam and serial ports. Linux V4L device paths are opened
directly through the V4L2 backend rather than GStreamer. Robot-mode capture is
320x240 at 15 FPS because the camera is a safety veto, not a displayed steering
sensor; this reduces Pi 3B USB buffer and CPU pressure.

### MPU-6050 yaw sensing

The MPU-6050 is connected directly to Pi I2C bus 1 at address `0x68`:

| MPU-6050 | Pi physical pin |
| --- | --- |
| VCC | 1 (3.3 V) |
| SDA | 3 (GPIO2/SDA1) |
| SCL | 5 (GPIO3/SCL1) |
| GND | 6 |
| AD0 | 9 (GND, selects `0x68`) |

The installed board is flat with components up and rotated 180 degrees from
the robot frame: its pin-header edge faces forward and its `MPU-6050` text
edge faces rearward. The runtime therefore defaults to
`--imu-mount-yaw-deg 180`, which reverses sensor X/Y while retaining Z. Override
only if the physical mount changes:

```bash
export VISIONFSD_IMU_MOUNT_YAW_DEG=180
```

The first stationary seconds of the existing 25-second standby calibrate gyro
bias. Keep the chassis still until the dashboard changes from
`MPU-6050 CALIBRATING` to `MPU-6050 LIVE`. Calibration rejects samples with
excessive motion. The IMU is advisory: disconnecting it changes the dashboard
to command-yaw fallback rather than creating a no-motion boot failure.

The gyro improves short-term turn measurement and smooths excessive yaw. Its
accelerometer is not integrated into position because chassis vibration,
gravity error, and the front/side mounting offset would create rapid drift.
The MPU-6050 has no magnetometer, so it cannot provide absolute heading.

### LiDAR + IMU SLAM-lite local map

The LD19-only panel uses a rolling **SLAM-lite** local map. It keeps a 6 m local
occupancy sketch at 2.5 cm per cell, integrates every valid current scan return
with vectorized NumPy operations, and compares successive scans in 360
one-degree angular bins. Bright current-scan points remain distinct from the
fading dead-reckoned history. Metre range rings make nearby geometry easier to
read.
A heading correction is applied only when that comparison has enough
non-ambiguous support. Between accepted LD19 matches, a live MPU-6050 supplies
measured yaw rate; if it is unavailable, the mapper uses conservative
commanded-motion prediction. This increases useful local detail without
turning a map estimate into a motor-control input.

This is intentionally advisory. The current LiDAR sectors remain the only
range input to steering and safety—the map cannot command the motors. Without
wheel encoders or an absolute heading/position reference, it is not global
localization, true metric SLAM, loop closure, or a guarantee of room coverage.
It is useful for a steadier local map and for showing when LiDAR heading
agreement is weak.

The LD19's 0-degree direction must physically point forward. If your mount is
rotated, set its correction before launching, for example:

```bash
export VISIONFSD_LIDAR_FRONT_OFFSET_DEG=90
```

Use `-90`, `90`, or `180` only when that matches the actual mounting rotation;
the dashboard's `F`, `FL`, and `FR` readings make a wrong orientation visible.

### Boot automatically

The default installer creates this Pi Desktop autostart file:

```text
~/.config/autostart/visionfsd-robot.desktop
```

With Raspberry Pi OS **Desktop** configured to auto-login, connecting power
starts the visual robot runtime after the graphical desktop appears; it still
holds STOP for 25 seconds. Use `--no-robot-autostart` with `install.sh` if you
do not want that. Raspberry Pi OS Lite has no graphical autostart session, so
it needs a separate headless service and does not show the visualizer.

The short normal update command is:

```bash
bash ~/visionfsd-pi/pi3b/update.sh codex/pi3b-runtime
```

It preserves the Pi release branch, updates only the sparse Pi checkout, and
does not download Windows/CAD files.

## Run from this checkout

```bash
cd pi3b
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
bash ./sync_primary_model.sh
.venv/bin/python visionfsd_pi.py --camera 0 --model models/vehicle_efficientdet_lite0_int8.tflite \
  --fallback-model models/vehicle_ssd_mobilenet_v1.tflite
```

The bottom of every visual screen has large touchscreen controls: **Quit**,
**Screen 1** (world), **Screen 2** (camera), and **Screen 3** (split).
They also work with a regular mouse. `1`, `2`, `3` select world, camera,
split; `S` saves a screenshot; `Q`/`Esc` quits. The world panel is a low-cost
OpenCV pseudo-3D view with a centred ego vehicle and two lane boundaries.
Both visual panels show the version from `pi3b/VERSION`.
The lanes glow bright white only when its low-rate, classical lane pass has a
fresh, geometrically valid left-and-right pair; otherwise they stay dim. The
same perspective lane geometry places vehicles into ego, left, or right lanes
in the world panel. It is not desktop OpenGL and it is not a driving
measurement.

## Performance

`--fps 25` is the Pi 3B display target; `20` remains available for thermally or
power-limited installations. Detector work stays in a newest-frame background
worker, so a slower neural pass cannot queue old frames or deliberately lower
camera/display resolution. The HUD reports display and detection FPS
separately: a 25 FPS display is not claimed as 25 FPS inference. Acceptance
requires a sustained Pi benchmark with no thermal throttling. If CPU inference
does not sustain the goal after input/model tuning, add a USB accelerator.

The default three LiteRT threads are the first performance candidate on the
Pi's four-core CPU, while OpenCV remains single-threaded. All scene classes
reuse the same EfficientDet result. Input and output tensor access avoids
redundant copies, the 25-result EfficientDet postprocessor is bounded, the
256 px lane pass runs in a single-slot background worker, and split view
renders 44% fewer pixels without changing the 640x480 camera input.
Missed display deadlines reset immediately instead of burst-rendering catch-up
frames. Do not claim 25 FPS **detection** unless the HUD's `DETECT` rate reaches
it during a sustained physical-Pi run.

Google's LiteRT guidance says thread count must be benchmarked with the whole
application because additional inference threads can contend with other work.
Raspberry Pi also documents CPU throttling near its thermal limit. Validate a
10-minute run with adequate cooling and power before treating any result as
sustained performance.

## Desktop provenance

| Pi component | Desktop reference | Adaptation |
|---|---|---|
| `LatestCamera` | `src/webcam_capture_proc.py` | Linux V4L2, newest frame only |
| `AsyncDetector` | `src/object_perception.py` | LiteRT, one pending frame |
| `TargetSelector` | `src/visionfsd_3d.py` | small tracker, scale-aware stable IDs, temporal class fusion, strict one-vehicle lead policy |
| range/bearing | `src/visionfsd.py` | car-width pinhole estimate only |
| split display | `src/visionfsd_3d.py` | OpenCV pseudo-3D, no OpenGL |

## Validation

```bash
python3 -m unittest discover -s tests -v
./run.sh --camera 0 --test-seconds 60 --benchmark-report logs/benchmark.json
```

The synthetic tests cover consecutive confirmation, rapid false-object expiry,
lane-aware target selection, label and box stability, lane extraction, version
rendering, reduced split-view size, and detector output decoding. The benchmark
records display/detection rates plus preprocess, invoke, postprocess, render,
capture, and end-to-end timings.

## LD19 LiDAR visualizer

The optional LD19 tool is a separate, **read-only** 360-degree point-cloud
viewer. It reads the LD19's documented 230400-baud UART stream through its USB
serial adapter and never sends motor or configuration commands.

It keeps only the latest return for each one-degree direction and expires it
after **300 ms** by default, rather than drawing a long history trail. New
obstacles therefore appear on the next physical scan and removed obstacles
clear quickly. The LD19 itself rotates at about 10 Hz, so its physical scan
period still sets a lower latency limit of roughly 100 ms. Conservative nearby
return clusters suppress isolated speckle and mark geometric obstacles with a
range estimate. They are not car, pedestrian, or sign classifications: one
horizontal 2D LiDAR cannot make that distinction.

Connect the LD19's supplied communication cable to its supplied USB serial
adapter, then connect that adapter to a Raspberry Pi USB port. Run:

```bash
cd ~/visionfsd-pi/pi3b
bash ./run_lidar.sh
```

It automatically selects a single USB serial adapter. If several serial
adapters are connected, specify the Pi port explicitly, normally
`/dev/ttyUSB0`:

```bash
bash ./run_lidar.sh --port /dev/ttyUSB0
```

For unusual reflective, dark, or close-range environments, tune the viewer
without changing code:

```bash
bash ./run_lidar.sh --persistence 0.25 --min-confidence 5 --min-range-mm 60
```

Press `Q` or `Esc` to close the viewer. The coloured dots are recent range
returns in the horizontal scan plane; `FRONT` is the LD19 zero-angle direction.
`CRC ERRORS` should remain at zero or very low. A missing serial port means
the USB-UART adapter or its operating-system driver needs attention, not that
the visualizer needs a different baud rate.
