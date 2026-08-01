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
Pi USB webcam ──────────────> Pi (optical-flow pose cue + live health gate)
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
2. Robot mode does not run object or person detection. Low-resolution webcam
   optical flow provides a bounded non-IMU yaw cue during turns and reduces
   false commanded map translation when a high-confidence textured view shows
   no movement. It is not odometry. Camera frames are not rendered, keeping
   camera/display work small on Pi 3B.
3. During standby, the mapper distinguishes observed free space from unknown
   space and persistent obstacle returns. A frontier planner selects a reachable
   unexplored boundary, plans around inflated obstacles, and supplies a
   persistent look-ahead route. The cached route advances its waypoint every
   control cycle instead of waiting for the next full replan. When all current
   frontiers are exhausted, it patrols the least-visited reachable mapped space.
4. The LD19 begins a direction-locked forward arc when the straight inflated
   corridor drops below 1.1 m. Steering can use up to a 28-PWM wheel split,
   both motors stay above the loaded-wheel stall region, and steering changes
   are slew-limited. A very-high-confidence close return is retained even when
   a thin obstacle occupies only one angular bin; weaker isolated speckle is ignored.
   The frontier heading only biases among currently safe full-body corridors.
5. A close straight return does not trigger recovery while another body-width
   forward corridor is open; the controller follows that corridor as a
   continuous differential arc. Only when every candidate corridor is below
   28 cm (or ultrasonic is below 22 cm) does it run a finite recovery sequence:
   a short LiDAR-cleared reverse curve with
   both wheels driven, a slow one-wheel reverse turn toward the best full
   body-width LiDAR corridor, then a short forward commit and immediate return
   to live corridor planning. A calibrated USB LSM6DS3 or GPIO MPU-6050 releases the turn after
   measured yaw reaches the clear corridor and hard-limits each turn to
   88 degrees; a 2.4-second bound applies if IMU yaw is unavailable. It may try
   the opposite side once. It stops as `STOP:BOXED_IN` when neither bounded
   attempt has a safe side/rear path, and automatically rechecks materially
   changed geometry. Rear safety is a robot-width swept corridor rather than
   the closest point in a broad rear sector. When that corridor is clear but
   both sides are ambiguous, one short straight reverse search obtains a new
   view before declaring itself boxed in. If reversing is blocked but one
   complete turn sweep has at least 34 cm clearance, it begins the same bounded
   slow turn directly instead of giving up despite that opening.
6. Uno ultrasonic hard-stop (under 18 cm) always wins and blocks forward
   motor commands even if the Pi fails.
7. A live, calibrated USB LSM6DS3 or GPIO MPU-6050 supplies measured yaw rate to the local mapper
   and recovery controller, and progressively removes steering split above
   38 deg/s, reaching zero additional split at 55 deg/s. If the IMU is absent
   or stale, navigation continues with command-predicted yaw corrected by
   successive LD19 scans. Recovery turns use that corrected pose heading when
   available, retaining the same 2.4-second hard timeout as the final bound.

This keeps roles separate: LD19 geometry chooses an open direction, the camera
supplies only bounded optical-flow pose cues, and the Uno enforces the final
close-range stop. Camera output never overrides measured LiDAR or ultrasonic
safety.

The motor battery in the described 9 V-style holder is not adequate for
autonomous testing: voltage sag can still cause buzzing, stuttering, or a
controller reset regardless of this software. Replace it with the kit's rated
7.4 V 2x18650 pack or a 6xAA NiMH pack before testing this runtime on the floor.

The runtime no longer translates steering into legacy `L`/`R` commands while
waiting for `CAPS DRIVE`; those commands are fast counter-rotating pivots. It
holds STOP unless the dashboard reports `UNO DIFFERENTIAL`. The Pi retries the
capability request every 0.5 seconds until it receives that exact response.
The full-screen LiDAR-only dashboard shows commanded PWM, Uno-reported actual PWM, the Uno
ultrasonic `blocked` flag, `IMU LSM6DS3 USB CALIBRATING/LIVE/STALE`, frontier/patrol
mode, target bearing/range, frontier count, observed-map coverage, and latest
planner time. It also reports camera-flow confidence. The occupancy grid uses
about 2.1 cm cells, up to 240 current-scan free-space rays, and useful LD19
returns up to 5.8 m where map bounds permit. These software changes do not
increase the LD19's physical range or create true odometry.

Camera startup defaults to `auto`. The runtime tries stable V4L by-id paths and
camera indexes 0 through 7. If no webcam currently delivers frames, the LiDAR
dashboard remains open in `CAMERA STALE` safe-STOP mode and retries instead of
terminating the complete robot runtime. Optical-flow pose assistance assumes the
webcam is rigidly mounted and faces forward; low-confidence flow is ignored.

`run_robot.sh` holds a process lock before opening USB devices. This prevents
XDG and compositor autostart entries from launching two robot processes that
compete for the same webcam and serial ports. Linux V4L device paths are opened
directly through the V4L2 backend rather than GStreamer. Robot-mode capture is
320x240 at 15 FPS. Optical flow uses a 160x120 grayscale copy and the camera
image is not displayed; this limits Pi 3B USB, display, and CPU pressure.

### Automatic USB LSM6DS3 yaw sensing

The preferred IMU path is:

```text
LSM6DS3 STEMMA QT -> MCP2221A I2C -> MCP2221A USB-C -> Pi USB-A
```

Keep the board rigid, flat, and component-side up. The normal updater installs
the Linux prerequisites and Python USB transport, writes persistent MCP2221A
USB permissions, and prevents the optional kernel MCP2221 driver from competing
with the userspace transport. That system setup is recorded once and skipped on
later updates. Every robot boot still opens and validates the sensor because an
IMU cannot remain open across a power cycle.

At startup the runtime automatically checks both normal LSM6DS3 I2C addresses,
`0x6A` and `0x6B`, and accepts the sensor only when its identity register returns
`0x69`. No manual `modprobe`, I2C scan, or launch command is required. During the
25-second stationary standby, the dashboard should change from
`IMU LSM6DS3 USB CALIBRATING` to `IMU LSM6DS3 USB LIVE`.

### GPIO MPU-6050 fallback

The previous MPU-6050 remains supported as an automatic fallback on Pi I2C bus
1 at address `0x68`:

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
bias. Calibration rejects samples with excessive motion. USB LSM6DS3 is
preferred, GPIO MPU-6050 is second, and `LD19+COMMAND POSE ACTIVE` remains the
automatic fallback if neither IMU is usable. Missing IMU hardware never creates
a no-motion boot failure.

The gyro improves short-term turn measurement and smooths excessive yaw. Its
accelerometer is not integrated into position because chassis vibration,
gravity error, and the front/side mounting offset would create rapid drift.
Neither supported IMU has a magnetometer, so neither provides absolute heading.

### LiDAR + IMU exploration map

The LD19-only panel keeps an 8 m occupancy map at 2.5 cm per cell. Every valid
scan marks both obstacle endpoints and the observed free ray leading to each
endpoint. Repeated free observations clear stale hit evidence; persistent hits
remain obstacles. Bright current-scan points remain distinct from mapped
history, while dark known-free cells are distinguishable from unknown space.
Metre range rings make nearby geometry easier to read.

The frontier explorer inflates obstacles by the robot body and safety margin,
finds the free-space component connected to the robot, clusters reachable
free/unknown boundaries, and runs bounded A* to the selected target. It guides
the local planner along a cached look-ahead route whose waypoint advances every
control cycle. If map inflation temporarily places the estimated pose just
outside free space, planning reconnects to nearby known free space while live
LiDAR remains authoritative. If no reachable frontier remains, it patrols
low-visit mapped cells to expose missed openings. Full replans run every 0.6
seconds with a 45 ms A* budget. The current motor decision is sent before that
advisory search, and a timed-out replan retains the last safe route rather than
pausing the robot or dropping the Uno's 350 ms watchdog.

A heading correction is applied only when successive 360-bin LD19 scans have
enough non-ambiguous support. Between accepted matches, a live USB LSM6DS3 or GPIO MPU-6050 supplies
measured yaw rate; if it is unavailable, the mapper uses conservative
commanded-motion prediction. Bounded scan-to-map correlation also corrects
small translation errors when a moving scan uniquely agrees with established
obstacle geometry.

The estimated map supplies exploration intent, not safety permission. Current
LD19 body-width corridors remain the range authority for steering, and the Uno
ultrasonic remains the final forward hard stop. Without wheel encoders, loop
closure, or an absolute position reference, this is not true metric SLAM and
cannot guarantee complete coverage or recovery from accumulated pose drift.

All mapping and exploration features remain enabled without either IMU:
free/occupied/unknown mapping, translation correlation, frontier detection,
inflated-grid A*, persistent waypoints, patrol, live-corridor overrides, and
reverse/turn/replan recovery. The degraded yaw source is shown as
`COMMAND+LD19`; it is less accurate during feature-poor or rapidly changing
scenes than live IMU yaw, but it does not disable autonomous movement.

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
