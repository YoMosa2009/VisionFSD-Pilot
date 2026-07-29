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
corridor profile**: for each of 37 candidate headings the planner computes how
far a rectangle of the robot's own width can travel before anything enters its
path. That is what lets it say "there is a 0.5 m gap 20 degrees to the right,
3 m deep" — something a five-sector summary structurally cannot express, and
the reason it can now curve around an object instead of treating a whole side
as blocked.

The chosen heading comes from that profile, preferring straight ahead, with a
switch margin so scan noise cannot make it weave between two near-tied options.
Headings within 50 degrees become a smooth arc; anything wider squares up with
a brief pivot first. Turning output is normalised so the outer wheel never
exceeds the governed speed, because scaling a turn *up* makes every corner
faster than driving straight.

If the robot still stutters on carpet, raise `--speed` before suspecting the
planner. A 9 V PP3 alkaline is not a usable motor supply here: its internal
resistance collapses under an amp of motor current. Use the kit's 2x18650
holder (7.4 V) or 6x AA NiMH.

Run a supervised first test on blocks, wheels free, then on an empty floor:

```bash
cd ~/visionfsd-pi && bash ./pi3b/run_robot.sh
```

At boot the robot runtime starts in a **25-second STOP standby**. It will not
move during that interval. Afterwards its authority order is fixed:

1. A stale Uno, LD19, or webcam stops the robot; it will not drive blind.
2. A confirmed person in the camera's forward path stops it. The camera draws
   its confirmed-person boxes in the robot display.
3. The LD19 begins a direction-locked forward arc away from a central obstacle
   below 86 cm. Turning is capped to a small PWM split, both motors stay above
   the loaded-wheel stall region, and an isolated LiDAR speckle cannot change
   the corridor plan. A clear one-metre forward corridor is preferred over a
   merely longer side corridor so the robot keeps making forward progress.
4. At 42 cm (or an ultrasonic return below 22 cm), it makes one short,
   LiDAR-cleared reverse curve with both wheels driven. It does not pivot in
   place. If the rear is not LiDAR-clear, or that bounded recovery does not
   restore front clearance, it stops and latches that stop instead of repeating
   the reverse maneuver. Once clear, it briefly commits to the selected escape
   side so new scan noise cannot immediately send it back toward the obstacle.
5. Uno ultrasonic hard-stop (under 18 cm) always wins and blocks forward
   motor commands even if the Pi fails.

This keeps roles separate: LD19 geometry chooses an open direction, the camera
prevents movement toward confirmed people, and the Uno enforces the final
close-range stop. A camera classification never overrides measured range data.

The motor battery in the described 9 V-style holder is not adequate for
autonomous testing: voltage sag can still cause buzzing, stuttering, or a
controller reset regardless of this software. Replace it with the kit's rated
7.4 V 2x18650 pack or a 6xAA NiMH pack before testing this runtime on the floor.

The runtime no longer translates steering into legacy `L`/`R` commands while
waiting for `CAPS DRIVE`; those commands are fast counter-rotating pivots. It
holds STOP unless the dashboard reports `UNO DIFFERENTIAL`.

### LiDAR SLAM-lite local map

The LD19 panel now uses a small rolling **SLAM-lite** local map. It keeps a
6 m local occupancy sketch at 5 cm per cell, integrates the latest useful
indoor returns, and compares successive scans in 90 four-degree angular bins.
A heading correction is applied only when that comparison has enough
non-ambiguous support; otherwise it stays with conservative commanded-motion
prediction. This increases useful local detail without turning a map estimate
into a motor-control input.

This is intentionally advisory. The current LiDAR sectors remain the only
range input to steering and safety—the map cannot command the motors. Without
wheel encoders or an IMU, it is not global localization, true metric SLAM,
loop closure, or a guarantee of room coverage. It is useful for a steadier
local map and for showing when LiDAR heading agreement is weak.

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
