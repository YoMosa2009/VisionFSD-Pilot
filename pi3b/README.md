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
avoid its continuous battery draw, motor PWM is capped at 105, and every
motion command expires after 350 ms. The Pi sends bounded differential motor
commands, allowing gentle forward arcs instead of only straight/pivot motion.
The Uno blocks forward travel below 18 cm even if the Pi crashes or sends a
bad command. **Re-flash this sketch after each robot-firmware update.**

The ultrasonic sensor is sampled on its own fixed 60 ms cadence, filtered with
a 3-sample median, and must read short **twice in a row** before it gates
forward travel. Sampling no longer happens inside a `DRIVE` command, so a motor
update is never delayed by an echo timeout. Reverse and pivots are never gated,
so the Pi can always drive out of a close-range situation.

**Re-flashing is required for v1.0.0, not optional.** The planner makes tight
turns by idling the inner wheel, so it sends commands such as `DRIVE 105 0`.
Firmware before v1.0.0 only gated a command when *both* wheels were positive,
so on old firmware those close-quarters turns would drive past the 18 cm
ultrasonic guard entirely. The current sketch gates any command with a forward
component.

Run a supervised first test on blocks, wheels free, then on an empty floor:

```bash
cd ~/visionfsd-pi && bash ./pi3b/run_robot.sh
```

At boot the robot runtime starts in a **25-second STOP standby**. It will not
move during that interval. Afterwards its authority order is fixed:

1. A stale Uno, LD19, or webcam stops the robot; it will not drive blind.
2. A confirmed person in the camera's forward path stops it. The camera draws
   its confirmed-person boxes in the robot display.
3. The LD19 chooses where to drive, using the robot's own body dimensions.
4. Uno ultrasonic hard-stop (under 18 cm) always wins and blocks forward
   motor commands even if the Pi fails.

This keeps roles separate: LD19 geometry chooses an open direction, the camera
prevents movement toward confirmed people, and the Uno enforces the final
close-range stop. A camera classification never overrides measured range data.

### How the navigation works

The planner treats the robot as a rectangle rather than a point. For each of
106 candidate headings across a 210-degree fan it computes how far a body of
the configured width can travel before anything enters the swept corridor. That
single geometric test is what lets it enter gaps it actually fits through,
refuse gaps it does not, and respect a chair leg that occupies one degree of
the scan.

Among headings with real room, it picks the best score from: usable clear
distance (which **saturates**, so "far enough" beats "roomiest"), a strong
preference for going straight, a penalty for turning the same way it has been
turning, a penalty for world headings it has already spent time driving, a
small bonus for unmeasured directions, and a penalty for confirmed people. The
saturation matters: without it the robot rotates toward whatever is roomiest
and gently spins in the middle of a room instead of crossing it.

Speed is proportional to the clear distance ahead. Headings within 42 degrees
are taken as a smooth differential arc; anything wider first turns on the spot
toward a measured heading target, then commits to driving for 0.7 s so it
cannot shuffle between turns without ever moving.

When nothing ahead is drivable it runs an escape ladder rather than stopping:
curved reverse burst, then a quarter-turn pivot, then a short straight nudge
backwards, then a tighter pivot, and only then HOLD. HOLD is retried every
cycle, so it is a state and not a dead end.

Because the chassis has no encoders and no IMU, heading change is estimated by
correlating successive LiDAR range profiles. That measured turn rate feeds the
anti-orbit penalty and detects a **stall** — the Pi commanding motion while the
world does not change, which is what a sagging motor battery looks like. A
stall triggers the same escape ladder.

The ultrasonic sensor is cross-checked against the LiDAR. If it keeps reporting
something much closer than the LiDAR can see, it is marked `UNTRUSTED` on the
dashboard and ignored for planning; the Uno's independent 18 cm hard stop is
unaffected. When trusted it may only *shorten* travel within its own narrow
cone. It can no longer trigger a turn, which is what previously trapped the
robot in a pivot loop in front of an obstacle.

### Tell it its own size

Defaults are the measured chassis: 0.14 m wide, 0.15 m long, LD19 assumed
centred. If yours differs, measure the widest point with the wheels included
and set it before launching, or gap choices go wrong in both directions:

```bash
export VISIONFSD_ROBOT_WIDTH_M=0.14
export VISIONFSD_ROBOT_LENGTH_M=0.15
export VISIONFSD_LIDAR_OFFSET_M=0.0
export VISIONFSD_SAFETY_MARGIN_M=0.040
```

`VISIONFSD_LIDAR_OFFSET_M` is how far the LD19 sits ahead of the middle of the
robot, negative if behind. The startup line prints the resulting corridor
half-width and pivot radius.

### Scan-matching SLAM

Pose comes from matching each LiDAR revolution against the map built from
previous revolutions, using a chamfer score over an OpenCV distance transform.
Commanded PWM is only the centre of the search window; the match decides the
answer, which is why the estimate survives the motor battery sagging.

It is genuinely simultaneous localization and mapping, and just as genuinely a
small one: **no loop closure and no pose-graph optimization.** Error creeps and
is never corrected by revisiting a place, so treat the map as a good sketch of
the room the robot is in now, not a survey. It is deliberately confined to
advisory work. The occupancy grid never reaches the corridor geometry or the
escape ladder; obstacle avoidance always runs on the live scan. A wrong pose
therefore makes the robot explore badly, never drive into furniture.

Two failure handlers matter in practice. A match residual above 13 cm marks the
pose `SEARCHING` on the dashboard, freezes map updates, and drops the
exploration bias to zero. If that persists for about three seconds the tracker
throws the map away and rebuilds around the current position, rather than
dead-reckoning itself out of the room.

Known-free cells that touch unknown space form **frontiers**, and their
direction becomes a scored preference in the planner. That is what turns
"drives around safely" into "works through the room".

Disable the whole subsystem with `--no-slam` if you want the older purely
reactive behaviour.

### What the camera adds

The LD19 measures one horizontal plane at its mounting height, so a shoe, a
cable, or a book is invisible to it. The camera covers that gap with a floor
guard: it learns the floor's appearance from the strip directly ahead and flags
sustained off-colour clutter low in the frame.

Without a calibrated camera height and tilt there is no honest way to turn that
into metres, so it never produces a distance or a bearing and never steers. Its
only authority is a forward speed cap, shown as `FLOOR n%` and `LOW-OBSTACLE`
on the dashboard. A patterned rug can still trigger it; `--no-low-obstacle-guard`
turns it off. Confirmed people remain a full stop, and their bearing biases
steering away before a stop becomes necessary.

### Reading the dashboard

The right panel is the SLAM occupancy map: blue-ish cells are measured
obstacles, grey-green is confirmed free space, dark is unknown, and the trail
is where the robot believes it has been. The green/blue fan is the per-heading
body-inflated travel limit and the yellow arrow is the chosen heading.

The header lines carry planner state and reason; LD19 liveness, point count,
spin rate and CRC error count; forward and rear clearance; ultrasonic trust;
measured yaw rate, accumulated turn and recovery count; SLAM state, residual
and cost in milliseconds; frontier bearing and weight; and floor coverage.

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

The short normal update command is still:

```bash
bash ~/visionfsd-pi/pi3b/update.sh
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

The robot navigation tests run without any hardware and cover the body-inflated
corridor test (including a gap one robot accepts and a wider robot rejects),
thin-obstacle survival, rear clearance, LiDAR yaw estimation, stall detection,
ultrasonic distrust, the anti-orbit penalty, the escape ladder, and the motor
deadband. Several encode field failures directly: a close ultrasonic reading
with clear LiDAR must still make forward progress, a dead end must produce a
reverse rather than a stop, and an open room must not weave.

The SLAM tests cover occupancy integration, the distance field, map recentring,
frontier direction, recovery of a known translation and rotation, sweep
deskewing, divergence detection, and the map restart. The camera guard is
tested on a plain floor and on a planted object.

None of this is a hardware measurement. A closed-loop simulation of the control
law in a synthetic room is used to check behaviour that unit tests cannot:
coverage, absence of orbiting, and pose error against ground truth including a
modelled intermittent 55% drivetrain slip. Simulated results do not transfer
directly to a real floor with real wheel slip and real LiDAR noise.

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
