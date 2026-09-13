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

### Automatic updates on boot

`run_robot.sh` checks for an update before starting the runtime. From v1.9.17:

- Fetch retries up to three times, with a default 12-second timeout per attempt
  and two seconds between attempts, to allow the boot network to become ready.
- The branch comes from `.install-ref`; a missing file defaults to this robot's
  deployment branch, `codex/pi3b-runtime`.
- Permission-only working-tree changes from installation do not block updates.
  Staged or unstaged tracked content edits do block them and remain untouched.
  Untracked files are preserved; checkout conflicts abort the update.
- Updates use the exact fetched commit, compile its Python files before checkout,
  and restart the launcher once while retaining the process lock. The launcher
  and manual updater parse their complete shell bodies before replacing files.
- Boot updates do not run sudo, pip or model downloads. A changed requirements
  file requires the normal manual `update.sh` command, and is reported explicitly.
- Offline or rejected updates keep the installed code. A post-checkout failure
  rolls back. A failed rollback withholds startup and records `update-blocked`;
  a successful manual update clears that marker.

Read `pi3b/logs/update-status.txt` for the latest result, installed commit and
reason for any skipped update. `pi3b/logs/robot.log` also records the startup
version and commit. A successful GitHub push does **not** verify Pi installation.
Set `VISIONFSD_AUTO_UPDATE=0` to disable checks, or
`VISIONFSD_AUTO_UPDATE_TIMEOUT_S` to change the fetch timeout (1�30 seconds).

For a Pi still running v1.9.14, perform this recovery once while the robot is
stopped, then reboot. It uses the current recovery script rather than the
possibly broken updater already installed:

```bash
curl -fsSL https://raw.githubusercontent.com/YoMosa2009/VisionFSD-Pilot/codex/pi3b-runtime/pi3b/recover-update.sh -o /tmp/visionfsd-recover.sh &&
  bash /tmp/visionfsd-recover.sh && sudo reboot
```

Then confirm the dashboard version and inspect:

```bash
cat ~/visionfsd-pi/pi3b/VERSION
cat ~/visionfsd-pi/pi3b/logs/update-status.txt
git -C ~/visionfsd-pi rev-parse HEAD
tail -n 100 ~/visionfsd-pi/pi3b/logs/robot.log
```

## OSOYOO Model 3 robot mode

The supported physical robot configuration is intentionally specific:

| Part | Connected role |
|---|---|
| OSOYOO Model 3 kit | Differential-drive chassis, front static ultrasonic sensor, motor shield, and motors |
| Raspberry Pi 3B | Runs the Pi planner, mapper, visualizer, and USB device discovery |
| LD19 | 360-degree horizontal range sensing through its USB-UART adapter |
| LSM6DS3 | Gyro/accelerometer, connected only through the MCP2221A USB-I2C adapter |
| Arduino Uno | Runs `robot/firmware/visionfsd_pi_autonomy/visionfsd_pi_autonomy.ino` and applies bounded differential motor commands |

The Pi deployment is a sparse checkout containing only `pi3b/` and
`robot/firmware/visionfsd_pi_autonomy/`. The desktop runtime, CAD, and manual
drive files are not deployed to the robot.

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

The chosen heading comes from that profile. Each candidate is scored by the
minimum clearance across an 11-degree opening, so one long ray between nearby
objects cannot masquerade as a usable route. It prefers straight ahead when
that broad opening has at least 1.25 m of clear travel, with a switch margin so
scan noise cannot make it weave between two near-tied options. The selected corridor receives
arc-lead compensation because a differential-drive chassis cannot instantly
assume a new straight heading. Avoidance uses a forward arc with both wheels
powered, never a fast counter-rotating pivot. The outer wheel receives bounded
steering headroom while the inside wheel remains at or above its loaded floor.
Before applying that arc, the planner also limits speed by the tightest
body-width opening through every heading it must sweep from its present steering
angle to the requested one. That guards the moving turn itself, not only the
straight course after the turn completes.

The control path uses at most one current LD19 revolution of point history,
scaled from the scanner's reported speed and capped at 180 ms. Older points
are excluded from both the dashboard and drive decisions; a scan stream older
than 250 ms is a safety stop, not valid perception.

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

1. A stale Uno, LD19, or webcam stops the robot; it will not drive blind. After
   the webcam has produced a live frame, a USB reset receives at most one second
   of last-frame grace so one device event does not create a motor pulse. The
   robot still requires a real webcam frame before initially driving. If an IMU
   is detected during startup, `IMU CALIBRATING` also holds STOP until calibration
   reaches `IMU LIVE`. An absent IMU still selects the supported non-IMU mode.
2. Robot mode does not run object or person detection. Low-resolution webcam
   optical flow provides a bounded non-IMU yaw cue during turns and reduces
   false commanded map translation when a high-confidence textured view shows
   no movement. It is not odometry. Camera frames are not rendered, keeping
   camera/display work small on Pi 3B.
3. During standby, the mapper distinguishes observed free space from unknown
   space and persistent obstacle returns. A frontier planner selects a reachable
   unexplored boundary, rewards wider approaches, plans around inflated obstacles,
   adds a graded cost near those obstacles, and supplies a
   persistent look-ahead route. The cached route advances its waypoint every
   control cycle instead of waiting for the next full replan. When all current
   frontiers are exhausted, it patrols the least-visited reachable mapped space.
4. The LD19 begins a direction-locked forward arc when the straight inflated
   corridor drops below 1.25 m. Steering can use up to a 28-PWM wheel split,
   both motors stay above the loaded-wheel stall region, and steering changes
   are slew-limited. A very-high-confidence close return is retained even when
   a thin obstacle occupies only one angular bin; weaker isolated speckle is ignored.
   The frontier heading only biases among currently safe full-body corridors.
5. While the straight path is safely clear, the controller follows open
   corridors as continuous differential arcs. If the straight body corridor
   reaches 40 cm, it pivots toward the best broad opening instead of continuing
   closer. If every candidate corridor is below 38 cm, or the Pi ultrasonic
   reading is below 26 cm, it runs a finite recovery sequence:
   a short LiDAR-cleared reverse curve with
   both wheels driven, a slow one-wheel reverse turn toward the best full
   body-width LiDAR corridor, then a short forward commit and immediate return
   to live corridor planning. Which side counts as "open" for that turn is
   scored with the same windowed body-width minimum used for forward path
   selection rather than the single farthest ray in the sweep, so one lucky
   LiDAR return narrower than the chassis can no longer look like a viable
   escape direction. A calibrated USB LSM6DS3 releases the turn after
   measured yaw reaches the clear corridor and hard-limits each turn to
   88 degrees; a 2.4-second bound applies if IMU yaw is unavailable. It may try
   the opposite side once. It stops as `STOP:BOXED_IN` when neither bounded
   attempt has a safe side/rear path, and automatically rechecks materially
   changed geometry. Rear safety is a robot-width swept corridor rather than
   the closest point in a broad rear sector. When that corridor is clear but
   both sides are ambiguous, one short straight reverse search obtains a new
   view before declaring itself boxed in. If reversing is blocked but one
   complete turn sweep has at least 34 cm clearance, it begins the same bounded
   low-PWM centre pivot directly instead of giving up despite that opening.
   The 38/40 cm
   thresholds initiate manoeuvring; they do not mark every surrounding return
   as boxed-in, and the narrower 30 cm escape-corridor threshold remains
   available during recovery.
6. Uno ultrasonic hard-stop (under 18 cm) always wins and blocks forward
   motor commands even if the Pi fails.
7. A live, calibrated USB LSM6DS3 supplies measured yaw rate
   and integrated yaw change to the local mapper
   and recovery controller, and progressively removes steering split above
   38 deg/s, reaching zero additional split at 55 deg/s. If the IMU is absent
   or stale, navigation continues with command-predicted yaw corrected by
   successive LD19 scans. Recovery turns use that corrected pose heading when
   available, retaining the same 2.4-second hard timeout as the final bound.
8. A stuck detector runs alongside the corridor/escape logic above. It has no
   way to see the floor, so it infers "is the chassis actually responding" from
   independent evidence instead: LD19 scan-to-obstacle progress against a
   nearby tracked return, camera optical flow while translating or pivoting, measured IMU
   yaw rate while pivoting, and the Uno's own `blocked` flag. Each source only
   ever votes that the chassis is or is not moving when it has a genuinely
   fresh, currently-applicable signal; otherwise it abstains, and abstention
   alone never counts as evidence of being stuck. A declaration requires two
   independent `NOT_MOVING` sources with no `MOVING` vote. In particular, an
   LD19 result is counted only once per newly captured scan, never once per
   fast control-loop iteration. If a commanded drive keeps failing this test,
   it tries a different LiDAR-checked maneuver (the opposite turn side, a
   reverse, or a center pivot) instead of repeating the command that is not
   working. After three maneuvers it pauses as `STOP:STUCK_*_RETRYING` for
   three seconds, then starts another bounded, LiDAR-checked recovery burst.
   This avoids both continuous grinding and the previous unexplained
   25-second no-action latch. IMU vibration still distinguishes slipping from
   stalled only for the operator; it never gates detection or recovery.

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
planner time. It also reports IMU motion/vibration magnitude and camera-flow
confidence. The `POLICY` line reports `STUCK_RECOVER_<maneuver>:<attempt>` while
the stuck detector is trying an alternate maneuver, and `STOP:STUCK_*_RETRYING`
during its three-second safe pause; the log's `NAV`/`NAV_EVENT` lines carry
the same information plus a per-source `M`/`N`/`U` (moving/not-moving/unknown)
breakdown for diagnosing which evidence source triggered it. The occupancy grid uses
about 2.1 cm cells, up to 240 current-scan free-space rays, and useful LD19
returns up to 5.8 m where map bounds permit. These software changes do not
increase the LD19's physical range or create true odometry.

Camera startup defaults to `auto`. The runtime tries stable V4L by-id paths and
camera indexes 0 through 7. When a USB IMU is connected, webcam capture is
deferred until IMU calibration completes. The runtime allows five seconds to
confirm that no IMU is present before starting the webcam in non-IMU mode. A
detected but unfinished IMU does not use that fallback. This keeps webcam
streaming and optical flow from competing with MCP2221 calibration on the Pi 3B.
Optical flow is skipped while both motors are stopped. If no
webcam currently delivers frames, the LiDAR
dashboard remains open in `CAMERA STALE` safe-STOP mode and retries instead of
terminating the complete robot runtime. A reopened camera gets a fresh two-second
startup window; an old frame timestamp cannot force it back into a permanent
reconnect loop. Optical-flow pose assistance assumes the
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

The installed orientation is specifically: component side up, with the left
STEMMA connector and `SCX`/`SDX` end shown in the supplied photo facing the
robot's front. In that position the breakout's printed `+X` points rearward and
its printed `+Y` points toward the robot's right. The runtime's existing 180
degree yaw transform converts those into robot `+X` forward and `+Y` left while
leaving `+Z` upward. The board must remain rigid and flat.

The normal updater installs
the Linux prerequisites and Python USB transport, writes persistent MCP2221A
USB permissions, and prevents the optional kernel MCP2221 driver from competing
with the userspace transport. That system setup is recorded once and skipped on
later updates. Every robot boot still opens and validates the sensor because an
IMU cannot remain open across a power cycle.

At startup the runtime automatically checks both normal LSM6DS3 I2C addresses,
`0x6A` and `0x6B`, accepts the sensor only when its identity register returns
`0x6A` for the installed LSM6DS3TR-C, and verifies every critical configuration
register after writing it.
The sampler ignores cycles without both new gyro and accelerometer data, uses
the datasheet's 256 LSB/degree C temperature conversion, performs trimmed-mean
stationary calibration, and slowly tracks gyro bias only during confirmed
stationary periods. Handling jolts and commanded motion pause unfinished
calibration without deleting already collected still samples. IMU reads run on
a dedicated 50 Hz sampler thread, independent of camera, display, and planner
latency. The USB LSM6DS3 needs 40 accepted samples, approximately 0.8 seconds
at that sampler rate. Acceptance uses total acceleration magnitude and a broad
handling-rate bound; it does not require the board to be perfectly level or
reject a stable zero-rate bias merely because that is the bias being measured.
Final validation uses trimmed gyro variance so isolated edge samples cannot
poison the whole window. A rejected aggregate window advances as a rolling
still-sample window rather than clearing to zero. If webcam insertion briefly resets the MCP2221,
the USB LSM6DS3 remains the selected calibration source and retains its partial
progress while reconnecting. A dashboard `HOLD` suffix identifies motion,
rejected samples, missing fresh data, or an unstable window. No manual
`modprobe`, I2C scan, or launch command is required. During the
25-second stationary standby, the dashboard should change from
`IMU LSM6DS3 USB CALIBRATING` to `IMU LSM6DS3 USB LIVE`.

v1.9.13 makes a temporary calibration pause diagnosable: the dashboard and
`pi3b/logs/robot.log` now show the active gate (`MOTION`, `SAMPLE`, `WAIT DATA`,
or `UNSTABLE`), acceleration magnitude, and peak gyro rate once per second.
This is diagnostics only; it does not change calibration thresholds or motor
authority.

v1.9.14 improves live safety behavior without relaxing any safety gate: it
expires LD19 points on a bounded current-scan history, evaluates clearance
through the headings a moving arc actually sweeps, and requires corroborated
fresh evidence before declaring the chassis stuck. A failed recovery now uses
a short visible retry pause instead of a long, unexplained latch.

v1.9.15 fixes a serial race that could transmit an expired STOP after a newer
DRIVE. Command selection and transmission now share one ordering lock; the
0.50-second Pi lease and 350 ms Uno watchdog remain unchanged. NAV logs include
`control_gap_ms`, `lease_stops`, and `uno_timeouts` to separate planner latency
from Uno watchdog stops. Unconsumed serial lines no longer accumulate in an
unbounded queue.

Stuck evidence now refers to the previously applied output, rejects camera flow
older than 450 ms, and lets a stationary camera corroborate a stalled IMU pivot.
A failed reverse selects another available maneuver; each recovery burst avoids
repeating already tried maneuvers. Observed motion lets a bounded maneuver finish
rather than immediately switching back to the failed route. Every cycle retains
live safety checks, including side clearance during reverse arcs. A robot with
insufficient trustworthy motion evidence can still fail to detect a stall.

Map guidance toward marginal corridors is reduced in favor of broad live LD19
openings. Waypoint shortcuts follow A*'s no-corner-cut rule, and a waypoint fallback
cannot cross a blocked route. Bounded reconnection from an approximate pose inside
map inflation remains advisory and subject to live LD19 clearance.

The USB IMU wrapper expires cached measurements independently of the sampling
thread. Integration uses actual sample intervals while fresh and skips missing
yaw history across gaps longer than 250 ms. Sampling accounts for USB read time
instead of adding a fixed sleep after every read. `imu_age_ms`, `imu_dt_ms`, and
`imu_gaps` expose freshness and timing in NAV logs. Mount correction, calibration
acceptance thresholds, non-IMU fallback, and the camera startup gate are preserved.
These changes do not provide position from acceleration or absolute heading.

Software regression tests cover these changes; physical Pi driving, collision
avoidance, carpet recovery, USB timing and motor-battery performance remain
unverified. Software cannot compensate for a motor battery that sags under load.
For the next supervised boot test, confirm dashboard v1.9.15 and live sensors,
then compare uninterrupted driving, a blocked reverse with a clear side, a stalled
pivot, and route choice around offset obstacles. Keep the resulting NAV log.

### v1.9.16: purposeful camera cues and finer LD19 geometry

The forward-facing USB webcam now supplies quality-checked image-motion cues at
160x120 with at most 80 features. Forward/backward optical-flow consistency and
a robust similarity fit reject bad tracks; features must span at least half of
a 4x3 image grid. A small moving patch, insufficient texture, or lost tracks
abstain from motion evidence. High-confidence stationary views reduce false
commanded translation in both forward and reverse map prediction. Camera yaw
remains a bounded non-IMU cue, calculated from approximate image bearings.

Sustained, confident scene expansion during forward travel lowers base cruise
speed within the existing moving PWM band. It needs distinct frames spanning
at least 150 ms and expires automatically; turning, reverse and weak evidence
abstain. `VISION_APPROACH` in POLICY identifies this advisory slowdown. Camera
cues cannot authorize a path or bypass LD19/ultrasonic/camera-liveness stops.
Image expansion is not calibrated distance or a dependable collision detector;
autofocus, camera pitch, changing illumination and moving objects can affect it.
Floor visibility is useful texture, but no calibrated floor-plane model or
rug/object recognition is claimed. No neural/person detector was added.

The robot now retains 720 half-degree angular bins and projects accepted raw
angles/ranges into body-width corridors. This preserves more of the LD19's
returned detail; sensor scan rate, range and physical resolution are unchanged.
Older high-confidence returns cannot replace newer geometry, and equal-time
returns in one bin retain the nearest accepted surface. Sector filtering is
vectorized, and Cartesian rotations replace per-ray/per-heading trigonometry.

The 8 m, 384-cell map retains its approximately 2.1 cm cells. Every accepted
endpoint is marked observed, even when free-space rays are subsampled. Multiple
returns in one cell contribute only once per map update. Free-space carving is
capped at 240 rays and translation matching at 180 samples. With fresh IMU data,
map input receives bounded rotational compensation using return receipt ages
and recent yaw rate (at most 10 degrees; stale points/rates abstain). This is
approximate scan-smear reduction, not full motion compensation or metric SLAM.
Immediate safety continues to use raw live LiDAR geometry. Geometry remains a
2D slice at sensor height; obstacles above/below that plane can be missed.

NAV logs add `cam_quality`, `cam_tracks`, `cam_coverage`, `cam_expand`, `cam_ms`,
`visual_slow`, and `lidar_points`. `LOW_TEXTURE`, `LOCAL_FEATURES`, `TRACK_LOST`,
or `INCONSISTENT` mean visual navigation evidence is unavailable, even while
the camera is successfully delivering live frames. The dashboard shows camera
tracking quality without rendering video. Capture remains 320x240 at 15 fps,
with calibration deferral and automatic retry preserved.

Verification: 210 software tests, including synthetic image expansion, localized
motion rejection, reverse no-motion evidence, a frame-to-policy slowdown test,
thin body-edge returns, bounded ray integration and map yaw-compensation tests.
Desktop processing checks do not establish Pi timing or physical driving behavior.
For the next supervised boot test, confirm v1.9.16, inspect tracking quality in a
textured scene, and compare map detail and turn smearing around chair legs and
offset obstacles. Retain the NAV log, especially `cam_ms`, `control_gap_ms`,
`lease_stops`, and `uno_timeouts`. Motor-battery limitations still apply.

Implementation references: [OpenCV sparse optical flow](https://docs.opencv.org/4.12.0/dc/d6b/group__video__track.html)
and [LDROBOT SDK data processing](https://github.com/ldrobotSensorTeam/ldlidar_sdk/blob/master/src/ldlidar_dataprocess.cpp).

### v1.9.17: recover stalled USB calibration and repair boot updates

The v1.9.14 photo shows calibration at 50% with `HOLD ACCEPTING`, stopped
motors and deferred camera. It does not establish the underlying USB fault.
The [Blinka MCP2221 transport](https://github.com/adafruit/Adafruit_Blinka/blob/main/src/adafruit_blinka/microcontroller/mcp2221/mcp2221.py)
uses a blocking HID read without a timeout and has an unbounded I2C status loop.
Previously either could strand the sampling thread indefinitely.

USB access now runs in one spawned process, isolated from navigation.
A transaction exceeding 0.75 seconds, or adapter initialization exceeding eight
seconds, terminates that process and releases its USB handle. The existing
one-second reconnect path creates a new owner. Accepted calibration samples
and calibrated bias stay in the parent sampler. No abandoned reader competes
for the adapter. Stale calibration reports `USB WAIT`; a failed transaction
reports `USB RETRY` and logs the error. A recognized IMU with a configuration
fault stays detected and cannot be mistaken for an absent optional IMU.
Calibration acceptance thresholds and the motor readiness gate are unchanged.

The installer previously chmod'ed launchers stored in Git as 100644. This
created permission-only modifications on Linux, and the old boot updater could
skip every subsequent update. Script modes and update checks are now consistent;
see the boot-update section above for recovery and diagnostics.

Regression tests deliberately block USB initialization/reads, verify reader
termination and replacement, and resume calibration after a simulated timeout.
Update tests use real temporary Git repositories for fetch, checkout, repeat
boot, offline retries, permission changes, edited files and rollback. A Bash
launcher integration test runs the actual update and re-exec path with a dummy
runtime and stand-ins for Linux utilities absent on Windows. These desktop tests
do not verify Pi USB timing, Linux lock behavior, physical calibration or driving.

Next supervised test: confirm **v1.9.17**, keep the robot stationary while the
IMU calibrates, then confirm camera readiness and motion. If it stays stopped,
retain the startup/update log and IMU error lines instead of waiting ten minutes.

### v1.9.18: keep rolling, notice stalls, notice being moved, watch remotely

Four behaviours reported from supervised driving of v1.9.17, and the code
paths each one traces back to.

**Drive/stop/drive pulsing, and ignoring an obviously clear route.** Any
straight corridor shorter than `CLOSE_FRONT_TURN_M` (0.40 m) went directly to
the pivot-in-place escape machine, even when the corridor profile showed two
metres of open floor twenty degrees away. Escaping stops the chassis, pivots,
then re-plans - which is exactly the observed stutter. A **steer-around band**
now sits between cruising and escaping: with the scan not blocked all round,
the planner curves past the obstruction at crawl speed instead. It is gated on
a genuinely broad alternative (`STEER_AROUND_MIN_OPENING_M`, 0.55 m), a real
turn away from the obstruction (8 degrees), and enough swept clearance for the
arc itself. Nothing underneath it changed: the Uno's ~18 cm hard stop, the
Pi's 26 cm ultrasonic recovery trigger and the all-round `CLOSE_LIDAR_M`
check all still run first and still force a real escape.

Escape pivots were also under-driven. `_pivot_crawl` used `min_move_pwm`,
which is calibrated for straight rolling; a pivot additionally has to scrub
both tyres. On carpet the wheel buzzed without rotating, the turn hit its
2.4 s timeout, and the policy reported `STOP:BOXED_IN` - indistinguishable
from a genuinely blocked turn. Pivots now add `ESCAPE_PIVOT_BOOST_PWM`.

**Stalls that never registered.** Every previous motion source had a blind
spot, and they overlapped badly in open floor: the IMU could only vote while
turning, the Uno only when `blocked`, and the LD19 progress check only while
something sat within 1.2 m. In the middle of a room a rug stall therefore
gathered at most one vote, and a declaration needs two. Three sources are
added. A **whole-scan range signature** (`robot_motion.py`) compares complete
LD19 revolutions, so it works at any range - a wedged chassis reproduces the
same scan indefinitely. The **Uno ultrasonic cone** is tracked across a
commanded forward drive. **Accelerometer energy** votes NOT_MOVING when it
sits at the stationary floor measured during calibration; it never votes
MOVING, because a stalled motor buzzing against a rug produces plenty of
energy. Recovery also gets one extra full round of maneuvers before latching,
since a caster against a threshold often frees itself from a slightly
different attitude.

**Not knowing it had been moved.** Every evidence source is defined relative
to a commanded drive, so a chassis latched at zero PWM and then lifted by hand
could not observe its own rescue: it resumed the escape phase, heading
commitment and route it had latched on, all computed for a position it was no
longer in. Two independent signals now detect displacement - accelerometer
energy or a changed resting attitude while the motors are commanded stopped,
and a scan that changes further between consecutive revolutions than any
drivable speed allows. Either one stops the chassis, clears the escape and
stuck state, drops the steering commitment, and resets the occupancy map and
the planner, because nothing on this robot can relate the old map frame to
the new one. The dashboard reports `REORIENTING` while it settles.

**Watching it.** The dashboard now also carries an `INTENT` line (what it is
about to do, in words), a `MOTION` readout of the per-source stuck votes, the
planned A* route drawn as a polyline from the chassis marker, and an arrow for
the steering actually being applied - which can differ from the route while
the local corridor planner curves around something the map has not resolved.
The same rendered dashboard is served read-only over HTTP at
`http://<pi>:8080/` (see below), and the runtime no longer requires a desktop
session, so it can start and stream with no HDMI attached.

The IMU itself gained attitude and motion cues: chassis tilt (the angle
between gravity and the board Z axis, which is independent of the mount-yaw
convention, so it cannot be wrong because the forward axis was guessed), a
tilt rate, gravity-removed planar and total acceleration, the stationary
energy floor learned during calibration, and the yaw change integrated over
roughly the last second. None of these is integrated into a velocity or a
position: with no wheel encoders and no absolute reference this sensor would
drift within seconds, so they stay first-order observations.

Desktop regression tests cover the scan-signature tracker, the steer-around
gates, the new evidence sources, displacement handling and map reset, the
route overlay, the IMU cues, and the stream's backlog/staleness/isolation
behaviour. **None of this has been run on the robot.** These tests do not
verify PWM levels against real carpet, pivot torque, LD19 timing, Pi 3B load
with a stream attached, or Wi-Fi behaviour.

Next supervised test: confirm **v1.9.18** on the dashboard, then check in
order - continuous driving without move-stop-move pulses; a deliberate rug or
threshold stall registering as `STUCK` rather than silently failing; lifting
the chassis mid-run and seeing `REORIENTING` with the map rebuilt; the route
polyline matching where it actually goes; and the phone view staying live
while walking the robot around.

### Remote dashboard view

The runtime serves the same dashboard it draws on the Pi monitor as a
view-only web page, so it can be watched from a phone or laptop on the same
network:

```
http://<pi-address>:8080/
```

The page shows a LIVE/STALE/DISCONNECTED badge and the age of the picture, so
a frozen view can never be mistaken for a stopped robot. There are no control
endpoints; nothing on the page can command the chassis.

Design constraints: the control loop only hands over a reference to an
already-rendered frame under a short lock, and JPEG encoding, socket writes
and client handling all run on other threads. Exactly one encoded frame is
retained, so a client on slow Wi-Fi misses intermediate frames rather than
building a backlog. A port that cannot be bound degrades to "no remote view"
and never blocks the runtime.

Environment overrides in `run_robot.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VISIONFSD_WEB_PORT` | `8080` | Listening port |
| `VISIONFSD_WEB_FPS` | `5` | Stream frame rate (clamped to 0.5..15) |
| `VISIONFSD_NO_WEB` | unset | Set to `1` to disable streaming entirely |

The runtime also starts without a desktop session. With no `DISPLAY` or
`WAYLAND_DISPLAY` it logs that it is headless, skips the local window, and
keeps driving and streaming normally.

### USB LSM6DS3 mounting

The USB LSM6DS3 through the MCP2221A USB-I2C adapter is the only supported
IMU configuration. A missing or disconnected LSM6DS3 goes straight to the
existing, fully supported `LD19+COMMAND POSE ACTIVE` non-IMU mode after the
probe interval.

The installed board is flat with components up and rotated 180 degrees from
the robot frame: its pin-header edge faces forward and its `LSM6DS3` text
edge faces rearward. The runtime therefore defaults to
`--imu-mount-yaw-deg 180`, which reverses sensor X/Y while retaining Z. Override
only if the physical mount changes:

```bash
export VISIONFSD_IMU_MOUNT_YAW_DEG=180
```

The first stationary seconds of the existing 25-second standby calibrate gyro
bias. Calibration rejects samples with excessive motion and pauses rather than
resetting valid progress. Missing IMU hardware never creates a no-motion boot
failure. A v1.9.10 attempt to tighten the aggregate stationary-calibration
acceptance bar based on the LSM6DS3's datasheet noise specs was reverted in
v1.9.11: physical testing showed `IMU CALIBRATING` stalling short of 100%
indefinitely, because the real sensor's noise did not reliably fit inside the
tighter bar. Do not retighten this without hardware-in-the-loop verification.

The gyro improves short-term turn measurement and smooths excessive yaw. Its
accelerometer is not integrated into position because chassis vibration,
gravity error, and the front/side mounting offset would create rapid drift.
The LSM6DS3 has no magnetometer, so it does not provide absolute heading.

### LiDAR + IMU exploration map

The LD19-only panel keeps an 8 m occupancy map at about 2.1 cm per cell. Its
6 m display viewport follows the estimated chassis position and keeps the robot
marker centred. The underlying grid is a fixed-size sliding window, not a
buffer anchored at the start position: once the estimated pose approaches
the edge of the 8 m buffer, the grid recenters around the robot (shifting its
contents and blanking the band that scrolled in from the far side) rather than
pinning the pose at the boundary. Earlier versions clamped position at that
boundary instead, which froze the dead-reckoned pose while the robot kept
moving physically -- every new scan then projected onto that stale position,
so the map stopped updating and the display showed a growing region that was
never drawn. The trade-off of the sliding window is that ground the robot
already covered can scroll out of the buffer on a long one-direction traverse,
so `coverage_ratio` and the least-visited patrol mode are relative to the
current window, not a persistent record of the whole session. Every valid
scan marks both obstacle endpoints and the observed free ray leading to each
endpoint. Repeated free observations clear stale hit evidence; persistent hits
remain obstacles. Bright current-scan points remain distinct from mapped
history, while dark known-free cells are distinguishable from unknown space.
Metre range rings make nearby geometry easier to read.

The frontier explorer inflates obstacles by the robot body plus a 7.5 cm lateral
safety margin, finds the free-space component connected to the robot, clusters
reachable free/unknown boundaries, and runs bounded A* to the selected target.
Frontier utility rewards obstacle clearance, while a graded A* traversal cost
prefers the middle of available space without converting narrow free passages
into blocked cells. Target scoring also penalizes total path length and excess
detour length so a slightly larger distant frontier does not override a clear,
efficient nearby opening. It guides
the local planner along a cached look-ahead route whose waypoint advances every
control cycle. If map inflation temporarily places the estimated pose just
outside free space, planning reconnects to nearby known free space while live
LiDAR remains authoritative. If no reachable frontier remains, it patrols
low-visit mapped cells to expose missed openings. Full replans run every 0.6
seconds with a 45 ms A* budget. The current motor decision is sent before that
advisory search, and a timed-out replan retains the last safe route rather than
pausing the robot or dropping the Uno's 350 ms watchdog.

The navigation design follows established local-navigation principles: broad
polar openings rather than single range rays, dynamically safe progress and
clearance objectives, reachable free/unknown frontiers, and graded obstacle
costs. Reference material:

- [Vector Field Histogram](https://public.websites.umich.edu/~ykoren/uploads/The_Vector_Field_HistogramuFast_Obstacle_Avoidance.pdf)
- [Dynamic Window Approach](https://rse-lab.cs.washington.edu/abstracts/colli-ieee.abstract.html)
- [Frontier-Based Exploration](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated2/yamauchi_frontier_explor.pdf)
- [Nav2 cost-aware planning](https://docs.ros.org/en/ros2_packages/humble/api/nav2_theta_star_planner/)
- [ST MotionGC gyroscope calibration](https://www.st.com/resource/en/user_manual/dm00372763-getting-started-with-motiongc-gyroscope-calibration-library-in-xcubemems1-expansion-for-stm32cube-stmicroelectronics.pdf)

A heading correction is applied only when successive 360-bin LD19 scans have
enough non-ambiguous support. Between accepted matches, a live USB LSM6DS3 supplies
measured yaw rate and its asynchronously integrated yaw change; if it is
unavailable, the mapper uses conservative
commanded-motion prediction. Bounded scan-to-map correlation also corrects
small translation errors when a moving scan uniquely agrees with established
obstacle geometry.

The dashboard's IMU `MOTION` value is filtered deviation from one-g total
acceleration. It exposes handling, vibration, or wheel-slip symptoms but is not
integrated into position. Without encoders or an external position reference,
accelerometer integration would drift too rapidly to improve this chassis map.

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
holds STOP for 25 seconds. Before that, `run_robot.sh` runs a bounded
auto-update check (see "Automatic updates on boot" above) so a fresh boot
picks up the latest installed release without a manual `update.sh` run. Use
`--no-robot-autostart` with `install.sh` if you do not want autostart at all.
Raspberry Pi OS Lite has no graphical autostart session, so it needs a
separate headless service and does not show the visualizer.

On the standard Pi Desktop Wayland session, the launcher selects Qt's XWayland
backend and reapplies fullscreen during the first rendered frames. This avoids
the small 640-pixel OpenCV window and makes the LiDAR dashboard occupy the
connected display. Set `QT_QPA_PLATFORM` manually only when using a different
custom desktop backend.

The launcher discovers the Uno by its exact `2341:0043` USB identity and the
LD19 adapter by its exact CP210x `10C4:EA60` identity. It does not hardcode
`/dev/ttyACM0`, because Linux may assign a different ACM number after a USB
reset. If an active Uno port returns a serial I/O error, the runtime keeps the
motors stopped, rediscovers the current node, waits for the Uno reset, and
repeats `CAPS` before differential motion can resume. The dashboard remains
running during this recovery. `VISIONFSD_ARDUINO_PORT` and
`VISIONFSD_LIDAR_PORT` remain optional manual overrides for other hardware.

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
