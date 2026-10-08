# VisionFSD robot — handoff

Everything a new session needs to work on the indoor autonomous robot: where
the system lives, how it is built, how to change it safely, and the full list
of problems reported from driving it, with the reasons each one existed.

Written at **v1.9.23**. Nothing in the runtime has been verified on the robot
by the sessions that wrote it; every claim below is from desktop tests,
simulation, or the operator's own reports.

**The goal, in the operator's words: drop → run → go.** Put the robot down
anywhere in the house, switch it on, and have it drive sensibly and adapt to
what is around it, with no per-room setup. It does not do this yet; §5.14 says
why, and §8 says what not to promise.

---

## 1. Where the system lives

| Thing | Value |
|---|---|
| Repository | https://github.com/YoMosa2009/VisionFSD-Pilot |
| Deployment branch | `codex/pi3b-runtime` (**not** `main`) |
| Runtime directory in the repo | `pi3b/` |
| Install path on the Pi | `~/visionfsd-pi` |
| Phone/desktop dashboard | `http://<pi-ip>:8080` on the same Wi-Fi |
| Local Windows clone | `C:\Users\user\source\repos\VisionFSD-Pilot` |
| Uno firmware sketch | `robot/firmware/visionfsd_pi_autonomy/visionfsd_pi_autonomy.ino` |

Update the robot over the air:

```bash
bash ~/visionfsd-pi/pi3b/update.sh codex/pi3b-runtime
```

`run_robot.sh` also auto-updates once per boot before starting the runtime. It
is fail-safe: offline, a dirty working tree, or a failed update leaves the
installed version running. Version string lives in `pi3b/VERSION` and is shown
in the window title and on the dashboard.

The HDMI screen still works exactly as before. The dashboard is an addition,
not a replacement; both show the same view.

---

## 2. Hardware

- **Chassis:** OSOYOO Model 3 kit, 4 DC motors, no encoders. Measured
  **9 in wide × 10.5 in long** (0.229 m × 0.267 m).
- **Arduino Uno R3 + OSOYOO motor shield** — drives the motors, owns the
  ultrasonic sensor, enforces its own safety stop. Powered from its own
  battery pack. **Never power the motors from the Pi.**
- **Raspberry Pi 3B, 64-bit Raspberry Pi OS** — runs everything else. Talks to
  the Uno over USB serial at 115200.
- **FHL-LD19 LiDAR** over USB serial — 360°, ~4500 points/s, up to 12 m. This
  is the robot's main sense of the world.
- **USB webcam** — the live view on the dashboard, plus a cheap non-neural
  motion cue. No neural or person detection runs in robot mode. See §8 for the
  decision about what this camera is *for*.
- **LSM6DS3 IMU** over an MCP2221A USB-to-I²C bridge — gyro yaw rate and
  accelerometer, used as corroboration only. **Currently not working on the
  robot — see §5.15.**
- **Ultrasonic sensor**, fixed forward-facing, trigger on D3, echo on D2,
  wired to the Uno.

Hardware limits worth knowing before proposing a fix:

- **No encoders.** The robot never truly knows how far it has moved. Position
  is inferred from LiDAR scan matching, which drifts.
- **The Uno will not drive the motors below 105 PWM** (`MIN_EFFECTIVE_PWM` in
  the firmware). Below that the motors buzz and the loaded chassis does not
  move. The Pi cannot command a slower crawl without reflashing the Uno.
- **IMU yaw is a rate, integrated.** It drifts and is not an absolute heading.
  It must never be treated as a compass, and accelerometer data must never be
  integrated into a position.

---

## 3. How the software is organised

All under `pi3b/`:

| File | Role |
|---|---|
| `robot_autonomy.py` | The runtime. Main loop, policy, escape and recovery states, safety, motor output, CLI. Everything else is a library it calls. |
| `robot_scan.py` | The LD19 scan as numpy arrays (`ScanFrame`), built once per new batch of packets, plus the display-only artefact flag. |
| `robot_local_planner.py` | Short-horizon arc planner: candidate arcs, clearance, stopping distance, look-ahead, gap finding, obstacle memory, oscillation watchdog. |
| `robot_explorer.py` | Global planner: frontier exploration on a coarse grid, goal commitment and blacklisting, A*, roaming goals, background worker thread. |
| `robot_slam_lite.py` | 12 m occupancy map, 576 cells, free-space wedges, scan-match translation, memory decay. |
| `robot_tracking.py` | Moving-object segmentation, association, alpha-beta filtering, closest-approach prediction, yielding. |
| `robot_motion.py` | Motion estimation and evidence that the robot is or is not moving. |
| `robot_camera_motion.py` | Camera-based motion cue used as one stuck-detection vote. |
| `robot_imu.py` | LSM6DS3 over MCP2221A: yaw rate, calibration, USB recovery. |
| `robot_web.py` + `web/index.html` | Dashboard server: LiDAR/plan view, camera MJPEG stream, manual control over WebSocket, telemetry. |
| `robot_update.py`, `update.sh`, `auto_update.sh`, `recover-update.sh` | OTA update and rollback. |
| `lidar_visualizer.py` | Standalone LiDAR viewer, separate from the robot runtime. |
| `tests/` | 475 desktop tests. |

### The control loop

Runs every 25 ms (`CONTROL_PERIOD_S`). Each tick:

1. Read the newest LD19 frame (cached per sequence number, so no work is
   repeated when the LiDAR has produced nothing new), the Uno status
   (including the ultrasonic distance), the camera motion cue and the IMU.
2. Update the map, the obstacle memory and the moving-object tracker.
3. Ask the planner what to do. The arc planner re-plans every 70 ms
   (`ARC_REPLAN_PERIOD_S`) and is bounded so it can never starve the Uno lease.
4. Convert the chosen arc into left/right wheel PWM, rate-limited.
5. Send the command. The Uno stops the motors if it does not hear from the Pi
   within 350 ms, so the loop must keep talking.

### Safety, in order of authority

1. **Uno firmware** — hard stop at 18 cm on the ultrasonic, and the 350 ms
   command timeout. The Pi cannot override this.
2. **Emergency brake in the runtime** — 0.30 m on LiDAR, 30 cm on ultrasonic.
3. **Arc admissibility** — an arc is only allowed if the robot could stop
   within the clear part of it, plus a 0.25 m buffer.
4. **Everything else** (goals, exploration, gaps) is preference, not
   permission.

The LD19 and the ultrasonic keep final authority over whether the robot may
move. Nothing added should be allowed to talk the robot into driving where
those two say it should not.

### Key numbers (v1.9.23)

```
MIN_MOVE_PWM              105   movement floor, set by the firmware
DEFAULT_CRUISE_PWM        112   "fast", earned only in open space
TURN_OUTER_CAP_PWM        127   no wheel goes above this while turning
MAX_STEERING_RATE_DPS      45   steering slew
MAX_PWM_RATE_PER_S         50   drive-level slew
CONTROL_PERIOD_S        0.025
ARC_REPLAN_PERIOD_S      0.07
EMERGENCY_BRAKE_M        0.30
ROBOT_FOOTPRINT      two circles, r = 0.1323 m, offset 0.0667 m
```

---

## 4. Working on this safely

**Git rules.** Commit and push verified changes to `origin/codex/pi3b-runtime`.
Preserve unrelated files. Never use destructive git commands
(`git reset --hard`, `git checkout -- .`). Do not stage or modify these unless
explicitly asked: `.serena/`, `robot/bringup/`, `robot/cad/`,
`robot/firmware/visionfsd_manual_drive/`, `robot/manual_control.py`,
`tools/OpenSCAD-2021.01-x86-64.zip`, `tools/OpenSCAD/`.

**Before pushing anything that changes robot behaviour:**

1. Add regression tests for the specific behaviour.
2. Run the full suite: `cd pi3b && ../.venv/Scripts/python.exe -m unittest discover -s tests`
3. Static checks: `compileall`, `pyflakes`, `bash -n` on shell scripts,
   `git diff --check`.
4. Bump `pi3b/VERSION` and add a README section explaining what changed and
   why, including honest limits.
5. Commit only the intended files, then push.

**Design constraints.**

- Keep the non-IMU mode working; the IMU is optional corroboration.
- No neural or person-detection models in robot mode, and no heavy
  dependencies. The Pi 3B has four slow cores and 1 GB of RAM.
- Keep the planner bounded so it cannot starve the Uno command lease.
- Do not claim hardware behaviour is verified unless it was actually tested on
  the robot.

**Simulation.** A closed-loop simulator was used to compare versions: the real
control loop, a simulated LD19 sweep with mixed pixels, the Uno firmware floor,
ramp and lease, chassis lag, and exact rectangle clearance. It has no wheel
slip, perfect odometry and clean geometry, so it *understates* collisions —
use it to compare versions, never to predict real behaviour. The scripts live
in the session scratchpad, not in the repo.

---

## 5. Every issue reported, and why it existed

These are the problems the operator reported from actually driving the robot,
in the order they were raised. "Status" describes what was done; none of it is
hardware-verified.

### 5.1 Motors pulsed between driving and stopping

**Reported:** the robot alternated between moving and stopping instead of
driving smoothly.

**Why:** the planner was choosing drive levels in the band below the loaded
chassis's movement floor. The firmware will not move below 105 PWM, so
commands in that band produced buzzing and no motion, followed by a correction
above the floor, repeatedly. A second contributor was the Uno's 350 ms command
timeout: any pause in the control loop cut the motors.

**Status:** any commanded wheel value is now either zero or at least 105, the
loop holds the lease, and output changes are rate-limited.

### 5.2 Stuck detection was weak

**Reported:** the robot would be physically stuck and keep trying the same
thing.

**Why:** with no encoders there is no direct "am I moving" signal, and the
original detector leaned on too few sources, so it either missed real stalls
or fired on noise.

**Status:** stalls are declared by vote — LiDAR scan-match displacement, camera
motion cue, IMU yaw rate, and commanded-versus-observed motion — requiring
several votes from at least two independent sources inside a 1.2 s window,
then a bounded recovery sequence (reverse, pivot, retry) with a limited number
of attempts.

### 5.3 The robot did not know it had been picked up or moved

**Reported:** after being moved by hand it carried on as if nothing had
happened, acting on commands from before the move.

**Why:** the map and the planner assumed continuity. A large sudden change in
the scan looked like a bad match rather than a teleport, and queued intent was
not invalidated.

**Status:** a large unexplained displacement now shifts or resets the map and
drops stale intent.

### 5.4 Spinning and confusion in open space

**Reported:** the robot span in place or dithered even where there was clearly
room.

**Why:** two roughly equal options score almost identically. The planner
re-decided from scratch every cycle, so turning left made the right option look
better, and back again — a classic local-planner limit cycle. Nothing detected
it because the robot never stopped moving.

**Status:** steering smoothness is scored, gap choices are committed to for a
period, and a progress watchdog (after Nav2's oscillation critic and TEB
recovery) locks a turn direction when turning without getting anywhere is
detected.

### 5.5 Driving far too fast, hitting walls, ultrasonic not saving it

**Reported (v1.9.19 era):** moving very fast, bumping into walls, and the
ultrasonic sensor could not stop it in time.

**Why:** speed was chosen without reference to stopping distance. The
ultrasonic is a single narrow forward beam mounted low; it cannot see a wall
approached at an angle, or a table edge, and it has almost no reaction time at
speed.

**Status:** an arc is only admissible if the robot could stop inside its clear
portion plus a buffer, and speed is governed by that. See 5.13 — this came
back in v1.9.22 for a different reason.

### 5.6 Chassis model was wrong

**Reported:** the operator supplied the real dimensions, 9 in × 10.5 in.

**Why:** the planner had been using a narrower footprint, about 60% too narrow.
It therefore believed it fitted through gaps it does not fit through, committed
to them, arrived, and improvised.

**Status:** a two-circle footprint derived from the measured chassis
(r = 0.1323 m, offset 0.0667 m) is used by the arc planner, and since v1.9.22
the global planner inflates obstacles by the same amount. A mismatch between
those two layers is a recurring class of bug here — if a new layer reasons
about clearance, it must use the same footprint.

### 5.7 Hitting things once or twice per run, and control latency

**Why (latency):** the dashboard and planner work shared the control thread, so
a slow frame delayed motor commands.

**Status:** LiDAR ingestion is cached per sequence, exploration planning runs
on a background worker, and the dashboard streams independently of the control
loop. Contacts were reduced but never eliminated — see 5.13 and 5.14.

### 5.8 Sticky continuous turning into rooms

**Reported:** once it started turning into a room it kept turning.

**Why:** the arc score rewarded arc length, and an arc that curls tightly stays
"clear" for its whole length while going nowhere. Curling therefore scored
better than crossing open floor.

**Status:** arcs are scored by forward progress, not arc length, with an
explicit penalty for arcs that run into something.

### 5.9 Two openings caused random driving and standing still

**Reported:** faced with two openings it drove randomly, span, and stayed in
one place.

**Why:** the same tie-breaking failure as 5.4, plus gap width being measured
at the wrong place — the width was taken at depth rather than at the mouth of
the gap, so a doorway looked wide enough from an angle where it was not.

**Status:** gap width is measured as the chord at the mouth (v1.9.21), with
hysteresis and a commitment window so a chosen opening is pursued.

### 5.10 Planning did not look far enough ahead; committing to bad plans

**Reported:** it needed to plan further ahead and stop committing to plans it
should not in multi-turn spaces.

**Why:** the local planner only sees about 1.6 m, and the global route was
being followed as a single look-ahead waypoint heading. At a junction where
the route turns, driving straight scores excellent "forward progress" while
landing off the route.

**Status:** candidate arcs are scored by distance made good *along* the planned
route and distance *off* it (the substance of Nav2 DWB's PathDist/PathAlign
critics), and goals are committed to with a progress timeout and blacklisting
(after explore_lite). A cost-to-go field was prototyped and rejected: too slow
for a Pi 3B.

### 5.11 Circling the same area in large open rooms

**Why:** with nothing nearby, every direction is equally clear, so the local
planner has no reason to prefer any of them and drifts back over ground it has
already covered.

**Status:** frontier exploration on a coarse grid provides a goal — head for
the boundary between known-free and unknown space — with roaming goals when no
frontier remains.

### 5.12 Moving objects (people walking past)

**Why:** everything was treated as static, so a person walking through left a
smear of phantom walls in the obstacle memory.

**Status:** returns are segmented and tracked with an alpha-beta filter,
predicted forward, and yielded to for at most 2 s (so a person standing in a
doorway cannot park the robot). Obstacle memory gained raytrace clearing, so a
remembered return is forgotten when the live scan sees past it. Honest limit:
without encoders, pose drift can make static things look like they are moving,
so motion is only declared after several sightings, a minimum speed, a minimum
distance actually travelled, and never while spinning quickly.

### 5.13 v1.9.22 regression: far too fast, repeatedly hitting things, short-term choices

**Reported:** a downgrade — driving *way* too fast instead of smooth and chill,
repeatedly driving into things, and failing to find genuinely clear openings
up ahead, looking short-term instead of planning a safe trajectory from the
LD19 data.

**Why (the main cause):** v1.9.22 added a mixed-pixel filter — a port of
LDROBOT's own NEAR_FILTER — that groups returns by a 3% range jump and discards
lone returns as edge artefacts, **and the planner used the filtered returns**.
Along a wall seen at a glancing angle, neighbouring returns legitimately differ
in range by far more than 3%, so each one looked isolated and was discarded. In
a synthetic LD19 sweep that removed 68–95% of such a wall at 1–2 m and **all**
of it at 2–3 m. The planner could not see walls the robot was driving alongside
until they were within 0.6 m, where a near-field exemption kept them. Those
walls therefore looked like open space: the robot earned cruise speed toward
them and found them far too late. That single fault explains all three
symptoms — the speed, the contacts, and the apparent short-sightedness.

**Contributing causes:**

- **No look-ahead past the arc.** Arcs are ~1.6 m long and were scored only on
  themselves. Two arcs equally clear for their length scored equally even when
  one ended facing a wall and the other faced a long corridor.
- **Speed preference was effectively free.** The outer wheel was allowed to run
  up to 136–139 PWM in sharp turns while the inner wheel sat at the floor, so
  turns were fast as well as tight.

**Status (v1.9.23):**

- Planning, memory, tracking and mapping use **every** LD19 return again. The
  artefact flag survives for the phone view only and is now judged by spatial
  isolation — weak *and* far in space from both angular neighbours — which
  keeps glancing walls and still flags doorway ghosts.
- Each arc is scored on clear corridor (0.52 m wide, out to 4 m) ahead of its
  end, along the heading it ends on. This is switched off while a global route
  is being followed, because the route already sees the whole map and a
  straight corridor must not outvote a planned turn.
- Cruise (112) is earned only with ≥3 m of clear corridor ahead and steering
  within 10°. No wheel exceeds 127 PWM while turning, a real turn drops to the
  movement floor (after Regulated Pure Pursuit), and the planner's turn model
  uses the capped split so it cannot plan arcs the wheels cannot follow.
  Steering slew 75 → 45 °/s, drive ramp 120 → 50 PWM/s, pivot boost 18 → 12.

Simulated comparison (3 rooms × 2 starts × 60 s; no version made contact in
simulation, which is why the table is a comparison and not a prediction):

| Version | Closest approach | Cells covered | Mean PWM | Max wheel PWM |
|---|---|---|---|---|
| v1.9.21 | 0.12 m | 42.7 | 106.9 | 136 |
| v1.9.22 | 0.04 m | 30.3 | 104.1 | 136 |
| v1.9.23 | 0.10 m | 38.7 | 106.2 | 127 |

### 5.14 Open: the whole-home test — it does not read complex spaces well

**Reported:** exploring the operator's home — different room shapes, open
areas, furniture — it fails to take the paths that are obviously the logical
ones, and gets stuck in places where there is a plain visible opening.

This is the headline problem, and the one the other navigation issues (5.4,
5.8, 5.9, 5.10, 5.11, 5.13) are all pieces of. It is stated separately here
because fixing any one piece has not fixed the whole, and a future session
should treat it as the actual goal rather than as a list of symptoms.

**The goal, in the operator's words: drop → run → go.** Put the robot down
anywhere in the house, switch it on, and have it drive sensibly and adapt to
whatever is around it, with no per-room setup.

**Why it keeps falling short — the honest structural reasons:**

- **The robot's horizon is about 1.6 m.** That is the arc planner's reach. A
  "logical path" through a house is a 5–15 m decision: through this doorway,
  round that couch, into the hall. The global planner (frontier exploration on
  a 6.25 cm grid, 12 m map) is meant to supply that, but it plans on a map
  built without encoders, so it is only as good as the scan matching underneath
  it. Local and global disagree more often in cluttered, varied geometry than
  in the simple test rooms.
- **A 2-D LiDAR at one height sees a slice of the room, not the room.** Chair
  and table legs appear as scattered dots with gaps between them that look
  like openings; a couch reads as a wall; anything above or below the scan
  plane does not exist at all. "Obvious opening" to a person and "obvious
  opening" to this sensor are genuinely different things.
- **Drift with no encoders.** Every goal, frontier and remembered obstacle is
  positioned by scan matching. In a varied home, matching is worse than in a
  corridor, so goals slowly land in the wrong place and the robot pursues them
  anyway.
- **Scoring is a weighted sum.** Progress, clearance, smoothness, the route,
  and now look-ahead are combined with hand-set weights. Weights tuned to fix
  one behaviour shift another; several versions in this history fixed one
  symptom and introduced the next. This is the main reason changes must be
  measured, not guessed.
- **It has no memory of what it already learned about a place.** The map decays
  (12 s half-life) and is 12 m wide, so a house is never held in mind as a
  whole. Every room is met fresh.

**What would actually move this forward**, in rough order of value per unit of
Pi 3B CPU:

1. A larger, persistent map, so the global planner reasons about the house
   rather than a 12 m window.
2. Better scan matching (rotation as well as translation), because everything
   above depends on pose quality.
3. Replacing hand-tuned score weights with something measured across many
   simulated rooms, so a change can be shown to help overall rather than in one
   scenario.

None of these are small. A future session should not promise "drop it anywhere
and it works" before they exist.

### 5.15 Open: the IMU is not working

**Reported:** the dashboard shows **IMU OFF**, while the light on the IMU board
itself is lit.

**What that status means, exactly:** the runtime reports `OFF` when
`imu.connected` is false — the sampler never successfully opened the sensor.
The other states are `CAL` (connected, still calibrating), `STALE` (connected,
samples too old) and `LIVE`. So `OFF` means the Pi never got as far as reading
the sensor's identity register, not that the sensor is faulty or unpowered.
**The board's LED only shows it has USB power**, which it gets from the
MCP2221A adapter regardless of whether any software ever talks to it. A lit LED
and `OFF` together are exactly what a communication failure looks like.

**Likely causes, most likely first:**

1. **The Linux `hid_mcp2221` kernel driver claimed the adapter.** The runtime
   talks to the MCP2221A through Blinka over raw HID; if the kernel's own
   driver grabs the USB HID interface first, opening it fails.
   `pi3b/setup_mcp2221.sh` exists precisely to blacklist that module and
   install a udev rule — if it was never run on this Pi, or was run before the
   adapter was plugged in, this is the most likely cause.
2. **Permissions.** Without the udev rule (`MODE="0666"` for VID `04d8`,
   PID `00dd`), the robot user cannot open the HID device.
3. **Missing Python dependencies** — `hidapi` and `adafruit-blinka` from
   `requirements.txt`.
4. **Wiring or address.** The link probes the LSM6DS3's `WHO_AM_I` at its
   candidate addresses; wrong SDA/SCL, no pull-ups, or a different chip variant
   fails that check. The recorded error text distinguishes "adapter not found"
   from "sensor not found at 0x6A/0x6B".

**Why it has not been diagnosed:** the exact failure string *is* captured
(`IMUState.error`), and it reaches the HDMI window and `logs/robot.log` — but
**it is not sent to the dashboard**, which only shows the four-state chip. From
the phone, "OFF" is all the operator can see. Sending that error text to the
dashboard is a small change and the obvious first step.

**Diagnosis on the Pi:**

```bash
lsusb | grep -i 04d8              # is the MCP2221A enumerated?
lsmod | grep hid_mcp2221          # kernel driver claiming it? (should be empty)
bash ~/visionfsd-pi/pi3b/setup_mcp2221.sh
grep -i -E "imu|lsm6|mcp2221" ~/visionfsd-pi/pi3b/logs/robot.log | tail -20
```

**Impact while it is off:** by design, none of the safety or navigation
behaviour depends on it. The IMU is corroboration: one vote in stuck detection,
short-term yaw prediction, turn-rate limiting, and a tilt/pick-up cue. Non-IMU
mode is a supported configuration and must stay that way. Practically, stuck
detection has one fewer independent source, so it leans harder on LiDAR
displacement and the camera cue.

### 5.16 Open: pivoting in tight spots

**Observed in simulation, not yet reported from driving.** From one tight
starting position in the simulated living room, v1.9.23 spent about 10 s of 60
turning in place, logged as `ESCAPE_DIRECT_TURN_NO_SAFE_ARC`. It still escaped,
covered more ground than v1.9.22 did from the same spot, and never latched as
stuck. A looser turn cap showed the same behaviour, so the 127 cap is not the
cause: no forward arc fits, so the robot pivots, which is the intended safe
response.

**Watch for:** turning in place for more than ~10–15 s with an opening visible
on the LiDAR view, especially alternating left and right. On carpet, real
pivots may be slower than simulated ones, and v1.9.23 pivots more gently than
v1.9.22 did.

**Likely fix if it shows up:** allow tighter, slower arcs at crawl speed so the
robot can creep out instead of stopping to pivot.

---

## 6. Dashboard and manual control

`http://<pi-ip>:8080`, same Wi-Fi, view-only by default, no login. Treat it as
trusted-network-only.

- **LiDAR and plan view** — the same visualisation as the HDMI screen: returns,
  the map, the planned arc and the robot's intent, drawn client-side on a
  canvas so the Pi only ships data.
- **Camera tab** — MJPEG at about 5 fps, with the same controls available.
- **Manual control** — stop, resume, and hold-to-drive arrows over a WebSocket
  (`/ws`), with a POST fallback.

Manual control history: taps originally produced a fixed ~45° turn and felt
delayed and unreliable, because each press was a discrete queued command. It is
now proportional to how long the control is held, with pointer capture, a
0.35 s command TTL, and release-to-stop. Browser-verified against the real
dashboard server: 26% magnitude at 150 ms, 91% at 1 s, STOP within 60 ms of
release.

A stale indicator shows when frames stop arriving. The dashboard never queues
control commands, and it starts headless without a monitor attached.

---

## 7. What the camera is for — decided

**Decision: the camera is a remote view first, and a cheap motion cue second.
No VLM, no neural layer, no second perception stack.** Navigation is LiDAR's
job. This matches what the operator asked for: be able to sit in another room,
watch what the robot sees, and take control if wanted.

**What runs today, and what it actually costs.** There is **no VLM and no
neural model in robot mode** — that was considered and never built, so there is
no VLM load to remove. The camera does two things:

1. **Streams to the dashboard** — MJPEG at about 5 fps, 320×240. Only encoded
   when someone is watching (the server counts viewers).
2. **A non-neural motion cue** (`robot_camera_motion.py`) — tracks at most 80
   corner features on a 160×120 grayscale image with Lucas-Kanade optical flow,
   and abstains unless the tracks are well-distributed and consistent both
   ways. Per-frame cost is logged as `cam_ms` in `robot.log`, so the real
   number on the robot can be read rather than guessed.

**Why keep the motion cue rather than strip the camera to a pure view.** It is
one of the independent votes in stuck detection, which was issue 5.2 — and with
the IMU currently off (5.15), dropping it would leave stuck detection resting
almost entirely on LiDAR displacement. It is a small, bounded, non-neural
computation on a tiny image, not a perception stack. If measurement on the
robot shows it is genuinely expensive, the right fix is to run it at a lower
rate, not to delete the only camera-side evidence that the wheels are turning.

**One thing to fix on principle:** the camera can currently slow the robot down
(`visual_caution`). That is the camera influencing motion, and the rule for
this system is that LD19 and the ultrasonic hold safety authority. A future
session should review whether that path still earns its place.

**If a VLM is ever revisited:** a Pi 3B cannot run one at control-loop rates,
so it would have to be advisory, off the critical path, heavily rate-limited,
and structurally unable to make the robot less cautious. Stacking a second
perception system on top of a LiDAR stack that still does not read complex
rooms well (5.14) would add load and a second source of disagreement, not
understanding. **Fix the LiDAR path first.**

---

## 8. Honest limits

- **Nothing in v1.9.20 through v1.9.23 has been verified on the robot** by the
  sessions that wrote it. Tests and simulation only.
- The simulator has no wheel slip, perfect odometry and clean geometry. It did
  not reproduce the collisions the operator saw with v1.9.22.
- Real LD19 mixed-pixel behaviour, pose drift while tracking, Pi 3B CPU
  headroom with everything running, and Wi-Fi latency are all unverified.
- Do not promise this will work reliably in an arbitrary environment. It is an
  indoor robot with one 2-D LiDAR, one narrow ultrasonic beam, no encoders, and
  a movement floor it cannot go below.
