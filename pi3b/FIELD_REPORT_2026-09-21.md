# Field report — 2026-09-21, v1.9.26

The first run of this robot measured from outside: full telemetry recorded over
the dashboard WebSocket, plus the launcher log pulled through the `/log.txt`
endpoint added in v1.9.26.

**This is measured data from the real robot**, not simulation. It is the first
such record in this project.

- Runtime: v1.9.26, commit `99de7d0`, confirmed by the boot log.
- Duration: 885 s (14 min 45 s), 1686 telemetry messages at ~1.9/s, 15 log
  snapshots. One session; a short sample.
- The operator lifted the robot to a safer spot several times during the run,
  and reports seeing it both get stuck and take sensible actions.
- Environment: the operator's home, mixed autonomous and manual driving.
- Raw data: session scratchpad `drive_logs/` (telemetry JSONL, log snapshots,
  `analysis_full.txt`). Not committed: it is large and machine-specific.

---

## 1. The finding that matters: the control loop runs at about 1 Hz

`control_gap_ms` is the measured interval between control decisions. It is
meant to be 25 ms (`CONTROL_PERIOD_S`).

| Window | median | p95 | max |
|---|---|---|---|
| Standby, before any dashboard viewer | 282–1373 (sampled) | — | 1373 |
| Driving, first snapshot | 889 ms | 1151 ms | 1946 ms |
| Driving, last snapshot | 484 ms | 1101 ms | 1243 ms |

**The loop is 20–35× slower than designed.** Everything below follows from it.

`lease_stops` reached **663 within the first minutes**. Every gap longer than
`UNO_CONTROL_LEASE_S` (0.25 s) expires the Pi-side command lease and the motors
stop until the next decision arrives. That is the stop/go pulsing, measured
from the robot itself: **267 stop/go transitions, 18.1 per minute**.

`uno_timeouts` stayed at **0** throughout, and LiDAR, camera and Uno health
never reported not-ok in 1686 messages. The serial link and the sensors are
fine. This is Pi-side compute.

Note the lease is not the root cause. At the old 500 ms lease these stalls
would still cut the motors, just less often. Raising it back would hide the
symptom and leave the robot acting on second-old decisions.

### Consequence: global planning almost never finishes

| explore mode | share of samples |
|---|---|
| `PLAN_TIMEOUT` | 1042 (62%) |
| `BUILDING_MAP` | 506 (30%) |
| `FRONTIER` (a real goal) | **59 (3.5%)** |
| `PATROL` | 42 |
| `NO_REACHABLE_TARGET` | 26 |

`plan_ms` median 124.2, p95 284.8, max 399.4 — against a 120 ms budget. The
frontier planner is timing out nearly every cycle, so for ~96% of the run the
robot had **no long-term destination at all**.

This is the honest explanation of "it fails to take the paths that are
genuinely logical". It is not mainly choosing badly between openings. It is
mostly driving with no destination, on decisions that are hundreds of
milliseconds stale - which is also why it circulates inside one area rather
than crossing the house.

### Consequence: it circulates rather than getting anywhere

**Corrected after operator review.** An earlier draft of this report said the
robot "went nowhere", from 63.3 m of path against 0.4 m of net displacement
over the whole run. That measure is invalid: there were **25 distinct
displacement events** (the operator lifting the robot to a safer spot, plus map
resets), and each breaks pose continuity, so start-to-end displacement across
15 minutes measures nothing. The operator's own observation — that it moved
around the living room and sometimes took sensible actions — is consistent with
the data once it is broken down properly.

Per minute, excluding steps over 1 m as pickups or resets:

| Minutes | Path travelled | Heading change |
|---|---|---|
| 0–5 | 0.2–2.1 m/min | up to 305°/min |
| 6–14 | 2.6–8.1 m/min | mostly modest |

Motors were commanded non-zero in **72%** of messages. So it does drive. The
pathology is the shape of that driving:

- The first six minutes are near-stationary with large rotation — minute 1 is
  0.2 m of travel against 244° of turning.
- Later minutes travel 4–8 m each, but net displacement per minute stays
  between 0.0 and 2.3 m: it circulates inside one area.
- One unbroken episode of `DRIVE:-18deg` lasting **123.5 s**, during which it
  travelled 1.7 m while its heading went 49° → 284° on a constant (105, 120)
  command. Two minutes of turning nearly in place.

Note that pose here is scan-matched and drifts, so per-minute path lengths are
estimates. The 1 Hz control-loop finding above does not depend on pose at all.

### It also drives very close to things

- Planned arc clearance: median 0.15 m, min 0.10 m.
- Nearest LiDAR return 0.08 m; under 0.25 m in 210 messages.
- Speed was at the 105 PWM floor for the entire run, and `reach1.60m
  stop0.09m` shows the stopping-distance governor working as designed.

So v1.9.23's speed work holds: this is no longer a speed problem. Being close
is now a consequence of stale decisions, not of driving fast.

### Time spent not driving

`STOP:DISPLACED_SETTLING` 65.6 s and `STOP:DISPLACED_REORIENT` 18.1 s — 9.5% of
the run spent believing it had been displaced. With the loop at 1 Hz, scan
matching compares frames nearly a second apart while the robot moves, which
will produce exactly this. Also `STOP:BOXED_IN` 12.6 s, and no admissible arc
in 30% of messages.

---

## 2. The IMU fault is now diagnosed

From the boot log:

```
MCP2221 rules checked and driver conflict cleared; sensor connection still
requires a successful identity read.
IMU USB error: MCP2221 USB timeout; reopening adapter
```

Then `MCP2221 USB timeout; reopening adapter` on **every one of 1675 messages**,
with `imu_age_ms` climbing to 945217 — not one successful sample in 15 minutes.

**What this rules out:** permissions, the udev rule, and the `hid_mcp2221`
kernel driver. The v1.9.25 setup repair ran and did its job. The adapter is
enumerated and the runtime opens it.

**Corrected by run 3.** This section originally concluded the I²C
transaction to the LSM6DS3 was timing out, and blamed wiring. That was not
established: the old message was identical for two different failures. With
v1.9.28's distinct messages the robot reported *adapter worker did not start
within 8.0 s* - the helper process that owns the USB adapter never started,
so the sensor was never even asked. The likely cause is software: the helper
was started with "spawn", which re-imports the whole runtime first. v1.9.29
forks it instead. Wiring remains possible, but is unproven either way until
the helper starts.

Impact: none on safety. Stuck detection loses one of its votes.

---

## 3. What has not been established

- **Where the loop time actually goes.** The suspects are the inline HDMI/
  dashboard render every 100 ms (a 640 px map render plus panel draw on the
  control thread), map integration, mover tracking, and full-telemetry
  serialisation of every LiDAR return. None of this is measured per stage
  yet — that is the first job, not a guess to act on.
- **How much the recording itself cost.** Full telemetry makes the Pi
  serialise every return a few times a second. Standby gaps of 282–1373 ms
  were logged *before* any recorder connected, so the stalls predate it, but
  the recording may have made them worse. A light-level or no-viewer
  comparison has not been run.
- Wheel slip, real stopping distance, thermal or power throttling over a long
  run (`throttled=0x0` at boot only).

---

## Run 2 — 2026-09-23, v1.9.27

A controlled 3-minute capture: boot log, then 60 s with no viewer, 60 s with
light telemetry, 60 s with full telemetry. No HDMI monitor and no other viewer
attached. Autonomous driving throughout. Runtime v1.9.27, commit `c69cc23`.

**Being watched is not the cost.** Loop rate 2.9 Hz with no viewer, 3.2 Hz
light, 3.7 Hz full. That settles the caveat in section 3.

**Where the time goes**, mean per tick (no-viewer phase):

| Stage | mean | worst |
|---|---|---|
| `sense` | 215 ms | 385 ms |
| `slam` | 122 ms | 253 ms |
| `render` | 29 ms | 211 ms — with no monitor and no viewer |
| `perceive` | 18 ms | 69 ms |
| `decide` | 17 ms | 63 ms |

Desktop profiling on the recorded sweeps traced `sense` mostly to the six
sector-clearance checks (n x n comparisons from Python point objects) and
`slam` to full-grid operations on every scan. `render` was drawing a window
nobody could see: the Pi runs a desktop session with nothing plugged in.

**The Pi was power-throttled**: `throttled=0x50005` at boot — under-voltage
and CPU throttling both active. The first run booted `0x0`. This slows every
stage and is a power-supply problem, not a software one.

**v1.9.27 helped, partly.** Loop 2.9-3.7 Hz against about 1-2 Hz before, and
the robot had a real frontier goal in 35-50% of samples against 3.5%; global
planning timeouts fell from 62% to 4-20%.

**The robot was wiping its memory every few seconds.** 33 "picked up" verdicts
in 3 minutes, each resetting the map and all visit history. Replaying the
recorded scans of both runs, 96 of 100 false verdicts are explained by the
robot having turned between the compared scans and 4 by loop stalls. This is
the main cause of the operator's report that it did not remember where it had
been. Fixed in v1.9.28.

The IMU still reported `MCP2221 USB timeout`, as expected with no wiring
change; v1.9.28 makes the message say which step timed out.

## Run 3 — 2026-09-23 evening, v1.9.28

Same 3-minute phased capture, no monitor, no other viewer. Operator watching:
lots of spinning, staying in one area, not finding open paths, poor at knowing
when stuck.

| | Run 2 (v1.9.27) | Run 3 (v1.9.28) |
|---|---|---|
| Loop rate (light/full phases) | 3.2-3.7 Hz | 6.6-6.7 Hz |
| Median control gap | 260-304 ms | 153-154 ms |
| Lease stops (light/full phases) | 95-132 | **0** |
| "Picked up" map wipes, whole run | 33 | 1 |
| `slam` on the control thread | 83-122 ms | 0.2 ms |
| `render` with no screen or viewer | 14-29 ms | 0.3-0.7 ms |
| Planner timed out | 4-20% | **~85%** |

`sense` (95-120 ms) and `perceive` (47-76 ms) are now the largest control-thread
costs. Power flags `0x50000`: under-voltage and throttling *had occurred*
since boot, not active at the moment of logging.

**Findings:**

- **No destination, again.** The planner timed out in ~85% of plans; a plan
  needs several desktop-equivalent hundreds of milliseconds on a Pi, nearly
  all Python A*. With no goal the robot turns toward the nearest opening,
  often behind it (bearings of -120 to +145 degrees). Fixed in v1.9.29.
- **No admissible arc in 52% of samples.** When blocked, the nearest return in
  the robot's lane ahead was a median 0.55 m away: genuine clutter, not
  ghosts.
- **Moving-object yields in ~22% of samples**, mostly slow (median 0.07 m/s).
  Mover counts do not differ between turning and straight driving, so they are
  not a pose-lag artefact. Some likely were the operator nearby.
- **Stall detection could not confirm anything**: only the whole-scan LiDAR
  source ever voted; the rule needs two. Fixed in v1.9.29.
- **IMU**: *adapter worker did not start within 8.0 s* - the helper process,
  not the sensor. Fixed in v1.9.29 as far as software can tell.

## 4. Plan for the next release

In order. Nothing here is a navigation-behaviour change: at 1 Hz, tuning
navigation is tuning noise.

1. **Measure per-stage loop time on the robot.** Add bounded timing around each
   main-loop stage (LiDAR ingest, scan motion, movers, map integrate, decide,
   render, telemetry), logged once a second and exposed in telemetry. Ship it,
   run it, read it. This turns the suspect list into a ranked one.
2. **Get rendering off the control thread.** The HDMI panel and the dashboard
   image are advisory; they must never sit between a scan and a motor command.
   Render from a snapshot on a worker thread, and skip rendering entirely when
   no window is shown and no viewer is watching.
3. **Cut the fixed per-tick cost**, ranked by step 1's numbers, with the target
   being a control gap p95 under 100 ms.
4. **Re-measure, then revisit the lease.** Once the loop is fast, decide the
   lease on evidence rather than restoring 500 ms to mask stalls.
5. **Only then** return to navigation: with a real frontier goal present most
   of the time rather than 3.5%, the planner's actual decision quality can be
   judged for the first time.

Each step follows the standing rules: regressions first, full suite, static
checks, VERSION bump, README, push to `origin/codex/pi3b-runtime`.
