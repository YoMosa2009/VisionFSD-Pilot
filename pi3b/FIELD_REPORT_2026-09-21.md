# Field report — 2026-09-21, v1.9.26

The first run of this robot measured from outside: full telemetry recorded over
the dashboard WebSocket, plus the launcher log pulled through the `/log.txt`
endpoint added in v1.9.26.

**This is measured data from the real robot**, not simulation. It is the first
such record in this project.

- Runtime: v1.9.26, commit `99de7d0`, confirmed by the boot log.
- Duration: 885 s (14.7 min), 1686 telemetry messages, 15 log snapshots.
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
milliseconds stale.

### Consequence: it goes nowhere

- Path length **63.3 m**, net displacement **0.4 m**.
- Moved less than 0.35 m in 12 s for **67%** of the run.
- One unbroken episode of `DRIVE:-18deg` lasting **123.5 s**. A constant
  steering angle held for two minutes is a circle.

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

**What remains:** the I²C transaction to the LSM6DS3 times out. That is the
adapter↔sensor link, not software — wiring (SDA/SCL/VCC/GND), missing I²C
pull-up resistors, a damaged breakout, or a bad USB cable. It should be
checked physically. No further software change will fix a link that never
answers.

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
