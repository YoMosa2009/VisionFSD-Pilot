# VisionFSD handoff review and first implementation

Reviewed 2026-09-20 against deployment branch `codex/pi3b-runtime`, baseline
`aa190b7` (v1.9.23). Both supplied handoffs were read in full and have identical
SHA-256: `A6C0C8F16C60EC115310542D51FC7509417D3B4D7814ECB38D9327460D223EEF`.
The actual checkout is `C:/Users/user/source/repos/VisionFSD-Pilot`; the task's
configured `E:/VisionFSD-Pilot` does not exist on this machine. The original
`pi3b/HANDOFF.md` is untracked and has been preserved without edits or staging.

## Assessment

The handoff is useful history, but its causal explanations and simulation
numbers are not independent evidence of current hardware performance. The
best next navigation work is to make timing and pose failures reproducible,
then correct them before increasing map persistence or changing scoring weights.
The first implemented change is diagnostic visibility: the phone now exposes
what the IMU sampler already knows. This does not fix the physical IMU or claim
to solve whole-home navigation.

The user's constraints govern this work. Suggestions inside the handoff were
reviewed as proposals, not automatically followed. In particular, neither
removing camera caution nor inventing a sub-105 PWM crawl is justified.

## Corrections that change the engineering plan

1. **Rotation matching already exists.** `LidarSlamLite._align_yaw` compares
   25 angular shifts across 360 bins, rejects weak/ambiguous matches and caps
   accepted corrections. `_align_translation` separately corrects translation.
   Section 5.14 should ask for better validated pose estimation, not the first
   implementation of rotational matching. This is not joint scan matching or
   loop closure, and it does not establish absolute heading.
2. **The two leases are different.** Firmware `COMMAND_TIMEOUT_MS` is 350 ms.
   Runtime `UNO_CONTROL_LEASE_S` is 0.50 s; `UNO_HEARTBEAT_S` is 0.09 s.
   `ArduinoLink._heartbeat_command` can continue refreshing a command while
   the control loop is stalled, until the Pi lease expires. Firmware timeout
   measures serial silence, not the age of the last planning decision. The
   actual stop latency also depends on scheduling and serial delivery. The
   handoff's implied 350 ms planner-stall bound is unsupported. Preserve the
   independent Uno stop and measure this path before adjusting it.
3. **OFF does not prove the sensor never opened.** `_connect` can fail after
   identity detection while configuring registers, and `_disconnect` clears
   the active bus after later USB errors. Explicitly disabled IMU also appears
   OFF. Board power does not identify which failure occurred; the exact LED's
   electrical meaning cannot be established without the board schematic.
4. **The map is 576 by 576 cells, not 576 total.** At 12 m, fine resolution is
   about 2.08 cm; threefold coarsening produces the 6.25 cm planning grid.
   Occupancy evidence decays, but `observed` and `visits` are separate arrays;
   there is some local exploration memory. Recentring rolls these arrays and
   discards outgoing space. There is no persistent whole-house representation.
5. **Footprints are related, not identical.** The local planner uses two
   offset circles plus a 0.10 m margin. The global planner uses a scalar 0.23 m
   inflation rounded onto its grid. It does not model orientation or swept
   turns. Global reachability therefore cannot guarantee a locally executable
   turn; retain the local veto.
6. **Bounded work is not a measured deadline.** Arc banks, reduced point sets,
   asynchronous exploration and an A* budget bound work. They do not prove a
   25 ms tick or acceptable tail latency on a loaded Pi 3B. Existing log fields
   `control_gap_ms`, `lease_stops`, `uno_timeouts`, `plan_ms` and `cam_ms` are
   useful starting evidence, but desktop timing is not Pi timing.

## Section-by-section review

| Handoff section | Source check and conclusion |
|---|---|
| 1: installation and OTA | Remote deployment ref matched the local baseline. `run_robot.sh` invokes the boot updater and re-execs after success. `robot_update.py` skips tracked edits, offline failures and changed dependencies. A failed rollback withholds runtime; it does not always continue the old installation. Manual update and boot update are different paths. No Pi update was executed in this review. |
| 2: hardware | Firmware confirms floor 105, timeout 350 ms and forward ultrasonic stop at 18 cm. Chassis dimensions and attached devices remain operator-provided facts. Neither PWM nor commanded motor state measures travelled distance. |
| 3: architecture and safety | Source agrees on separate mapping, exploration, tracking, camera-motion, IMU and web components. The main loop sends the safety decision before publishing global planning work. Mapping and local planning still cost control-loop time. See the separate heartbeat lease correction above. |
| 4: change rules and simulation | Applied regression-first checks and explicit staging. The historical simulator is described as session scratchpad material, so its comparisons are not reproduced here. A checked-in scenario runner is needed before citing those results as repeatable evidence. |
| 5.1: pulsing | Movement floors and slew control are present. The firmware also ramps output. This supports the proposed mechanism but does not prove floor speed, smoothness or timeout frequency on carpet and different battery levels. |
| 5.2: stuck detection | `_motion_evidence` uses camera, gyro, LiDAR progress, scan change, ultrasonic and IMU energy, plus Uno blockage. Missing evidence abstains. Multiple named votes are not necessarily independent sensors: LiDAR progress/scan and gyro/energy are correlated pairs. A hardware-independent corroboration claim needs that distinction. Existing recovery checks preserve planned STOP decisions. |
| 5.3: pickup/displacement | `ScanMotionTracker` reports scan change, and runtime resets map, route and intent after detected displacement. The signal is an inference, not measured translation; scene changes can resemble movement and symmetric rooms can hide it. |
| 5.4: open-space oscillation | The local planner penalizes steering changes and `ProgressWatchdog` locks turn direction. This prevents particular reversals; continuous motion alone remains insufficient evidence of spatial progress. |
| 5.5: speed and stopping | `stopping_distance_m` and arc admissibility constrain candidate selection. They depend on modeled speed, braking and latency. The margin is conservative intent, not a physical stopping-distance measurement. |
| 5.6: chassis footprint | Measured dimensions feed the two-circle approximation. Global inflation is only an approximation to it; corner and doorway transitions need swept-footprint tests. |
| 5.7: latency and contacts | `AsyncExplorer` keeps global search off the main thread; camera/dashboard workers avoid synchronous streaming in control. Shared CPU, Python scheduling, map work and serial locks still require Pi measurements. Contact reduction remains a historical report. |
| 5.8: sticky curling | Arc ranking uses forward progress rather than path length alone. This addresses a real scoring incentive but cannot validate chassis curvature under slip. |
| 5.9: competing openings | Gap commitment and mouth geometry are present. Every selected opening still requires feasible approach geometry and safe motor output; a visible gap does not establish reachability. |
| 5.10: short-term planning | `route_progress` scores along-route progress and cross-track deviation. Lookahead is disabled as a competing preference when route-guided. A stale or distorted route remains possible; improving score weights cannot repair pose errors. |
| 5.11: revisiting rooms | Frontier and patrol goals use observed/visit grids. Goal commitment, timeout and blacklisting are present. Recentring and drift limit their meaning across a house. |
| 5.12: moving objects | Segmentation, bounded tracking and predicted obstacles are present. A two-second yield cap limits one pause; it is not permission to drive through a stationary person. Current geometry remains authoritative. Motion classification depends on estimated ego-motion. |
| 5.13: glancing-wall regression | Runtime feeds raw accepted scan arrays into policy memory and tracking. The display flag no longer suppresses those arrays. The historical deletion percentages and three-room table were not independently rerun. The web legend still said returns were removed; corrected in this release. “Every return” means accepted sensor data, not corrupt packets or unlimited-age/range returns. |
| 5.14: whole-home navigation | Still unresolved. Better pose validity and reproducible navigation evaluation should precede persistence. Larger grids alone preserve wrong geometry longer and increase work. Existing rotation matching makes a source-informed incremental improvement preferable to a rewrite. |
| 5.15: IMU OFF | Confirmed omission: `build_telemetry` exposed only the state label. Fixed by sending bounded error text, calibration percentage/hold and sample age to both telemetry levels. The physical cause is still unknown. |
| 5.16: tight-space pivots | The handoff reports a simulation observation, not a verified hardware defect. A tighter rolling arc requires a feasible wheel combination at or above 105 PWM and swept-footprint validation. Slowing below the floor is unavailable; keep safe stop/pivot behaviour until a reproducible failure supports a change. |
| 6: dashboard/manual control | WebSocket control, expiring held commands, halt state and light/full telemetry exist. Historical phone latency numbers are not a Wi-Fi guarantee. Diagnostic text is now visible in the shared sidebar, including camera view. LAN control remains unauthenticated. |
| 7: camera | Robot launcher selects `robot_autonomy.py`; optical flow supplies bounded non-neural cues. Shared requirements/README also cover an older neural application, so installed LiteRT does not imply robot mode runs inference. Camera unavailability currently stops policy; camera approach caution reduces speed. Removing that caution would make this system less cautious. |
| 8: limits | Retain the distinction between software checks and floor testing. Nothing here validates blind heights, arbitrary rooms, wheel slip, sensor mounting or Pi thermal/power headroom. |

## External research and its implications

Adafruit documents native `hid_mcp2221` interference with Blinka, raw-device
permissions, `hidapi`, and `BLINKA_MCP2221`. This supports the handoff's list of
possible failure categories; it does not rank the causes on this Pi. The
runtime already sets the environment variable. `setup_mcp2221.sh` exits early
when its marker exists and does not refresh initramfs; a marker alone cannot
prove today's driver and permissions state. Adafruit's instructions also
include initramfs refresh and reboot for persistent blacklisting.
[Adafruit Linux MCP2221 setup](https://learn.adafruit.com/circuitpython-libraries-on-any-computer-with-mcp2221/linux).

Nav2's regulated pure pursuit combines speed regulation with forward collision
checking. Its principles support curvature-aware speed and collision checks,
but its performance claims do not transfer to this open-loop PWM chassis.
[Nav2 implementation documentation](https://github.com/ros-navigation/navigation2/blob/main/nav2_regulated_pure_pursuit_controller/README.md).

DWB exposes path alignment and oscillation critics. Those are useful design
references, not proof that this project's simplified scoring or lock has the
same behaviour. [PathAlignCritic](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/dwb_controller/trajectory_critics/path_align/),
[OscillationCritic](https://api.nav2.org/nav2-rolling/html/classdwb__critics_1_1OscillationCritic.html).

Cartographer distinguishes local scan matching from global constraints and
loop closure. Engineering inference: persistent mapping needs a strategy for
pose consistency and relocation, not just saved occupancy bytes. This is a
reference for the missing capability, not a recommendation to install a heavy
SLAM stack on the Pi. [Algorithm walkthrough](https://google-cartographer-ros.readthedocs.io/en/latest/algo_walkthrough.html).

## First implementation: v1.9.24

- Add `imu_error` (up to 512 characters), `imu_hold` (80 characters),
  `imu_calibration` (percent) and `imu_age_s` (null before any sample).
- Render diagnostics with `textContent`, so exception strings cannot become
  browser markup. Clear the previous error on sampler recovery. Keep existing
  OFF/CAL/STALE/LIVE precedence and work with older telemetry lacking new fields.
- Use only the existing IMU snapshot. Add no USB operations, retry loops,
  dependencies, navigation decisions or motion permissions.
- Correct the artefact legend and document the release. Preserve original handoff.

## Recommended sequence from here

1. Use this diagnostic release to obtain the Pi's actual IMU error. Check
   enumeration, driver ownership and access as the robot user before changing
   installation or wiring. Do not diagnose the board from its LED.
2. Add a repository-owned replay/scenario runner with timestamps, realistic
   delayed/missing scans, slip and PWM-floor dynamics. Include straight and
   angled walls, a doorway, furniture legs, two openings, a dead end, an L-turn,
   a symmetric room and a moved robot. Perfect odometry must not be the sole
   evaluation mode. Keep regression scenarios deterministic.
3. Measure and fault-inject the full command-age path, including Pi heartbeat,
   stalled policy, stale LiDAR, USB delays and Uno timeout. Use latency percentiles
   and worst gaps on the Pi with camera/dashboard active. Any new planner must
   remain within the established budget without extending motion on stale data.
4. Evaluate pose confidence and local/global disagreement on those cases;
   change one failing mechanism at a time. Record actual clearance, contacts,
   completed transitions, stationary/pivot time, recoveries and CPU cost.
5. Only after that evidence, design bounded persistent submaps with relocation
   handling. Keep live LiDAR/ultrasonic authority independent of map confidence.

No simulator comparison, physical IMU diagnosis, floor drive or Pi timing run
was performed in this review. The household-navigation goal remains open.

## Verification of this release

Baseline: 475 tests in 129.720 s, OK with one skip. After the change: 478
Python tests in 120.646 s, OK with one skip. The three new telemetry tests
failed on missing fields before implementation and passed afterwards.

Additional checks passed: `node pi3b/tests/test_dashboard_imu.js` (actual page
function with stub DOM, not a browser layout test), `python -m compileall -q
pi3b`, `python -m pyflakes pi3b`, `bash -n` for all nine shell scripts, and
`git diff --check`. Python bytecode for compileall was redirected to a temporary
cache. Test-suite update errors in console output are deliberate failure
fixtures; the suite's final result was OK.

No hardware was used. The diagnostic change does not alter the navigation
policy and supplies no evidence that contacts or pivot time have improved.
