# VisionFSD: complete change record and handoff

Date: 2026-09-20  
Scope: work from v1.9.23 through v1.9.25 on `codex/pi3b-runtime`.

## 1. Current state

Two releases were implemented, tested, committed and pushed to the deployment branch:

| Release | Commit | Purpose |
|---|---|---|
| Baseline v1.9.23 | `aa190b7f798d64492e38008c3898afab3f639cc1` | Starting point for the review |
| v1.9.24 | `f08c632ab64e1a26438b889876f2132625d37b1d` | Expose IMU diagnostics and correct the LiDAR legend |
| v1.9.25 | `36e17f32e8d097da846147198c74c656a38a135a` | Correct navigation geometry, memory, stale guidance, command age and MCP2221 setup |

The remote deployment ref was verified against the full v1.9.25 commit after pushing. This confirms publication for OTA, **not installation or execution on the robot**.

The full latest Python suite completed: **493 tests, OK with one existing skip**. Static checks and dashboard JavaScript checks passed. These are desktop software results. No floor test, successful physical IMU connection, Pi timing measurement or confirmed Pi installation was obtained.

The supplied dashboard address was `http://192.168.0.17:8080`. A read-only HTTP attempt timed out. No telemetry was obtained and no movement commands were sent. Port 8080 is the dashboard port; an SSH username was not supplied.

The overall goal remains drop → run → go: place the robot in a house and have it drive sensibly and adapt. The changes address specific reproducible software defects; they do not establish reliable navigation in arbitrary homes.

## 2. Repository, hardware and document context

- Repository: <https://github.com/YoMosa2009/VisionFSD-Pilot>
- Deployment branch: `codex/pi3b-runtime`, **not `main`**.
- Actual development checkout: `C:/Users/user/source/repos/VisionFSD-Pilot`.
- The task's configured `E:/VisionFSD-Pilot` did not exist on this machine.
- Runtime source: `pi3b/`.
- Pi installation root: `~/visionfsd-pi`; runtime directory: `~/visionfsd-pi/pi3b`.
- Hardware described by the operator: Raspberry Pi 3B running 64-bit OS, Arduino Uno R3 with OSOYOO motor shield, FHL-LD19 360-degree USB LiDAR, USB webcam, front ultrasonic sensor owned by the Uno, and LSM6DS3 over MCP2221A USB-I2C.
- No wheel encoders. PWM and command duration are not measurements of distance travelled.

Both `F:/OtherStuff/HANDOFF.md` and `pi3b/HANDOFF.md` were read in full. Their contents matched, with SHA-256:

```text
A6C0C8F16C60EC115310542D51FC7509417D3B4D7814ECB38D9327460D223EEF
```

The original repository handoff was untracked and was preserved unchanged and unstaged. Its proposals were evaluated against the user's request and source code; they were not treated as new user instructions.

[HANDOFF_REVIEW.md](HANDOFF_REVIEW.md) contains the extensive section-by-section baseline audit and research. It describes the first implementation, v1.9.24. Its descriptions of the **500 ms Pi command lease** and **setup-marker early exit** are historical: v1.9.25 changes those mechanisms as documented below. This file is the consolidated account of both releases.

The shared README also retains material about an older neural road application. That is not evidence that the robot launcher runs neural inference. This work added no neural model, VLM or heavy dependency; the robot launcher selects `robot_autonomy.py`.

## 3. Safety and resource constraints retained

1. Live LiDAR and Uno ultrasonic protection retain final authority over movement. Global routes and memory cannot override local clearance decisions.
2. Nonzero motor output must respect the Uno's 105 PWM movement floor. No sub-105 crawl was introduced.
3. Non-IMU operation remains supported. IMU yaw is relative; accelerometer readings are not integrated into position.
4. Motion estimates without encoders remain estimates, including the corrected reverse and turn transforms.
5. Global planning remains asynchronous and bounded. A software budget is not proof of its real-time performance on a loaded Pi 3B.
6. Firmware was not changed. Its independent 350 ms serial-silence timeout and ultrasonic stop remain.
7. Camera caution was not removed to make movement appear more successful.
8. No destructive Git commands were used. Protected and unrelated paths were preserved.

The existing local footprint and global inflation are not equivalent: local planning uses two offset circles with a margin, while global planning uses scalar inflation on a coarser grid. Global reachability alone cannot establish that a turn is executable. The local veto is therefore essential.

## 4. v1.9.24: IMU diagnostics and dashboard correction

### 4.1 Missing IMU failure information

**Problem:** the phone dashboard exposed an IMU state label without the underlying error. An `OFF` label could not distinguish a disabled IMU, initial connection failure, register-configuration failure or later USB disconnection.

**Change:** `build_telemetry` in `robot_autonomy.py` now supplies these fields in both light and full health telemetry:

| Field | Meaning and bound |
|---|---|
| `imu_error` | Existing sampler error, limited to 512 characters |
| `imu_hold` | Calibration hold reason, limited to 80 characters |
| `imu_calibration` | Rounded calibration progress percentage |
| `imu_age_s` | Nonnegative age of the latest sample, or null before any sample |

Telemetry reads the existing snapshot. It performs no new USB reads, retries or setup operations.

The shared dashboard IMU card displays this information, including in camera view. Existing `OFF` / `CAL` / `STALE` / `LIVE` state precedence is retained. Old errors clear after recovery, and older telemetry without the additional fields remains supported.

Error strings are rendered with `textContent`, so exception text cannot become HTML markup.

**Evidence:** three Python telemetry regressions cover failure/recovery/disconnection, calibration/sample age, and bounded text/missing sample age. A Node test executes the actual page function with a stub DOM, including markup-looking error strings and missing fields. It also checks page JavaScript syntax.

**Limit:** this is not a real-browser layout test or a physical IMU repair. `OFF` still does not prove that the device never opened. Board power alone does not prove USB-I2C communication.

### 4.2 Misleading LiDAR legend

The dashboard legend previously said suspected edge artefacts were removed. It now says:

> Possible edge artefact (still used for safety)

This makes the display agree with the existing safety path, which retains accepted returns. It does not change packet validation, sensor range/age acceptance or motion decisions.

### 4.3 Review artifact

Added [HANDOFF_REVIEW.md](HANDOFF_REVIEW.md): architecture checks, all original handoff sections, corrections to assumptions, research references, limitations and the recommended engineering sequence.

No motor behavior changed in v1.9.24.

## 5. v1.9.25: navigation and runtime fixes

### 5.1 Incorrect opening bearings across the circular scan boundary

**Location:** `robot_local_planner.py`, `find_gap`.

The circular mask is rotated before finding runs. The center calculation incorrectly halved the rotation offset:

```python
# Before
(start + index + end) / 2 + .5
# After
start + (index + end) / 2 + .5
```

This could report an opening at the wrong bearing and let recovery declare alignment while still pointing away from the actual opening.

Regression coverage sweeps opening bearings from -170 to +170 degrees in 10-degree increments and checks rotational consistency. A simplified closed-loop gap-pivot case previously declared alignment about 45 degrees away from the real opening; the corrected implementation meets the test's 20-degree tolerance.

The closed-loop test supplies a specified pivot response. It is not a complete simulator of the robot, wheel slip, battery condition or actual turn speed. The fix does not add tighter rolling arcs or reduce clearance margins.

### 5.2 Inconsistent non-IMU turn prediction

**Location:** `robot_autonomy.py`, `AutonomousPolicy._track_local_motion`.

Obstacle memory and the committed opening previously used different command-based turn estimates. Their geometry could drift apart during the same pivot.

The runtime now calculates one yaw step and uses it for both local obstacle-memory transformation and committed-opening/watchdog tracking. It uses an available IMU yaw delta; otherwise it uses the existing chassis command-based turn model.

A regression checks that a non-IMU counter-rotation transforms memory and the committed opening by the same angular step.

This fixes internal consistency. It does not make an open-loop turn estimate a measured heading or remove slip error.

### 5.3 Reverse motion was missing from obstacle-memory translation

**Location:** `robot_local_planner.py`, `LocalPlanner.track_motion`.

The memory transform reused a forward-only speed function that clamps reverse motion to zero. That behavior is appropriate for the forward speed governor but wrong when transforming remembered obstacles after reversing.

Memory translation now uses signed average wheel command, modeled top speed and elapsed time. A remembered front obstacle therefore moves farther away in robot-relative coordinates during predicted reverse motion.

The regression covers reverse commands at -105 PWM on both wheels and accounts for memory-grid quantization. Actual reverse travel remains unknown without measurements.

### 5.4 Integer rounding shortened occupancy-map memory

**Location:** `robot_slam_lite.py`, map integration, shift and reset paths.

Occupancy decay was repeatedly truncated to integers. Frequent updates therefore erased evidence faster than the configured half-life implied. A cell starting at 200 fell to 66 after 12 seconds at 10 Hz instead of approximately 100.

A float32 fractional remainder now preserves sub-integer evidence between scans. The visible occupancy grid remains uint8. Fractional state shifts with the map and resets with it; reset also clears the decay timestamp. Live ray-based free-space clearing remains enabled.

Tests integrate the actual grid at 0.1, 0.5 and 1.0 second intervals and verify approximately the same 12-second half-life. Separate coverage checks shift/reset behavior.

At the default 576 × 576 map size, the extra persistent array costs 1,327,104 bytes, about 1.27 MiB. This is a correction to the existing local rolling map, not a larger map or persistent house map.

### 5.5 Unbudgeted committed-goal and fallback searches

**Location:** `robot_explorer.py`, `FrontierExplorer`.

Some committed-goal and no-progress fallback searches did not receive the common deadline. Searching alternatives first also spent the available budget before validating the current commitment.

The planner now validates the committed goal first and passes the same absolute deadline through the relevant A* paths. The existing 120 ms budget and 12,000-visit search bound remain.

Regression coverage checks deadline propagation for ordinary commitment and stalled-goal replanning. Actual Pi scheduling and search tail latency remain unmeasured.

### 5.6 Unsafe assumptions when reusing a cached global route

On timeout, cached guidance is now retained only when the previous route is active, commitment is valid, goal progress is sufficiently recent, and the current reachable-grid mask equals the cached one.

Otherwise the goal is dropped with a timeout/unreachable state and retried later. A newly blocked route cannot be retained merely because replanning ran out of time.

The regression introduces a new obstacle while forcing timeout and checks that guidance is withdrawn. Existing coverage also preserves valid cached guidance when the reachable mask is unchanged.

### 5.7 Global guidance could outlive its map snapshot

**Location:** `robot_explorer.py`, `AsyncExplorer`.

Map snapshots now carry their publication time. Active guidance based on a snapshot older than 1.5 seconds is returned as inactive `STALE_MAP`. The worker skips stale snapshots and publishes inactive `PLANNER_ERROR` on an exception for the current epoch. Successful processing clears the prior error.

Existing epoch invalidation remains in place. Withdrawing global guidance leaves local sensor-gated planning responsible for safe behavior; it is not permission to bypass safety checks.

A regression checks that an old global snapshot cannot continue supplying active steering guidance.

### 5.8 Heartbeats could extend an old planning command too long

**Location:** `robot_autonomy.py`, `UNO_CONTROL_LEASE_S`.

The Pi heartbeat can resend a command even when the main planning loop has stalled. Consequently, the Uno's 350 ms serial-silence timeout alone does not bound the age of the planning decision: heartbeats are still serial traffic.

The Pi-side command lease was shortened from **500 ms to 250 ms**. The heartbeat interval remains **90 ms**. The Uno's separate **350 ms** silence timeout is unchanged.

Regression coverage verifies expiration and STOP behavior. An older test explicitly expecting a command to survive for 450 ms was updated to the new contract: DRIVE at 200 ms, STOP at 260 ms, and no repeated fresh command at 450 ms.

These values are software thresholds, not a guaranteed physical stopping time. USB delivery, scheduling, braking and Uno behavior still need measurement together on the robot.

## 6. v1.9.25: bounded MCP2221 setup repair

### 6.1 Why the old setup could stay broken

`setup_mcp2221.sh` previously returned early when `.mcp2221-system-v1` existed. A successful historical setup did not prove that present permissions or kernel-driver ownership were still correct.

The marker is now informational. Setup checks actual rules and driver state on every invocation.

### 6.2 What setup now does

- Reconciles device-specific USB and hidraw rules for vendor `04d8`, product `00dd`.
- Retains the existing `MODE="0666"` permission policy and explicitly covers hidraw nodes.
- Ensures `blacklist hid_mcp2221` is present in `/etc/modprobe.d/visionfsd-mcp2221.conf`.
- Unloads a currently loaded `hid_mcp2221` driver; an unload failure is no longer masked as success.
- Reloads udev rules and retriggers USB and hidraw add events for existing devices.
- Writes the completion marker only after successful completion.
- Avoids rewriting already-correct rule content.

The permissions permit local users to read/write the matching device. Driver unloading and the persistent rules/blacklist are privileged host changes. The operator explicitly approved the bounded boot repair after these changes were explained.

### 6.3 Boot versus manual setup

| Path | Behavior |
|---|---|
| Boot: `setup_mcp2221.sh --repair` | Uses `sudo -n`; no password prompt, package installation, network work or initramfs rebuild |
| Manual: `bash setup_mcp2221.sh` | Can prompt for sudo; checks/install existing required system packages if missing; refreshes initramfs when adding the blacklist and the utility exists |

`run_robot.sh` attempts repair after OTA/re-execution and before runtime hardware ownership:

```bash
timeout --kill-after=1s 8s bash "$ROOT/setup_mcp2221.sh" --repair
```

The timeout is eight seconds with a one-second forced-kill grace. Failed sudo, a busy driver or timeout is logged; optional non-IMU startup continues. The repair is conditional on the script and `timeout` being available. This is not a transactional rollback of privileged changes already completed before a later failure.

### 6.4 Verification and limits

Five isolated tests execute copies of the actual shell script with temporary paths and fake privileged commands:

1. An old marker does not bypass missing rules or a loaded driver.
2. Correct rules are not rewritten by a repeated repair.
3. Denied sudo makes no setup-file changes and does not report success.
4. Failed driver unload is not reported as successful setup.
5. A deliberately hanging setup is timed out and the stub runtime still starts.

The launcher test uses the real timeout path. Desktop testing did not modify actual Linux device permissions or unload a physical driver.

This repair addresses plausible setup failure mechanisms, not a confirmed diagnosis of this Pi. Wiring, USB health, Python dependencies, I2C communication and sensor identity remain possible causes. A successful setup message is not a successful sensor read. The dashboard error is still needed to identify a remaining failure.

## 7. Original open problems: current disposition

| Original section | Work completed | Still open |
|---|---|---|
| 5.14: whole-home navigation | Fixed opening geometry, consistent memory transforms, occupancy decay, search budgets and stale guidance | Reliable pose, relocation, persistent mapping and repeatable household performance remain unproved |
| 5.15: IMU OFF | Exposed exact diagnostics; repaired stale setup-marker behavior and added bounded boot reconciliation | Actual physical fault and successful IMU operation remain unverified |
| 5.16: tight-space pivots | Corrected a reproducible false-bearing cause of excessive/misaligned pivoting; added a closed-loop regression | No new tighter rolling-arc policy; real chassis clearance, slip and pivot duration need floor testing |

Rotation matching already existed before this work: `LidarSlamLite._align_yaw` compares angular shifts and rejects weak/ambiguous matches. Translation matching also exists separately. These fixes did not introduce joint scan matching, loop closure or absolute localization.

The map is 576 × 576 cells over a 12 m local window, approximately 2.08 cm fine resolution with a coarser planning grid. Observed/visit arrays provide local exploration history. Recentring discards outgoing space; decay correction does not create a persistent whole-house representation.

No historical claims about contact reduction, successful room transitions or simulation percentages were independently reproduced here.

## 8. Verification record

| Stage | Result |
|---|---|
| Baseline v1.9.23 | 475 Python tests, 129.720 s, OK with one skip |
| v1.9.24 | 478 Python tests, 120.646 s, OK with one skip |
| Final v1.9.25 | 493 Python tests, 98.838 s, OK with one skip |
| Python compilation | `compileall` passed |
| Python static analysis | Pyflakes passed |
| Shell syntax | `bash -n` passed for all nine shell scripts |
| Dashboard JavaScript | Node regression and extracted page syntax checks passed |
| Patch whitespace | `git diff --check` and staged diff check passed |
| Publication | Deployment branch push completed; remote commit matched `36e17f3…` |
| Hardware | Not tested; dashboard request timed out |

The first v1.9.25 full run exposed the legacy test expecting the old half-second lease. Its expectation was changed to the intentionally stricter lease, and the complete suite was rerun successfully. The table reports the final results, not an uninterrupted first-pass success.

The 15 additional v1.9.25 Python regressions comprise ten navigation/runtime tests and five setup/launcher tests. v1.9.24 added three telemetry tests plus the separate JavaScript test. Test-count differences are not measures of coverage completeness.

An optional desktop microbenchmark of 50 integrations with 360 synthetic returns reported median 12.62 ms, p95 18.67 ms and maximum 19.57 ms. It was not a Pi benchmark, a full control-loop benchmark or a before/after performance comparison.

### Repeating software checks

On a development machine with the required dependencies, from the repository root:

```bash
cd pi3b
python -m unittest discover -s tests
cd ..
python -m compileall -q pi3b
python -m pyflakes pi3b
for script in pi3b/*.sh; do bash -n "$script" || exit; done
node pi3b/tests/test_dashboard_imu.js
git diff --check
```

Use the environment's actual Python executable. This desktop checkout used root `.venv/Scripts/python.exe`; the installed Pi runtime uses `pi3b/.venv/bin/python`. Node is for development checks, not a newly required robot runtime dependency. Compileall bytecode was redirected to a temporary cache during release validation.

## 9. Changed-file inventory

Paths below are relative to the repository root. Only intended release files were staged.

| File | Changes |
|---|---|
| `pi3b/robot_autonomy.py` | IMU health telemetry, shared motion yaw step, 250 ms command lease |
| `pi3b/robot_local_planner.py` | Circular opening-center formula and signed reverse memory translation |
| `pi3b/robot_slam_lite.py` | Fractional occupancy evidence, map-shift/reset handling |
| `pi3b/robot_explorer.py` | Shared search deadlines, committed-goal priority, cached-route validation, snapshot age and error handling |
| `pi3b/setup_mcp2221.sh` | Actual-state reconciliation, hidraw permissions, driver handling, manual/repair distinction |
| `pi3b/run_robot.sh` | Bounded pre-runtime MCP2221 repair |
| `pi3b/web/index.html` | IMU diagnostic display and corrected LiDAR legend |
| `pi3b/tests/test_dashboard_link.py` | Telemetry regressions |
| `pi3b/tests/test_dashboard_imu.js` | Dashboard-function regression and JavaScript syntax checking |
| `pi3b/tests/test_navigation_geometry_regressions.py` | Ten geometry, map, planning and command-age regressions |
| `pi3b/tests/test_mcp2221_setup.py` | Five isolated setup/launcher regressions |
| `pi3b/tests/test_robot_autonomy.py` | Updated heartbeat lease expectation |
| `pi3b/README.md` | Both release descriptions, reasons, verification and honest limitations |
| `pi3b/VERSION` | Bumped through 1.9.24 to 1.9.25 |
| `pi3b/HANDOFF_REVIEW.md` | Baseline audit and research |

This consolidated Markdown file was created after the two release commits. It is documentation, not an additional behavior release.

## 10. OTA and installation verification

The behavior update has already been pushed to `origin/codex/pi3b-runtime`. The boot launcher checks for updates, but an inaccessible Pi cannot be assumed to have fetched or restarted into the new version.

Recommended path: with the robot stopped, install the deployment branch if necessary, then restart through its existing launcher/autostart arrangement and verify the resulting startup log. Do not start a second independent runtime alongside an existing process.

Manual update on the Pi:

```bash
bash ~/visionfsd-pi/pi3b/update.sh codex/pi3b-runtime
```

Check installed files:

```bash
cat ~/visionfsd-pi/pi3b/VERSION
git -C ~/visionfsd-pi rev-parse HEAD
```

For this release, expect `1.9.25` and commit `36e17f32e8d097da846147198c74c656a38a135a`, unless a later release has since been published. The updater may install a detached HEAD; a blank current-branch display alone does not indicate failure.

Installed files do not identify the version of an already-running process. After a controlled restart, inspect:

```bash
tail -n 120 ~/visionfsd-pi/pi3b/logs/robot.log
```

The launcher records the startup version and commit. It also logs the boot repair result. The preceding run is retained in `robot.previous.log` in the same directory.

Open the dashboard at the operator-provided address, `http://192.168.0.17:8080`, if still correct. Record the exact IMU state, error, calibration hold/progress and sample age. Do not infer repair success merely from receiving OTA or from the adapter's LED.

If boot logs specifically report that noninteractive sudo prevented setup, the existing manual setup entry point is:

```bash
bash ~/visionfsd-pi/pi3b/setup_mcp2221.sh
```

Run setup while the runtime is stopped. Unlike boot repair, manual setup can install missing system packages and refresh initramfs under the conditions described above. An unknown remaining error should be diagnosed before repeatedly changing permissions or wiring.

## 11. Recommended next engineering work

Follow one sequence:

1. Confirm the Pi is actually running this release and capture the exact IMU diagnostic. Resolve the observed physical/setup fault rather than assuming the boot repair fixed it.
2. Verify sensor freshness and independent stop behavior with the robot controlled and supervised before free driving. Measure the complete command-age path with camera/dashboard active, including stalled planning and USB delays.
3. Record a supervised reproduction of the navigation failure with timestamps, sensor data and motor commands. Measure clearance, stationary/pivot time, recoveries and actual progress; do not substitute commanded travel for ground truth.
4. Extend a repository-owned replay/scenario runner with delayed/missing scans, slip, PWM-floor dynamics, doorways, dead ends, competing openings, symmetric rooms and displacement. The added pivot regression is only one bounded scenario.
5. Improve pose confidence and local/global consistency from those failures before adding persistent submaps or broader map retention.

Still unverified: physical IMU communication, Pi CPU/thermal/power headroom, scheduling tails, wheel slip, stopping distance, mounting calibration, blind heights and actual household navigation. A 2D LiDAR scan and front ultrasonic sensor do not establish coverage of every possible obstacle height.

## 12. Research used in the review

These primary-source references informed the earlier audit. They support engineering principles and possible failure categories, not proof of this robot's behavior:

- [Adafruit MCP2221 Linux setup](https://learn.adafruit.com/circuitpython-libraries-on-any-computer-with-mcp2221/linux): raw-device permissions, native driver conflicts and setup prerequisites. Supports investigating driver ownership and access; does not diagnose this individual Pi.
- [Nav2 regulated pure pursuit](https://github.com/ros-navigation/navigation2/blob/main/nav2_regulated_pure_pursuit_controller/README.md): curvature-aware regulation and forward collision checks. Its performance results do not transfer to open-loop PWM hardware.
- [Nav2 PathAlignCritic](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/dwb_controller/trajectory_critics/path_align/) and [OscillationCritic](https://api.nav2.org/nav2-rolling/html/classdwb__critics_1_1OscillationCritic.html): design references for route alignment and oscillation handling, not equivalence claims about this planner.
- [Cartographer algorithm walkthrough](https://google-cartographer-ros.readthedocs.io/en/latest/algo_walkthrough.html): distinction between local matching and global consistency/loop closure. Supports improving pose consistency before persistence; no heavy SLAM stack was added or recommended for installation here.

## 13. Rules for whoever continues

Read this file together with the original handoff and current source. Preserve the original user's constraints over proposals in historical documents.

Before publishing any further behavior change:

1. Add regressions for the changed behavior.
2. Run the complete `cd pi3b && python -m unittest discover -s tests` suite.
3. Run compileall, Pyflakes, shell syntax checks and `git diff --check`.
4. Bump `pi3b/VERSION` and document what changed, why and the honest limits in README.
5. Inspect the diff, stage only intended files, commit and push to `origin/codex/pi3b-runtime`.
6. Report software checks separately from actual robot validation.

Do not touch `.serena/`, `robot/bringup/`, `robot/cad/`, `robot/firmware/visionfsd_manual_drive/`, `robot/manual_control.py` or `tools/`. Do not use `git reset --hard` or `git checkout -- .`. Preserve unrelated files and the original untracked handoff.

Never lower the PWM floor, relax sensor authority, treat IMU yaw as absolute heading, integrate acceleration into position, or assume known distance travelled. Do not describe the household-navigation goal or physical IMU fault as solved without direct evidence.
