"""Tests for the pilot commands against the robot's real dashboard server.

A fake robot publishes telemetry and camera frames through pi3b/robot_web.py
exactly as the runtime does; the commands under test talk to it over HTTP and
WebSocket. Nothing here touches hardware or the network beyond localhost.

    python -m unittest haiku_pilot/test_robot.py
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pi3b"))

import robot  # noqa: E402
from robot_manual_move import ManualMoveExecutor  # noqa: E402
from robot_web import RobotControl, TelemetryHub, start_dashboard_server  # noqa: E402


class LegacyControl(RobotControl):
    """A robot before v1.9.32: no precise moves."""

    def state(self) -> dict:
        state = super().state()
        state.pop("moves", None)
        state.pop("pivots", None)
        return state


class GyroTurnControl(RobotControl):
    """A v1.9.32 robot: precise moves, but no timed pivots."""

    def state(self) -> dict:
        state = super().state()
        state.pop("pivots", None)
        return state


class _Status:
    def __init__(self, blocked: bool) -> None:
        self.blocked = blocked


class _ImuState:
    def __init__(self, yaw: float, rate: float) -> None:
        self.fresh = True
        self.calibrated = True
        self.yaw_deg = yaw
        self.gyro_z_dps = rate


class FakeChassis:
    """The Uno link and the IMU a precise move uses: wheel output turns the
    chassis at a PWM-dependent rate and it coasts briefly after a stop."""

    def __init__(self, robot: "FakeRobot") -> None:
        self.robot = robot
        self.differential_ready = True
        self.output = (0, 0)
        self.yaw = 0.0
        self.rate = 0.0
        # Position in the simulated room (FakeRobot(room=True)), metres.
        self.x = 1.6
        self.y = 1.4
        # Straight drives veer like the real chassis: forward clockwise.
        self.veer_dps = 0.0
        self.imu_live = True
        self._at = time.monotonic()
        self._lock = threading.Lock()

    def _advance(self) -> None:
        now = time.monotonic()
        elapsed, self._at = now - self._at, now
        left, right = self.output
        if left * right < 0:
            target = (right - left) / 2.0 * 1.25            # ~150 deg/s at 120 PWM
        elif left * right > 0:
            veer = -self.veer_dps if left > 0 else self.veer_dps
            target = 0.6 * (right - left) + veer
        else:
            target = 0.0
        # First-order response, ~60 ms: spins up and coasts down.
        blend = min(1.0, elapsed / 0.06)
        self.rate += (target - self.rate) * blend
        self.yaw = (self.yaw + self.rate * elapsed + 180.0) % 360.0 - 180.0
        if left * right > 0:
            speed = 0.25 if left > 0 else -0.25
            radians = math.radians(self.yaw)
            self.x += -math.sin(radians) * speed * elapsed
            self.y += math.cos(radians) * speed * elapsed

    def publish_drive(self, left: int, right: int, owner=None) -> None:
        with self._lock:
            self._advance()
            self.output = (left, right)
        if (left, right) != (0, 0):
            letter = "F" if left > 0 and right > 0 else "B" if left < 0 and right < 0 else (
                "L" if left < right else "R")
            self.robot.commands.append(letter)

    def claim(self, owner, until) -> None:
        pass

    def release(self, owner) -> None:
        pass

    def status(self):
        return _Status(self.robot.uno_blocked)

    def pose(self) -> tuple[float, float, float]:
        with self._lock:
            self._advance()
            return self.x, self.y, self.yaw

    def state(self):
        with self._lock:
            self._advance()
            state = _ImuState(self.yaw, self.rate)
        state.fresh = self.imu_live
        return state


class FakeRobot:
    """Publishes telemetry like the runtime, and applies the manual-mode
    forward guard the real policy applies."""

    def __init__(self, ahead_m: float = 2.0, behind_m: float = 1.5, scan_every_s: float = 0.0,
                 legacy: bool = False, room: bool = False) -> None:
        # room=True: the scan is ray-cast from the chassis's simulated pose
        # in a 5 x 4 m room, so LiDAR-measured moves can be checked.
        self.room = room
        self.ahead_m = ahead_m
        self.behind_m = behind_m
        # A busy robot sheds full telemetry: LiDAR points only this often.
        self.scan_every_s = scan_every_s
        self._next_scan_at = 0.0
        self.uno_blocked = False
        self.frame_number = 0
        self.frame_value = 0
        self.silent = False          # stop publishing telemetry (a Wi-Fi stall)
        self.commands: list[str] = []
        # Pivots (LiDAR-measured turns) need a scan that turns with the
        # chassis, so only the simulated room offers them.
        self.control = (LegacyControl() if legacy else
                        RobotControl() if room else GyroTurnControl())
        # The robot's real precise-move executor, on a simulated chassis.
        self.chassis = FakeChassis(self)
        self.mover = None if legacy else ManualMoveExecutor(
            self.control, self.chassis, self.chassis,
            lambda command, magnitude: {
                "F": (110, 110), "B": (-110, -110),
                "L": (-int(105 + 40 * magnitude), int(105 + 40 * magnitude)),
                "R": (int(105 + 40 * magnitude), -int(105 + 40 * magnitude)),
            }[command],
        )
        self.hub = TelemetryHub()
        started = start_dashboard_server(
            port=0, version="9.9.9", control=self.control, camera_fps=10.0, telemetry=self.hub,
        )
        if started is None:
            raise unittest.SkipTest("could not bind a dashboard port")
        self.stream, self.server = started
        self.address = f"127.0.0.1:{self.server._server.server_address[1]}"
        self.forward_blocked = False
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _scan(self) -> dict:
        if self.room:
            x, y, yaw = self.chassis.pose()
            points = _room_points(x, y, yaw)
            return {"x": [int(round(px * 100)) for px, _py in points],
                    "y": [int(round(py * 100)) for _px, py in points]}
        xs, ys = [], []
        for x in np.arange(-1.0, 1.0, 0.03):
            xs.append(x)
            ys.append(self.ahead_m + 0.13)       # wall ahead (LiDAR to wall)
            xs.append(x)
            ys.append(-(self.behind_m + 0.13))   # wall behind
        return {"x": [int(round(v * 100)) for v in xs], "y": [int(round(v * 100)) for v in ys]}

    def _run(self) -> None:
        while not self._stop.is_set():
            command, _magnitude = self.control.manual_input()
            if command != "STOP":
                self.commands.append(command)
            reason = "EXPLORE_FRONTIER"
            if self.control.manual:
                reason = f"MANUAL:{command}"
                if command == "F" and self.forward_blocked:
                    reason = "STOP:MANUAL_FORWARD_BLOCKED"
            if self.control.halted:
                reason = "STOP:HALTED_BY_OPERATOR"
            light = {
                "v": "9.9.9", "reason": reason, "pose": {"x": 3.0, "y": 3.0, "h": 0.0},
                "health": {"range": {"front_m": self.ahead_m, "rear_m": self.behind_m,
                                     "ultra_cm": int(self.ahead_m * 100), "uno_blocked": self.uno_blocked}},
                "control": self.control.state(),
            }
            now = time.monotonic()
            if self.silent:
                time.sleep(0.05)
                continue
            if now >= self._next_scan_at:
                self.hub.publish(light, dict(light, scan=self._scan()))
                self._next_scan_at = now + self.scan_every_s
            else:
                self.hub.publish(light, None)
            if self.server.camera is not None and self.server.camera.viewers > 0:
                self.frame_number += 1
                # Each frame a distinct solid shade, so a test can tell which
                # frame the pilot saved.
                self.frame_value = 30 + (self.frame_number * 40) % 200
                self.server.camera.publish(
                    np.full((240, 320, 3), self.frame_value, dtype=np.uint8)
                )
            time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        if self.mover is not None:
            self.mover.close()
        self.server.close()


def run(address: str, *args: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = robot.run(["--robot", address, *args])
    return code, out.getvalue()


class PilotCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = FakeRobot()
        self.addCleanup(self.robot.close)

    def test_observe_reports_the_senses_and_saves_the_camera_frame(self) -> None:
        if os.path.exists(robot.VIEW_PATH):
            os.remove(robot.VIEW_PATH)
        code, out = run(self.robot.address, "observe")
        self.assertEqual(code, 0, out)
        self.assertIn("Clear in your own lane: ahead 2.00 m (across your whole width), "
                      "behind 1.50 m (across your whole width)", out)
        self.assertIn("Ultrasonic straight ahead: 200 cm", out)
        self.assertIn("autonomous", out)
        self.assertTrue(os.path.exists(robot.VIEW_PATH))
        with open(robot.VIEW_PATH, "rb") as handle:
            self.assertEqual(handle.read(2), b"\xff\xd8")
        self.assertIn("LiDAR map saved:", out)
        self.assertIn("Open corridors", out)
        with open(robot.LIDAR_PATH, "rb") as handle:
            self.assertEqual(handle.read(4), b"\x89PNG")

    def test_no_move_without_manual_control(self) -> None:
        code, out = run(self.robot.address, "drive", "forward", "0.5")
        self.assertEqual(code, 5, out)
        self.assertIn("Manual Control is off", out)
        self.assertEqual(self.robot.commands, [])

    def test_manual_on_then_a_drive_moves_and_stops(self) -> None:
        code, out = run(self.robot.address, "manual", "on")
        self.assertEqual(code, 0, out)
        self.assertTrue(self.robot.control.manual)
        code, out = run(self.robot.address, "drive", "forward", "0.4", "0.5", "--say", "testing")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"Did: drive forward for 0\.4\d s")
        self.assertIn("F", self.robot.commands)
        time.sleep(0.4)
        self.assertEqual(self.robot.control.manual_input()[0], "STOP")

    def test_forward_close_to_something_is_allowed(self) -> None:
        """Manual Control has no proximity limit (v1.9.31, operator request)."""
        self.robot.ahead_m = 0.05
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "drive", "forward", "0.3")
        self.assertEqual(code, 0, out)
        self.assertIn("F", self.robot.commands)

    def test_reversing_close_to_something_is_allowed(self) -> None:
        self.robot.behind_m = 0.03
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "drive", "backward", "0.3")
        self.assertEqual(code, 0, out)
        self.assertIn("B", self.robot.commands)

    def test_the_arduinos_18_cm_stop_is_reported(self) -> None:
        """Only reflashing can remove the firmware's own forward stop; a move
        it holds back must say so, not look like a move that did nothing."""
        run(self.robot.address, "manual", "on")
        self.robot.uno_blocked = True
        code, out = run(self.robot.address, "drive", "forward", "1.0")
        self.assertEqual(code, 4, out)
        self.assertIn("18 cm ultrasonic stop", out)

    def test_a_person_pressing_stop_ends_the_move(self) -> None:
        run(self.robot.address, "manual", "on")
        threading.Timer(0.4, self.robot.control.halt).start()
        code, out = run(self.robot.address, "turn", "left", "180", "0.0")
        self.assertEqual(code, 5, out)
        self.assertIn("STOP was pressed", out)

    def test_manual_on_is_refused_while_a_person_has_stopped_the_robot(self) -> None:
        self.robot.control.halt()
        code, out = run(self.robot.address, "manual", "on")
        self.assertEqual(code, 3, out)
        self.assertFalse(self.robot.control.manual)

    def test_moves_are_capped(self) -> None:
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "turn", "right", "900", "0.5")
        self.assertEqual(code, 0, out)
        self.assertIn("asked 180 degrees", out)
        code, out = run(self.robot.address, "drive", "backward", "9")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"for 2\.0\d s")

    def test_manual_off_hands_the_robot_back(self) -> None:
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "manual", "off")
        self.assertEqual(code, 0, out)
        self.assertFalse(self.robot.control.manual)

    def test_an_unreachable_robot_is_a_clear_error(self) -> None:
        code, out = run("127.0.0.1:9", "observe")
        self.assertEqual(code, 2, out)
        self.assertIn("cannot reach the robot", out)


def _room_points(robot_x: float = 1.6, robot_y: float = 1.4,
                 yaw_deg: float = 0.0) -> list[tuple[float, float]]:
    """A 5 x 4 m room with a doorway in the right-hand wall and a box in it,
    as LiDAR points in the robot frame (robot facing +y when yaw is 0; yaw
    positive = turned left)."""
    walls = [((0, 0), (5, 0)), ((0, 0), (0, 4)), ((0, 4), (5, 4)),
             ((5, 0), (5, 1.8)), ((5, 2.7), (5, 4)),
             ((0.8, 3.0), (1.4, 3.0)), ((1.4, 3.0), (1.4, 3.5))]
    yaw = math.radians(yaw_deg)
    points = []
    for step in range(450):
        angle = math.radians(step * 0.8)
        rx, ry = math.sin(angle), math.cos(angle)
        dx = rx * math.cos(yaw) - ry * math.sin(yaw)
        dy = rx * math.sin(yaw) + ry * math.cos(yaw)
        best = None
        for (x1, y1), (x2, y2) in walls:
            ex, ey = x2 - x1, y2 - y1
            denominator = dx * ey - dy * ex
            if abs(denominator) < 1e-9:
                continue
            t = ((x1 - robot_x) * ey - (y1 - robot_y) * ex) / denominator
            u = ((x1 - robot_x) * dy - (y1 - robot_y) * dx) / denominator
            if t > 0.05 and 0 <= u <= 1:
                best = t if best is None else min(best, t)
        if best is not None and best < 8:
            points.append((rx * best, ry * best))
    return points


class BusyRobotTests(unittest.TestCase):
    """2026-10-07: the robot's loop slowed to 2.4 Hz and it sent LiDAR points
    only every ~2 s. The pilot demanded two fresh scans within 4 s and failed
    every command with "no fresh LiDAR data"."""

    def setUp(self) -> None:
        self.robot = FakeRobot(scan_every_s=2.0)
        self.addCleanup(self.robot.close)

    def test_observe_and_move_still_work(self) -> None:
        code, out = run(self.robot.address, "manual", "on")
        self.assertEqual(code, 0, out)
        code, out = run(self.robot.address, "drive", "forward", "0.3")
        self.assertEqual(code, 0, out)
        self.assertIn("F", self.robot.commands)

class NoScanTests(unittest.TestCase):
    """2026-10-07, after a restart: the robot sent no LiDAR scan for the first
    8.5 s after the pilot connected, and the pilot refused every move. A late
    scan must never block a move; safety uses the range readings."""

    def setUp(self) -> None:
        self.robot = FakeRobot(scan_every_s=3600.0)   # one scan at start, then none
        self.addCleanup(self.robot.close)
        time.sleep(0.3)                                # the hub now holds light only
        self._wait = robot.SCAN_WAIT_S
        robot.SCAN_WAIT_S = 0.5
        self.addCleanup(setattr, robot, "SCAN_WAIT_S", self._wait)

    def test_moves_work_without_any_scan(self) -> None:
        code, out = run(self.robot.address, "manual", "on")
        self.assertEqual(code, 0, out)
        self.assertIn("no scan received yet", out)
        code, out = run(self.robot.address, "drive", "forward", "0.3")
        self.assertEqual(code, 0, out)
        self.assertIn("F", self.robot.commands)
        self.assertIn("Clear in your own lane: ahead 2.00 m", out)

class FieldIssueTests(unittest.TestCase):
    """2026-10-08, from the operator's AI-pilot session."""

    def setUp(self) -> None:
        self.robot = FakeRobot()
        self.addCleanup(self.robot.close)
        run(self.robot.address, "manual", "on")

    def test_a_v1_9_32_robot_still_turns_by_gyro(self) -> None:
        """2026-10-08 (2): a 0.1 s turn rotated 93 degrees and the same turn
        anywhere from 0 to 178. A turn is now in degrees, run by the robot and
        ended by its gyro."""
        for asked in (15, 45, 90):
            start = self.robot.chassis.yaw
            code, out = run(self.robot.address, "turn", "left", str(asked))
            self.assertEqual(code, 0, out)
            turned = abs((self.robot.chassis.yaw - start + 180.0) % 360.0 - 180.0)
            self.assertAlmostEqual(turned, asked, delta=8.0, msg=out)
            self.assertIn(f"asked {asked} degrees", out)
            self.assertIn("measured by the robot's gyro", out)

    def test_a_turn_without_the_gyro_is_marked_as_an_estimate(self) -> None:
        self.robot.chassis.imu_live = False
        code, out = run(self.robot.address, "turn", "right", "40")
        self.assertEqual(code, 0, out)
        self.assertIn("ESTIMATED from time", out)

    def test_the_arduinos_stop_ends_a_precise_drive(self) -> None:
        self.robot.uno_blocked = True
        code, out = run(self.robot.address, "drive", "forward", "1.5")
        self.assertEqual(code, 4, out)
        self.assertIn("18 cm ultrasonic stop", out)

    def test_the_camera_image_is_not_the_cached_frame(self) -> None:
        import cv2

        run(self.robot.address, "observe")
        time.sleep(0.5)                       # nobody watching: no new frames
        stale_value = self.robot.frame_value  # what the robot has cached
        run(self.robot.address, "observe")
        image = cv2.imread(robot.VIEW_PATH)
        self.assertIsNotNone(image)
        self.assertGreater(abs(float(image.mean()) - stale_value), 10.0)

    def test_a_short_telemetry_pause_does_not_end_a_move(self) -> None:
        """Gaps of ~0.9 s are normal; 1.5 s ended moves as a 'dropout'."""
        self.assertGreaterEqual(robot.TELEMETRY_STALE_S, 3.0)

    def test_a_low_object_only_the_ultrasonic_sees_is_flagged(self) -> None:
        """2026-10-08: LiDAR lane 1.68 m clear, a 10 cm bottle 35 cm ahead."""
        wall = {"x": [x for x in range(-20, 21, 3)], "y": [181] * 14}
        summary = robot.summarize({"scan": wall, "health": {"range": {"ultra_cm": 35}}})
        text = robot.describe(summary, {"manual": True})
        self.assertIn("WARNING: the ultrasonic sees something 35 cm ahead", text)
        agreeing = robot.summarize({"scan": wall, "health": {"range": {"ultra_cm": 170}}})
        self.assertNotIn("WARNING", robot.describe(agreeing, {"manual": True}))

    def test_the_lane_says_where_the_nearest_thing_is(self) -> None:
        summary = robot.summarize({"scan": {"x": [12, 12], "y": [20, 22]},
                                   "health": {"range": {"ultra_cm": 35}}})
        self.assertEqual(summary["lane_ahead_m"], 0.07)
        text = robot.describe(summary, {"manual": True})
        self.assertIn("right corner", text)
        self.assertIn("narrow beam", text)


class MeasuredMoveTests(unittest.TestCase):
    """2026-10-09 (3): gyro-ended turns over- and undershot by up to 27 deg,
    drives veered, and the pilot lost track of the capsule and the bucket.
    Turns are now LiDAR-measured pivots; moves are measured and tracked."""

    def setUp(self) -> None:
        folder = tempfile.mkdtemp()
        for name in ("PIVOT_MODEL_PATH", "TRACK_PATH"):
            original = getattr(robot, name)
            setattr(robot, name, os.path.join(folder, os.path.basename(original)))
            self.addCleanup(setattr, robot, name, original)
        self.robot = FakeRobot(room=True)
        self.addCleanup(self.robot.close)
        run(self.robot.address, "manual", "on")

    def _turned(self, before: float) -> float:
        return (self.robot.chassis.yaw - before + 180.0) % 360.0 - 180.0

    def test_turns_land_within_a_few_degrees(self) -> None:
        for direction, asked in (("left", 30), ("right", 10), ("left", 5), ("right", 60)):
            before = self.robot.chassis.yaw
            code, out = run(self.robot.address, "turn", direction, str(asked))
            self.assertEqual(code, 0, out)
            self.assertIn("measured by LiDAR", out)
            actual = self._turned(before) * (1 if direction == "left" else -1)
            self.assertAlmostEqual(actual, asked, delta=robot.TURN_TOLERANCE_DEG + 1.0, msg=out)

    def test_a_drive_is_measured_and_tracked(self) -> None:
        x0, y0, _yaw = self.robot.chassis.pose()
        code, out = run(self.robot.address, "drive", "forward", "1.0")
        self.assertEqual(code, 0, out)
        x1, y1, _yaw = self.robot.chassis.pose()
        moved = math.hypot(x1 - x0, y1 - y0)
        self.assertIn("Measured by LiDAR:", out)
        self.assertRegex(out, r"Measured by LiDAR: 0\.\d\d m forward")
        reported = float(out.split("Measured by LiDAR: ")[1].split(" m")[0])
        self.assertAlmostEqual(reported, moved, delta=0.02)
        self.assertIn("in 2 steps", out)
        track = robot.Track(robot.TRACK_PATH)
        self.assertAlmostEqual(track.y, moved, delta=0.03)
        self.assertIn("Since `manual on` (2 moves)", out)

    def test_a_marked_target_is_found_again_after_turning(self) -> None:
        code, out = run(self.robot.address, "mark", "capsule", "right", "30", "1.0")
        self.assertEqual(code, 0, out)
        code, out = run(self.robot.address, "turn", "right", "30")
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"Marked 'capsule': about 1\.0\d m away, (straight ahead|[0-4] deg (left|right))")
        code, out = run(self.robot.address, "unmark", "capsule")
        self.assertIn("Forgot 'capsule'", out)

    def test_a_veering_drive_is_straightened(self) -> None:
        """2026-10-09 (4): straight drives still ended 4-14 deg off."""
        self.robot.chassis.veer_dps = 12.0
        before = self.robot.chassis.yaw
        code, out = run(self.robot.address, "drive", "forward", "1.4")
        self.assertEqual(code, 0, out)
        self.assertIn("Veer turned back out", out)
        self.assertLess(abs(self._turned(before)), robot.STRAIGHTEN_DEG + 1.5, out)

    def test_straightening_can_be_turned_off(self) -> None:
        self.robot.chassis.veer_dps = 12.0
        before = self.robot.chassis.yaw
        code, out = run(self.robot.address, "drive", "forward", "1.4", "--no-straighten")
        self.assertEqual(code, 0, out)
        self.assertGreater(abs(self._turned(before)), 8.0, out)

    def test_drifted_tracking_is_re_anchored_to_an_earlier_view(self) -> None:
        """93 chained moves left the marks off; the pose is now re-anchored
        against still scans remembered along the way."""
        track = robot.Track(robot.TRACK_PATH)
        self.assertEqual(len(track.keyframes), 1)          # from `manual on`
        track.x += 0.10                                    # simulated drift
        track.heading += 4.0
        track.save()
        before = self.robot.chassis.yaw
        code, out = run(self.robot.address, "turn", "left", "10")
        self.assertEqual(code, 0, out)
        self.assertIn("re-anchored", out)
        track = robot.Track(robot.TRACK_PATH)
        self.assertAlmostEqual(track.x, 0.0, delta=0.03)
        self.assertAlmostEqual(track.heading, self._turned(before), delta=1.5)

    def test_a_wifi_stall_is_not_reported_as_a_person_taking_over(self) -> None:
        for name, value in (("MOVE_LINK_GRACE_S", 1.0), ("TELEMETRY_WAIT_S", 1.5)):
            original = getattr(robot, name)
            setattr(robot, name, value)
            self.addCleanup(setattr, robot, name, original)
        threading.Timer(0.6, setattr, (self.robot, "silent", True)).start()
        code, out = run(self.robot.address, "drive", "forward", "2.0", "--no-straighten")
        self.assertEqual(code, 6, out)
        self.assertIn("CONNECTION:", out)
        self.assertIn("nobody took over", out)
        self.robot.silent = False

    def test_a_person_pressing_stop_ends_a_measured_turn(self) -> None:
        threading.Timer(0.3, self.robot.control.halt).start()
        code, out = run(self.robot.address, "turn", "left", "180", "0.0")
        self.assertEqual(code, 5, out)
        self.assertIn("STOP was pressed", out)

    def test_every_report_saves_images_under_new_names(self) -> None:
        names = []
        for _ in range(2):
            code, out = run(self.robot.address, "observe")
            self.assertEqual(code, 0, out)
            line = next(line for line in out.splitlines() if line.startswith("Camera image saved:"))
            names.append(line.split(": ", 1)[1].split("  ")[0])
            time.sleep(0.01)
        self.assertNotEqual(names[0], names[1])
        self.assertTrue(all(os.path.exists(name) for name in names))


class LegacyRobotTests(unittest.TestCase):
    """A robot not yet restarted onto v1.9.32 still gets roughly timed moves."""

    def setUp(self) -> None:
        self.robot = FakeRobot(legacy=True)
        self.addCleanup(self.robot.close)
        run(self.robot.address, "manual", "on")

    def test_moves_fall_back_to_holding_the_button(self) -> None:
        code, out = run(self.robot.address, "turn", "left", "30")
        self.assertEqual(code, 0, out)
        self.assertIn("L", self.robot.commands)
        self.assertIn("older than v1.9.32", out)

    def test_the_robots_own_refusal_cuts_a_move_short(self) -> None:
        self.robot.forward_blocked = True
        started = time.monotonic()
        code, out = run(self.robot.address, "drive", "forward", "2.0")
        self.assertEqual(code, 4, out)
        self.assertIn("cut the move short", out)
        self.assertLess(time.monotonic() - started, 6.0)


class OpeningTests(unittest.TestCase):
    def test_a_doorway_is_the_first_opening_listed(self) -> None:
        openings = robot.find_openings(_room_points())
        self.assertTrue(openings)
        first = openings[0]
        # The doorway is to the right, between 1.8 and 2.7 m up the right wall.
        self.assertGreater(first["bearing"], 45)
        self.assertLess(first["bearing"], 110)
        self.assertGreaterEqual(first["clear_m"], 3.0)

    def test_an_open_room_is_not_one_360_degree_opening(self) -> None:
        """Every direction clears the minimum in an open room; the list must
        still single out the directions that run furthest."""
        openings = robot.find_openings(_room_points())
        self.assertTrue(all(item["span"] < 360 for item in openings))
        self.assertLessEqual(len(robot.describe_openings(openings).split(";")), 4)

    def test_a_blocked_lane_is_not_an_opening(self) -> None:
        wall = [(x / 100.0, 0.35) for x in range(-200, 201, 3)]
        for item in robot.find_openings(wall):
            self.assertGreater(abs(item["bearing"]), 30)

    def test_the_picture_is_drawn(self) -> None:
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "lidar.png")
        points = _room_points()
        telemetry = {"scan": {"x": [int(x * 100) for x, _ in points],
                              "y": [int(y * 100) for _, y in points]}}
        summary = robot.summarize(telemetry)
        saved = robot.draw_lidar(telemetry, summary, robot.find_openings(points), path)
        self.assertEqual(saved, path)
        self.assertGreater(os.path.getsize(path), 1000)


class PilotDisplayTests(unittest.TestCase):
    def test_the_dashboard_shows_what_the_pilot_is_doing(self) -> None:
        control = RobotControl()
        control.note_pilot("heading for the doorway", "claude-haiku-5-5")
        self.assertEqual(control.state()["pilot"]["note"], "heading for the doorway")
        control.note_pilot("")
        self.assertNotIn("pilot", control.state())


if __name__ == "__main__":
    unittest.main()
