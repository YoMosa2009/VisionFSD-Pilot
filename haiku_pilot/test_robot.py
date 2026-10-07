"""Tests for the pilot commands against the robot's real dashboard server.

A fake robot publishes telemetry and camera frames through pi3b/robot_web.py
exactly as the runtime does; the commands under test talk to it over HTTP and
WebSocket. Nothing here touches hardware or the network beyond localhost.

    python -m unittest haiku_pilot/test_robot.py
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import threading
import time
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pi3b"))

import robot  # noqa: E402
from robot_web import RobotControl, TelemetryHub, start_dashboard_server  # noqa: E402


class FakeRobot:
    """Publishes telemetry like the runtime, and applies the manual-mode
    forward guard the real policy applies."""

    def __init__(self, ahead_m: float = 2.0, behind_m: float = 1.5, scan_every_s: float = 0.0) -> None:
        self.ahead_m = ahead_m
        self.behind_m = behind_m
        # A busy robot sheds full telemetry: LiDAR points only this often.
        self.scan_every_s = scan_every_s
        self._next_scan_at = 0.0
        self.control = RobotControl()
        self.hub = TelemetryHub()
        started = start_dashboard_server(
            port=0, version="9.9.9", control=self.control, camera_fps=10.0, telemetry=self.hub,
        )
        if started is None:
            raise unittest.SkipTest("could not bind a dashboard port")
        self.stream, self.server = started
        self.address = f"127.0.0.1:{self.server._server.server_address[1]}"
        self.commands: list[str] = []
        self.forward_blocked = False
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _scan(self) -> dict:
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
                                     "ultra_cm": int(self.ahead_m * 100), "uno_blocked": False}},
                "control": self.control.state(),
            }
            now = time.monotonic()
            if now >= self._next_scan_at:
                self.hub.publish(light, dict(light, scan=self._scan()))
                self._next_scan_at = now + self.scan_every_s
            else:
                self.hub.publish(light, None)
            if self.server.camera is not None and self.server.camera.viewers > 0:
                self.server.camera.publish(np.full((240, 320, 3), 90, dtype=np.uint8))
            time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
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
        self.assertIn("Clear in your own lane: ahead 2.00 m, behind 1.50 m", out)
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
        self.assertIn("Did: drive forward for 0.4 s", out)
        self.assertIn("F", self.robot.commands)
        time.sleep(0.4)
        self.assertEqual(self.robot.control.manual_input()[0], "STOP")

    def test_forward_into_something_close_is_refused_before_moving(self) -> None:
        self.robot.ahead_m = 0.25
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "drive", "forward", "1.0")
        self.assertEqual(code, 3, out)
        self.assertIn("REFUSED, nothing moved", out)
        self.assertNotIn("F", self.robot.commands)

    def test_stale_cached_telemetry_cannot_authorise_a_move(self) -> None:
        """A new connection first gets the robot's last cached telemetry. An
        obstacle that appeared since must still stop the move."""
        run(self.robot.address, "manual", "on")
        time.sleep(0.3)                 # the hub now caches a clear lane
        self.robot.ahead_m = 0.25       # then something appears ahead
        code, out = run(self.robot.address, "drive", "forward", "1.0")
        self.assertEqual(code, 3, out)
        self.assertNotIn("F", self.robot.commands)

    def test_reversing_into_something_close_is_refused(self) -> None:
        self.robot.behind_m = 0.2
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "drive", "backward", "1.0")
        self.assertEqual(code, 3, out)
        self.assertNotIn("B", self.robot.commands)

    def test_the_robots_own_refusal_cuts_a_move_short(self) -> None:
        run(self.robot.address, "manual", "on")
        self.robot.forward_blocked = True
        started = time.monotonic()
        code, out = run(self.robot.address, "drive", "forward", "2.0")
        self.assertEqual(code, 4, out)
        self.assertIn("cut the move short", out)
        self.assertLess(time.monotonic() - started, 6.0)

    def test_a_person_pressing_stop_ends_the_move(self) -> None:
        run(self.robot.address, "manual", "on")
        threading.Timer(0.4, self.robot.control.halt).start()
        code, out = run(self.robot.address, "turn", "left", "1.5")
        self.assertEqual(code, 5, out)
        self.assertIn("STOP was pressed", out)

    def test_manual_on_is_refused_while_a_person_has_stopped_the_robot(self) -> None:
        self.robot.control.halt()
        code, out = run(self.robot.address, "manual", "on")
        self.assertEqual(code, 3, out)
        self.assertFalse(self.robot.control.manual)

    def test_moves_are_capped(self) -> None:
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "turn", "right", "9", "0.5")
        self.assertEqual(code, 0, out)
        self.assertIn(f"for {robot.TURN_MAX_S:.1f} s", out)

    def test_manual_off_hands_the_robot_back(self) -> None:
        run(self.robot.address, "manual", "on")
        code, out = run(self.robot.address, "manual", "off")
        self.assertEqual(code, 0, out)
        self.assertFalse(self.robot.control.manual)

    def test_an_unreachable_robot_is_a_clear_error(self) -> None:
        code, out = run("127.0.0.1:9", "observe")
        self.assertEqual(code, 2, out)
        self.assertIn("cannot reach the robot", out)


def _room_points(robot_x: float = 1.6, robot_y: float = 1.4) -> list[tuple[float, float]]:
    """A 5 x 4 m room with a doorway in the right-hand wall, as LiDAR points
    in the robot frame (robot facing +y)."""
    import math

    walls = [((0, 0), (5, 0)), ((0, 0), (0, 4)), ((0, 4), (5, 4)),
             ((5, 0), (5, 1.8)), ((5, 2.7), (5, 4))]
    points = []
    for step in range(450):
        angle = math.radians(step * 0.8)
        dx, dy = math.sin(angle), math.cos(angle)
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
            points.append((dx * best, dy * best))
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

    def test_an_obstacle_between_scans_still_blocks_the_move(self) -> None:
        """Safety uses the light messages' range readings, which arrive
        several times a second, not the scan, which may be seconds old."""
        run(self.robot.address, "manual", "on")
        self.robot.ahead_m = 0.25
        code, out = run(self.robot.address, "drive", "forward", "1.0")
        self.assertEqual(code, 3, out)
        self.assertNotIn("F", self.robot.commands)


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

    def test_range_readings_still_block_a_move_without_a_scan(self) -> None:
        run(self.robot.address, "manual", "on")
        self.robot.ahead_m = 0.25
        code, out = run(self.robot.address, "drive", "forward", "1.0")
        self.assertEqual(code, 3, out)
        self.assertNotIn("F", self.robot.commands)


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
