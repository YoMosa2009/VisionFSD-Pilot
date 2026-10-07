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

    def __init__(self, ahead_m: float = 2.0, behind_m: float = 1.5) -> None:
        self.ahead_m = ahead_m
        self.behind_m = behind_m
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
            self.hub.publish(light, dict(light, scan=self._scan()))
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


class PilotDisplayTests(unittest.TestCase):
    def test_the_dashboard_shows_what_the_pilot_is_doing(self) -> None:
        control = RobotControl()
        control.note_pilot("heading for the doorway", "claude-haiku-5-5")
        self.assertEqual(control.state()["pilot"]["note"], "heading for the doorway")
        control.note_pilot("")
        self.assertNotIn("pilot", control.state())


if __name__ == "__main__":
    unittest.main()
