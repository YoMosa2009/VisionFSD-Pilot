"""Run the real control loop against simulated hardware.

Every other test exercises one component. This one runs ``main()`` - the
loop that wires the LD19, the Uno, the map, the planner thread, the tracker,
the policy and the phone dashboard together - for a couple of seconds against
stand-ins for the hardware, and checks that it drives, plans, streams
telemetry and shuts down cleanly. It exists because wiring mistakes in that
loop are invisible to component tests and would otherwise first appear on the
robot.
"""

from __future__ import annotations

import math
import pathlib
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import robot_autonomy
from lidar_visualizer import LidarPoint
from robot_autonomy import ArduinoStatus, SectorClearance, corridor_profile, _CLEARANCE_HEADINGS
from robot_camera_motion import CameraMotionState
from robot_scan import frame_from_points
from robot_web import TelemetryHub


def _room_points(now: float) -> list:
    points = []
    for index, angle in enumerate(np.arange(0.0, 360.0, 0.8)):
        radians = math.radians(angle)
        distance = min(
            2.5 / max(abs(math.cos(radians)), 1e-6),
            2.0 / max(abs(math.sin(radians)), 1e-6),
        )
        points.append((index, LidarPoint(float(angle), int(distance * 1000), 210, now)))
    return points


class _FakeLD19:
    def __init__(self, *_args, **_kwargs) -> None:
        self.seq = 0

    def snapshot(self):
        return _room_points(time.monotonic()), True

    def scan_frame(self):
        self.seq += 1
        return frame_from_points(_room_points(time.monotonic()), self.seq, 3600.0)

    def clearance(self):
        now = time.monotonic()
        points = _room_points(now)
        profiles = corridor_profile(points, _CLEARANCE_HEADINGS)
        return SectorClearance(
            2.0, 2.0, 2.0, True, 2.0, 2.0, profiles[:-1], float(profiles[-1]), now
        )

    def close(self) -> None:
        return None


class _FakeUno:
    def __init__(self, *_args, **_kwargs) -> None:
        self.differential_ready = True
        self.drive_lease_expirations = 0
        self.uno_watchdog_stops = 0
        self.commands: list[tuple[int, int]] = []

    def poll_capabilities(self, _now: float) -> None:
        return None

    def status(self) -> ArduinoStatus:
        return ArduinoStatus(front_cm=None, motion="S", received_at=time.monotonic())

    def publish_drive(self, left: int, right: int) -> None:
        self.commands.append((left, right))

    def send(self, _command: str) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeCamera:
    def __init__(self, *_args, **_kwargs) -> None:
        self.motion = CameraMotionState()

    def start(self, _now=None) -> None:
        return None

    def tick(self, *_args) -> None:
        return None

    def ready(self, _now: float) -> bool:
        return True

    def annotated_frame(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class _WatchedHub(TelemetryHub):
    """A hub that always has a full subscriber and remembers what it got."""

    instances: list["_WatchedHub"] = []

    def __init__(self) -> None:
        super().__init__()
        self.published = 0
        self.last_full = None
        self.maps = 0
        _WatchedHub.instances.append(self)

    def wants(self, level: str) -> bool:
        return True

    def publish(self, light, full=None) -> None:
        super().publish(light, full)
        self.published += 1
        self.last_full = full

    def publish_map(self, png: bytes) -> None:
        super().publish_map(png)
        self.maps += 1


class RuntimeSmokeTests(unittest.TestCase):
    def test_main_loop_drives_plans_streams_and_stops_cleanly(self) -> None:
        uno = _FakeUno()
        handlers = {}

        def capture_signal(signum, handler):
            handlers[signum] = handler

        argv = [
            "robot_autonomy.py",
            "--no-display",
            "--no-imu",
            "--standby-seconds",
            "0",
            "--web-port",
            "0",
        ]
        result = {}
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(robot_autonomy, "discover_arduino_port", return_value="/dev/ttyACM0"),
            mock.patch.object(robot_autonomy, "discover_ld19_port", return_value="/dev/ttyUSB0"),
            mock.patch.object(robot_autonomy, "ArduinoLink", return_value=uno),
            mock.patch.object(robot_autonomy, "LD19Link", _FakeLD19),
            mock.patch.object(robot_autonomy, "CameraSafety", _FakeCamera),
            mock.patch.object(robot_autonomy, "TelemetryHub", _WatchedHub),
            mock.patch.object(robot_autonomy.signal, "signal", side_effect=capture_signal),
        ):
            def stop_later() -> None:
                time.sleep(3.0)
                handler = handlers.get(robot_autonomy.signal.SIGTERM)
                if handler is not None:
                    handler(robot_autonomy.signal.SIGTERM, None)

            stopper = threading.Thread(target=stop_later, daemon=True)
            stopper.start()
            started = time.monotonic()
            result["exit"] = robot_autonomy.main()
            result["elapsed"] = time.monotonic() - started

        self.assertEqual(result["exit"], 0)
        self.assertLess(result["elapsed"], 10.0)
        self.assertTrue(uno.commands, "the loop never published a drive command")
        self.assertTrue(
            any(left > 0 and right > 0 for left, right in uno.commands),
            "an open simulated room should produce forward driving",
        )
        hub = _WatchedHub.instances[-1]
        self.assertGreater(hub.published, 5)
        self.assertIsNotNone(hub.last_full)
        self.assertIn("scan", hub.last_full)
        self.assertGreater(len(hub.last_full["scan"]["x"]), 300)
        self.assertGreaterEqual(hub.maps, 1)


if __name__ == "__main__":
    unittest.main()
