"""Regression tests for the v1.9.22 phone dashboard link.

The WebSocket carries both the LiDAR visualiser's telemetry and manual
control, so it is tested the way a browser uses it: a real socket, a real
handshake, masked client frames, and the robot-side control state checked
after each message.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import socket
import struct
import sys
import time
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar_visualizer import LidarPoint
from robot_autonomy import (
    ArduinoStatus,
    AutonomousPolicy,
    SectorClearance,
    build_telemetry,
    encode_map_png,
)
from robot_explorer import ExplorationState
from robot_imu import IMUState
from robot_scan import frame_from_points
from robot_slam_lite import LidarSlamLite
from robot_web import (
    RobotControl,
    TelemetryHub,
    encode_frame,
    read_frame,
    start_dashboard_server,
    websocket_accept_key,
)


class _Client:
    """A minimal browser-like WebSocket client."""

    def __init__(self, port: int) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            "GET /ws HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(1)
        self.status_line = response.split(b"\r\n", 1)[0]
        self.accept = None
        for line in response.split(b"\r\n"):
            if line.lower().startswith(b"sec-websocket-accept:"):
                self.accept = line.split(b":", 1)[1].strip().decode("ascii")
        self.key = key

    def send(self, message: dict) -> None:
        payload = json.dumps(message).encode("utf-8")
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", 0x81, 0x80 | length)
        else:
            header = struct.pack("!BBH", 0x81, 0x80 | 126, length)
        self.sock.sendall(header + mask + masked)

    def receive(self, kind: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            opcode, payload = read_frame(self.sock)
            if opcode != 0x1:
                continue
            message = json.loads(payload)
            if message.get("type") == kind:
                return message
        raise AssertionError(f"no {kind} message")

    def close(self) -> None:
        try:
            self.sock.sendall(struct.pack("!BB", 0x88, 0x80) + b"\x00\x00\x00\x00")
        except OSError:
            pass
        self.sock.close()


class FramingTests(unittest.TestCase):
    def test_accept_key_matches_the_rfc_6455_example(self) -> None:
        self.assertEqual(
            websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
        )

    def test_server_frames_carry_every_length_encoding(self) -> None:
        for size in (5, 300, 70000):
            frame = encode_frame(b"x" * size)
            self.assertEqual(frame[0], 0x81)
            if size < 126:
                self.assertEqual(frame[1], size)
            elif size < 65536:
                self.assertEqual(struct.unpack("!H", frame[2:4])[0], size)
            else:
                self.assertEqual(struct.unpack("!Q", frame[2:10])[0], size)


class WebSocketSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = RobotControl()
        self.hub = TelemetryHub()
        started = start_dashboard_server(
            port=0, fps=15.0, version="test", control=self.control,
            camera_fps=10.0, telemetry=self.hub,
        )
        if started is None:
            self.skipTest("could not bind a dashboard port")
        self.stream, self.server = started
        self.addCleanup(self.server.close)
        self.port = self.server._server.server_address[1]

    def _client(self) -> _Client:
        client = _Client(self.port)
        self.addCleanup(client.close)
        return client

    def test_handshake_is_accepted(self) -> None:
        client = self._client()
        self.assertIn(b"101", client.status_line)
        self.assertEqual(client.accept, websocket_accept_key(client.key))
        hello = client.receive("hello")
        self.assertEqual(hello["version"], "test")

    def test_manual_driving_over_the_socket(self) -> None:
        client = self._client()
        client.receive("hello")
        client.send({"type": "control", "manual": True})
        client.receive("control")
        client.send({"type": "control", "drive": "L", "mag": 0.4})
        deadline = time.monotonic() + 2.0
        while self.control.manual_input()[0] != "L" and time.monotonic() < deadline:
            time.sleep(0.01)
        command, magnitude = self.control.manual_input()
        self.assertEqual(command, "L")
        self.assertAlmostEqual(magnitude, 0.4, places=2)

    def test_a_dropped_socket_stops_a_held_command(self) -> None:
        """A closed tab must not leave the robot turning until the command
        happens to expire."""
        client = _Client(self.port)
        client.receive("hello")
        client.send({"type": "control", "manual": True})
        client.send({"type": "control", "drive": "F", "mag": 1.0})
        deadline = time.monotonic() + 2.0
        while self.control.manual_input()[0] != "F" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.control.manual_input()[0], "F")
        client.sock.close()
        deadline = time.monotonic() + 2.0
        while self.control.manual_command() != "STOP" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.control.manual_command(), "STOP")

    def test_halt_over_the_socket(self) -> None:
        client = self._client()
        client.receive("hello")
        client.send({"type": "control", "halt": True})
        state = client.receive("control")
        self.assertTrue(state["halted"])
        self.assertTrue(self.control.halted)

    def test_subscribers_receive_telemetry_and_are_counted(self) -> None:
        self.assertFalse(self.hub.wants("full"))
        client = self._client()
        client.receive("hello")
        client.send({"type": "subscribe", "level": "full"})
        deadline = time.monotonic() + 2.0
        while not self.hub.wants("full") and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.hub.wants("full"))
        self.hub.publish({"intent": "EXPLORING"}, {"intent": "EXPLORING", "scan": {"x": [1], "y": [2]}})
        message = client.receive("telemetry")
        self.assertEqual(message["level"], "full")
        self.assertEqual(message["scan"]["x"], [1])
        client.close()
        deadline = time.monotonic() + 2.0
        while self.hub.wants("full") and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.hub.wants("full"))

    def test_ping_is_answered(self) -> None:
        client = self._client()
        client.receive("hello")
        client.send({"type": "ping", "c": 123.5})
        self.assertEqual(client.receive("pong")["c"], 123.5)

    def test_camera_stream_counts_viewers(self) -> None:
        camera = self.server.camera
        self.assertEqual(camera.viewers, 0)
        response = urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/camera.mjpg", timeout=5.0
        )
        self.addCleanup(response.close)
        deadline = time.monotonic() + 2.0
        while camera.viewers == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(camera.viewers, 1)
        camera.publish(np.zeros((24, 32, 3), dtype=np.uint8))
        self.assertIn(b"--visionfsdframe", response.read(64))

    def test_map_is_unavailable_until_published(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/map.png", timeout=5.0)
        self.assertEqual(caught.exception.code, 503)
        ok, png = cv2.imencode(".png", np.zeros((4, 4), dtype=np.uint8))
        self.hub.publish_map(png.tobytes())
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/map.png", timeout=5.0) as response:
            self.assertEqual(response.headers["Content-Type"], "image/png")

    def test_page_has_all_three_views_and_the_controls(self) -> None:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5.0) as response:
            page = response.read()
        for marker in (b'data-tab="lidar"', b'data-tab="camera"', b'data-tab="dashboard"',
                       b'id="halt"', b'id="resume"', b'id="manual"', b'data-drive="R"',
                       b"setPointerCapture", b"/ws"):
            self.assertIn(marker, page)


class TelemetryBuilderTests(unittest.TestCase):
    def _inputs(self):
        now = time.monotonic()
        points = [
            (i, LidarPoint(float(a), 2000, 210, now))
            for i, a in enumerate(np.arange(0.0, 360.0, 0.8))
        ]
        scan = frame_from_points(points, 1, 3600.0)
        mapper = LidarSlamLite()
        mapper._integrate_points(points, now)
        mapper._map_updates = 3
        policy = AutonomousPolicy(0.0, 112)
        policy.observe_pose(mapper.state())
        angles, ranges = scan.kept()
        policy.observe_scan(angles, ranges, now, now)
        status = ArduinoStatus(front_cm=None, motion="S", received_at=now)
        clearance = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0, rear_m=2.0, scan_at=now)
        policy.decide(clearance, status, False, now)
        return policy, scan, mapper, status, clearance, now

    def test_light_telemetry_is_small_and_has_no_arrays(self) -> None:
        policy, scan, mapper, status, clearance, now = self._inputs()
        light, full = build_telemetry(
            policy, scan, ExplorationState(), mapper.state(), IMUState(), status,
            clearance, True, True, mapper, now, False,
        )
        self.assertIsNone(full)
        self.assertNotIn("scan", light)
        self.assertLess(len(json.dumps(light)), 2000)

    def test_full_telemetry_carries_every_return(self) -> None:
        policy, scan, mapper, status, clearance, now = self._inputs()
        _light, full = build_telemetry(
            policy, scan, ExplorationState(), mapper.state(), IMUState(), status,
            clearance, True, True, mapper, now, True,
        )
        self.assertEqual(len(full["scan"]["x"]), scan.size)
        self.assertEqual(len(full["scan"]["k"]), scan.size)
        self.assertIn("foot", full)
        self.assertAlmostEqual(full["foot"]["w"], 22.9, places=1)
        # A whole revolution at centimetre precision stays a few kilobytes.
        self.assertLess(len(json.dumps(full)), 30000)

    def test_map_png_decodes_at_half_resolution(self) -> None:
        _policy, _scan, mapper, *_rest = self._inputs()
        png = encode_map_png(mapper)
        image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(image.shape, (mapper.cells // 2, mapper.cells // 2))
        self.assertGreater(int(np.count_nonzero(image)), 0)


if __name__ == "__main__":
    unittest.main()
