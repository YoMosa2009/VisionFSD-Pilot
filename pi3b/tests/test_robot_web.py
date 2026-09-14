"""Regression tests for the view-only dashboard stream.

The stream exists so the robot can be watched from a phone instead of an
HDMI cable. Its hard requirement is that it can never affect driving: no
blocking of the control loop, no unbounded frame backlog, and no exception
escaping into the runtime.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_web import (
    STALE_AFTER_S,
    DashboardStream,
    RobotControl,
    local_ip_address,
    start_dashboard_server,
)


def _frame(value: int = 40) -> np.ndarray:
    frame = np.full((64, 64, 3), value, dtype=np.uint8)
    frame[10:20, 10:20] = 255
    return frame


def _wait_for_encoded(stream: DashboardStream, timeout: float = 5.0) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jpeg, _at = stream.latest()
        if jpeg is not None:
            return jpeg
        time.sleep(0.02)
    raise AssertionError("no frame was encoded within the timeout")


class DashboardStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = DashboardStream(fps=15.0, version="test")
        self.addCleanup(self.stream.close)

    def test_published_frame_becomes_a_jpeg(self) -> None:
        self.stream.publish(_frame())
        jpeg = _wait_for_encoded(self.stream)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_publish_returns_without_encoding_on_the_caller(self) -> None:
        """The control loop must pay a lock acquisition, not a JPEG encode."""
        started = time.perf_counter()
        for _ in range(50):
            self.stream.publish(_frame())
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.25)

    def test_only_the_newest_frame_is_retained(self) -> None:
        """No accumulating backlog: a slow encoder or slow Wi-Fi drops
        intermediate frames instead of queueing them."""
        slow = DashboardStream(fps=2.0)
        self.addCleanup(slow.close)
        for value in range(60):
            slow.publish(_frame(value % 250))
        pending = slow._pending
        # A single frame is held, never a queue of them.
        self.assertTrue(pending is None or isinstance(pending, np.ndarray))
        _wait_for_encoded(slow)
        # At 2 fps a burst of sixty publishes cannot have produced sixty
        # encodes; the intermediate frames were dropped, not buffered.
        self.assertLess(slow._encoded_count, 10)

    def test_status_reports_staleness_honestly(self) -> None:
        empty = self.stream.status()
        self.assertFalse(empty["has_frame"])
        self.assertTrue(empty["stale"])

        self.stream.publish(_frame())
        _wait_for_encoded(self.stream)
        fresh = self.stream.status()
        self.assertTrue(fresh["has_frame"])
        self.assertFalse(fresh["stale"])
        self.assertEqual(fresh["version"], "test")
        self.assertEqual(fresh["stale_after_s"], STALE_AFTER_S)

    def test_frame_rate_is_clamped_to_a_sane_range(self) -> None:
        slow = DashboardStream(fps=0.01)
        self.addCleanup(slow.close)
        fast = DashboardStream(fps=500.0)
        self.addCleanup(fast.close)
        self.assertGreaterEqual(slow.fps, 0.5)
        self.assertLessEqual(fast.fps, 15.0)

    def test_wait_for_frame_times_out_without_blocking_forever(self) -> None:
        started = time.monotonic()
        jpeg, _at = self.stream.wait_for_frame(0.0, timeout=0.2)
        self.assertIsNone(jpeg)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_a_bad_frame_does_not_kill_the_encoder(self) -> None:
        self.stream.publish(np.zeros((0, 0, 3), dtype=np.uint8))
        time.sleep(0.2)
        self.stream.publish(_frame())
        self.assertIsNotNone(_wait_for_encoded(self.stream))


class DashboardControlEndpointTests(unittest.TestCase):
    """The one non-read-only part of the dashboard. It has to fail safe and
    must never be reachable by accident."""

    def setUp(self) -> None:
        self.control = RobotControl()
        started = start_dashboard_server(
            port=0, fps=15.0, version="test", control=self.control
        )
        if started is None:
            self.skipTest("could not bind a dashboard port")
        self.stream, self.server = started
        self.addCleanup(self.server.close)
        self.port = self.server._server.server_address[1]

    def _post(self, payload: dict):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/control",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return json.loads(response.read())

    def test_halt_and_resume_round_trip(self) -> None:
        self.assertTrue(self._post({"halt": True})["halted"])
        self.assertTrue(self.control.halted)
        self.assertFalse(self._post({"resume": True})["halted"])
        self.assertFalse(self.control.halted)

    def test_manual_mode_gates_driving(self) -> None:
        self._post({"drive": "F"})
        self.assertEqual(self.control.manual_command(), "STOP")
        self._post({"manual": True})
        self._post({"drive": "F"})
        self.assertEqual(self.control.manual_command(), "F")

    def test_leaving_manual_mode_clears_the_command(self) -> None:
        self._post({"manual": True})
        self._post({"drive": "B"})
        self._post({"manual": False})
        self.assertEqual(self.control.manual_command(), "STOP")

    def test_a_garbage_body_does_not_crash_the_server(self) -> None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/control",
            data=b"not json at all",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            self.assertEqual(response.status, 200)
        self.assertFalse(self.control.halted)

    def test_status_reports_the_control_state(self) -> None:
        self._post({"halt": True})
        url = f"http://127.0.0.1:{self.port}/status.json"
        with urllib.request.urlopen(url, timeout=5.0) as response:
            payload = json.loads(response.read())
        self.assertTrue(payload["halted"])
        self.assertFalse(payload["manual"])

    def test_posting_anywhere_else_is_rejected(self) -> None:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/motors",
            data=b"{}",
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5.0)
        self.assertEqual(caught.exception.code, 404)


class DashboardControlDisabledTests(unittest.TestCase):
    def test_control_is_unavailable_without_a_bridge(self) -> None:
        started = start_dashboard_server(port=0, fps=15.0, version="test")
        if started is None:
            self.skipTest("could not bind a dashboard port")
        _stream, server = started
        self.addCleanup(server.close)
        port = server._server.server_address[1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/control", data=b"{}", method="POST"
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5.0)
        self.assertEqual(caught.exception.code, 503)


class DashboardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        started = start_dashboard_server(port=0, fps=15.0, version="9.9.9")
        if started is None:
            self.skipTest("could not bind a dashboard port")
        self.stream, self.server = started
        self.addCleanup(self.server.close)
        self.port = self.server._server.server_address[1]

    def _get(self, path: str, timeout: float = 5.0):
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read(), response.headers

    def test_index_page_is_served(self) -> None:
        status, body, headers = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"VisionFSD robot", body)
        # The operator controls ship with the page.
        self.assertIn(b'id="halt"', body)
        self.assertIn(b'id="manual"', body)
        self.assertIn(b'data-drive="F"', body)
        # The staleness threshold is substituted into the page, never left
        # as the literal placeholder.
        self.assertNotIn(b"STALE_AFTER_PLACEHOLDER", body)

    def test_status_endpoint_is_json(self) -> None:
        status, body, _headers = self._get("/status.json")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["version"], "9.9.9")
        self.assertIn("frame_age_s", payload)

    def test_frame_endpoint_reports_no_content_before_a_render(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._get("/frame.jpg")
        self.assertEqual(caught.exception.code, 503)

    def test_frame_endpoint_serves_a_published_frame(self) -> None:
        self.stream.publish(_frame())
        _wait_for_encoded(self.stream)
        status, body, headers = self._get("/frame.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertTrue(body.startswith(b"\xff\xd8"))

    def test_unknown_paths_are_rejected(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._get("/motors?left=255")
        self.assertEqual(caught.exception.code, 404)

    def test_stream_sends_multipart_frames(self) -> None:
        self.stream.publish(_frame())
        _wait_for_encoded(self.stream)
        url = f"http://127.0.0.1:{self.port}/stream.mjpg"
        response = urllib.request.urlopen(url, timeout=5.0)
        self.addCleanup(response.close)
        self.assertIn("multipart/x-mixed-replace", response.headers["Content-Type"])

        chunk = response.read(64)
        self.assertIn(b"--visionfsdframe", chunk)

    def test_a_client_disconnecting_mid_stream_is_harmless(self) -> None:
        self.stream.publish(_frame())
        _wait_for_encoded(self.stream)
        url = f"http://127.0.0.1:{self.port}/stream.mjpg"
        response = urllib.request.urlopen(url, timeout=5.0)
        response.read(32)
        response.close()
        time.sleep(0.2)

        # The server still answers other requests afterwards.
        status, _body, _headers = self._get("/status.json")
        self.assertEqual(status, 200)

    def test_server_threads_are_daemons(self) -> None:
        """A hung viewer must never keep the runtime process alive after the
        robot loop exits."""
        self.assertTrue(self.server._thread.daemon)
        self.assertTrue(self.server._server.daemon_threads)

    def test_port_already_in_use_degrades_instead_of_raising(self) -> None:
        again = start_dashboard_server(port=self.port, fps=5.0)
        if again is not None:
            again[1].close()
            self.skipTest("platform allows rebinding this port")
        self.assertIsNone(again)


class LocalAddressTests(unittest.TestCase):
    def test_local_address_is_a_string(self) -> None:
        address = local_ip_address()
        self.assertIsInstance(address, str)
        self.assertTrue(address)


class ThreadHygieneTests(unittest.TestCase):
    def test_closing_the_stream_stops_its_encoder(self) -> None:
        stream = DashboardStream(fps=15.0)
        stream.publish(_frame())
        _wait_for_encoded(stream)
        stream.close()
        time.sleep(0.3)
        names = [thread.name for thread in threading.enumerate()]
        self.assertLessEqual(names.count("dashboard-encoder"), 1)


if __name__ == "__main__":
    unittest.main()
