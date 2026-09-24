"""Regressions for the map worker (v1.9.28).

The map moved off the control thread after the 2026-09-23 field run measured
it at 83-203 ms of every control tick. Moving it must not change what it
builds: these check that the worker reproduces the inline map exactly, and
that pickups, recentring and the view behave.
"""

from __future__ import annotations

import math
import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar_visualizer import LidarPoint
from robot_map_worker import MapWorker, MotionSample
from robot_slam_lite import LidarSlamLite


def _room_scan(x: float, y: float, heading_deg: float, stamp: float):
    """Returns from a 4 x 3 m room with a box in it, seen from (x, y)."""
    walls = [((0, 0), (4, 0)), ((4, 0), (4, 3)), ((4, 3), (0, 3)), ((0, 3), (0, 0)),
             ((2.2, 1.2), (2.7, 1.2)), ((2.7, 1.2), (2.7, 1.6))]
    points = []
    for index, angle in enumerate(np.arange(0.0, 360.0, 0.8)):
        world = math.radians(angle + heading_deg)
        dx, dy = math.sin(world), math.cos(world)
        best = 6.0
        for (x1, y1), (x2, y2) in walls:
            ex, ey = x2 - x1, y2 - y1
            denominator = dx * ey - dy * ex
            if abs(denominator) < 1e-9:
                continue
            t = ((x1 - x) * ey - (y1 - y) * ex) / denominator
            u = ((x1 - x) * dy - (y1 - y) * dx) / denominator
            if t > 0.05 and 0.0 <= u <= 1.0:
                best = min(best, t)
        if best < 5.8:
            points.append((index, LidarPoint(float(angle), int(best * 1000), 200, stamp)))
    return points


def _ticks(count: int = 160):
    """Control ticks at 25 ms with a new scan every fourth tick, driving and
    turning, as the runtime submits them."""
    x, y, heading = 1.0, 1.0, 0.0
    commands = [(105, 105)] * 40 + [(105, 127)] * 30 + [(105, 105)] * 50 + [(127, 105)] * 40
    scan = None
    for tick in range(count):
        now = 10.0 + tick * 0.025
        left, right = commands[tick % len(commands)]
        speed = (left + right) / 2.0 / 255.0 * 0.26
        heading += (left - right) / 255.0 * 130.0 * 0.025
        x += math.sin(math.radians(heading)) * speed * 0.025
        y += math.cos(math.radians(heading)) * speed * 0.025
        new = tick % 4 == 0
        if new:
            scan = _room_scan(x, y, heading, now)
        yield left, right, now, scan, new


class MapWorkerEquivalenceTests(unittest.TestCase):
    def test_the_worker_builds_exactly_the_inline_map(self) -> None:
        inline = LidarSlamLite()
        worked = LidarSlamLite()
        worker = MapWorker(worked, threaded=False)
        for left, right, now, scan, new in _ticks():
            inline.update(scan, left, right, now, scan_stamp_hint=scan[0][1].captured_at)
            worker.submit(MotionSample(
                left, right, now,
                points=scan if new else None,
                scan_stamp_hint=scan[0][1].captured_at if new else None,
            ))
        self.assertGreater(inline.state().map_updates, 20)
        np.testing.assert_array_equal(inline.grid, worked.grid)
        np.testing.assert_array_equal(inline.observed, worked.observed)
        np.testing.assert_array_equal(inline.visits, worked.visits)
        self.assertEqual(inline.x, worked.x)
        self.assertEqual(inline.y, worked.y)
        self.assertEqual(inline.heading, worked.heading)
        self.assertEqual(worker.state().map_updates, inline.state().map_updates)

    def test_motion_is_never_skipped_when_scans_are(self) -> None:
        """A worker that falls behind may skip older scans in a batch, but
        every tick's motion still reaches dead reckoning."""
        inline = LidarSlamLite()
        worked = LidarSlamLite()
        worker = MapWorker(worked, threaded=False)
        samples = []
        for left, right, now, scan, new in _ticks(40):
            inline.integrate_motion(left, right, now)
            samples.append(MotionSample(left, right, now))
        # Queue everything, then process it as one batch.
        with worker._queue_lock:
            worker._pending.extend(samples)
        worker._drain()
        self.assertAlmostEqual(inline.x, worked.x, places=9)
        self.assertAlmostEqual(inline.heading, worked.heading, places=9)
        self.assertEqual(worker.processed, 40)


class MapWorkerBehaviourTests(unittest.TestCase):
    def test_a_pickup_resets_the_map_and_drops_queued_motion(self) -> None:
        worked = LidarSlamLite()
        worker = MapWorker(worked, threaded=False)
        for left, right, now, scan, new in _ticks(40):
            worker.submit(MotionSample(left, right, now, points=scan if new else None,
                                       scan_stamp_hint=scan[0][1].captured_at if new else None))
        self.assertGreater(np.count_nonzero(worked.observed), 0)
        worker.request_reset()
        self.assertEqual(np.count_nonzero(worked.observed), 0)
        self.assertEqual(np.count_nonzero(worked.grid), 0)
        self.assertEqual(worked.x, worked.metres / 2.0)

    def test_a_pre_reset_map_is_not_handed_to_the_planner(self) -> None:
        explorer = _RecordingExplorer()
        worker = MapWorker(LidarSlamLite(), explorer=explorer, threaded=False)
        # Simulate the reset request arriving while a batch is being processed.
        original = worker.local_map.integrate_motion

        def integrate_then_reset(*args, **kwargs):
            original(*args, **kwargs)
            with worker._queue_lock:
                worker._reset_requested = True

        worker.local_map.integrate_motion = integrate_then_reset
        worker.submit(MotionSample(105, 105, 1.0))
        self.assertEqual(explorer.published, 0)

    def test_recentre_shifts_reach_the_explorer(self) -> None:
        explorer = _RecordingExplorer()
        worked = LidarSlamLite()
        worker = MapWorker(worked, explorer=explorer, threaded=False)
        # Drive straight for long enough to push the robot off the grid centre.
        now = 1.0
        for _ in range(4000):
            now += 0.05
            worker.submit(MotionSample(255, 255, now))
            if worked._recenter_count:
                break
        self.assertGreater(worked._recenter_count, 0)
        self.assertEqual(len(explorer.shifts), worked._recenter_count)

    def test_the_view_never_waits_for_a_busy_worker(self) -> None:
        worker = MapWorker(LidarSlamLite(), threaded=False)
        with worker._map_lock:
            started = time.perf_counter()
            with worker.borrow() as view:
                self.assertIsNone(view)
            self.assertLess(time.perf_counter() - started, 0.05)
        with worker.borrow() as view:
            self.assertIsNotNone(view)

    def test_the_threaded_worker_keeps_up_and_stops_cleanly(self) -> None:
        worked = LidarSlamLite()
        worker = MapWorker(worked)
        try:
            for left, right, now, scan, new in _ticks(80):
                worker.submit(MotionSample(left, right, now, points=scan if new else None,
                                           scan_stamp_hint=scan[0][1].captured_at if new else None))
            deadline = time.monotonic() + 5.0
            while worker.processed < 80 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(worker.processed, 80)
            self.assertGreater(worker.state().map_updates, 0)
        finally:
            worker.close()
        self.assertFalse(any(t.name == "map-worker" and t.is_alive() for t in threading.enumerate()))


class _RecordingExplorer:
    def __init__(self) -> None:
        self.published = 0
        self.shifts: list[tuple[int, int]] = []

    def publish(self, *args, **kwargs) -> None:
        self.published += 1

    def shift(self, rows: int, cols: int, _scale: float) -> None:
        self.shifts.append((rows, cols))

    def invalidate(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
