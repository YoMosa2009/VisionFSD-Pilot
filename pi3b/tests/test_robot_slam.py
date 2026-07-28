"""Hardware-free tests for scan-matching SLAM and the camera floor guard.

The pose these produce is advisory: it biases exploration and draws the map,
and must never reach the corridor geometry.  The tests below therefore check
both that it works and that it knows when it has failed.
"""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_slam import (
    OccupancyMap,
    ScanMatcher,
    SlamConfig,
    SlamTracker,
    deskew,
    points_to_robot_frame,
)


ROOM = (3.2, 2.6)  # half-extents in metres about the world origin


def raycast_room(x: float, y: float, heading_deg: float, count: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Ranges to a rectangular room with a pillar, from a pose inside it."""
    bearings = np.linspace(-180.0, 179.0, count, dtype=np.float32)
    world = np.radians(heading_deg + bearings)
    dx, dy = np.sin(world), np.cos(world)
    best = np.full(count, np.inf)
    for bound, delta, origin in ((ROOM[0], dx, x), (-ROOM[0], dx, x),
                                 (ROOM[1], dy, y), (-ROOM[1], dy, y)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (bound - origin) / delta
        best = np.minimum(best, np.where((t > 0) & np.isfinite(t), t, np.inf))
    # A pillar breaks the room's symmetry so the match has something to lock on.
    px, py, radius = 0.9, -0.4, 0.16
    fx, fy = x - px, y - py
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * c
    root = np.where(disc >= 0, (-b - np.sqrt(np.where(disc >= 0, disc, 0.0))) / 2.0, np.inf)
    best = np.minimum(best, np.where(root > 0, root, np.inf))
    return bearings, np.where(best <= 6.0, best, np.nan).astype(np.float32)


def integrate_pose(occupancy: OccupancyMap, x: float, y: float, heading: float) -> None:
    bearings, ranges = raycast_room(x, y, heading)
    keep = np.isfinite(ranges)
    points = points_to_robot_frame(bearings[keep], ranges[keep])
    angle = math.radians(heading)
    cos, sin = math.cos(angle), math.sin(angle)
    world_x = x + points[:, 0] * cos + points[:, 1] * sin
    world_y = y - points[:, 0] * sin + points[:, 1] * cos
    occupancy.integrate(x, y, world_x, world_y)


class OccupancyMapTests(unittest.TestCase):
    def test_beam_marks_its_endpoint_occupied_and_its_path_free(self) -> None:
        occupancy = OccupancyMap(SlamConfig())
        for _ in range(4):
            occupancy.integrate(0.0, 0.0, np.array([1.0], np.float32), np.array([0.0], np.float32))
        end_row, end_col = occupancy.world_to_cell(np.array([1.0]), np.array([0.0]))
        mid_row, mid_col = occupancy.world_to_cell(np.array([0.5]), np.array([0.0]))
        self.assertGreater(occupancy.grid[end_row[0], end_col[0]], 0.0)
        self.assertLess(occupancy.grid[mid_row[0], mid_col[0]], 0.0)

    def test_distance_field_is_zero_on_an_obstacle_and_grows_away_from_it(self) -> None:
        occupancy = OccupancyMap(SlamConfig())
        for _ in range(4):
            occupancy.integrate(0.0, 0.0, np.array([1.0], np.float32), np.array([0.0], np.float32))
        field = occupancy.distance_field()
        end_row, end_col = occupancy.world_to_cell(np.array([1.0]), np.array([0.0]))
        far_row, far_col = occupancy.world_to_cell(np.array([1.0]), np.array([1.0]))
        self.assertAlmostEqual(float(field[end_row[0], end_col[0]]), 0.0, places=3)
        self.assertGreater(float(field[far_row[0], far_col[0]]), 0.7)

    def test_recentring_keeps_the_robot_near_the_middle(self) -> None:
        config = SlamConfig()
        occupancy = OccupancyMap(config)
        integrate_pose(occupancy, 0.0, 0.0, 0.0)
        occupancy.recentre(2.4, 1.8)
        rows, cols = occupancy.world_to_cell(np.array([2.4]), np.array([1.8]))
        centre = config.cells // 2
        self.assertLess(abs(int(rows[0]) - centre), config.cells * 0.2)
        self.assertLess(abs(int(cols[0]) - centre), config.cells * 0.2)

    def test_frontier_points_toward_unexplored_space(self) -> None:
        occupancy = OccupancyMap(SlamConfig())
        # Carve free space only on the +x side, leaving -x unknown.
        for _ in range(3):
            for bearing in np.linspace(30.0, 150.0, 60):
                radians = math.radians(bearing)
                occupancy.integrate(0.0, 0.0,
                                    np.array([math.sin(radians) * 1.6], np.float32),
                                    np.array([math.cos(radians) * 1.6], np.float32))
        bearing, weight = occupancy.frontier_bearing(0.0, 0.0, 0.0)
        self.assertGreater(weight, 0.0)
        self.assertGreater(bearing, 0.0)  # frontier lies to the robot's right


class ScanMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SlamConfig()
        self.occupancy = OccupancyMap(self.config)
        for _ in range(3):
            integrate_pose(self.occupancy, 0.0, 0.0, 0.0)
        self.matcher = ScanMatcher(self.config)

    def _scan_at(self, x: float, y: float, heading: float) -> np.ndarray:
        bearings, ranges = raycast_room(x, y, heading)
        keep = np.isfinite(ranges)
        return points_to_robot_frame(bearings[keep], ranges[keep])

    def test_recovers_a_pure_translation(self) -> None:
        truth = (0.09, -0.06, 0.0)
        points = self._scan_at(*truth)
        (x, y, heading), residual = self.matcher.match(self.occupancy, points, (0.0, 0.0, 0.0))
        self.assertLess(math.hypot(x - truth[0], y - truth[1]), 0.045)
        self.assertLess(residual, 0.06)

    def test_recovers_a_rotation(self) -> None:
        truth = (0.0, 0.0, 5.0)
        points = self._scan_at(*truth)
        (x, y, heading), residual = self.matcher.match(self.occupancy, points, (0.0, 0.0, 0.0))
        self.assertLess(abs((heading - 5.0 + 180.0) % 360.0 - 180.0), 2.5)
        self.assertLess(residual, 0.06)

    def test_a_scan_from_a_different_room_reports_a_large_residual(self) -> None:
        rubbish = points_to_robot_frame(
            np.linspace(-180.0, 179.0, 200, dtype=np.float32),
            np.full(200, 0.45, dtype=np.float32),
        )
        _pose, residual = self.matcher.match(self.occupancy, rubbish, (0.0, 0.0, 0.0))
        self.assertGreater(residual, self.config.divergence_m)


class DeskewTests(unittest.TestCase):
    def test_rotation_during_a_sweep_is_undone(self) -> None:
        points = points_to_robot_frame(np.array([0.0], np.float32), np.array([1.0], np.float32))
        ages = np.array([0.1], np.float32)
        corrected = deskew(points, ages, yaw_rate_dps=90.0, forward_speed_ms=0.0)
        # 90 deg/s for 100 ms: the return was taken 9 degrees before "now".
        bearing = math.degrees(math.atan2(corrected[0, 0], corrected[0, 1]))
        self.assertAlmostEqual(bearing, -9.0, places=1)

    def test_no_motion_leaves_points_untouched(self) -> None:
        points = points_to_robot_frame(np.array([15.0, -40.0], np.float32),
                                       np.array([1.0, 2.0], np.float32))
        corrected = deskew(points, np.array([0.05, 0.09], np.float32), 0.0, 0.0)
        self.assertTrue(np.allclose(points, corrected, atol=1e-6))


class SlamTrackerTests(unittest.TestCase):
    @staticmethod
    def _feed(tracker: SlamTracker, x: float, y: float, heading: float, elapsed: float = 0.22,
              yaw_rate: float = 0.0, forward: float = 0.0) -> None:
        bearings, ranges = raycast_room(x, y, heading)
        ages = np.zeros_like(bearings)
        tracker.update(bearings, ranges, ages, yaw_rate, forward, elapsed)

    def test_tracks_a_short_straight_move(self) -> None:
        tracker = SlamTracker()
        self._feed(tracker, 0.0, 0.0, 0.0)
        for step in range(1, 5):
            self._feed(tracker, 0.0, 0.05 * step, 0.0, forward=0.22)
        state = tracker.state()
        self.assertLess(abs(state.y - 0.20), 0.06)
        self.assertLess(abs(state.x), 0.06)

    def test_divergence_marks_the_pose_untrusted_and_stops_mapping(self) -> None:
        tracker = SlamTracker()
        # Past the bootstrap sweeps, so there is a map worth matching against.
        for _ in range(SlamConfig().bootstrap_updates + 2):
            self._feed(tracker, 0.0, 0.0, 0.0)
        updates_before = tracker.updates
        # A scan that cannot be explained by the map at any nearby pose.
        tracker.update(np.linspace(-180.0, 179.0, 200, dtype=np.float32),
                       np.full(200, 0.4, dtype=np.float32),
                       np.zeros(200, dtype=np.float32), 0.0, 0.0, 0.22)
        self.assertFalse(tracker.state().trusted)
        self.assertEqual(tracker.updates, updates_before)
        self.assertEqual(tracker.frontier(), (0.0, 0.0))

    def test_sustained_divergence_restarts_the_map_instead_of_drifting_away(self) -> None:
        config = SlamConfig()
        tracker = SlamTracker(config)
        for _ in range(config.bootstrap_updates + 2):
            self._feed(tracker, 0.0, 0.0, 0.0)
        for _ in range(config.max_lost_ticks):
            tracker.update(np.linspace(-180.0, 179.0, 200, dtype=np.float32),
                           np.full(200, 0.4, dtype=np.float32),
                           np.zeros(200, dtype=np.float32), 0.0, 0.0, 0.22)
        self.assertEqual(tracker.restarts, 1)
        self.assertEqual(tracker.updates, 0)
        self.assertFalse(tracker.state().trusted)

    def test_frontier_is_withheld_until_the_pose_is_trusted(self) -> None:
        tracker = SlamTracker()
        tracker.trusted = False
        tracker.frontier_bearing_deg, tracker.frontier_weight = 40.0, 0.9
        self.assertEqual(tracker.frontier(), (0.0, 0.0))

    def test_too_few_returns_is_ignored_rather_than_guessed(self) -> None:
        tracker = SlamTracker()
        tracker.update(np.linspace(-180.0, 179.0, 12, dtype=np.float32),
                       np.full(12, 1.0, dtype=np.float32),
                       np.zeros(12, dtype=np.float32), 0.0, 0.0, 0.22)
        self.assertEqual(tracker.updates, 0)


class LowObstacleGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        from robot_vision import LowObstacleGuard
        self.guard_class = LowObstacleGuard

    @staticmethod
    def _floor(colour: tuple[int, int, int] = (120, 118, 115)) -> np.ndarray:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = colour
        return frame

    def test_plain_floor_does_not_slow_the_robot(self) -> None:
        guard = self.guard_class(sustain_frames=2)
        for _ in range(6):
            scale = guard.update(self._floor())
        self.assertEqual(scale, 1.0)
        self.assertFalse(guard.blocked)

    def test_a_sustained_object_on_the_floor_slows_it(self) -> None:
        guard = self.guard_class(sustain_frames=2)
        frame = self._floor()
        # A strongly off-colour block low and central: a shoe the LiDAR plane
        # passes straight over.
        frame[150:238, 110:210] = (40, 60, 210)
        for _ in range(6):
            scale = guard.update(frame)
        self.assertTrue(guard.blocked)
        self.assertLess(scale, 1.0)

    def test_disabled_guard_never_reports(self) -> None:
        guard = self.guard_class(enabled=False)
        frame = self._floor()
        frame[150:238, 110:210] = (40, 60, 210)
        for _ in range(6):
            scale = guard.update(frame)
        self.assertEqual(scale, 1.0)
        self.assertFalse(guard.blocked)


if __name__ == "__main__":
    unittest.main()
