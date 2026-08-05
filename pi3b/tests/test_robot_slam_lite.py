from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar_visualizer import LidarPoint
from robot_slam_lite import LidarSlamLite


class SlamLiteTests(unittest.TestCase):
    def test_default_map_uses_about_two_centimetre_cells(self) -> None:
        mapper = LidarSlamLite()
        self.assertAlmostEqual(mapper.metres / mapper.cells, 0.0208, places=3)
        self.assertEqual(mapper.BIN_COUNT, 360)

    def test_one_degree_bins_keep_the_nearest_duplicate_return(self) -> None:
        points = [
            (0, LidarPoint(10.2, 1400, 90, 1.0)),
            (1, LidarPoint(10.8, 900, 90, 1.1)),
        ]
        bins, newest = LidarSlamLite.bins_from_points(points)
        self.assertAlmostEqual(float(bins[10]), 0.9, places=5)
        self.assertEqual(newest, 1.1)

    def test_angular_match_accepts_a_clear_shift(self) -> None:
        mapper = LidarSlamLite()
        previous = np.arange(mapper.BIN_COUNT, dtype=np.float32) * 0.04 + 0.5
        mapper._previous_bins = previous
        current = np.roll(previous, 2)
        correction, confidence, accepted = mapper._align_yaw(
            current, expected_delta_deg=-2 * mapper.BIN_DEGREES
        )
        self.assertTrue(accepted)
        self.assertGreater(confidence, 0.16)
        self.assertAlmostEqual(correction, 0.0, places=3)

    def test_angular_match_rejects_ambiguous_flat_room(self) -> None:
        mapper = LidarSlamLite()
        mapper._previous_bins = np.full(mapper.BIN_COUNT, 1.5, dtype=np.float32)
        correction, confidence, accepted = mapper._align_yaw(
            np.full(mapper.BIN_COUNT, 1.5, dtype=np.float32), expected_delta_deg=0.0
        )
        self.assertFalse(accepted)
        self.assertEqual(correction, 0.0)
        self.assertEqual(confidence, 0.0)

    def test_map_integrates_only_a_new_physical_scan(self) -> None:
        mapper = LidarSlamLite()
        points = [
            (0, LidarPoint(0.0, 1000, 90, 1.0)),
            (1, LidarPoint(5.0, 1010, 90, 1.0)),
            (2, LidarPoint(10.0, 1020, 90, 1.0)),
        ]
        first = mapper.update(points, 0, 0, 1.05)
        second = mapper.update(points, 0, 0, 1.08)
        self.assertEqual(first.map_updates, 1)
        self.assertEqual(second.map_updates, 1)

    def test_map_integrates_all_distinct_scan_returns(self) -> None:
        mapper = LidarSlamLite()
        points = [
            (index, LidarPoint(float(angle), 1000, 90, 1.0))
            for index, angle in enumerate(range(0, 180, 20))
        ]
        mapper._integrate_points(points)
        self.assertGreaterEqual(int(np.count_nonzero(mapper.grid)), len(points))
        self.assertEqual(mapper._latest_hits.shape[0], len(points))

    def test_scan_rays_mark_free_space_as_observed(self) -> None:
        mapper = LidarSlamLite(cells=120, metres=3.0)
        points = [
            (index, LidarPoint(float(angle), 1000, 90, 1.0))
            for index, angle in enumerate((-45, 0, 45))
        ]
        mapper._integrate_points(points)

        self.assertGreater(int(np.count_nonzero(mapper.observed)), 60)
        self.assertGreater(int(np.count_nonzero(mapper.visits)), 1)

    def test_scan_to_map_translation_corrects_predicted_position(self) -> None:
        mapper = LidarSlamLite(cells=320, metres=8.0)
        mapper._map_updates = 3
        true_x = true_y = 4.0
        points = []
        scale = mapper.cells / mapper.metres
        for index, angle in enumerate(range(0, 360, 10)):
            distance = (
                1.3
                + 0.25 * math.sin(math.radians(angle * 2))
                + 0.12 * math.cos(math.radians(angle * 5))
            )
            points.append((
                index,
                LidarPoint(float(angle), int(distance * 1000), 100, 1.0),
            ))
            radians = math.radians(angle)
            col = int((true_x + math.sin(radians) * distance) * scale)
            row = int((true_y - math.cos(radians) * distance) * scale)
            mapper.grid[row, col] = 255

        mapper.x = true_x + 0.075
        mapper.y = true_y
        delta_x, delta_y, confidence, accepted = mapper._align_translation(points)

        self.assertTrue(accepted)
        self.assertLess(delta_x, 0.0)
        self.assertAlmostEqual(delta_x, -0.075, places=2)
        self.assertAlmostEqual(delta_y, 0.0, places=2)
        self.assertGreater(confidence, 0.5)

    def test_live_imu_rate_replaces_commanded_yaw_prediction(self) -> None:
        mapper = LidarSlamLite()
        mapper.integrate_motion(0, 0, 1.0)
        mapper.integrate_motion(0, 0, 1.1, imu_yaw_rate_dps=30.0)
        self.assertAlmostEqual(mapper.heading, 357.0, places=3)
        self.assertTrue(mapper._using_imu)
        self.assertEqual(mapper.state().yaw_source, "IMU+LD19")

    def test_integrated_imu_yaw_delta_beats_main_loop_rate_estimate(self) -> None:
        mapper = LidarSlamLite()
        mapper.integrate_motion(
            0, 0, 1.0, imu_yaw_rate_dps=180.0, imu_yaw_deg=0.0
        )
        mapper.integrate_motion(
            0, 0, 1.1, imu_yaw_rate_dps=180.0, imu_yaw_deg=4.0
        )

        self.assertAlmostEqual(mapper.heading, 356.0, places=3)

    def test_render_keeps_robot_marker_at_viewport_centre(self) -> None:
        mapper = LidarSlamLite()
        mapper.x = 0.25
        mapper.y = 7.75

        panel = mapper.render(size=300)
        centre_region = panel[143:158, 143:158]
        green_marker = (
            (centre_region[:, :, 1] > 190)
            & (centre_region[:, :, 2] < 180)
        )

        self.assertTrue(bool(np.any(green_marker)))

    def test_missing_imu_reports_command_plus_ld19_yaw(self) -> None:
        mapper = LidarSlamLite()
        mapper.integrate_motion(105, 0, 1.0)
        mapper.integrate_motion(105, 0, 1.1, imu_yaw_rate_dps=None)

        self.assertEqual(mapper.state().yaw_source, "COMMAND+LD19")

    def test_camera_flow_refines_command_yaw_when_direction_agrees(self) -> None:
        mapper = LidarSlamLite()
        mapper.integrate_motion(118, 90, 1.0)
        mapper.integrate_motion(
            118, 90, 1.1, camera_yaw_rate_dps=40.0
        )

        self.assertEqual(mapper.state().yaw_source, "COMMAND+CAMERA+LD19")
        command_only_delta = ((118 - 90) / 255.0) * 130.0 * 0.1
        self.assertGreater(mapper.heading, command_only_delta)

    def test_camera_no_motion_reduces_uncertain_forward_prediction(self) -> None:
        normal = LidarSlamLite()
        reduced = LidarSlamLite()
        normal.integrate_motion(118, 118, 1.0)
        reduced.integrate_motion(118, 118, 1.0)
        normal.integrate_motion(118, 118, 1.1)
        reduced.integrate_motion(
            118, 118, 1.1, camera_translation_scale=0.20
        )

        normal_distance = 4.0 - normal.y
        reduced_distance = 4.0 - reduced.y
        self.assertAlmostEqual(reduced_distance / normal_distance, 0.20, places=2)

    def test_recenter_moves_grid_content_by_the_same_shift_as_the_robot(self) -> None:
        mapper = LidarSlamLite(cells=100, metres=5.0)
        scale = mapper.cells / mapper.metres
        mapper.x = 2.5
        mapper.y = 0.10  # row ~2, well inside the 20-cell edge margin
        marker_row, marker_col = 5, 50
        mapper.grid[marker_row, marker_col] = 200
        robot_row_before = int(round(mapper.y * scale))

        shifted = mapper._recenter_if_needed()

        self.assertTrue(shifted)
        expected_shift_row = mapper.cells // 2 - robot_row_before
        self.assertEqual(mapper.grid[marker_row + expected_shift_row, marker_col], 200)
        self.assertAlmostEqual(mapper.y, 0.10 + expected_shift_row / scale, places=6)
        self.assertEqual(int(round(mapper.y * scale)), mapper.cells // 2)

    def test_recenter_preserves_nearby_content_but_clears_the_far_wrapped_band(self) -> None:
        mapper = LidarSlamLite(cells=100, metres=5.0)
        mapper.grid[10, 50] = 200  # near the robot -- should survive, shifted
        mapper.grid[95, 50] = 255  # far side -- would wrap in via np.roll
        mapper.x = 2.5
        mapper.y = 0.10
        scale = mapper.cells / mapper.metres
        robot_row_before = int(round(mapper.y * scale))
        shift_row = mapper.cells // 2 - robot_row_before

        mapper._recenter_if_needed()

        self.assertEqual(mapper.grid[10 + shift_row, 50], 200)
        # Content that wrapped in from the far side of the array is stale
        # (outside the robot's actual vicinity) and must be blanked rather
        # than appearing as a phantom obstacle near the new position.
        self.assertEqual(int(np.count_nonzero(mapper.grid[:shift_row, :])), 0)

    def test_sustained_travel_recenters_instead_of_clamping_pose(self) -> None:
        """A robot travelling far in one direction must not have its pose
        pinned at the grid edge -- that was the visualizer's black-void bug,
        where new scans kept projecting onto a stale, frozen position once
        the old hard clamp engaged."""
        mapper = LidarSlamLite(cells=100, metres=5.0)
        mapper.heading = 90.0  # straight +x travel
        now = 1.0
        for _ in range(600):
            now += 0.05
            mapper.integrate_motion(200, 200, now)

        scale = mapper.cells / mapper.metres
        margin_cells = int(round(mapper.cells * mapper.RECENTER_MARGIN_FRACTION))
        robot_col = mapper.x * scale

        self.assertGreater(mapper._recenter_count, 0)
        self.assertGreaterEqual(robot_col, margin_cells - 1)
        self.assertLessEqual(robot_col, mapper.cells - margin_cells + 1)

    def test_state_reports_recenter_count(self) -> None:
        mapper = LidarSlamLite(cells=100, metres=5.0)
        self.assertEqual(mapper.state().recenter_count, 0)
        mapper.heading = 90.0
        now = 1.0
        for _ in range(400):
            now += 0.05
            mapper.integrate_motion(200, 200, now)

        self.assertGreater(mapper.state().recenter_count, 0)


if __name__ == "__main__":
    unittest.main()
