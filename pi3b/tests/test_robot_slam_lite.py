from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar_visualizer import LidarPoint
from robot_slam_lite import LidarSlamLite


class SlamLiteTests(unittest.TestCase):
    def test_angular_match_accepts_a_clear_shift(self) -> None:
        mapper = LidarSlamLite()
        previous = np.arange(mapper.BIN_COUNT, dtype=np.float32) * 0.04 + 0.5
        mapper._previous_bins = previous
        current = np.roll(previous, 2)
        correction, confidence, accepted = mapper._align_yaw(current, expected_delta_deg=-10.0)
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


if __name__ == "__main__":
    unittest.main()
