"""Regression tests for whole-scan LD19 motion evidence.

These cover the gap that made a stalled chassis invisible in an open room:
every previous evidence source needed either a nearby tracked range, a
commanded turn, or good camera texture, so a robot spinning its wheels on a
rug in the middle of a floor gathered no votes at all.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_motion import (
    MIN_OVERLAP_BINS,
    SIGNATURE_BINS,
    ScanMotionTracker,
    range_signature,
    signature_change_m,
)


class _Point:
    def __init__(self, angle_deg: float, distance_mm: float) -> None:
        self.angle_deg = angle_deg
        self.distance_mm = distance_mm


def _scan(distance_m: float, count: int = SIGNATURE_BINS) -> list:
    step = 360.0 / count
    return [
        (index, _Point(index * step + step * 0.5, distance_m * 1000.0))
        for index in range(count)
    ]


def _ramped_scan(base_m: float, slope_m: float) -> list:
    step = 360.0 / SIGNATURE_BINS
    return [
        (
            index,
            _Point(
                index * step + step * 0.5,
                (base_m + slope_m * index) * 1000.0,
            ),
        )
        for index in range(SIGNATURE_BINS)
    ]


class RangeSignatureTests(unittest.TestCase):
    def test_empty_scan_is_all_unknown(self) -> None:
        signature = range_signature([])
        self.assertEqual(signature.size, SIGNATURE_BINS)
        self.assertTrue(np.all(np.isnan(signature)))

    def test_missing_sectors_stay_nan_not_zero(self) -> None:
        """A sector with no return means "no information". Filling it with a
        zero range would look like an obstacle pressed against the chassis."""
        points = [(0, _Point(5.0, 1000.0)), (1, _Point(6.0, 1100.0))]
        signature = range_signature(points)
        self.assertFalse(np.isnan(signature[1]))
        self.assertTrue(np.isnan(signature[40]))

    def test_bin_keeps_the_nearest_return(self) -> None:
        points = [(0, _Point(2.0, 1500.0)), (1, _Point(3.0, 900.0))]
        signature = range_signature(points)
        self.assertAlmostEqual(float(signature[0]), 0.9, places=3)

    def test_out_of_range_returns_are_discarded(self) -> None:
        points = [(0, _Point(2.0, 20.0)), (1, _Point(3.0, 9000.0))]
        self.assertTrue(np.all(np.isnan(range_signature(points))))

    def test_change_needs_enough_shared_geometry(self) -> None:
        current = np.full(SIGNATURE_BINS, np.nan, dtype=np.float32)
        reference = np.full(SIGNATURE_BINS, np.nan, dtype=np.float32)
        current[:4] = 1.0
        reference[:4] = 2.0
        change, overlap, _fraction = signature_change_m(current, reference)
        self.assertLess(overlap, MIN_OVERLAP_BINS)
        self.assertEqual(change, 0.0)


class ScanMotionTrackerTests(unittest.TestCase):
    def test_identical_scans_report_not_moving(self) -> None:
        tracker = ScanMotionTracker()
        scan = range_signature(_scan(2.5))
        tracker.update(scan, 0.0)
        result = tracker.update(range_signature(_scan(2.5)), 0.8)

        self.assertEqual(result.verdict, "NOT_MOVING")
        self.assertFalse(result.displaced)

    def test_open_room_stall_is_detected_far_from_any_wall(self) -> None:
        """The regression this module exists for: every range is metres away,
        so no per-sector progress check can say anything, yet the scan is
        provably unchanged."""
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_scan(4.0)), 0.0)
        result = tracker.update(range_signature(_scan(4.0)), 0.9)

        self.assertEqual(result.verdict, "NOT_MOVING")

    def test_changing_scans_report_moving(self) -> None:
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_scan(2.5)), 0.0)
        result = tracker.update(range_signature(_scan(2.3)), 0.8)

        self.assertEqual(result.verdict, "MOVING")

    def test_small_range_noise_is_not_movement(self) -> None:
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_scan(2.50)), 0.0)
        result = tracker.update(range_signature(_scan(2.51)), 0.8)

        self.assertEqual(result.verdict, "NOT_MOVING")

    def test_short_window_returns_unknown(self) -> None:
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_scan(2.5)), 0.0)
        result = tracker.update(range_signature(_scan(2.5)), 0.10)

        self.assertEqual(result.verdict, "UNKNOWN")

    def test_replayed_scan_timestamp_is_ignored(self) -> None:
        """The control loop runs far faster than the LD19. Re-reading one
        cached revolution must never accumulate as repeated evidence."""
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_scan(2.5)), 0.0)
        first = tracker.update(range_signature(_scan(2.5)), 0.8)
        repeat = tracker.update(range_signature(_scan(2.5)), 0.8)

        self.assertEqual(first.verdict, "NOT_MOVING")
        self.assertEqual(repeat.verdict, "UNKNOWN")

    def test_being_carried_is_flagged_as_displacement(self) -> None:
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_ramped_scan(1.0, 0.02)), 0.0)
        result = tracker.update(range_signature(_ramped_scan(3.0, 0.02)), 0.1)

        self.assertTrue(result.displaced)

    def test_ordinary_driving_is_not_displacement(self) -> None:
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_scan(2.50)), 0.0)
        result = tracker.update(range_signature(_scan(2.44)), 0.1)

        self.assertFalse(result.displaced)

    def test_reset_drops_all_history(self) -> None:
        tracker = ScanMotionTracker()
        tracker.update(range_signature(_scan(2.5)), 0.0)
        tracker.reset()
        result = tracker.update(range_signature(_scan(2.5)), 0.8)

        self.assertEqual(result.verdict, "UNKNOWN")
        self.assertFalse(result.displaced)

    def test_missing_scan_timestamp_is_unknown(self) -> None:
        tracker = ScanMotionTracker()
        self.assertEqual(
            tracker.update(range_signature(_scan(2.5)), None).verdict, "UNKNOWN"
        )


if __name__ == "__main__":
    unittest.main()
