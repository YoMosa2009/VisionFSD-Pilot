"""Low-cost LiDAR mapping with cautious yaw correction for VisionFSD Pi.

The OSOYOO chassis has no encoders or IMU, so full metric SLAM is not an
honest claim.  This module intentionally does less: it keeps a small rolling
occupancy map from the LD19 and applies a heading correction only when two
successive scans have a clear, low-residual angular match.  Translation remains
commanded-motion dead reckoning and the map is advisory only; it is never an
input to the motor-safety decision path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class SlamLiteState:
    heading_deg: float
    yaw_confidence: float
    yaw_correction_deg: float
    matched: bool
    map_updates: int


class LidarSlamLite:
    """A Pi 3B-friendly, advisory local LiDAR mapper.

    The angular matcher uses 72 five-degree bins and tests only 17 possible
    shifts.  That is inexpensive enough to run beside camera inference, while
    rejecting ambiguous matches instead of inventing a pose.
    """

    BIN_COUNT = 72
    BIN_DEGREES = 360.0 / BIN_COUNT
    MIN_MATCH_BINS = 24
    MAX_SCAN_AGE_S = 0.35

    def __init__(self, cells: int = 100, metres: float = 5.0) -> None:
        self.cells = cells
        self.metres = metres
        self.grid = np.zeros((cells, cells), dtype=np.uint8)
        self.x = metres / 2.0
        self.y = metres / 2.0
        self.heading = 0.0
        self._last_motion_at: float | None = None
        self._last_scan_stamp = -1.0
        self._previous_bins: np.ndarray | None = None
        self._yaw_since_scan = 0.0
        self._yaw_confidence = 0.0
        self._last_correction = 0.0
        self._matched = False
        self._map_updates = 0

    @staticmethod
    def _signed_angle(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    @classmethod
    def bins_from_points(cls, points: Iterable[tuple[int, object]]) -> tuple[np.ndarray, float]:
        """Build robust five-degree range bins and return the newest timestamp."""
        buckets: list[list[float]] = [[] for _ in range(cls.BIN_COUNT)]
        newest = -1.0
        for _index, point in points:
            distance = float(point.distance_mm) / 1000.0
            if not 0.10 <= distance <= 4.5:
                continue
            bin_index = int((float(point.angle_deg) % 360.0) / cls.BIN_DEGREES) % cls.BIN_COUNT
            buckets[bin_index].append(distance)
            newest = max(newest, float(point.captured_at))
        values = np.full(cls.BIN_COUNT, np.nan, dtype=np.float32)
        for index, samples in enumerate(buckets):
            if samples:
                values[index] = float(np.median(samples))
        return values, newest

    def _align_yaw(self, current: np.ndarray, expected_delta_deg: float) -> tuple[float, float, bool]:
        """Return (correction, confidence, accepted) for the current scan.

        A positive bin shift means a stationary world feature moved right in
        the robot frame, which corresponds to a left (negative) robot heading
        change in this project's coordinate convention.
        """
        if self._previous_bins is None:
            return 0.0, 0.0, False
        choices: list[tuple[float, int, int]] = []
        for shift in range(-8, 9):
            shifted_previous = np.roll(self._previous_bins, shift)
            valid = np.isfinite(current) & np.isfinite(shifted_previous)
            count = int(np.count_nonzero(valid))
            if count < self.MIN_MATCH_BINS:
                continue
            # Median absolute range error is robust to a moving person or a
            # newly seen chair occupying only part of the scan.
            residual = float(np.median(np.abs(current[valid] - shifted_previous[valid])))
            choices.append((residual, shift, count))
        if len(choices) < 2:
            return 0.0, 0.0, False
        choices.sort(key=lambda item: item[0])
        best_residual, best_shift, count = choices[0]
        runner_residual = choices[1][0]
        separation = runner_residual - best_residual
        confidence = float(np.clip((separation / 0.035) * (count / self.BIN_COUNT), 0.0, 1.0))
        observed_delta = -best_shift * self.BIN_DEGREES
        mismatch = self._signed_angle(observed_delta - expected_delta_deg)
        # A strong mismatch means either translation dominated the scan or the
        # room geometry was ambiguous.  Do not turn a map display guess into a
        # pose correction in that case.
        accepted = best_residual <= 0.20 and confidence >= 0.16 and abs(mismatch) <= 20.0
        if not accepted:
            return 0.0, confidence, False
        return float(np.clip(mismatch * 0.35, -5.0, 5.0)), confidence, True

    def integrate_motion(self, left_pwm: int, right_pwm: int, now: float) -> None:
        if self._last_motion_at is None:
            self._last_motion_at = now
            return
        elapsed = min(0.20, max(0.0, now - self._last_motion_at))
        self._last_motion_at = now
        # These deliberately conservative values are only a prediction used by
        # the visual local map.  They are not odometry and never control drive.
        linear_mps = ((left_pwm + right_pwm) * 0.5 / 105.0) * 0.10
        turn_rate_dps = ((left_pwm - right_pwm) / 105.0) * 96.0
        yaw_delta = turn_rate_dps * elapsed
        self.heading = (self.heading + yaw_delta) % 360.0
        self._yaw_since_scan += yaw_delta
        radians = math.radians(self.heading)
        self.x += math.sin(radians) * linear_mps * elapsed
        self.y -= math.cos(radians) * linear_mps * elapsed

    def _integrate_points(self, points: Iterable[tuple[int, object]]) -> None:
        scale = self.cells / self.metres
        for _index, point in list(points)[::3]:
            distance = float(point.distance_mm) / 1000.0
            if not 0.10 <= distance <= self.metres / 1.5:
                continue
            angle = math.radians(float(point.angle_deg) + self.heading)
            x = self.x + math.sin(angle) * distance
            y = self.y - math.cos(angle) * distance
            col, row = int(x * scale), int(y * scale)
            if 0 <= row < self.cells and 0 <= col < self.cells:
                self.grid[row, col] = min(255, int(self.grid[row, col]) + 32)
        self.grid = (self.grid.astype(np.float32) * 0.992).astype(np.uint8)
        self._map_updates += 1

    def update(self, points: list[tuple[int, object]], left_pwm: int, right_pwm: int,
               now: float) -> SlamLiteState:
        self.integrate_motion(left_pwm, right_pwm, now)
        bins, scan_stamp = self.bins_from_points(points)
        self._matched = False
        new_scan = scan_stamp > self._last_scan_stamp + 0.035 and now - scan_stamp <= self.MAX_SCAN_AGE_S
        if new_scan:
            correction, confidence, accepted = self._align_yaw(bins, self._yaw_since_scan)
            self._yaw_confidence = self._yaw_confidence * 0.65 + confidence * 0.35
            self._last_correction = correction if accepted else 0.0
            if accepted:
                self.heading = (self.heading + correction) % 360.0
                self._matched = True
            self._previous_bins = bins
            self._last_scan_stamp = scan_stamp
            self._yaw_since_scan = 0.0
            # Integrate once per physical LD19 update, not once per control
            # loop.  This reduces Pi work and prevents a single scan from
            # becoming artificially certain just because the planner runs fast.
            self._integrate_points(points)
        return self.state()

    def state(self) -> SlamLiteState:
        return SlamLiteState(self.heading, self._yaw_confidence, self._last_correction,
                             self._matched, self._map_updates)

    def render(self, size: int = 500) -> np.ndarray:
        image = cv2.resize(self.grid, (size, size), interpolation=cv2.INTER_NEAREST)
        panel = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        px = int(np.clip(self.x / self.metres * size, 0, size - 1))
        py = int(np.clip(self.y / self.metres * size, 0, size - 1))
        radians = math.radians(self.heading)
        tip = (int(px + math.sin(radians) * 20), int(py - math.cos(radians) * 20))
        cv2.circle(panel, (px, py), 8, (80, 240, 100), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, (px, py), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        label = "SLAM-LITE LOCAL MAP - ADVISORY"
        detail = f"yaw match {self._yaw_confidence:.2f} correction {self._last_correction:+.1f} deg"
        cv2.putText(panel, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (235, 245, 250), 1, cv2.LINE_AA)
        cv2.putText(panel, detail, (12, 47), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (190, 215, 230), 1, cv2.LINE_AA)
        return panel
