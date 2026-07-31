"""Low-cost LiDAR mapping with cautious yaw correction for VisionFSD Pi.

The MPU-6050 supplies short-term yaw rate when live, and the LD19 corrects that
prediction only when successive scans have clear, low-residual agreement.
Translation starts from commanded-motion prediction and receives only bounded,
unique scan-to-map corrections because the chassis has no wheel encoders.  The
map supplies exploration guidance but never overrides current-scan motor safety.
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
    x_m: float = 0.0
    y_m: float = 0.0
    translation_confidence: float = 0.0
    translation_correction_m: float = 0.0
    translation_matched: bool = False
    observed_cells: int = 0


class LidarSlamLite:
    """A Pi 3B-friendly LiDAR/IMU exploration mapper.

    The angular matcher uses 360 one-degree bins and tests nearby shifts.
    Vectorized scan integration keeps this inexpensive beside camera inference, while
    preserving more useful LD19 geometry and rejecting ambiguous matches
    instead of inventing a pose.
    """

    BIN_COUNT = 360
    BIN_DEGREES = 360.0 / BIN_COUNT
    MIN_MATCH_BINS = 75
    MAX_SCAN_AGE_S = 0.35

    def __init__(self, cells: int = 320, metres: float = 8.0) -> None:
        self.cells = cells
        self.metres = metres
        self.grid = np.zeros((cells, cells), dtype=np.uint8)
        self.observed = np.zeros((cells, cells), dtype=np.uint8)
        self.visits = np.zeros((cells, cells), dtype=np.uint16)
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
        self._latest_hits = np.empty((0, 2), dtype=np.int32)
        self._using_imu = False
        self._translation_confidence = 0.0
        self._translation_correction_m = 0.0
        self._translation_matched = False

    @staticmethod
    def _signed_angle(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    @classmethod
    def bins_from_points(cls, points: Iterable[tuple[int, object]]) -> tuple[np.ndarray, float]:
        """Build robust one-degree range bins and return the newest timestamp."""
        point_list = list(points)
        values = np.full(cls.BIN_COUNT, np.inf, dtype=np.float32)
        if not point_list:
            values.fill(np.nan)
            return values, -1.0
        distances = np.fromiter(
            (float(point.distance_mm) / 1000.0 for _index, point in point_list),
            dtype=np.float32,
            count=len(point_list),
        )
        angles = np.fromiter(
            (float(point.angle_deg) for _index, point in point_list),
            dtype=np.float32,
            count=len(point_list),
        )
        stamps = np.fromiter(
            (float(point.captured_at) for _index, point in point_list),
            dtype=np.float64,
            count=len(point_list),
        )
        valid = (distances >= 0.10) & (distances <= 5.8)
        if not np.any(valid):
            values.fill(np.nan)
            return values, -1.0
        bins = ((angles[valid] % 360.0) / cls.BIN_DEGREES).astype(np.int16) % cls.BIN_COUNT
        np.minimum.at(values, bins, distances[valid])
        values[~np.isfinite(values)] = np.nan
        newest = float(np.max(stamps[valid]))
        return values, newest

    def _align_yaw(self, current: np.ndarray, expected_delta_deg: float) -> tuple[float, float, bool]:
        """Return (correction, confidence, accepted) for the current scan.

        A positive bin shift means a stationary world feature moved right in
        the robot frame, which corresponds to a left (negative) robot heading
        change in this project's coordinate convention.
        """
        if self._previous_bins is None:
            return 0.0, 0.0, False
        shifts = np.arange(-12, 13, dtype=np.int16)
        shifted = np.stack(
            [np.roll(self._previous_bins, int(shift)) for shift in shifts],
            axis=0,
        )
        valid = np.isfinite(current)[None, :] & np.isfinite(shifted)
        counts = np.count_nonzero(valid, axis=1)
        eligible = np.flatnonzero(counts >= self.MIN_MATCH_BINS)
        if eligible.size < 2:
            return 0.0, 0.0, False
        # Evaluate every nearby yaw shift in one NumPy operation.  Median
        # absolute range error remains robust to a moving person or a newly
        # seen chair, without a Python median loop on the Pi.
        residual_matrix = np.where(
            valid[eligible],
            np.abs(shifted[eligible] - current[None, :]),
            np.nan,
        )
        residuals = np.nanmedian(residual_matrix, axis=1)
        order = np.argsort(residuals)
        best_row = int(eligible[order[0]])
        best_residual = float(residuals[order[0]])
        runner_residual = float(residuals[order[1]])
        best_shift = int(shifts[best_row])
        count = int(counts[best_row])
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

    def integrate_motion(
        self,
        left_pwm: int,
        right_pwm: int,
        now: float,
        imu_yaw_rate_dps: float | None = None,
    ) -> None:
        if self._last_motion_at is None:
            self._last_motion_at = now
            return
        elapsed = min(0.20, max(0.0, now - self._last_motion_at))
        self._last_motion_at = now
        # These deliberately conservative values are only a prediction used by
        # the visual local map.  They are not odometry and never control drive.
        linear_mps = ((left_pwm + right_pwm) * 0.5 / 255.0) * 0.26
        self._using_imu = imu_yaw_rate_dps is not None
        # The mapper's heading increases clockwise; robot-frame +Z gyro is
        # counter-clockwise.  Negate the IMU rate to preserve map convention.
        turn_rate_dps = (
            -imu_yaw_rate_dps
            if imu_yaw_rate_dps is not None
            else ((left_pwm - right_pwm) / 255.0) * 130.0
        )
        yaw_delta = turn_rate_dps * elapsed
        self.heading = (self.heading + yaw_delta) % 360.0
        self._yaw_since_scan += yaw_delta
        radians = math.radians(self.heading)
        self.x += math.sin(radians) * linear_mps * elapsed
        self.y -= math.cos(radians) * linear_mps * elapsed
        margin = 0.20
        self.x = float(np.clip(self.x, margin, self.metres - margin))
        self.y = float(np.clip(self.y, margin, self.metres - margin))

    def _align_translation(
        self, points: list[tuple[int, object]]
    ) -> tuple[float, float, float, bool]:
        """Correlate current scan hits against the map near the predicted pose."""
        if self._map_updates < 3 or not points:
            return 0.0, 0.0, 0.0, False
        distances = np.fromiter(
            (float(point.distance_mm) / 1000.0 for _index, point in points),
            dtype=np.float32,
            count=len(points),
        )
        angles = np.fromiter(
            (float(point.angle_deg) for _index, point in points),
            dtype=np.float32,
            count=len(points),
        )
        confidence = np.fromiter(
            (float(getattr(point, "confidence", 0.0)) for _index, point in points),
            dtype=np.float32,
            count=len(points),
        )
        valid = (
            (distances >= 0.15)
            & (distances <= min(4.5, self.metres * 0.55))
            & (confidence >= 35.0)
        )
        distances = distances[valid]
        angles = angles[valid]
        if distances.size < 35:
            return 0.0, 0.0, 0.0, False
        stride = max(1, distances.size // 140)
        distances = distances[::stride]
        angles = angles[::stride]
        scale = self.cells / self.metres
        radians = np.radians(angles + self.heading)
        base_cols = ((self.x + np.sin(radians) * distances) * scale).astype(np.int32)
        base_rows = ((self.y - np.cos(radians) * distances) * scale).astype(np.int32)
        margin_cells = 5
        inside = (
            (base_rows >= margin_cells)
            & (base_rows < self.cells - margin_cells)
            & (base_cols >= margin_cells)
            & (base_cols < self.cells - margin_cells)
        )
        base_rows = base_rows[inside]
        base_cols = base_cols[inside]
        if base_rows.size < 30:
            return 0.0, 0.0, 0.0, False

        candidates: list[tuple[float, int, int]] = []
        for row_offset in range(-4, 5):
            for col_offset in range(-4, 5):
                score = float(np.mean(
                    self.grid[
                        base_rows + row_offset,
                        base_cols + col_offset,
                    ]
                ))
                candidates.append((score, row_offset, col_offset))
        candidates.sort(reverse=True)
        best_score, best_row, best_col = candidates[0]
        zero_score = next(
            score
            for score, row_offset, col_offset in candidates
            if row_offset == 0 and col_offset == 0
        )
        second_score = candidates[1][0]
        uniqueness = best_score - second_score
        improvement = best_score - zero_score
        accepted = (
            (best_row != 0 or best_col != 0)
            and best_score >= 7.0
            and improvement >= 1.0
            and uniqueness >= 0.08
        )
        match_confidence = float(np.clip(
            (best_score / 24.0) * 0.65 + (max(0.0, uniqueness) / 2.0) * 0.35,
            0.0,
            1.0,
        ))
        if not accepted:
            return 0.0, 0.0, match_confidence, False
        return (
            best_col / scale,
            best_row / scale,
            match_confidence,
            True,
        )

    def _integrate_points(self, points: Iterable[tuple[int, object]]) -> None:
        point_list = list(points)
        scale = self.cells / self.metres
        accumulator = (self.grid.astype(np.uint16) * 248) // 250
        if point_list:
            distances = np.fromiter(
                (float(point.distance_mm) / 1000.0 for _index, point in point_list),
                dtype=np.float32,
                count=len(point_list),
            )
            angles = np.fromiter(
                (float(point.angle_deg) for _index, point in point_list),
                dtype=np.float32,
                count=len(point_list),
            )
            confidence = np.fromiter(
                (
                    float(getattr(point, "confidence", 0.0))
                    for _index, point in point_list
                ),
                dtype=np.float32,
                count=len(point_list),
            )
            valid = (
                (distances >= 0.10)
                & (distances <= self.metres * 0.62)
                & (confidence >= 20.0)
            )
            distances = distances[valid]
            radians = np.radians(angles[valid] + self.heading)
            cols = ((self.x + np.sin(radians) * distances) * scale).astype(np.int32)
            rows = ((self.y - np.cos(radians) * distances) * scale).astype(np.int32)
            inside = (rows >= 0) & (rows < self.cells) & (cols >= 0) & (cols < self.cells)
            hit_rows = rows[inside]
            hit_cols = cols[inside]
            self._latest_hits = np.column_stack((hit_rows, hit_cols))

            visible = np.zeros_like(self.observed)
            robot_col = int(np.clip(round(self.x * scale), 0, self.cells - 1))
            robot_row = int(np.clip(round(self.y * scale), 0, self.cells - 1))
            ray_count = hit_rows.size
            ray_stride = max(1, ray_count // 180)
            for hit_row, hit_col in zip(
                hit_rows[::ray_stride], hit_cols[::ray_stride]
            ):
                cv2.line(
                    visible,
                    (robot_col, robot_row),
                    (int(hit_col), int(hit_row)),
                    255,
                    1,
                    cv2.LINE_8,
                )
            self.observed[visible > 0] = 255

            endpoint_mask = np.zeros_like(self.observed)
            endpoint_mask[hit_rows, hit_cols] = 255
            free_mask = (visible > 0) & (endpoint_mask == 0)
            accumulator[free_mask] = np.maximum(
                accumulator[free_mask].astype(np.int16) - 4, 0
            ).astype(np.uint16)
            np.add.at(accumulator, (hit_rows, hit_cols), 12)

            visit_mask = np.zeros_like(self.observed)
            cv2.circle(
                visit_mask,
                (robot_col, robot_row),
                max(1, int(round(0.12 * scale))),
                1,
                -1,
            )
            visit_cells = visit_mask > 0
            self.visits[visit_cells] = np.minimum(
                self.visits[visit_cells].astype(np.uint32) + 1,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        else:
            self._latest_hits = np.empty((0, 2), dtype=np.int32)
        self.grid = np.minimum(accumulator, 255).astype(np.uint8)
        self._map_updates += 1

    def update(
        self,
        points: list[tuple[int, object]],
        left_pwm: int,
        right_pwm: int,
        now: float,
        imu_yaw_rate_dps: float | None = None,
    ) -> SlamLiteState:
        self.integrate_motion(left_pwm, right_pwm, now, imu_yaw_rate_dps)
        bins, scan_stamp = self.bins_from_points(points)
        self._matched = False
        self._translation_matched = False
        self._translation_correction_m = 0.0
        new_scan = scan_stamp > self._last_scan_stamp + 0.035 and now - scan_stamp <= self.MAX_SCAN_AGE_S
        if new_scan:
            correction, confidence, accepted = self._align_yaw(bins, self._yaw_since_scan)
            self._yaw_confidence = self._yaw_confidence * 0.65 + confidence * 0.35
            self._last_correction = correction if accepted else 0.0
            if accepted:
                self.heading = (self.heading + correction) % 360.0
                self._matched = True
            translation_x = translation_y = translation_confidence = 0.0
            translation_accepted = False
            if left_pwm != 0 or right_pwm != 0:
                (
                    translation_x,
                    translation_y,
                    translation_confidence,
                    translation_accepted,
                ) = self._align_translation(points)
            self._translation_confidence = (
                self._translation_confidence * 0.72
                + translation_confidence * 0.28
            )
            if translation_accepted:
                self.x += translation_x
                self.y += translation_y
                self._translation_correction_m = math.hypot(
                    translation_x, translation_y
                )
                self._translation_matched = True
            self._previous_bins = bins
            self._last_scan_stamp = scan_stamp
            self._yaw_since_scan = 0.0
            # Integrate once per physical LD19 update, not once per control
            # loop.  This reduces Pi work and prevents a single scan from
            # becoming artificially certain just because the planner runs fast.
            self._integrate_points(points)
        return self.state()

    def state(self) -> SlamLiteState:
        return SlamLiteState(
            self.heading,
            self._yaw_confidence,
            self._last_correction,
            self._matched,
            self._map_updates,
            self.x,
            self.y,
            self._translation_confidence,
            self._translation_correction_m,
            self._translation_matched,
            int(np.count_nonzero(self.observed)),
        )

    def render(
        self,
        size: int = 500,
        target_xy: tuple[float, float] | None = None,
        waypoint_xy: tuple[float, float] | None = None,
    ) -> np.ndarray:
        image = cv2.resize(self.grid, (size, size), interpolation=cv2.INTER_NEAREST)
        panel = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        observed = cv2.resize(
            self.observed, (size, size), interpolation=cv2.INTER_NEAREST
        )
        panel[(observed > 0) & (image < 18)] = (25, 34, 42)
        if self._latest_hits.size:
            hit_y = np.clip(
                ((self._latest_hits[:, 0] + 0.5) * size / self.cells).astype(np.int32),
                1,
                size - 2,
            )
            hit_x = np.clip(
                ((self._latest_hits[:, 1] + 0.5) * size / self.cells).astype(np.int32),
                1,
                size - 2,
            )
            # Bright current-scan points stay metrically sharper than the
            # fading dead-reckoned history underneath them.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    panel[hit_y + dy, hit_x + dx] = (0, 220, 255)
        px = int(np.clip(self.x / self.metres * size, 0, size - 1))
        py = int(np.clip(self.y / self.metres * size, 0, size - 1))
        pixels_per_metre = size / self.metres
        for metres in range(1, int(self.metres / 2.0) + 1):
            cv2.circle(panel, (px, py), int(metres * pixels_per_metre),
                       (45, 65, 75), 1, cv2.LINE_AA)
        radians = math.radians(self.heading)
        tip = (int(px + math.sin(radians) * 20), int(py - math.cos(radians) * 20))
        if target_xy is not None:
            target = (
                int(np.clip(target_xy[0] / self.metres * size, 0, size - 1)),
                int(np.clip(target_xy[1] / self.metres * size, 0, size - 1)),
            )
            cv2.circle(panel, target, 9, (225, 90, 225), 2, cv2.LINE_AA)
        if waypoint_xy is not None:
            waypoint = (
                int(np.clip(waypoint_xy[0] / self.metres * size, 0, size - 1)),
                int(np.clip(waypoint_xy[1] / self.metres * size, 0, size - 1)),
            )
            cv2.line(panel, (px, py), waypoint, (80, 210, 255), 1, cv2.LINE_AA)
            cv2.circle(panel, waypoint, 6, (80, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, (px, py), 8, (80, 240, 100), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, (px, py), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        label = "LIDAR+IMU EXPLORATION MAP - ESTIMATED POSE"
        yaw_source = "IMU" if self._using_imu else "COMMAND"
        detail = (
            f"{self.metres / self.cells * 100:.1f} cm/cell  scans {self._map_updates}  "
            f"yaw {yaw_source} {self._yaw_confidence:.2f}  "
            f"xy match {self._translation_confidence:.2f}"
        )
        cv2.rectangle(panel, (0, size - 56), (size, size), (14, 22, 31), -1)
        cv2.putText(panel, label, (12, size - 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (235, 245, 250), 1, cv2.LINE_AA)
        cv2.putText(panel, detail, (12, size - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (190, 215, 230), 1, cv2.LINE_AA)
        return panel
