"""Low-cost LiDAR mapping with cautious yaw correction for VisionFSD Pi.

The USB LSM6DS3 IMU supplies short-term yaw rate when live, and the LD19
corrects that prediction only when successive scans have clear, low-residual
agreement.
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

from lidar_visualizer import LidarPoint


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
    yaw_source: str = "COMMAND+LD19"
    recenter_count: int = 0


class LidarSlamLite:
    """A Pi 3B-friendly LiDAR/IMU exploration mapper.

    The angular matcher uses 360 one-degree bins and tests nearby shifts.
    Vectorized scan integration keeps this inexpensive beside camera motion analysis, while
    preserving more useful LD19 geometry and rejecting ambiguous matches
    instead of inventing a pose.
    """

    BIN_COUNT = 360
    BIN_DEGREES = 360.0 / BIN_COUNT
    MIN_MATCH_BINS = 75
    MAX_SCAN_AGE_S = 0.35
    # Recenter once the robot enters the outer fraction of the grid on either
    # axis, well before integrate_motion's safety clamp could ever bind. This
    # keeps the map a sliding window around the robot instead of a fixed
    # buffer anchored at the start position: without it, a robot that
    # travelled far enough from its origin would have its dead-reckoned pose
    # pinned at the array edge while it kept moving physically, so every new
    # scan projected onto a stale position and the map stopped updating.
    RECENTER_MARGIN_FRACTION = 0.20
    # Fold at most one LD19 revolution (about 100 ms) into the map at a time.
    #
    # The live scan is a sliding window refreshed every packet batch, so the
    # previous 35 ms gate let the same returns be integrated two or three
    # times per revolution. That spent Pi 3B time re-deskewing and re-matching
    # identical data, and it made occupancy evidence grow with the control
    # loop rate rather than with independent observations.
    SCAN_INTEGRATION_PERIOD_S = 0.085
    # Occupancy evidence outside the current view fades with this half-life.
    #
    # Fading used to be applied once per integration, so the effective memory
    # depended on how often the loop happened to integrate - about 3.4 s at the
    # old rate. Walls that briefly left the LD19's view were forgotten before
    # the planner could route around them again. Time-based fading keeps a
    # deliberate, rate-independent memory: long enough to plan multi-turn
    # routes through space not currently in view, short enough that
    # dead-reckoning drift cannot accumulate into a permanently wrong map.
    MEMORY_HALF_LIFE_S = 12.0

    def __init__(self, cells: int = 576, metres: float = 12.0) -> None:
        self.cells = cells
        self.metres = metres
        self.grid = np.zeros((cells, cells), dtype=np.uint8)
        # Retain sub-unit occupancy between scans. Flooring the decay every
        # revolution otherwise erases weak walls at the scan rate, not in seconds.
        self._decay_remainder = np.zeros((cells, cells), dtype=np.float32)
        self.observed = np.zeros((cells, cells), dtype=np.uint8)
        self.visits = np.zeros((cells, cells), dtype=np.uint16)
        self.x = metres / 2.0
        self.y = metres / 2.0
        self.heading = 0.0
        self._recenter_count = 0
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
        self._last_imu_yaw_deg: float | None = None
        self._using_camera = False
        self._translation_confidence = 0.0
        self._translation_correction_m = 0.0
        self._translation_matched = False
        self._last_decay_at: float | None = None
        self.last_shift_cells = (0, 0)

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
        residual_by_row = np.full(shifts.size, np.nan, dtype=np.float32)
        residual_by_row[eligible] = residuals
        order = np.argsort(residuals)
        best_row = int(eligible[order[0]])
        best_residual = float(residuals[order[0]])
        runner_residual = float(residuals[order[1]])
        best_shift = float(shifts[best_row])
        if 0 < best_row < shifts.size - 1:
            left_residual = float(residual_by_row[best_row - 1])
            right_residual = float(residual_by_row[best_row + 1])
            denominator = left_residual - 2.0 * best_residual + right_residual
            if np.isfinite(left_residual + right_residual) and denominator > 1e-6:
                best_shift += float(np.clip(
                    0.5 * (left_residual - right_residual) / denominator,
                    -0.5,
                    0.5,
                ))
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
        camera_yaw_rate_dps: float | None = None,
        camera_translation_scale: float = 1.0,
        imu_yaw_deg: float | None = None,
    ) -> None:
        if self._last_motion_at is None:
            self._last_motion_at = now
            self._last_imu_yaw_deg = imu_yaw_deg
            return
        elapsed = min(0.20, max(0.0, now - self._last_motion_at))
        self._last_motion_at = now
        # These deliberately conservative values are only a prediction used by
        # the visual local map.  They are not odometry and never control drive.
        linear_mps = (
            ((left_pwm + right_pwm) * 0.5 / 255.0)
            * 0.26
            * float(np.clip(camera_translation_scale, 0.15, 1.0))
        )
        self._using_imu = imu_yaw_rate_dps is not None
        self._using_camera = False
        # The mapper's heading increases clockwise; robot-frame +Z gyro is
        # counter-clockwise.  Negate the IMU rate to preserve map convention.
        command_turn_rate = ((left_pwm - right_pwm) / 255.0) * 130.0
        if imu_yaw_rate_dps is not None:
            if imu_yaw_deg is not None and self._last_imu_yaw_deg is not None:
                measured_delta = self._signed_angle(
                    imu_yaw_deg - self._last_imu_yaw_deg
                )
                # Reject an impossible discontinuity caused by a reconnect or
                # a stale async state. The LSM6DS3 has no absolute heading, so
                # this remains bounded short-term yaw, not global position.
                max_delta = max(3.0, 240.0 * elapsed)
                yaw_delta = -float(np.clip(
                    measured_delta, -max_delta, max_delta
                ))
            else:
                yaw_delta = -imu_yaw_rate_dps * elapsed
            self._last_imu_yaw_deg = imu_yaw_deg
        elif (
            camera_yaw_rate_dps is not None
            and abs(command_turn_rate) >= 4.0
            and command_turn_rate * camera_yaw_rate_dps > 0.0
        ):
            camera_rate = float(np.clip(camera_yaw_rate_dps, -180.0, 180.0))
            turn_rate_dps = command_turn_rate * 0.65 + camera_rate * 0.35
            self._using_camera = True
            yaw_delta = turn_rate_dps * elapsed
        else:
            turn_rate_dps = command_turn_rate
            yaw_delta = turn_rate_dps * elapsed
        if imu_yaw_rate_dps is None:
            self._last_imu_yaw_deg = None
        self.heading = (self.heading + yaw_delta) % 360.0
        self._yaw_since_scan += yaw_delta
        radians = math.radians(self.heading)
        self.x += math.sin(radians) * linear_mps * elapsed
        self.y -= math.cos(radians) * linear_mps * elapsed
        self._recenter_if_needed()

    def _recenter_if_needed(self) -> bool:
        """Keep the robot away from the fixed grid's edge by scrolling it.

        Shifts grid/observed/visits by an integer cell offset so the robot
        lands back near the centre, and moves self.x/self.y by the same
        distance so the world-frame pose stays numerically consistent with
        the new cell mapping.  Returns True when a shift happened, so the
        caller can invalidate anything else holding cell-index coordinates
        into this grid (frontier/route caches).
        """
        scale = self.cells / self.metres
        margin_cells = max(1, int(round(self.cells * self.RECENTER_MARGIN_FRACTION)))
        robot_row = int(round(self.y * scale))
        robot_col = int(round(self.x * scale))
        near_edge = (
            robot_row < margin_cells
            or robot_row > self.cells - 1 - margin_cells
            or robot_col < margin_cells
            or robot_col > self.cells - 1 - margin_cells
        )
        if not near_edge:
            return False
        centre = self.cells // 2
        shift_row = centre - robot_row
        shift_col = centre - robot_col
        self.grid = self._rolled_and_cleared(self.grid, shift_row, shift_col)
        self._decay_remainder = self._rolled_and_cleared(
            self._decay_remainder, shift_row, shift_col
        )
        self.observed = self._rolled_and_cleared(self.observed, shift_row, shift_col)
        self.visits = self._rolled_and_cleared(self.visits, shift_row, shift_col)
        if self._latest_hits.size:
            shifted = self._latest_hits + np.array([shift_row, shift_col], dtype=np.int32)
            inside = (
                (shifted[:, 0] >= 0) & (shifted[:, 0] < self.cells)
                & (shifted[:, 1] >= 0) & (shifted[:, 1] < self.cells)
            )
            self._latest_hits = shifted[inside]
        self.y += shift_row / scale
        self.x += shift_col / scale
        self._recenter_count += 1
        self.last_shift_cells = (shift_row, shift_col)
        return True

    @staticmethod
    def _rolled_and_cleared(array: np.ndarray, shift_row: int, shift_col: int) -> np.ndarray:
        """Roll array contents by an integer cell offset and blank the seam.

        np.roll wraps cyclically: the band that wraps in from the far side of
        the array is stale content from outside the robot's actual vicinity,
        not real geometry adjacent to its new position, so it must be zeroed
        rather than left to look like a phantom wall or false free space.
        """
        if shift_row == 0 and shift_col == 0:
            return array
        rolled = np.roll(array, (shift_row, shift_col), axis=(0, 1))
        if shift_row > 0:
            rolled[:shift_row, :] = 0
        elif shift_row < 0:
            rolled[shift_row:, :] = 0
        if shift_col > 0:
            rolled[:, :shift_col] = 0
        elif shift_col < 0:
            rolled[:, shift_col:] = 0
        return rolled

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
            & (distances <= min(5.6, self.metres * 0.72))
            & (confidence >= 35.0)
        )
        distances = distances[valid]
        angles = angles[valid]
        if distances.size < 35:
            return 0.0, 0.0, 0.0, False
        stride = max(1, math.ceil(distances.size / 180))
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

        # Score all 121 candidate offsets with one gather instead of 121
        # separate fancy-index operations; on a Pi 3B the Python loop cost
        # dominated the actual arithmetic.
        offsets = np.arange(-5, 6, dtype=np.int32)
        row_offsets = np.repeat(offsets, offsets.size)
        col_offsets = np.tile(offsets, offsets.size)
        gathered = self.grid[
            base_rows[None, :] + row_offsets[:, None],
            base_cols[None, :] + col_offsets[:, None],
        ]
        scores = gathered.mean(axis=1, dtype=np.float32)
        candidates: list[tuple[float, int, int]] = sorted(
            (
                (float(score), int(row_offset), int(col_offset))
                for score, row_offset, col_offset in zip(
                    scores, row_offsets, col_offsets
                )
            ),
            reverse=True,
        )
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

    #: Consecutive returns closer together than this, in degrees, are treated
    #: as sampling one continuous surface, so the wedge between them is
    #: observed free space. The LD19 samples every 0.8 degrees at 10 Hz; a
    #: wider gap is missing returns (dark or distant surfaces), which say
    #: nothing about the space in between.
    FREE_WEDGE_MAX_GAP_DEG = 2.5
    #: Range ratio beyond which two adjacent returns are different surfaces.
    FREE_WEDGE_MAX_RANGE_RATIO = 1.25

    def _trace_visible(
        self,
        visible: np.ndarray,
        robot_row: int,
        robot_col: int,
        rows: np.ndarray,
        cols: np.ndarray,
        angles_deg: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        """Mark everything the scan saw through as observed.

        Previously at most 240 rays were drawn one ``cv2.line`` call at a time,
        so on a 450-return revolution about half the measured free space was
        never marked, and the Python loop cost scaled with the scan. Here every
        return contributes: adjacent returns on one continuous surface fill
        the wedge between them in a single ``cv2.fillPoly`` call, and every
        ray is drawn in a single ``cv2.polylines`` call. Wedges are not filled
        across a depth jump or a run of missing returns, so a doorway's far
        side is not claimed as seen through its frame.
        """
        count = rows.size
        if count == 0:
            return
        origin = np.array([robot_col, robot_row], dtype=np.int32)
        ends = np.column_stack((cols, rows)).astype(np.int32)
        segments = np.stack(
            (np.broadcast_to(origin, ends.shape), ends), axis=1
        )
        cv2.polylines(visible, list(segments), False, 255, 1, cv2.LINE_8)
        if count < 2:
            return
        order = np.argsort(angles_deg)
        ordered_angles = angles_deg[order]
        ordered_ranges = distances[order]
        ordered_ends = ends[order]
        gap = np.diff(ordered_angles)
        near = np.minimum(ordered_ranges[:-1], ordered_ranges[1:])
        far = np.maximum(ordered_ranges[:-1], ordered_ranges[1:])
        continuous = (
            (gap <= self.FREE_WEDGE_MAX_GAP_DEG)
            & (far <= near * self.FREE_WEDGE_MAX_RANGE_RATIO)
        )
        if not np.any(continuous):
            return
        first = ordered_ends[:-1][continuous]
        second = ordered_ends[1:][continuous]
        wedges = np.stack(
            (np.broadcast_to(origin, first.shape), first, second), axis=1
        )
        cv2.fillPoly(visible, list(wedges), 255)

    def _decay_factor(self, now: float | None) -> float:
        """Fraction of occupancy evidence to keep since the last fade."""
        if now is None:
            # Direct callers without a clock (tests, tools) get one nominal
            # revolution of fading, matching the old per-call behaviour.
            return 0.5 ** (self.SCAN_INTEGRATION_PERIOD_S / self.MEMORY_HALF_LIFE_S)
        previous = self._last_decay_at
        self._last_decay_at = now
        if previous is None:
            return 1.0
        elapsed = min(2.0, max(0.0, now - previous))
        return 0.5 ** (elapsed / self.MEMORY_HALF_LIFE_S)

    def _integrate_points(
        self, points: Iterable[tuple[int, object]], now: float | None = None
    ) -> None:
        point_list = list(points)
        scale = self.cells / self.metres
        keep = self._decay_factor(now)
        accumulator = (self.grid.astype(np.float32) + self._decay_remainder) * keep
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
                & (distances <= self.metres * 0.72)
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
            self._trace_visible(
                visible,
                robot_row,
                robot_col,
                rows,
                cols,
                angles[valid],
                distances,
            )
            self.observed[visible > 0] = 255

            endpoint_mask = np.zeros_like(self.observed)
            endpoint_mask[hit_rows, hit_cols] = 255
            self.observed[endpoint_mask > 0] = 255
            free_mask = (visible > 0) & (endpoint_mask == 0)
            accumulator[free_mask] = np.maximum(
                accumulator[free_mask] - 4.0, 0.0
            )
            # Multiple high-resolution rays can land in one occupancy cell.
            # More samples preserve shape, not multiple independent confirmations.
            accumulator[endpoint_mask > 0] += 12

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
        np.minimum(accumulator, 255.0, out=accumulator)
        self.grid = accumulator.astype(np.uint8)
        self._decay_remainder = accumulator - self.grid
        self._map_updates += 1

    @classmethod
    def deskew_points(cls, points: list[tuple[int, object]], now: float,
                      imu_yaw_rate_dps: float | None) -> list[tuple[int, object]]:
        """Approximate rotational compensation for map input only.

        Receipt timestamps and a constant recent gyro rate are imperfect; cap
        the correction and retain raw ranges for the independent safety path.
        Positive IMU Z is counter-clockwise, opposite the map heading convention.
        """
        if imu_yaw_rate_dps is None or not math.isfinite(imu_yaw_rate_dps) or abs(imu_yaw_rate_dps) > 80.:
            return points
        corrected = []
        for index, point in points:
            age = now - point.captured_at
            correction = (max(-10., min(10., imu_yaw_rate_dps * age))
                          if 0.0 <= age <= cls.MAX_SCAN_AGE_S else 0.0)
            corrected.append((index, LidarPoint(
                (point.angle_deg + correction) % 360., point.distance_mm,
                getattr(point, "confidence", 0), point.captured_at,
            )))
        return corrected

    def update(
        self,
        points: list[tuple[int, object]],
        left_pwm: int,
        right_pwm: int,
        now: float,
        imu_yaw_rate_dps: float | None = None,
        camera_yaw_rate_dps: float | None = None,
        camera_translation_scale: float = 1.0,
        imu_yaw_deg: float | None = None,
        scan_stamp_hint: float | None = None,
    ) -> SlamLiteState:
        self.integrate_motion(
            left_pwm,
            right_pwm,
            now,
            imu_yaw_rate_dps,
            camera_yaw_rate_dps,
            camera_translation_scale,
            imu_yaw_deg,
        )
        self._matched = False
        self._translation_matched = False
        self._translation_correction_m = 0.0
        if (
            scan_stamp_hint is not None
            and scan_stamp_hint
            <= self._last_scan_stamp + self.SCAN_INTEGRATION_PERIOD_S
        ):
            # Nothing new enough to integrate. Deskewing and binning allocate
            # a fresh object per return, so skipping them on unchanged data is
            # most of the per-tick cost of keeping a map at all.
            return self.state()
        points = self.deskew_points(points, now, imu_yaw_rate_dps)
        bins, scan_stamp = self.bins_from_points(points)
        new_scan = (
            scan_stamp
            > self._last_scan_stamp + self.SCAN_INTEGRATION_PERIOD_S
            and now - scan_stamp <= self.MAX_SCAN_AGE_S
        )
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
            self._integrate_points(points, now)
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
            (
                "IMU+LD19"
                if self._using_imu
                else "COMMAND+CAMERA+LD19"
                if self._using_camera
                else "COMMAND+LD19"
            ),
            self._recenter_count,
        )

    def reset(self) -> None:
        """Discard the map and restart the pose at the centre of a new grid.

        Used when the chassis is picked up or shoved: the occupancy grid, the
        visit counts and the pose all describe a place the robot is no longer
        in, and there is no encoder or absolute reference that could relate
        the old frame to the new one. Starting clean is honest; carrying the
        old map forward would silently corrupt every later plan.
        """
        self.grid[:] = 0
        self._decay_remainder[:] = 0
        self._last_decay_at = None
        self.observed[:] = 0
        self.visits[:] = 0
        self.x = self.metres / 2.0
        self.y = self.metres / 2.0
        self.heading = 0.0
        self._last_motion_at = None
        self._last_scan_stamp = -1.0
        self._previous_bins = None
        self._yaw_since_scan = 0.0
        self._yaw_confidence = 0.0
        self._last_correction = 0.0
        self._matched = False
        self._map_updates = 0
        self._latest_hits = np.empty((0, 2), dtype=np.int32)
        self._last_imu_yaw_deg = None
        self._translation_confidence = 0.0
        self._translation_correction_m = 0.0
        self._translation_matched = False

    def _arc_pixels(
        self,
        arc_xy: tuple[tuple[float, float], ...],
        px: int,
        py: int,
        pixels_per_metre: float,
    ) -> np.ndarray | None:
        """Rotate a robot-frame arc into the world-aligned follow view."""
        if len(arc_xy) < 2:
            return None
        radians = math.radians(self.heading)
        cos = math.cos(radians)
        sin = math.sin(radians)
        points = [(float(px), float(py))]
        for offset_x, offset_y in arc_xy:
            # Robot frame is +y ahead, +x right; the panel is world-aligned
            # with the heading arrow drawn at self.heading.
            world_x = offset_x * cos + offset_y * sin
            world_y = offset_x * sin - offset_y * cos
            points.append(
                (
                    px + world_x * pixels_per_metre,
                    py + world_y * pixels_per_metre,
                )
            )
        return np.array(points, dtype=np.float32)

    @staticmethod
    def _draw_intent_ribbon(
        panel: np.ndarray,
        points: np.ndarray,
        color: tuple[int, int, int],
        clearance_m: float | None,
    ) -> None:
        """Draw the predicted trajectory as a tapering, fading ribbon.

        Thickness falls along the arc because the near end is what the robot
        is committed to and the far end is a prediction; a single flat line
        gave both equal visual weight. When the planner reports how much room
        the arc has, a tight one is tinted toward its warning colour so a
        squeeze is legible at a glance rather than only in the text rows.
        """
        tint = 1.0
        if clearance_m is not None:
            tint = float(np.clip(clearance_m / 0.45, 0.25, 1.0))
        drawn = (
            int(color[0] * tint + 70 * (1.0 - tint)),
            int(color[1] * tint + 120 * (1.0 - tint)),
            int(color[2] * tint + 255 * (1.0 - tint)),
        )
        integer_points = np.rint(points).astype(np.int32)
        segments = len(integer_points) - 1
        for index in range(segments):
            span = 1.0 - index / max(1, segments)
            thickness = max(1, int(round(1.0 + 5.0 * span)))
            shade = 0.45 + 0.55 * span
            segment_color = tuple(int(channel * shade) for channel in drawn)
            cv2.line(
                panel,
                tuple(integer_points[index]),
                tuple(integer_points[index + 1]),
                segment_color,
                thickness,
                cv2.LINE_AA,
            )
        cv2.arrowedLine(
            panel,
            tuple(integer_points[-2]),
            tuple(integer_points[-1]),
            drawn,
            2,
            cv2.LINE_AA,
            tipLength=0.55,
        )

    def render(
        self,
        size: int = 500,
        target_xy: tuple[float, float] | None = None,
        waypoint_xy: tuple[float, float] | None = None,
        path_xy: tuple[tuple[float, float], ...] = (),
        steering_deg: float | None = None,
        intent_color: tuple[int, int, int] = (120, 230, 255),
        arc_xy: tuple[tuple[float, float], ...] = (),
        arc_clearance_m: float | None = None,
    ) -> np.ndarray:
        # Keep the estimated chassis at the centre of a robot-following local
        # viewport. The underlying occupancy grid is a fixed-size sliding
        # window that recenters around the robot (see _recenter_if_needed),
        # not a buffer anchored at the start position, so it keeps covering
        # new ground no matter how far the robot travels from its origin.
        view_metres = min(6.0, self.metres)
        view_cells = self.cells * view_metres / self.metres
        pixels_per_cell = size / view_cells
        scale = self.cells / self.metres
        robot_col = self.x * scale
        robot_row = self.y * scale
        centre = size * 0.5
        transform = np.array(
            (
                (pixels_per_cell, 0.0, centre - robot_col * pixels_per_cell),
                (0.0, pixels_per_cell, centre - robot_row * pixels_per_cell),
            ),
            dtype=np.float32,
        )
        image = cv2.warpAffine(
            self.grid,
            transform,
            (size, size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        panel = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        observed = cv2.warpAffine(
            self.observed,
            transform,
            (size, size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        panel[(observed > 0) & (image < 18)] = (25, 34, 42)
        if self._latest_hits.size:
            hit_y = np.rint(
                centre + (self._latest_hits[:, 0] + 0.5 - robot_row)
                * pixels_per_cell
            ).astype(np.int32)
            hit_x = np.rint(
                centre + (self._latest_hits[:, 1] + 0.5 - robot_col)
                * pixels_per_cell
            ).astype(np.int32)
            visible_hits = (
                (hit_y >= 1) & (hit_y < size - 1)
                & (hit_x >= 1) & (hit_x < size - 1)
            )
            hit_y = hit_y[visible_hits]
            hit_x = hit_x[visible_hits]
            # Bright current-scan points stay metrically sharper than the
            # fading dead-reckoned history underneath them.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    panel[hit_y + dy, hit_x + dx] = (0, 220, 255)
        px = py = size // 2
        pixels_per_metre = size / view_metres
        for metres in range(1, int(view_metres / 2.0) + 1):
            cv2.circle(panel, (px, py), int(metres * pixels_per_metre),
                       (45, 65, 75), 1, cv2.LINE_AA)
        radians = math.radians(self.heading)
        tip = (int(px + math.sin(radians) * 20), int(py - math.cos(radians) * 20))
        # The planned route, drawn from the chassis outward so the intended
        # path is visible as a path rather than inferred from a single target
        # marker. Clipped to the viewport by polylines() itself.
        if len(path_xy) >= 2:
            points = np.array(
                [
                    (
                        px + (point[0] - self.x) * pixels_per_metre,
                        py + (point[1] - self.y) * pixels_per_metre,
                    )
                    for point in path_xy
                ],
                dtype=np.float32,
            )
            polyline = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                panel, [polyline], False, (40, 120, 150), 5, cv2.LINE_AA
            )
            cv2.polylines(
                panel, [polyline], False, intent_color, 2, cv2.LINE_AA
            )
        # The trajectory the chassis is actually about to follow over the next
        # planning horizon, drawn as a widening ribbon out of the robot
        # marker. This is the arc the local planner selected and committed to,
        # so it answers "what is it about to do" directly, where the old
        # single steering ray only showed an instantaneous angle.
        arc_points = self._arc_pixels(arc_xy, px, py, pixels_per_metre)
        if arc_points is not None:
            self._draw_intent_ribbon(
                panel, arc_points, intent_color, arc_clearance_m
            )
        elif steering_deg is not None:
            steer_radians = math.radians(self.heading + steering_deg)
            steer_tip = (
                int(px + math.sin(steer_radians) * 46),
                int(py - math.cos(steer_radians) * 46),
            )
            cv2.arrowedLine(
                panel, (px, py), steer_tip, intent_color, 2, cv2.LINE_AA,
                tipLength=0.22,
            )
        if target_xy is not None:
            target = (
                int(round(px + (target_xy[0] - self.x) * pixels_per_metre)),
                int(round(py + (target_xy[1] - self.y) * pixels_per_metre)),
            )
            if 0 <= target[0] < size and 0 <= target[1] < size:
                cv2.circle(panel, target, 9, (225, 90, 225), 2, cv2.LINE_AA)
        if waypoint_xy is not None:
            waypoint = (
                int(round(px + (waypoint_xy[0] - self.x) * pixels_per_metre)),
                int(round(py + (waypoint_xy[1] - self.y) * pixels_per_metre)),
            )
            if 0 <= waypoint[0] < size and 0 <= waypoint[1] < size:
                cv2.line(panel, (px, py), waypoint, (80, 210, 255), 1, cv2.LINE_AA)
                cv2.circle(panel, waypoint, 6, (80, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, (px, py), 8, (80, 240, 100), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, (px, py), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        label = "LIDAR+IMU EXPLORATION MAP - ROBOT FOLLOW VIEW"
        yaw_source = "IMU+LD19" if self._using_imu else "COMMAND+LD19"
        detail = (
            f"FOLLOW {view_metres:.1f}m  {self.metres / self.cells * 100:.1f} cm/cell  "
            f"scans {self._map_updates}  "
            f"yaw {yaw_source} {self._yaw_confidence:.2f}  "
            f"xy match {self._translation_confidence:.2f}"
        )
        cv2.rectangle(panel, (0, size - 56), (size, size), (14, 22, 31), -1)
        cv2.putText(panel, label, (12, size - 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (235, 245, 250), 1, cv2.LINE_AA)
        cv2.putText(panel, detail, (12, size - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (190, 215, 230), 1, cv2.LINE_AA)
        return panel
