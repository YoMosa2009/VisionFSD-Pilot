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
    Vectorized scan integration keeps this inexpensive beside camera inference, while
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

    def __init__(self, cells: int = 384, metres: float = 8.0) -> None:
        self.cells = cells
        self.metres = metres
        self.grid = np.zeros((cells, cells), dtype=np.uint8)
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
        stride = max(1, distances.size // 180)
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
        for row_offset in range(-5, 6):
            for col_offset in range(-5, 6):
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
            ray_count = hit_rows.size
            ray_stride = max(1, ray_count // 240)
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
        camera_yaw_rate_dps: float | None = None,
        camera_translation_scale: float = 1.0,
        imu_yaw_deg: float | None = None,
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
            (
                "IMU+LD19"
                if self._using_imu
                else "COMMAND+CAMERA+LD19"
                if self._using_camera
                else "COMMAND+LD19"
            ),
            self._recenter_count,
        )

    def render(
        self,
        size: int = 500,
        target_xy: tuple[float, float] | None = None,
        waypoint_xy: tuple[float, float] | None = None,
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
