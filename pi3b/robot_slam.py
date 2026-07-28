#!/usr/bin/env python3
"""Scan-matching SLAM and frontier exploration for the LD19 robot.

The chassis has no wheel encoders and no IMU, so pose comes from matching each
LiDAR revolution against the map built from previous revolutions.  That is real
simultaneous localization and mapping, but a deliberately small one: there is
no loop closure and no pose-graph optimization, so error accumulates slowly and
is never corrected by revisiting a place.  Treat the map as a good local sketch
of the current room, not a survey.

Matching uses a chamfer/distance-transform score rather than ICP with nearest
neighbour search, because ``cv2.distanceTransform`` does the expensive part in
C and the Pi 3B has no scipy.  A coarse-to-fine search over a small pose window
then costs only a few vectorised array lookups.

Safety boundary, deliberately hard: **nothing in this module may influence
obstacle avoidance.**  A diverged pose must never be able to shorten or extend
a travel limit.  SLAM only biases which way the robot prefers to explore, and
draws the map.  A wrong pose should make the robot wander badly, never drive
into furniture.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SlamConfig:
    resolution_m: float = 0.04
    # 160 cells at 4 cm is a 6.4 m window: big enough for an indoor room, and
    # the distance transform cost scales with its area on a Pi 3B.
    cells: int = 160
    # Log-odds increments.  Free space is added more timidly than occupancy so
    # that a single bad pose cannot erase a wall.
    hit_log_odds: float = 0.90
    miss_log_odds: float = 0.18
    log_odds_limit: float = 6.0
    # A single hit must be enough to make a cell occupied.  With a threshold
    # above hit_log_odds the distance field stays empty on the first sweeps,
    # every candidate pose scores the same clamped miss, and matching reports
    # divergence before the map has had a chance to exist.
    occupied_threshold: float = 0.45
    free_threshold: float = -0.5
    max_match_points: int = 150
    max_range_m: float = 3.6
    # Mean chamfer residual above this means the match did not explain the scan.
    divergence_m: float = 0.13
    recovery_ticks: int = 6
    # Integrate a few sweeps at the predicted pose before trusting any match:
    # there is nothing to match against until the map has structure.
    bootstrap_updates: int = 4
    # Sustained divergence means the pose is lost.  Continuing to dead-reckon
    # from it just walks the estimate out of the room, so start a fresh map.
    max_lost_ticks: int = 14


@dataclass
class SlamState:
    x: float
    y: float
    heading_deg: float
    residual_m: float
    trusted: bool
    matched_points: int
    updates: int


class OccupancyMap:
    """Log-odds occupancy grid that scrolls to keep the robot near its centre."""

    FIELD_REFRESH_UPDATES = 3

    def __init__(self, config: SlamConfig) -> None:
        self.config = config
        self.grid = np.zeros((config.cells, config.cells), dtype=np.float32)
        # World coordinate of grid cell (0, 0).
        self.origin_x = -config.cells * config.resolution_m / 2.0
        self.origin_y = -config.cells * config.resolution_m / 2.0
        self._distance_cache: np.ndarray | None = None
        self._cache_dirty = True
        self._since_field = 0

    def world_to_cell(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = self.config.resolution_m
        cols = ((xs - self.origin_x) / res).astype(np.int32)
        rows = ((ys - self.origin_y) / res).astype(np.int32)
        return rows, cols

    def recentre(self, x: float, y: float) -> None:
        """Scroll the grid when the robot drifts away from the middle.

        Rolling loses the far edge rather than growing memory without bound,
        which suits a robot that only ever needs the room it is currently in.
        """
        cells, res = self.config.cells, self.config.resolution_m
        centre = cells // 2
        rows, cols = self.world_to_cell(np.array([x]), np.array([y]))
        shift_row = centre - int(rows[0])
        shift_col = centre - int(cols[0])
        if abs(shift_row) < cells * 0.18 and abs(shift_col) < cells * 0.18:
            return
        self.grid = np.roll(self.grid, (shift_row, shift_col), axis=(0, 1))
        if shift_row > 0:
            self.grid[:shift_row, :] = 0.0
        elif shift_row < 0:
            self.grid[shift_row:, :] = 0.0
        if shift_col > 0:
            self.grid[:, :shift_col] = 0.0
        elif shift_col < 0:
            self.grid[:, shift_col:] = 0.0
        self.origin_x -= shift_col * res
        self.origin_y -= shift_row * res
        self._cache_dirty = True

    def distance_field(self) -> np.ndarray:
        """Distance in metres from each cell to the nearest occupied cell.

        The transform is the most expensive thing SLAM does, and a room does not
        change shape between sweeps, so it is refreshed every few integrations
        rather than every one.
        """
        self._since_field += 1
        if self._distance_cache is not None and self._since_field < self.FIELD_REFRESH_UPDATES:
            return self._distance_cache
        self._since_field = 0
        if self._distance_cache is None or self._cache_dirty:
            occupied = (self.grid >= self.config.occupied_threshold).astype(np.uint8)
            if not np.any(occupied):
                self._distance_cache = np.full(self.grid.shape, 9.0, dtype=np.float32)
            else:
                field = cv2.distanceTransform(1 - occupied, cv2.DIST_L2, 3)
                self._distance_cache = (field * self.config.resolution_m).astype(np.float32)
            self._cache_dirty = False
        return self._distance_cache

    def integrate(self, pose_x: float, pose_y: float, world_x: np.ndarray, world_y: np.ndarray) -> None:
        """Mark hits occupied and the space along each beam free."""
        config = self.config
        cells = config.cells
        rows, cols = self.world_to_cell(world_x, world_y)
        inside = (rows >= 0) & (rows < cells) & (cols >= 0) & (cols < cells)
        if not np.any(inside):
            return

        # Free space: sample along each beam and stop short of the endpoint so
        # the surface itself is not repeatedly cleared.
        steps = 32
        fractions = np.linspace(0.0, 0.92, steps, dtype=np.float32)[None, :]
        free_x = pose_x + (world_x[inside][:, None] - pose_x) * fractions
        free_y = pose_y + (world_y[inside][:, None] - pose_y) * fractions
        free_rows, free_cols = self.world_to_cell(free_x.ravel(), free_y.ravel())
        valid = (free_rows >= 0) & (free_rows < cells) & (free_cols >= 0) & (free_cols < cells)
        np.add.at(self.grid, (free_rows[valid], free_cols[valid]), -config.miss_log_odds)
        np.add.at(self.grid, (rows[inside], cols[inside]), config.hit_log_odds)
        np.clip(self.grid, -config.log_odds_limit, config.log_odds_limit, out=self.grid)
        self._cache_dirty = True

    def frontier_bearing(self, x: float, y: float, heading_deg: float) -> tuple[float, float]:
        """Direction of the nearest useful unexplored boundary.

        Returns (bearing relative to the robot in degrees, weight 0..1).  A
        frontier is a known-free cell touching unknown space: driving toward it
        is what turns aimless wandering into coverage.
        """
        config = self.config
        known_free = (self.grid <= config.free_threshold).astype(np.uint8)
        unknown = (np.abs(self.grid) < 0.2).astype(np.uint8)
        if not np.any(known_free) or not np.any(unknown):
            return 0.0, 0.0
        neighbours = cv2.dilate(unknown, np.ones((3, 3), np.uint8), iterations=1)
        frontier = known_free & neighbours
        rows, cols = np.nonzero(frontier)
        if rows.size == 0:
            return 0.0, 0.0
        res = config.resolution_m
        world_x = self.origin_x + (cols + 0.5) * res
        world_y = self.origin_y + (rows + 0.5) * res
        dx, dy = world_x - x, world_y - y
        distance = np.hypot(dx, dy)
        usable = (distance > 0.35) & (distance < 4.0)
        if not np.any(usable):
            return 0.0, 0.0
        dx, dy, distance = dx[usable], dy[usable], distance[usable]
        # Nearer frontiers matter more, and the sum over a cluster of cells
        # naturally outweighs an isolated speckle.
        weights = 1.0 / (distance + 0.5)
        world_bearing = np.degrees(np.arctan2(dx, dy))
        relative = (world_bearing - heading_deg + 180.0) % 360.0 - 180.0
        radians = np.radians(relative)
        mean_x = float(np.sum(np.cos(radians) * weights))
        mean_y = float(np.sum(np.sin(radians) * weights))
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0, 0.0
        bearing = math.degrees(math.atan2(mean_y, mean_x))
        # Spread-out frontiers cancel and produce a low weight, which is the
        # honest answer: there is no single direction worth preferring.
        coherence = math.hypot(mean_x, mean_y) / total
        return bearing, float(np.clip(coherence, 0.0, 1.0))

    def render(self, x: float, y: float, heading_deg: float, size: int = 460,
               trail: list[tuple[float, float]] | None = None) -> np.ndarray:
        occupied = np.clip(self.grid / self.config.occupied_threshold, 0.0, 1.0)
        free = np.clip(-self.grid / abs(self.config.free_threshold), 0.0, 1.0)
        panel = np.zeros((self.config.cells, self.config.cells, 3), dtype=np.uint8)
        panel[..., 0] = (44 + free * 46).astype(np.uint8)
        panel[..., 1] = (40 + free * 44 + occupied * 150).astype(np.uint8)
        panel[..., 2] = (36 + occupied * 210).astype(np.uint8)
        panel = cv2.flip(panel, 0)
        panel = cv2.resize(panel, (size, size), interpolation=cv2.INTER_NEAREST)

        scale = size / (self.config.cells * self.config.resolution_m)

        def to_pixel(wx: float, wy: float) -> tuple[int, int]:
            px = int((wx - self.origin_x) * scale)
            py = size - 1 - int((wy - self.origin_y) * scale)
            return int(np.clip(px, 0, size - 1)), int(np.clip(py, 0, size - 1))

        if trail:
            for index in range(1, len(trail)):
                cv2.line(panel, to_pixel(*trail[index - 1]), to_pixel(*trail[index]),
                         (120, 165, 95), 1, cv2.LINE_AA)
        px, py = to_pixel(x, y)
        radians = math.radians(heading_deg)
        tip = (int(px + math.sin(radians) * 22), int(py - math.cos(radians) * 22))
        cv2.circle(panel, (px, py), 6, (80, 240, 100), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, (px, py), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        return panel


def points_to_robot_frame(bearings_deg: np.ndarray, ranges_m: np.ndarray) -> np.ndarray:
    """(N,) polar in the robot frame -> (N,2) cartesian, x right, y forward."""
    radians = np.radians(bearings_deg)
    return np.stack((ranges_m * np.sin(radians), ranges_m * np.cos(radians)), axis=1).astype(np.float32)


def deskew(points: np.ndarray, ages_s: np.ndarray, yaw_rate_dps: float,
           forward_speed_ms: float) -> np.ndarray:
    """Undo the robot's own motion during one revolution.

    The LD19 sweeps for ~100 ms, so at 100 deg/s the start and end of a single
    revolution are nearly 10 degrees apart in the robot frame.  Matching skewed
    scans against a map biases every pose estimate during a turn.
    """
    if points.shape[0] == 0:
        return points
    # A return captured `age` ago was taken before the robot rotated by
    # yaw_rate*age, so in the current frame its bearing is that much lower.
    # Bearings run from forward toward the right, which is the opposite
    # handedness to the textbook rotation matrix.
    angle = np.radians(-yaw_rate_dps * ages_s).astype(np.float32)
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = points[:, 0], points[:, 1]
    rotated_x = x * cos + y * sin
    rotated_y = -x * sin + y * cos
    return np.stack((rotated_x, rotated_y - forward_speed_ms * ages_s), axis=1).astype(np.float32)


class ScanMatcher:
    """Coarse-to-fine chamfer alignment of a scan against the occupancy map."""

    COARSE_TRANSLATION = (-0.12, -0.06, 0.0, 0.06, 0.12)
    COARSE_ROTATION = (-6.0, -3.0, 0.0, 3.0, 6.0)
    FINE_TRANSLATION = (-0.03, -0.015, 0.0, 0.015, 0.03)
    FINE_ROTATION = (-1.5, -0.75, 0.0, 0.75, 1.5)

    def __init__(self, config: SlamConfig) -> None:
        self.config = config

    def _score(self, field: np.ndarray, occupancy: OccupancyMap, points: np.ndarray,
               poses: np.ndarray) -> np.ndarray:
        """Mean clamped distance-to-nearest-obstacle for every candidate pose."""
        radians = np.radians(poses[:, 2])
        cos, sin = np.cos(radians)[:, None], np.sin(radians)[:, None]
        a, b = points[None, :, 0], points[None, :, 1]
        world_x = poses[:, 0][:, None] + a * cos + b * sin
        world_y = poses[:, 1][:, None] - a * sin + b * cos
        rows, cols = occupancy.world_to_cell(world_x, world_y)
        cells = self.config.cells
        valid = (rows >= 0) & (rows < cells) & (cols >= 0) & (cols < cells)
        rows = np.clip(rows, 0, cells - 1)
        cols = np.clip(cols, 0, cells - 1)
        # Returns that land outside the map window say nothing about the fit, so
        # they are excluded rather than counted as misses.  Without this, any
        # room larger than the map scores a permanently high residual and the
        # tracker declares divergence in a room it is tracking perfectly well.
        # Poses that push most of the scan off the map are still rejected, so
        # "see less of the map" cannot become a way to win.
        clamped = np.minimum(np.where(valid, field[rows, cols], 0.0), 0.5)
        counted = valid.sum(axis=1)
        total = np.sum(np.where(valid, clamped, 0.0), axis=1)
        enough = counted >= max(8, int(points.shape[0] * 0.5))
        return np.where(enough, total / np.maximum(counted, 1), 1.0)

    def match(self, occupancy: OccupancyMap, points: np.ndarray,
              guess: tuple[float, float, float]) -> tuple[tuple[float, float, float], float]:
        field = occupancy.distance_field()
        best = np.array(guess, dtype=np.float64)
        best_cost = float("inf")
        for translations, rotations in ((self.COARSE_TRANSLATION, self.COARSE_ROTATION),
                                        (self.FINE_TRANSLATION, self.FINE_ROTATION)):
            offsets = np.array(
                [(dx, dy, dt) for dx in translations for dy in translations for dt in rotations],
                dtype=np.float64,
            )
            poses = offsets + best
            costs = self._score(field, occupancy, points, poses)
            index = int(np.argmin(costs))
            best, best_cost = poses[index], float(costs[index])
        return (float(best[0]), float(best[1]), float(best[2] % 360.0)), best_cost


class SlamTracker:
    """Owns the pose, the map, and the decision to stop trusting either."""

    def __init__(self, config: SlamConfig | None = None) -> None:
        self.config = config or SlamConfig()
        self.map = OccupancyMap(self.config)
        self.matcher = ScanMatcher(self.config)
        self.x = 0.0
        self.y = 0.0
        self.heading_deg = 0.0
        self.residual_m = 0.0
        self.trusted = False
        self.updates = 0
        self.matched_points = 0
        self.trail: list[tuple[float, float]] = []
        self._good_ticks = 0
        self._lost_ticks = 0
        self.restarts = 0
        self._lock = threading.Lock()
        self.frontier_bearing_deg = 0.0
        self.frontier_weight = 0.0

    def _restart(self, pose: tuple[float, float, float]) -> None:
        """Abandon a map the robot can no longer locate itself in.

        Everything built so far was referenced to a pose that has since drifted,
        so keeping it would keep poisoning the match.  A fresh map around the
        current position recovers within a few sweeps.
        """
        with self._lock:
            self.map = OccupancyMap(self.config)
            self.x, self.y, self.heading_deg = pose
            self.updates = 0
            self.trusted = False
            self.frontier_weight = 0.0
            self.residual_m = 0.0
            self.trail.clear()
            self._good_ticks = 0
            self._lost_ticks = 0
            self.restarts += 1

    def state(self) -> SlamState:
        with self._lock:
            return SlamState(self.x, self.y, self.heading_deg, self.residual_m,
                             self.trusted, self.matched_points, self.updates)

    def frontier(self) -> tuple[float, float]:
        with self._lock:
            if not self.trusted:
                return 0.0, 0.0
            return self.frontier_bearing_deg, self.frontier_weight

    def update(self, bearings_deg: np.ndarray, ranges_m: np.ndarray, ages_s: np.ndarray,
               yaw_rate_dps: float, forward_speed_ms: float, elapsed_s: float) -> None:
        config = self.config
        usable = np.isfinite(ranges_m) & (ranges_m > 0.10) & (ranges_m <= config.max_range_m)
        if int(np.count_nonzero(usable)) < 40:
            return
        bearings, ranges, ages = bearings_deg[usable], ranges_m[usable], ages_s[usable]
        if bearings.size > config.max_match_points:
            pick = np.linspace(0, bearings.size - 1, config.max_match_points).astype(np.int32)
            bearings, ranges, ages = bearings[pick], ranges[pick], ages[pick]
        points = deskew(points_to_robot_frame(bearings, ranges), ages, yaw_rate_dps, forward_speed_ms)

        # Prediction: commanded speed and measured yaw are only a starting
        # guess.  They are wrong exactly when the battery sags, which is why
        # the search window around them is generous.
        predicted_heading = self.heading_deg + yaw_rate_dps * elapsed_s
        travel = forward_speed_ms * elapsed_s
        radians = math.radians(predicted_heading)
        guess = (self.x + math.sin(radians) * travel, self.y + math.cos(radians) * travel,
                 predicted_heading)

        bootstrapping = self.updates < config.bootstrap_updates
        if bootstrapping:
            pose, residual = guess, 0.0
        else:
            pose, residual = self.matcher.match(self.map, points, guess)

        diverged = (not bootstrapping) and residual > config.divergence_m
        if diverged:
            self._good_ticks = 0
            self._lost_ticks += 1
        else:
            self._lost_ticks = 0
            self._good_ticks = min(config.recovery_ticks, self._good_ticks + 1)

        if self._lost_ticks >= config.max_lost_ticks:
            self._restart(guess)
            return

        with self._lock:
            self.residual_m = residual
            self.matched_points = int(points.shape[0])
            if diverged:
                # Keep the predicted pose so the display still moves, but do not
                # corrupt the map with a fit we could not verify.
                self.x, self.y, self.heading_deg = guess
                self.trusted = False
                self.frontier_weight = 0.0
                return
            self.x, self.y, self.heading_deg = pose
            self.trusted = (not bootstrapping) and self._good_ticks >= config.recovery_ticks

        angle = math.radians(pose[2])
        cos, sin = math.cos(angle), math.sin(angle)
        world_x = pose[0] + points[:, 0] * cos + points[:, 1] * sin
        world_y = pose[1] - points[:, 0] * sin + points[:, 1] * cos
        self.map.recentre(pose[0], pose[1])
        self.map.integrate(pose[0], pose[1], world_x, world_y)
        self.updates += 1

        if not self.trail or math.hypot(pose[0] - self.trail[-1][0], pose[1] - self.trail[-1][1]) > 0.06:
            self.trail.append((pose[0], pose[1]))
            del self.trail[:-400]
        if self.updates % 4 == 0:
            bearing, weight = self.map.frontier_bearing(pose[0], pose[1], pose[2])
            with self._lock:
                self.frontier_bearing_deg = bearing
                self.frontier_weight = weight if self.trusted else 0.0


class SlamWorker:
    """Runs SLAM off the planning thread and drops ticks rather than blocking."""

    def __init__(self, tracker: SlamTracker, period_s: float = 0.22) -> None:
        self.tracker = tracker
        self.period_s = period_s
        self._pending: tuple | None = None
        self._condition = threading.Condition()
        self._stop = False
        self.last_duration_ms = 0.0
        self._thread = threading.Thread(target=self._run, name="robot-slam", daemon=True)
        self._thread.start()

    def submit(self, bearings, ranges, ages, yaw_rate_dps, forward_speed_ms, elapsed_s) -> None:
        with self._condition:
            self._pending = (bearings, ranges, ages, yaw_rate_dps, forward_speed_ms, elapsed_s)
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait(0.5)
                if self._stop:
                    return
                work = self._pending
                self._pending = None
            started = time.monotonic()
            try:
                self.tracker.update(*work)
            except Exception:
                # SLAM is advisory.  It must never take the robot down with it.
                pass
            self.last_duration_ms = (time.monotonic() - started) * 1000.0
            time.sleep(max(0.0, self.period_s - (time.monotonic() - started)))

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify()
        self._thread.join(timeout=1.5)
