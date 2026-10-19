"""Moving-object detection and tracking from the LD19 scan.

Until now every LD19 return was treated as part of a still world. A person
walking past was a wall that happened to be in a different place each scan:
the arc planner routed around where they *were*, the obstacle memory left a
trail where they had *been*, and nothing anticipated where they were *going*.

This follows the established pattern for 2D LiDAR people and object tracking:
segment the scan into compact clusters, associate clusters to tracks by
nearest neighbour, and estimate each track's velocity with a light filter
(here an alpha-beta filter, the constant-gain form of a constant-velocity
Kalman filter). Moving tracks are then predicted forward under the
constant-velocity assumption, which is how predictive extensions of the
Dynamic Window Approach treat moving obstacles, and those predictions are
handed to the arc planner as obstacles at the positions they will occupy.

Tracking happens in the map frame, so the robot's own motion is removed before
velocities are estimated. That is also the honest limit of this module: the
chassis has no wheel encoders, so its own pose is dead-reckoned and corrected
by scan matching. Errors in that estimate make still objects appear to drift.
The thresholds below - a minimum speed, a minimum distance actually travelled,
several consecutive sightings, and no classification while the chassis is
spinning quickly - exist to keep that drift from being reported as motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Segmentation.
CLUSTER_MIN_POINTS = 3
CLUSTER_MAX_EXTENT_M = 0.80
CLUSTER_MIN_GAP_M = 0.10
CLUSTER_GAP_PER_METRE = 0.055
TRACK_MAX_RANGE_M = 5.0

# Association and filtering.
ASSOCIATION_GATE_M = 0.35
ALPHA = 0.55
BETA = 0.25
MAX_SPEED_MPS = 2.5
TRACK_TIMEOUT_S = 0.55

# Classification.
MOVING_MIN_HITS = 4
MOVING_MIN_SPEED_MPS = 0.20
MOVING_RELEASE_SPEED_MPS = 0.10
MOVING_MIN_TRAVEL_M = 0.30
MAX_CLASSIFY_YAW_RATE_DPS = 40.0

# Prediction handed to the planner.
PREDICTION_TIMES_S = (0.4, 0.8, 1.2, 1.6)
PREDICTION_RING_POINTS = 8
PREDICTION_PADDING_M = 0.05


@dataclass(frozen=True)
class TrackedObject:
    """A tracked cluster. Map-frame position and velocity, plus robot frame."""

    track_id: int
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    radius_m: float
    moving: bool
    hits: int
    robot_x_m: float = 0.0
    robot_y_m: float = 0.0
    robot_vx_mps: float = 0.0
    robot_vy_mps: float = 0.0

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx_mps, self.vy_mps)


@dataclass
class _Track:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    first_x: float
    first_y: float
    last_seen: float
    hits: int
    moving: bool = False


def to_robot_frame(
    dx: np.ndarray | float, dy: np.ndarray | float, heading_deg: float
):
    """Map-frame offset -> robot frame (+x right, +y ahead)."""
    heading = math.radians(heading_deg)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return dx * cos_h + dy * sin_h, dx * sin_h - dy * cos_h


def to_map_frame(
    rx: np.ndarray | float, ry: np.ndarray | float, heading_deg: float
):
    """Robot-frame offset -> map frame."""
    heading = math.radians(heading_deg)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return rx * cos_h + ry * sin_h, rx * sin_h - ry * cos_h


def segment_scan(
    angles_deg: np.ndarray, ranges_m: np.ndarray
) -> list[tuple[float, float, float]]:
    """Split one revolution into compact clusters: (x, y, radius), robot frame.

    Consecutive returns along the scan belong to one surface unless the gap
    between them is larger than the LD19's own point spacing would produce at
    that range. Clusters wider than CLUSTER_MAX_EXTENT_M are walls, furniture
    runs and the like - still scenery rather than candidates for motion.
    """
    usable = (ranges_m >= 0.10) & (ranges_m <= TRACK_MAX_RANGE_M)
    if np.count_nonzero(usable) < CLUSTER_MIN_POINTS:
        return []
    order = np.argsort(angles_deg[usable])
    radians = np.radians(angles_deg[usable][order])
    ranges = ranges_m[usable][order]
    x = np.sin(radians) * ranges
    y = np.cos(radians) * ranges
    step = np.hypot(np.diff(x), np.diff(y))
    allowed = np.maximum(
        CLUSTER_MIN_GAP_M, np.minimum(ranges[:-1], ranges[1:]) * CLUSTER_GAP_PER_METRE
    )
    breaks = np.flatnonzero(step > allowed) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [x.size]))
    clusters: list[tuple[float, float, float]] = []
    for start, end in zip(starts, ends):
        if end - start < CLUSTER_MIN_POINTS:
            continue
        cx = x[start:end]
        cy = y[start:end]
        extent = math.hypot(float(cx.max() - cx.min()), float(cy.max() - cy.min()))
        if extent > CLUSTER_MAX_EXTENT_M:
            continue
        clusters.append(
            (float(cx.mean()), float(cy.mean()), max(0.06, extent / 2.0))
        )
    return clusters


class MovingObjectTracker:
    """Nearest-neighbour association with an alpha-beta velocity filter."""

    def __init__(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_stamp: float | None = None
        self.objects: tuple[TrackedObject, ...] = ()

    def reset(self) -> None:
        self._tracks.clear()
        self._last_stamp = None
        self.objects = ()

    def update(
        self,
        angles_deg: np.ndarray,
        ranges_m: np.ndarray,
        stamp: float,
        pose_x_m: float,
        pose_y_m: float,
        heading_deg: float,
        yaw_rate_dps: float | None = None,
    ) -> tuple[TrackedObject, ...]:
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return self.objects
        dt = 0.1 if self._last_stamp is None else min(0.5, stamp - self._last_stamp)
        self._last_stamp = stamp
        spinning = yaw_rate_dps is not None and abs(yaw_rate_dps) > MAX_CLASSIFY_YAW_RATE_DPS

        detections = []
        for rx, ry, radius in segment_scan(angles_deg, ranges_m):
            mx, my = to_map_frame(rx, ry, heading_deg)
            detections.append((pose_x_m + mx, pose_y_m + my, radius))

        # Predict, then associate greedily nearest-first within a gate that
        # widens with each track's own speed.
        for track in self._tracks:
            track.x += track.vx * dt
            track.y += track.vy * dt
        pairs = []
        for track_index, track in enumerate(self._tracks):
            gate = ASSOCIATION_GATE_M + math.hypot(track.vx, track.vy) * dt
            for detection_index, (dx, dy, _radius) in enumerate(detections):
                distance = math.hypot(dx - track.x, dy - track.y)
                if distance <= gate:
                    pairs.append((distance, track_index, detection_index))
        pairs.sort()
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _distance, track_index, detection_index in pairs:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            track = self._tracks[track_index]
            dx, dy, radius = detections[detection_index]
            residual_x = dx - track.x
            residual_y = dy - track.y
            track.x += ALPHA * residual_x
            track.y += ALPHA * residual_y
            if dt > 1e-3:
                track.vx += BETA * residual_x / dt
                track.vy += BETA * residual_y / dt
            speed = math.hypot(track.vx, track.vy)
            if speed > MAX_SPEED_MPS:
                track.vx *= MAX_SPEED_MPS / speed
                track.vy *= MAX_SPEED_MPS / speed
                speed = MAX_SPEED_MPS
            track.radius = track.radius * 0.7 + radius * 0.3
            track.last_seen = stamp
            track.hits += 1
            travelled = math.hypot(track.x - track.first_x, track.y - track.first_y)
            if track.moving:
                track.moving = speed >= MOVING_RELEASE_SPEED_MPS
            elif not spinning:
                track.moving = (
                    track.hits >= MOVING_MIN_HITS
                    and speed >= MOVING_MIN_SPEED_MPS
                    and travelled >= MOVING_MIN_TRAVEL_M
                )
        for detection_index, (dx, dy, radius) in enumerate(detections):
            if detection_index in used_detections:
                continue
            self._tracks.append(
                _Track(
                    track_id=self._next_id,
                    x=dx,
                    y=dy,
                    vx=0.0,
                    vy=0.0,
                    radius=radius,
                    first_x=dx,
                    first_y=dy,
                    last_seen=stamp,
                    hits=1,
                )
            )
            self._next_id += 1
        self._tracks = [
            track for track in self._tracks if stamp - track.last_seen <= TRACK_TIMEOUT_S
        ]
        objects = []
        for track in self._tracks:
            rx, ry = to_robot_frame(track.x - pose_x_m, track.y - pose_y_m, heading_deg)
            rvx, rvy = to_robot_frame(track.vx, track.vy, heading_deg)
            objects.append(
                TrackedObject(
                    track_id=track.track_id,
                    x_m=track.x,
                    y_m=track.y,
                    vx_mps=track.vx,
                    vy_mps=track.vy,
                    radius_m=track.radius,
                    moving=track.moving,
                    hits=track.hits,
                    robot_x_m=float(rx),
                    robot_y_m=float(ry),
                    robot_vx_mps=float(rvx),
                    robot_vy_mps=float(rvy),
                )
            )
        self.objects = tuple(objects)
        return self.objects


def predicted_obstacles(
    objects: tuple[TrackedObject, ...],
    max_range_m: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Robot-frame points where moving objects are expected to be.

    Each moving object is drawn as a small ring at several future times, so an
    arc that would meet it later is scored against where it will be rather
    than where it was when the scan was taken. Stationary tracks add nothing:
    the scan and the memory already contain them.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ring = np.linspace(0.0, 2.0 * math.pi, PREDICTION_RING_POINTS, endpoint=False)
    for item in objects:
        if not item.moving or math.hypot(item.robot_x_m, item.robot_y_m) > max_range_m:
            continue
        radius = item.radius_m + PREDICTION_PADDING_M
        for seconds in PREDICTION_TIMES_S:
            cx = item.robot_x_m + item.robot_vx_mps * seconds
            cy = item.robot_y_m + item.robot_vy_mps * seconds
            xs.append((cx + np.cos(ring) * radius).astype(np.float32))
            ys.append((cy + np.sin(ring) * radius).astype(np.float32))
    if not xs:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    return np.concatenate(xs), np.concatenate(ys)


def closest_approach(
    item: TrackedObject, robot_speed_mps: float
) -> tuple[float, float]:
    """Time and distance of closest approach, both moving at constant velocity.

    The robot is taken to continue straight ahead (+y) at its current speed.
    Returns (seconds, metres); seconds is 0 when they are already separating.
    """
    px = item.robot_x_m
    py = item.robot_y_m
    vx = item.robot_vx_mps
    vy = item.robot_vy_mps - robot_speed_mps
    relative_sq = vx * vx + vy * vy
    if relative_sq < 1e-6:
        return 0.0, math.hypot(px, py)
    seconds = max(0.0, -(px * vx + py * vy) / relative_sq)
    return seconds, math.hypot(px + vx * seconds, py + vy * seconds)
