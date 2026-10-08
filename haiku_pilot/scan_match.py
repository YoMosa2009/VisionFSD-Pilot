"""How far the robot really turned and moved, from two LiDAR scans.

The robot's gyro is read over a slow USB bridge every 50-100 ms, so a short
or fast turn is measured coarsely, and the wheels slip by different amounts
on every floor and every push. Two scans taken while the robot stands still,
before and after a move, measure the move itself: the room has not moved,
so the transform that lays one scan onto the other is the robot's motion.

Points are in the robot frame as the robot publishes them: x to the right,
y straight ahead, metres. The result is the robot's own motion between the
scans: ``turn_deg`` positive to the LEFT (counter-clockwise), ``forward_m``
and ``right_m`` in the frame the robot had before the move.

Pure numpy; small enough for a few hundred points per scan.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Points nearer than this are the robot's own body or the object it pushes;
# farther than this the scan is too sparse to match well.
MIN_RANGE_M = 0.20
MAX_RANGE_M = 6.0
# A correspondence farther apart than this after alignment is not the same
# surface (a moved object, a person, a new view round a corner).
INLIER_M = 0.06
# Below this inlier fraction the match is not trusted.
MIN_INLIERS = 0.45
MIN_POINTS = 40


@dataclass(frozen=True)
class Motion:
    turn_deg: float      # + = turned left (counter-clockwise)
    forward_m: float     # + = moved forward
    right_m: float       # + = moved to the right
    inliers: float       # fraction of points that agree after alignment
    rms_m: float

    @property
    def trusted(self) -> bool:
        return self.inliers >= MIN_INLIERS


def points_from_scan(scan: dict | None) -> np.ndarray:
    """(N, 2) metres from a telemetry scan ({"x": [cm], "y": [cm]})."""
    if not scan:
        return np.zeros((0, 2))
    xs = np.asarray(scan.get("x") or [], dtype=float) / 100.0
    ys = np.asarray(scan.get("y") or [], dtype=float) / 100.0
    count = min(len(xs), len(ys))
    points = np.stack((xs[:count], ys[:count]), axis=1)
    ranges = np.hypot(points[:, 0], points[:, 1])
    return points[(ranges >= MIN_RANGE_M) & (ranges <= MAX_RANGE_M)]


def _rotation(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, -s], [s, c]])


def _nearest(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each source point: index of and distance to the nearest target point."""
    diff = source[:, None, :] - target[None, :, :]
    squared = np.einsum("ijk,ijk->ij", diff, diff)
    index = np.argmin(squared, axis=1)
    return index, np.sqrt(squared[np.arange(len(source)), index])


def _score(after: np.ndarray, before: np.ndarray, rotation: np.ndarray,
           shift: np.ndarray) -> float:
    moved = after @ rotation.T + shift
    _index, distance = _nearest(moved, before)
    return float(np.mean(np.minimum(distance, 0.25)))


def match(before_scan: dict | None, after_scan: dict | None,
          guess_turn_deg: float = 0.0, search_deg: float = 45.0,
          guess_forward_m: float = 0.0, search_forward_m: float = 0.0,
          guess_right_m: float = 0.0, search_step_deg: float = 1.0) -> Motion | None:
    """The robot's motion between two still scans, or None if unmeasurable.

    ``guess_turn_deg`` (left positive) centres the coarse search - the
    commanded or gyro angle - so a turn is not confused with a symmetric room.
    A drive also searches ``guess_forward_m`` +/- ``search_forward_m``, so a
    long move cannot settle into the wrong wall. ``guess_right_m`` starts
    the search off to the side (re-anchoring against an older scan).
    """
    before = points_from_scan(before_scan)
    after = points_from_scan(after_scan)
    if len(before) < MIN_POINTS or len(after) < MIN_POINTS:
        return None
    # p_before = R(theta) p_after + t, where theta is the robot's own turn.
    # Coarse: a grid of turns (and forward moves) scored on a subsample.
    sample = after[:: max(1, len(after) // 90)]
    forwards = (np.arange(-search_forward_m, search_forward_m + 1e-9, 0.05) + guess_forward_m
                if search_forward_m > 0 else np.array([guess_forward_m]))
    best = None
    for degrees in np.arange(guess_turn_deg - search_deg,
                             guess_turn_deg + search_deg + search_step_deg / 2, search_step_deg):
        rotation = _rotation(float(degrees))
        for forward in forwards:
            shift = np.array([float(guess_right_m), float(forward)])
            score = _score(sample, before, rotation, shift)
            if best is None or score < best[0]:
                best = (score, float(degrees), shift)
    rotation = _rotation(best[1])
    shift = best[2]
    # Fine: trimmed point-to-point ICP.
    for _ in range(30):
        moved = after @ rotation.T + shift
        index, distance = _nearest(moved, before)
        cutoff = max(INLIER_M, float(np.quantile(distance, 0.7)))
        keep = distance <= cutoff
        if keep.sum() < MIN_POINTS // 2:
            break
        source, target = after[keep], before[index[keep]]
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        covariance = (source - source_mean).T @ (target - target_mean)
        u, _s, vt = np.linalg.svd(covariance)
        new_rotation = vt.T @ u.T
        if np.linalg.det(new_rotation) < 0:
            vt[-1, :] *= -1
            new_rotation = vt.T @ u.T
        new_shift = target_mean - source_mean @ new_rotation.T
        converged = (np.abs(new_rotation - rotation).max() < 1e-6
                     and np.abs(new_shift - shift).max() < 1e-5)
        rotation, shift = new_rotation, new_shift
        if converged:
            break
    moved = after @ rotation.T + shift
    _index, distance = _nearest(moved, before)
    inliers = float(np.mean(distance <= INLIER_M))
    rms = float(np.sqrt(np.mean(np.minimum(distance, 0.25) ** 2)))
    turn = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    # The robot's origin after the move, in the before frame, is ``shift``.
    return Motion(turn_deg=round(turn, 1), forward_m=round(float(shift[1]), 3),
                  right_m=round(float(shift[0]), 3), inliers=round(inliers, 2),
                  rms_m=round(rms, 3))
