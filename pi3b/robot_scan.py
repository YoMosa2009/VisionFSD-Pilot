"""Array form of the live LD19 scan, with an edge-artefact flag for display.

**Cost.** Every consumer of the scan used to walk the list of point objects
again on every control tick, whether or not the LD19 had produced anything new.
``ScanFrame`` is built once per new batch of packets and shared as arrays.

**Edge artefacts are flagged, not removed.** A time-of-flight beam clipping an
edge can return a range between the edge and whatever is behind it. v1.9.22
removed those returns before planning, using a port of LDROBOT's own
NEAR_FILTER, which groups returns by a 3% range jump. That was a serious
mistake: along a wall seen at a glancing angle, consecutive returns
*legitimately* differ in range by more than 3%, so every one of them became a
lone "artefact" and was discarded. In simulation it removed 68-95% of a
wall's returns at 1-2 m and all of them at 2-3 m, for a wall the robot drove
alongside. The planner could not see those walls until they were within
0.6 m - which is how v1.9.22 came to drive quickly at openings that were walls.

Two changes follow from that:

* Planning, memory, tracking and mapping use **every** return again. A
  phantom return in a doorway only makes the planner more cautious; a missing
  real one causes a collision.
* The flag, now only for the phone view, is judged by *spatial isolation*: a
  return is an artefact candidate only when it is far in space from both of
  its angular neighbours and weak. A glancing wall's returns are spread in
  range but close together along the wall, so they are never flagged, while
  a ghost floating between two surfaces still is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MEASURE_FREQUENCY_HZ = 4500.0
DEFAULT_SPEED_DPS = 3600.0
# Only returns this close are candidates for the artefact flag.
FILTER_MAX_RANGE_MM = 5000.0
# A return that is spatially isolated needs at least this signal strength to
# be considered a real thin object rather than a ghost.
LONE_POINT_MIN_INTENSITY = 220.0
# Neighbouring returns on one surface can be spread by up to the sample step
# divided by the cosine of the angle of incidence. Tolerate surfaces seen up to
# this glancing angle before calling two returns separated.
GLANCING_LIMIT_DEG = 80.0
MIN_SEPARATION_M = 0.06
# Returns this close are never flagged.
NEAR_KEEP_M = 0.60


@dataclass(frozen=True)
class ScanFrame:
    """One view of the current scan window as arrays.

    ``keep`` is False for returns flagged as likely edge ghosts. It is for
    display only: planning uses every return. Arrays are never mutated after
    construction, so a frame can be shared across threads.
    """

    seq: int
    stamp: float
    angles_deg: np.ndarray
    ranges_m: np.ndarray
    intensity: np.ndarray
    keep: np.ndarray
    speed_dps: float

    @property
    def size(self) -> int:
        return int(self.angles_deg.size)

    def kept(self) -> tuple[np.ndarray, np.ndarray]:
        """Filtered (angles_deg, ranges_m)."""
        return self.angles_deg[self.keep], self.ranges_m[self.keep]

    def cartesian(self, kept_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Robot-frame (x, y) metres: +x right, +y ahead."""
        angles = self.angles_deg[self.keep] if kept_only else self.angles_deg
        ranges = self.ranges_m[self.keep] if kept_only else self.ranges_m
        radians = np.radians(angles)
        return (
            (np.sin(radians) * ranges).astype(np.float32),
            (np.cos(radians) * ranges).astype(np.float32),
        )


EMPTY_FRAME = ScanFrame(
    seq=-1,
    stamp=0.0,
    angles_deg=np.zeros(0, dtype=np.float32),
    ranges_m=np.zeros(0, dtype=np.float32),
    intensity=np.zeros(0, dtype=np.float32),
    keep=np.zeros(0, dtype=bool),
    speed_dps=0.0,
)


def mixed_pixel_keep_mask(
    angles_deg: np.ndarray,
    ranges_mm: np.ndarray,
    intensity: np.ndarray,
    speed_dps: float,
) -> np.ndarray:
    """True for returns that look like real surfaces; False for likely ghosts.

    A return is flagged only when it is weak, within the near field, and far in
    space from *both* of its angular neighbours. Judging separation in space
    rather than by a range jump is what keeps walls seen at glancing angles:
    their consecutive ranges differ a lot, but the points sit close together
    along the wall.
    """
    count = angles_deg.size
    keep = np.ones(count, dtype=bool)
    if count < 3:
        return keep
    order = np.argsort(angles_deg, kind="stable")
    radians = np.radians(angles_deg[order].astype(np.float64))
    ranges = ranges_mm[order].astype(np.float64) / 1000.0
    x = np.sin(radians) * ranges
    y = np.cos(radians) * ranges
    previous = np.roll(np.arange(count), 1)
    following = np.roll(np.arange(count), -1)
    gap_previous = np.hypot(x - x[previous], y - y[previous])
    gap_following = np.hypot(x - x[following], y - y[following])
    speed = speed_dps if speed_dps > 0.0 else DEFAULT_SPEED_DPS
    step = np.radians(speed / MEASURE_FREQUENCY_HZ * 1.5)
    allowed = np.maximum(
        MIN_SEPARATION_M,
        ranges * step / np.cos(np.radians(GLANCING_LIMIT_DEG)),
    )
    isolated = (gap_previous > allowed) & (gap_following > allowed)
    weak = intensity[order] < LONE_POINT_MIN_INTENSITY
    near = ranges_mm[order] < FILTER_MAX_RANGE_MM
    keep[order] = ~(isolated & weak & near)
    return keep


def frame_from_points(
    points,
    seq: int,
    speed_dps: float = 0.0,
    min_range_m: float = 0.02,
    max_range_m: float = 12.0,
) -> ScanFrame:
    """Build a frame from the (index, LidarPoint) list the link publishes."""
    if not points:
        return ScanFrame(
            seq=seq,
            stamp=0.0,
            angles_deg=EMPTY_FRAME.angles_deg,
            ranges_m=EMPTY_FRAME.ranges_m,
            intensity=EMPTY_FRAME.intensity,
            keep=EMPTY_FRAME.keep,
            speed_dps=speed_dps,
        )
    count = len(points)
    angles = np.fromiter(
        (point.angle_deg for _index, point in points), dtype=np.float32, count=count
    )
    ranges_mm = np.fromiter(
        (point.distance_mm for _index, point in points),
        dtype=np.float32,
        count=count,
    )
    intensity = np.fromiter(
        (getattr(point, "confidence", 0) for _index, point in points),
        dtype=np.float32,
        count=count,
    )
    stamps = np.fromiter(
        (point.captured_at for _index, point in points),
        dtype=np.float64,
        count=count,
    )
    ranges_m = ranges_mm / 1000.0
    valid = (ranges_m >= min_range_m) & (ranges_m <= max_range_m)
    angles = angles[valid]
    ranges_mm = ranges_mm[valid]
    ranges_m = ranges_m[valid]
    intensity = intensity[valid]
    keep = mixed_pixel_keep_mask(angles, ranges_mm, intensity, speed_dps)
    keep |= ranges_m <= NEAR_KEEP_M
    return ScanFrame(
        seq=seq,
        stamp=float(stamps.max()) if stamps.size else 0.0,
        angles_deg=angles,
        ranges_m=ranges_m,
        intensity=intensity,
        keep=keep,
        speed_dps=speed_dps,
    )
