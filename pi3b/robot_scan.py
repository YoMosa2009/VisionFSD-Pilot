"""Array form of the live LD19 scan, with the manufacturer's edge filter.

Two things motivate this module.

**Cost.** Every consumer of the scan - the arc planner's obstacle memory, the
whole-scan motion detector, the moving-object tracker, the web visualiser -
used to walk the list of point objects again on every control tick, whether or
not the LD19 had produced anything new. ``ScanFrame`` is built once per new
batch of packets and handed to all of them as plain arrays.

**Mixed pixels.** A time-of-flight beam that clips an edge returns a range
somewhere between the edge and whatever is behind it. Those phantom returns
float in doorways and beside furniture legs, which is exactly where the
planner is deciding whether a gap is passable. LDROBOT's own SDK removes them
for the LD06/LD19 with a near-field filter (``Tofbf``, NEAR_FILTER mode); this
is a direct port of that filter's grouping and intensity rules, vectorised.

The filter is not allowed to weaken near-field safety. It can drop a real thin,
dark object, so every return inside ``NEAR_KEEP_M`` is kept regardless of what
the filter says. Close to the chassis a phantom obstacle only costs caution; a
missed real one costs a collision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# LDROBOT Tofbf NEAR_FILTER constants for the LD06/LD19.
MEASURE_FREQUENCY_HZ = 4500.0
FILTER_MAX_RANGE_MM = 5000.0
GROUP_RANGE_JUMP_RATIO = 0.03
LARGE_GROUP_POINTS = 15
SMALL_GROUP_POINTS = 3
GROUP_MIN_MEAN_INTENSITY = 15.0
LONE_POINT_MIN_INTENSITY = 220.0
# Returns this close are never filtered. See the module docstring.
NEAR_KEEP_M = 0.60
DEFAULT_SPEED_DPS = 3600.0


@dataclass(frozen=True)
class ScanFrame:
    """One view of the current scan window as arrays.

    ``keep`` marks returns that survive the mixed-pixel filter (plus every
    near-field return). Arrays are never mutated after construction, so a
    frame can be shared across threads.
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
    """Return True for returns that are real surfaces, per LDROBOT's filter.

    Returns within 5 m are grouped along the scan: a new group starts when the
    angular gap exceeds two nominal sample spacings or the range jumps by more
    than 3%. Groups of more than 15 returns are real surfaces. Groups of 3-15
    need a mean intensity of at least 15. Groups of fewer than 3 are the
    classic mixed-pixel signature and each return needs intensity 220.
    """
    count = angles_deg.size
    keep = np.ones(count, dtype=bool)
    if count == 0:
        return keep
    speed = speed_dps if speed_dps > 0.0 else DEFAULT_SPEED_DPS
    max_gap_deg = speed / MEASURE_FREQUENCY_HZ * 2.0
    candidate = ranges_mm < FILTER_MAX_RANGE_MM
    indices = np.flatnonzero(candidate)
    if indices.size == 0:
        return keep
    order = indices[np.argsort(angles_deg[indices], kind="stable")]
    angles = angles_deg[order].astype(np.float64)
    ranges = ranges_mm[order].astype(np.float64)
    power = intensity[order].astype(np.float64)

    breaks = np.zeros(order.size, dtype=bool)
    if order.size > 1:
        breaks[1:] = (np.diff(angles) > max_gap_deg) | (
            np.abs(np.diff(ranges)) > ranges[:-1] * GROUP_RANGE_JUMP_RATIO
        )
    group = np.cumsum(breaks)
    # A surface that straddles 0 degrees is one group, not two.
    if order.size > 1 and group[-1] > 0:
        wrap_gap = angles[0] + 360.0 - angles[-1]
        if (
            wrap_gap <= max_gap_deg
            and abs(ranges[0] - ranges[-1]) <= ranges[-1] * GROUP_RANGE_JUMP_RATIO
        ):
            group[group == group[-1]] = 0
    sizes = np.bincount(group)
    mean_power = np.bincount(group, weights=power) / np.maximum(sizes, 1)
    point_size = sizes[group]
    point_mean = mean_power[group]
    survives = np.where(
        point_size > LARGE_GROUP_POINTS,
        True,
        np.where(
            point_size >= SMALL_GROUP_POINTS,
            point_mean >= GROUP_MIN_MEAN_INTENSITY,
            power >= LONE_POINT_MIN_INTENSITY,
        ),
    )
    keep[order] = survives
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
