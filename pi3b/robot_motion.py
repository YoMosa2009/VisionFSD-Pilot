"""Range-signature motion evidence for the Pi robot runtime.

The escape/stuck logic in :mod:`robot_autonomy` needs to know whether the
world is actually changing the way a commanded drive predicts.  The existing
evidence sources each have a blind spot:

* the webcam flow needs texture and reasonable light,
* the IMU can observe rotation but never translation,
* the Uno ultrasonic only reports a single forward cone, and
* tracking one LD19 sector (``front_m``) only works while something happens
  to be within about a metre.

That last gap is the important one.  In the middle of an open room every
tracked sector is far away, so the per-sector progress check returns UNKNOWN
and a genuine wheel stall on a rug can never gather the two independent votes
that a stuck declaration requires.

A whole-scan range signature closes that gap: translating or rotating the
chassis changes many angular bins at once, at any range, while a wedged
chassis reproduces nearly the same scan indefinitely.  The same comparison
also detects the opposite extreme - a scan that changes far more between two
consecutive revolutions than any drivable speed allows, which is what being
picked up and carried looks like.

This is deliberately a *motion detector*, not odometry.  It reports whether
the geometry around the robot is changing, never where the robot is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 5-degree bins keep the signature cheap to build and compare on a Pi 3B while
# still describing the whole revolution.
SIGNATURE_BINS = 72
MIN_VALID_RANGE_M = 0.08
MAX_VALID_RANGE_M = 5.80
# Below this many mutually valid bins the two scans do not describe enough
# shared geometry for their difference to mean anything.
MIN_OVERLAP_BINS = 18
# A verdict needs a window long enough for real travel to exceed LD19 range
# noise. The LD19 turns at roughly 10 Hz, so this spans several revolutions.
MIN_WINDOW_S = 0.55
MAX_WINDOW_S = 1.60
# Median bin change that counts as the robot having genuinely moved during a
# MIN_WINDOW_S window. Well above LD19 range noise, well below the travel of
# even the slowest commanded drive.
MOVED_CHANGE_M = 0.045
# Displacement (being carried) is judged between consecutive scans, where no
# drivable speed can rearrange this much geometry.
DISPLACED_CHANGE_M = 0.35
DISPLACED_BIN_FRACTION = 0.45


@dataclass(frozen=True)
class ScanMotionResult:
    """One comparison of the live scan against recent history."""

    verdict: str = "UNKNOWN"
    change_m: float = 0.0
    window_s: float = 0.0
    overlap_bins: int = 0
    displaced: bool = False
    displacement_m: float = 0.0


def range_signature(points, bins: int = SIGNATURE_BINS) -> np.ndarray:
    """Reduce one LD19 revolution to a fixed-length minimum-range vector.

    Bins without a usable return become NaN so a missing sector is treated as
    "no information", never as a range of zero.
    """
    signature = np.full(bins, np.nan, dtype=np.float32)
    if not points:
        return signature
    angles = np.fromiter(
        (float(point.angle_deg) for _index, point in points),
        dtype=np.float32,
        count=len(points),
    )
    ranges = np.fromiter(
        (float(point.distance_mm) / 1000.0 for _index, point in points),
        dtype=np.float32,
        count=len(points),
    )
    valid = (ranges >= MIN_VALID_RANGE_M) & (ranges <= MAX_VALID_RANGE_M)
    if not np.any(valid):
        return signature
    indices = np.floor(angles[valid] / (360.0 / bins)).astype(np.int32) % bins
    filled = np.full(bins, np.inf, dtype=np.float32)
    np.minimum.at(filled, indices, ranges[valid])
    occupied = np.isfinite(filled)
    signature[occupied] = filled[occupied]
    return signature


def signature_change_m(
    current: np.ndarray, reference: np.ndarray
) -> tuple[float, int, float]:
    """Median absolute bin change, the overlap size, and the changed fraction.

    A median rejects the handful of bins that legitimately appear or vanish at
    a sector edge, so one moving object crossing the scan cannot masquerade as
    the whole chassis moving.
    """
    overlap = np.isfinite(current) & np.isfinite(reference)
    count = int(np.count_nonzero(overlap))
    if count < MIN_OVERLAP_BINS:
        return 0.0, count, 0.0
    differences = np.abs(current[overlap] - reference[overlap])
    changed_fraction = float(
        np.count_nonzero(differences >= DISPLACED_CHANGE_M) / count
    )
    return float(np.median(differences)), count, changed_fraction


class ScanMotionTracker:
    """Turn successive range signatures into MOVING/NOT_MOVING/UNKNOWN.

    Two histories are kept for two different questions.  The immediately
    previous scan answers "was the chassis just displaced?", while a reference
    scan at least ``min_window_s`` old answers "has anything changed while we
    were driving?".
    """

    def __init__(
        self,
        min_window_s: float = MIN_WINDOW_S,
        moved_change_m: float = MOVED_CHANGE_M,
    ) -> None:
        self.min_window_s = min_window_s
        self.moved_change_m = moved_change_m
        self._previous: np.ndarray | None = None
        self._reference: np.ndarray | None = None
        self._reference_at = 0.0
        self._last_scan_at: float | None = None

    def reset(self) -> None:
        self._previous = None
        self._reference = None
        self._reference_at = 0.0
        self._last_scan_at = None

    def update(
        self, signature: np.ndarray | None, scan_at: float | None
    ) -> ScanMotionResult:
        if scan_at is None or signature is None:
            return ScanMotionResult()
        if self._last_scan_at is not None and scan_at <= self._last_scan_at:
            # The control loop runs far faster than the LD19. Re-reading one
            # cached revolution must never count as fresh evidence.
            return ScanMotionResult()
        self._last_scan_at = scan_at

        displaced = False
        displacement_m = 0.0
        if self._previous is not None:
            step_change, step_overlap, changed_fraction = signature_change_m(
                signature, self._previous
            )
            if step_overlap >= MIN_OVERLAP_BINS:
                displacement_m = step_change
                displaced = (
                    step_change >= DISPLACED_CHANGE_M
                    and changed_fraction >= DISPLACED_BIN_FRACTION
                )
        self._previous = signature

        if self._reference is None:
            self._reference = signature
            self._reference_at = scan_at
            return ScanMotionResult(
                displaced=displaced, displacement_m=displacement_m
            )
        window = scan_at - self._reference_at
        if window < self.min_window_s:
            return ScanMotionResult(
                displaced=displaced, displacement_m=displacement_m
            )
        change_m, overlap, _fraction = signature_change_m(
            signature, self._reference
        )
        self._reference = signature
        self._reference_at = scan_at
        if overlap < MIN_OVERLAP_BINS or window > MAX_WINDOW_S:
            # Too little shared geometry, or the reference aged out while the
            # loop was stalled; restart the window rather than guess.
            return ScanMotionResult(
                window_s=window,
                overlap_bins=overlap,
                displaced=displaced,
                displacement_m=displacement_m,
            )
        verdict = "MOVING" if change_m >= self.moved_change_m else "NOT_MOVING"
        return ScanMotionResult(
            verdict=verdict,
            change_m=change_m,
            window_s=window,
            overlap_bins=overlap,
            displaced=displaced,
            displacement_m=displacement_m,
        )
