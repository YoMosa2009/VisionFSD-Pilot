"""Lightweight environment intelligence for VisionFSD Pilot.

Uses only signals already produced by road/object perception (drivable mask,
tracks, path source, traffic-light ROI colour). No extra neural models —
designed to stay under ~1 ms on typical frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from visionfsd import (
    SIGNAL_CLASSES,
    VEHICLE_CLASSES,
    DetectedObject,
    RoadGeometry,
    ego_path_style,
    track_quality_score,
)


# Free-space cell status.
FS_FREE = "free"
FS_BLOCKED = "blocked"
FS_UNKNOWN = "unknown"

# Sticky scene modes.
SCENE_HIGHWAY = "highway"
SCENE_URBAN = "urban"
SCENE_INTERSECTION = "intersection"
SCENE_UNMARKED = "unmarked"
SCENE_OPPOSING = "opposing"
SCENE_UNKNOWN = "unknown"


@dataclass
class FreeSpaceBand:
    """Occupancy at one lookahead depth (left / center / right)."""

    distance_m: float
    left: str = FS_UNKNOWN
    center: str = FS_UNKNOWN
    right: str = FS_UNKNOWN

    def short(self) -> str:
        def _c(v: str) -> str:
            return {"free": "ok", "blocked": "blk", "unknown": "?"}.get(v, "?")
        return f"{int(self.distance_m)}m L:{_c(self.left)} C:{_c(self.center)} R:{_c(self.right)}"


@dataclass
class ControllingLight:
    track_id: int
    state: str | None  # red | yellow | green | None
    distance_m: float
    bearing_deg: float
    confidence: float


@dataclass
class EnvironmentSnapshot:
    mode: str = SCENE_UNKNOWN
    mode_confidence: float = 0.0
    free_space: list[FreeSpaceBand] = field(default_factory=list)
    controlling_light: ControllingLight | None = None
    path_source: str = "none"
    vehicle_count: int = 0
    signal_count: int = 0
    merge_like: int = 0
    summary: str = "ENV --"

    def free_space_line(self) -> str:
        if not self.free_space:
            return "FS --"
        # Prefer the nearest band for the HUD chip.
        return "FS " + self.free_space[0].short()


# Module cache (updated a few times per second from the render loop).
_LATEST: EnvironmentSnapshot = EnvironmentSnapshot()
_MODE_STICKY: str = SCENE_UNKNOWN
_MODE_HITS: dict[str, int] = {}
_MODE_HOLD: int = 0
# Free-space cell hysteresis (key = "10.0:left" etc.).
_FS_STICKY: dict[str, str] = {}
_FS_VOTES: dict[str, int] = {}
# Controlling-light identity stickiness.
_CTRL_LIGHT_ID: int | None = None
_CTRL_LIGHT_HITS: int = 0
_CTRL_CHALLENGER: int | None = None
_CTRL_CHALLENGE_HITS: int = 0


def latest_environment() -> EnvironmentSnapshot:
    return _LATEST


def prune_environment_caches(max_keys: int = 64) -> None:
    """Bound free-space sticky maps so long sessions do not grow without limit."""
    global _FS_STICKY, _FS_VOTES, _MODE_HITS
    if len(_FS_STICKY) > max_keys:
        # Drop oldest half by arbitrary order (values are cheap to relearn).
        for key in list(_FS_STICKY.keys())[: len(_FS_STICKY) // 2]:
            _FS_STICKY.pop(key, None)
            _FS_VOTES.pop(key, None)
    if len(_MODE_HITS) > 16:
        for key in list(_MODE_HITS.keys())[: len(_MODE_HITS) // 2]:
            _MODE_HITS.pop(key, None)


def _band_status(fraction: float) -> str:
    # Hysteresis-friendly thresholds (sticky helper may hold prior state).
    if fraction >= 0.42:
        return FS_FREE
    if fraction <= 0.10:
        return FS_BLOCKED
    return FS_UNKNOWN


def _sticky_band(key: str, raw: str, hold_frames: int = 5) -> str:
    """Require several agreeing samples before flipping free/blocked/?."""
    prev = _FS_STICKY.get(key)
    if prev is None or prev == raw:
        _FS_STICKY[key] = raw
        _FS_VOTES[key] = 0 if prev == raw else 0
        return raw
    # Unknown is soft — allow one-step transitions more easily.
    if raw == FS_UNKNOWN:
        _FS_VOTES[key] = _FS_VOTES.get(key, 0) + 1
        if _FS_VOTES[key] >= max(2, hold_frames // 2):
            _FS_STICKY[key] = raw
            _FS_VOTES[key] = 0
            return raw
        return prev
    _FS_VOTES[key] = _FS_VOTES.get(key, 0) + 1
    if _FS_VOTES[key] >= hold_frames:
        _FS_STICKY[key] = raw
        _FS_VOTES[key] = 0
        return raw
    return prev


def compute_free_space(
    road_geometry: RoadGeometry | None,
    objects: list[DetectedObject] | None = None,
    *,
    distances_m: tuple[float, ...] = (10.0, 20.0, 35.0),
    horizon_ratio: float = 0.52,
    camera_height_m: float = 1.25,
    fov_deg: float = 70.0,
) -> list[FreeSpaceBand]:
    """Sample drivable mask into L/C/R free-space bins at a few depths.

    Image rows are derived from a flat-road pinhole so 10/20/35 m map to the
    correct vertical positions. Vehicle feet in a bin force *blocked*.
    """
    bands: list[FreeSpaceBand] = []
    if road_geometry is None or road_geometry.drivable_mask is None:
        return [FreeSpaceBand(d) for d in distances_m]

    mask = np.asarray(road_geometry.drivable_mask > 0, dtype=np.uint8)
    h, w = mask.shape[:2]
    if h < 32 or w < 32:
        return [FreeSpaceBand(d) for d in distances_m]

    fw = road_geometry.frame_width or w
    fh = road_geometry.frame_height or h
    if mask.shape[0] != fh or mask.shape[1] != fw:
        # Mask may be analysis-resolution; scale sample coords into mask space.
        scale_x = w / max(1, fw)
        scale_y = h / max(1, fh)
    else:
        scale_x = scale_y = 1.0

    focal = (fw / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    horizon_y = float(np.clip(horizon_ratio, 0.30, 0.72) * fh)
    cx = fw * 0.5
    # Prefer road centre near bottom if available.
    if road_geometry.center_points is not None and len(road_geometry.center_points) > 0:
        cx = float(road_geometry.center_points[0, 0])

    # Pre-bin vehicle feet by approximate range for occupancy override.
    vehicle_feet: list[tuple[float, float]] = []  # (distance_m, foot_x)
    if objects:
        for obj in objects:
            if obj.label not in VEHICLE_CLASSES:
                continue
            vehicle_feet.append((float(obj.distance_m), (obj.box[0] + obj.box[2]) * 0.5))

    half_lane_px = fw * 0.075  # ~center corridor half-width in image
    side_px = fw * 0.11

    for dist in distances_m:
        ground_px = camera_height_m * focal / max(1.0, dist)
        row = horizon_y + ground_px
        row = float(np.clip(row, fh * 0.42, fh * 0.92))
        my = int(np.clip(round(row * scale_y), 1, h - 2))
        # Thick band for stability.
        y0, y1 = max(0, my - 2), min(h, my + 3)
        strip = mask[y0:y1, :]

        def _frac(x0: float, x1: float) -> float:
            a = int(np.clip(round(x0 * scale_x), 0, w - 1))
            b = int(np.clip(round(x1 * scale_x), a + 1, w))
            region = strip[:, a:b]
            if region.size == 0:
                return 0.0
            return float(np.count_nonzero(region)) / float(region.size)

        left_x0, left_x1 = cx - half_lane_px - side_px, cx - half_lane_px
        cen_x0, cen_x1 = cx - half_lane_px, cx + half_lane_px
        right_x0, right_x1 = cx + half_lane_px, cx + half_lane_px + side_px

        left_s = _band_status(_frac(left_x0, left_x1))
        cen_s = _band_status(_frac(cen_x0, cen_x1))
        right_s = _band_status(_frac(right_x0, right_x1))

        # Vehicles near this depth override free → blocked in their lateral bin.
        for v_dist, foot_x in vehicle_feet:
            if abs(v_dist - dist) > dist * 0.45 + 4.0:
                continue
            if left_x0 <= foot_x < left_x1:
                left_s = FS_BLOCKED
            elif cen_x0 <= foot_x < cen_x1:
                cen_s = FS_BLOCKED
            elif right_x0 <= foot_x < right_x1:
                right_s = FS_BLOCKED

        key = f"{dist:.0f}"
        left_s = _sticky_band(f"{key}:L", left_s)
        cen_s = _sticky_band(f"{key}:C", cen_s)
        right_s = _sticky_band(f"{key}:R", right_s)
        bands.append(FreeSpaceBand(float(dist), left_s, cen_s, right_s))
    return bands


def select_controlling_light(
    objects: list[DetectedObject],
) -> ControllingLight | None:
    """Pick the traffic light most relevant to the ego corridor ahead.

    Sticky identity: do not jump EGO-TL between track IDs every frame.
    """
    global _CTRL_LIGHT_ID, _CTRL_LIGHT_HITS, _CTRL_CHALLENGER, _CTRL_CHALLENGE_HITS
    candidates: list[tuple[float, DetectedObject]] = []
    for obj in objects:
        if obj.label != "traffic light":
            continue
        if not obj.observed and obj.missed_updates > 3:
            continue
        dist = float(obj.distance_m)
        bearing = abs(float(obj.bearing_deg))
        if dist < 6.0 or dist > 70.0:
            continue
        if bearing > 28.0:
            continue
        quality = obj.track_quality if obj.track_quality > 0 else track_quality_score(obj)
        score = (
            dist * 0.55
            + bearing * 1.15
            - float(obj.confidence) * 8.0
            - quality * 6.0
            - (4.0 if obj.signal_state else 0.0)
            + (3.0 if not obj.observed else 0.0)
        )
        if bearing < 12.0 and 10.0 <= dist <= 50.0:
            score -= 5.0
        if _CTRL_LIGHT_ID is not None and obj.track_id == _CTRL_LIGHT_ID:
            score -= 6.0  # incumbent advantage
        candidates.append((score, obj))
    if not candidates:
        _CTRL_LIGHT_HITS = max(0, _CTRL_LIGHT_HITS - 1)
        if _CTRL_LIGHT_HITS <= 0:
            _CTRL_LIGHT_ID = None
        return None

    candidates.sort(key=lambda item: item[0])
    best_score, best_obj = candidates[0]

    if _CTRL_LIGHT_ID is None:
        _CTRL_LIGHT_ID = best_obj.track_id
        _CTRL_LIGHT_HITS = 6
    elif best_obj.track_id == _CTRL_LIGHT_ID:
        _CTRL_LIGHT_HITS = min(16, _CTRL_LIGHT_HITS + 1)
        _CTRL_CHALLENGER = None
        _CTRL_CHALLENGE_HITS = 0
    else:
        # Challenger must win several updates before taking EGO-TL.
        if _CTRL_CHALLENGER == best_obj.track_id:
            _CTRL_CHALLENGE_HITS += 1
        else:
            _CTRL_CHALLENGER = best_obj.track_id
            _CTRL_CHALLENGE_HITS = 1
        incumbent = next((o for s, o in candidates if o.track_id == _CTRL_LIGHT_ID), None)
        if incumbent is None:
            _CTRL_LIGHT_HITS -= 1
            if _CTRL_LIGHT_HITS <= 0 or _CTRL_CHALLENGE_HITS >= 4:
                _CTRL_LIGHT_ID = best_obj.track_id
                _CTRL_LIGHT_HITS = 6
                _CTRL_CHALLENGE_HITS = 0
        elif _CTRL_CHALLENGE_HITS >= 8 and best_score < (
            next(s for s, o in candidates if o.track_id == _CTRL_LIGHT_ID) - 4.0
        ):
            _CTRL_LIGHT_ID = best_obj.track_id
            _CTRL_LIGHT_HITS = 6
            _CTRL_CHALLENGE_HITS = 0

    chosen = next((o for s, o in candidates if o.track_id == _CTRL_LIGHT_ID), best_obj)
    return ControllingLight(
        track_id=chosen.track_id,
        state=chosen.signal_state,
        distance_m=float(chosen.distance_m),
        bearing_deg=float(chosen.bearing_deg),
        confidence=float(chosen.confidence),
    )


def _vote_scene_mode(
    road_geometry: RoadGeometry | None,
    objects: list[DetectedObject],
    free_space: list[FreeSpaceBand],
    path_source: str,
    controlling: ControllingLight | None,
    oncoming_active: bool,
) -> tuple[str, float]:
    """Score scene modes from cheap cues; returns (mode, confidence)."""
    votes: dict[str, float] = {
        SCENE_HIGHWAY: 0.0,
        SCENE_URBAN: 0.0,
        SCENE_INTERSECTION: 0.0,
        SCENE_UNMARKED: 0.0,
        SCENE_OPPOSING: 0.0,
    }

    topology = (road_geometry.topology if road_geometry is not None else "unknown") or "unknown"
    vehicles = [o for o in objects if o.label in VEHICLE_CLASSES]
    signals = [o for o in objects if o.label in SIGNAL_CLASSES]
    vrus = [o for o in objects if o.label in {"person", "bicycle", "motorcycle"}]

    if oncoming_active:
        votes[SCENE_OPPOSING] += 4.0
    if any(o.closing_mps > 5.5 and abs(o.bearing_deg) < 16.0 for o in vehicles):
        votes[SCENE_OPPOSING] += 2.0

    if topology == "intersection" or (
        road_geometry is not None and road_geometry.intersection_y is not None
    ):
        votes[SCENE_INTERSECTION] += 4.5
    if controlling is not None and controlling.distance_m < 45.0:
        votes[SCENE_INTERSECTION] += 2.5
        if controlling.state == "red":
            votes[SCENE_INTERSECTION] += 1.5
    if any(o.label == "stop sign" and o.distance_m < 40.0 for o in signals):
        votes[SCENE_INTERSECTION] += 2.0

    if path_source in {"curbs", "asphalt", "hold"}:
        votes[SCENE_UNMARKED] += 3.5
    if path_source == "none":
        votes[SCENE_UNMARKED] += 1.5

    if path_source == "lanes" and topology in {"straight", "curve-left", "curve-right"}:
        votes[SCENE_HIGHWAY] += 2.5
    if len(vehicles) <= 2 and len(signals) == 0 and path_source == "lanes":
        votes[SCENE_HIGHWAY] += 2.0
    # Open center free-space at range → highway-ish.
    if free_space and free_space[-1].center == FS_FREE and free_space[0].center == FS_FREE:
        votes[SCENE_HIGHWAY] += 1.5

    if len(signals) >= 2 or len(vrus) >= 1:
        votes[SCENE_URBAN] += 2.5
    if len(vehicles) >= 2 and path_source in {"lanes", "curbs"}:
        votes[SCENE_URBAN] += 1.2
    if free_space and free_space[0].center == FS_BLOCKED:
        votes[SCENE_URBAN] += 1.0

    mode = max(votes, key=votes.get)
    total = sum(max(0.0, v) for v in votes.values()) + 1e-3
    conf = float(np.clip(votes[mode] / total, 0.0, 1.0))
    if votes[mode] < 1.5:
        return SCENE_UNKNOWN, conf * 0.5
    return mode, conf


def _sticky_mode(raw_mode: str, raw_conf: float) -> tuple[str, float]:
    """Hysteresis so the scene mode does not flicker every frame."""
    global _MODE_STICKY, _MODE_HITS, _MODE_HOLD
    for key in list(_MODE_HITS.keys()):
        if key != raw_mode:
            _MODE_HITS[key] = max(0, _MODE_HITS.get(key, 0) - 1)
    _MODE_HITS[raw_mode] = min(12, _MODE_HITS.get(raw_mode, 0) + 1)

    if _MODE_STICKY == SCENE_UNKNOWN or _MODE_HOLD <= 0:
        if _MODE_HITS.get(raw_mode, 0) >= 2:
            _MODE_STICKY = raw_mode
            _MODE_HOLD = 8
        return _MODE_STICKY if _MODE_STICKY != SCENE_UNKNOWN else raw_mode, raw_conf

    if raw_mode == _MODE_STICKY:
        _MODE_HOLD = min(16, _MODE_HOLD + 1)
        return _MODE_STICKY, max(raw_conf, 0.45)

    # Switch only after sustained disagreement.
    if _MODE_HITS.get(raw_mode, 0) >= 6 and raw_conf >= 0.35:
        _MODE_STICKY = raw_mode
        _MODE_HOLD = 8
        return _MODE_STICKY, raw_conf

    _MODE_HOLD = max(0, _MODE_HOLD - 1)
    return _MODE_STICKY, max(raw_conf * 0.7, 0.30)


def update_environment_intel(
    objects: list[DetectedObject],
    road_geometry: RoadGeometry | None,
    *,
    oncoming_active: bool = False,
    merge_count: int = 0,
    horizon_ratio: float = 0.52,
    camera_height_m: float = 1.25,
    fov_deg: float = 70.0,
) -> EnvironmentSnapshot:
    """Refresh free-space, scene mode, and controlling light (call throttled)."""
    global _LATEST
    path_source, _alpha, _fill, _edge = ego_path_style()
    free_space = compute_free_space(
        road_geometry, objects,
        horizon_ratio=horizon_ratio,
        camera_height_m=camera_height_m,
        fov_deg=fov_deg,
    )
    controlling = select_controlling_light(objects)
    raw_mode, raw_conf = _vote_scene_mode(
        road_geometry, objects, free_space, path_source, controlling, oncoming_active,
    )
    mode, conf = _sticky_mode(raw_mode, raw_conf)

    vehicles = sum(1 for o in objects if o.label in VEHICLE_CLASSES)
    signals = sum(1 for o in objects if o.label in SIGNAL_CLASSES)

    # Build one-line summary for HUDs.
    mode_tag = mode.upper()
    fs_tag = free_space[0].short() if free_space else "--"
    if controlling is not None:
        st = (controlling.state or "?").upper()
        tl_tag = f"TL {st} {controlling.distance_m:.0f}m"
    else:
        tl_tag = "TL --"
    summary = f"ENV {mode_tag}  |  FS {fs_tag}  |  {tl_tag}"

    _LATEST = EnvironmentSnapshot(
        mode=mode,
        mode_confidence=conf,
        free_space=free_space,
        controlling_light=controlling,
        path_source=path_source,
        vehicle_count=vehicles,
        signal_count=signals,
        merge_like=merge_count,
        summary=summary,
    )
    return _LATEST
