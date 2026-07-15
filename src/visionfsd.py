"""Read-only webcam driving-scene visualizer; never controls a vehicle."""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_DIR = PROJECT_ROOT / "models" / "ultralytics"
ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from screeninfo import get_monitors
except Exception:  # App still runs if optional monitor detection is unavailable.
    get_monitors = None


SCREENSHOT_DIR = PROJECT_ROOT / "logs" / "screenshots"

# Approximate real-world sizes (metres) used only for monocular distance estimates.
CLASS_WIDTH_METRES = {
    "person": 0.45, "bicycle": 0.60, "car": 1.80, "motorcycle": 0.75,
    "bus": 2.55, "truck": 2.55, "traffic light": 0.35, "stop sign": 0.75,
    "parking meter": 0.30, "fire hydrant": 0.40,
}
CLASS_HEIGHT_METRES = {
    "person": 1.70, "bicycle": 1.15, "car": 1.50, "motorcycle": 1.40,
    "bus": 3.00, "truck": 3.20, "train": 3.50,
    "traffic light": 1.10, "stop sign": 0.85,
    "parking meter": 1.20, "fire hydrant": 0.70,
}
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
SIGNAL_CLASSES = {"traffic light", "stop sign", "parking meter", "fire hydrant"}
VEHICLE_CLASSES = {"car", "truck", "bus", "train"}
FRAGILE_CLASSES = {"person", "bicycle", "traffic light", "stop sign", "parking meter", "fire hydrant"}
# YOLO11n is COCO-trained (80 classes); most of them never belong in a
# driving scene and can fire as low-confidence noise (kite, teddy bear,
# toothbrush, ...). Restrict to classes that are actually meaningful here.
RELEVANT_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "stop sign", "parking meter", "fire hydrant",
}


@dataclass
class DetectedObject:
    track_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    distance_m: float
    bearing_deg: float
    closing_mps: float
    image_velocity: tuple[float, float]
    mask_polygon: np.ndarray | None = None
    observed: bool = True
    missed_updates: int = 0
    stationary: bool = False
    # Optional traffic-light lamp colour from ROI scan: "red"|"yellow"|"green".
    signal_state: str | None = None
    # Display reliability 0..1 (hits, misses, conf). Used to pick meshes / lead car.
    track_quality: float = 0.0


@dataclass
class RoadMarking:
    """Lightweight road-paint / marking observation from classical vision."""
    kind: str          # crosswalk | lane_paint  (stop_line/arrow dropped — too noisy)
    confidence: float
    box: tuple[int, int, int, int]  # x1,y1,x2,y2 in full-frame pixels
    centroid: tuple[float, float]
    colour: str = "white"  # white | yellow


class MotionEstimator:
    """Maintains short history for display-only velocity and projected paths."""

    def __init__(self) -> None:
        self._history: dict[int, Deque[tuple[float, float, float, float]]] = defaultdict(lambda: deque(maxlen=12))
        self._filtered_range: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, track_id: int, center: tuple[float, float], distance_m: float,
               now: float) -> tuple[float, float, tuple[float, float]]:
        h = self._history[track_id]
        previous_range = self._filtered_range.get(track_id, distance_m)
        # Heavy range grounding — monocular width thrash was a main 3D flicker source.
        filtered_range = float(previous_range * 0.72 + distance_m * 0.28)
        # Deadband: ignore tiny range updates that rattle mesh depth.
        if abs(filtered_range - previous_range) < 0.45:
            filtered_range = previous_range
        self._filtered_range[track_id] = filtered_range
        self._last_seen[track_id] = now
        h.append((now, center[0], center[1], distance_m))
        closing_mps = 0.0
        image_velocity = (0.0, 0.0)
        if len(h) >= 2:
            # A short least-squares velocity is substantially less jittery than
            # differencing the two newest detector boxes.
            samples = list(h)[-6:]
            times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
            times -= times[0]
            if times[-1] >= 1e-3:
                xs = np.asarray([sample[1] for sample in samples], dtype=np.float64)
                ys = np.asarray([sample[2] for sample in samples], dtype=np.float64)
                ranges = np.asarray([sample[3] for sample in samples], dtype=np.float64)
                vx = float(np.polyfit(times, xs, 1)[0])
                vy = float(np.polyfit(times, ys, 1)[0])
                range_rate = float(np.polyfit(times, ranges, 1)[0])
                closing_mps = float(np.clip(-range_rate, -25.0, 25.0))
                image_velocity = (float(np.clip(vx, -1600.0, 1600.0)),
                                  float(np.clip(vy, -1600.0, 1600.0)))
        return filtered_range, closing_mps, image_velocity

    def prune(self, now: float, max_age_s: float = 4.0) -> int:
        """Drop tracks not updated recently (prevents multi-minute dict bloat)."""
        dead = [tid for tid, t in self._last_seen.items() if now - t > max_age_s]
        for tid in dead:
            self._history.pop(tid, None)
            self._filtered_range.pop(tid, None)
            self._last_seen.pop(tid, None)
        return len(dead)


def focal_length_px(frame_width: int, horizontal_fov_deg: float) -> float:
    return (frame_width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)


def estimate_distance_m(label: str, pixel_width: int, focal_px: float) -> float:
    """Return a coarse pinhole-model range estimate, clamped for a stable UI."""
    known_width = CLASS_WIDTH_METRES.get(label, 1.0)
    if pixel_width <= 1:
        return 150.0
    return float(np.clip((known_width * focal_px) / pixel_width, 1.0, 150.0))


def estimate_height_distance_m(label: str, pixel_height: int, focal_px: float) -> float:
    """Pinhole range from apparent height (helps when width is clipped/noisy)."""
    known_height = CLASS_HEIGHT_METRES.get(label, 1.5)
    if pixel_height <= 1:
        return 150.0
    return float(np.clip((known_height * focal_px) / pixel_height, 1.0, 150.0))


def ground_range_from_foot_y(
    foot_y: float,
    frame_height: int,
    focal_px: float,
    camera_height_m: float = 1.25,
    horizon_ratio: float = 0.52,
) -> float | None:
    """Flat-road pinhole range from a ground-contact image row, or None if invalid."""
    if frame_height <= 0 or focal_px <= 1.0:
        return None
    horizon_y = float(np.clip(horizon_ratio, 0.30, 0.72) * frame_height)
    ground_pixels = float(foot_y) - horizon_y
    if ground_pixels < frame_height * 0.03:
        return None
    return float(np.clip(camera_height_m * focal_px / ground_pixels, 1.0, 150.0))


def refine_distance_with_road_prior(
    obj: "DetectedObject",
    road_geometry: "RoadGeometry | None",
    *,
    focal_px: float,
    frame_height: int,
    camera_height_m: float = 1.25,
    horizon_ratio: float = 0.52,
) -> float:
    """Blend monocular size range with road-foot + lane-centre prior.

    Cheap CPU-only refinement (no depth NN). Vehicles near the ego lane trust
    the foot-row ground range more; off-lane keeps size-led range.
    """
    mono = float(obj.distance_m)
    if obj.label not in VEHICLE_CLASSES | {"motorcycle", "bicycle", "person"}:
        return mono
    x1, y1, x2, y2 = obj.box
    foot_x = (x1 + x2) * 0.5
    foot_y = float(y2)
    ground = ground_range_from_foot_y(
        foot_y, frame_height, focal_px, camera_height_m, horizon_ratio,
    )
    if ground is None:
        return mono

    # Lane membership from vision corridor (metres from ego-lane centre).
    abs_lat = 99.0
    if road_geometry is not None:
        lat = road_lateral_offset_m(foot_x, foot_y, road_geometry)
        if lat is not None:
            abs_lat = abs(float(lat))

    # Ego / near-lane: ground-led. Adjacent: balanced. Far side: size-led.
    if abs_lat <= 1.7:
        w_ground = 0.58
    elif abs_lat <= 5.2:
        w_ground = 0.38
    else:
        w_ground = 0.22
    # Reject wild ground outliers vs monocular (hood/horizon glitches).
    if abs(ground - mono) > max(12.0, mono * 0.85):
        w_ground *= 0.35
    fused = (1.0 - w_ground) * mono + w_ground * ground
    return float(np.clip(fused, 1.0, 150.0))


def apply_road_range_prior(
    objects: list["DetectedObject"],
    road_geometry: "RoadGeometry | None",
    *,
    frame_width: int,
    frame_height: int,
    fov_deg: float = 70.0,
    camera_height_m: float = 1.25,
    horizon_ratio: float = 0.52,
) -> list["DetectedObject"]:
    """Return objects with road-aware range (new DetectedObject instances)."""
    if not objects or frame_height <= 0 or frame_width <= 0:
        return objects
    focal = focal_length_px(frame_width, fov_deg)
    out: list[DetectedObject] = []
    for obj in objects:
        new_dist = refine_distance_with_road_prior(
            obj, road_geometry,
            focal_px=focal,
            frame_height=frame_height,
            camera_height_m=camera_height_m,
            horizon_ratio=horizon_ratio,
        )
        if abs(new_dist - obj.distance_m) < 0.05:
            out.append(obj)
            continue
        out.append(DetectedObject(
            obj.track_id, obj.label, obj.confidence, obj.box, new_dist,
            obj.bearing_deg, obj.closing_mps, obj.image_velocity, obj.mask_polygon,
            obj.observed, obj.missed_updates, obj.stationary, obj.signal_state,
            obj.track_quality,
        ))
    return out


class TemporalTrackFusion:
    """Light temporal EMA over stabilized tracks (box + range + closing).

    Runs after ByteTrack/TrackStabilizer. Cheap, no numpy solvers — safe for
    the async detect worker. Suppresses single-frame thrash in 3D/camera.
    """

    def __init__(self, box_alpha: float = 0.38, range_alpha: float = 0.32,
                 close_alpha: float = 0.40, max_age_s: float = 2.5) -> None:
        # state: tid -> (box_f32 x4, dist, closing, last_t)
        self._state: dict[int, tuple[np.ndarray, float, float, float]] = {}
        self._box_alpha = float(box_alpha)
        self._range_alpha = float(range_alpha)
        self._close_alpha = float(close_alpha)
        self._max_age_s = float(max_age_s)

    def update(self, objects: list["DetectedObject"], now: float) -> list["DetectedObject"]:
        if not objects:
            self._prune(now, set())
            return objects
        out: list[DetectedObject] = []
        seen: set[int] = set()
        for obj in objects:
            seen.add(obj.track_id)
            raw_box = np.asarray(obj.box, dtype=np.float32)
            prev = self._state.get(obj.track_id)
            if prev is None or not obj.observed:
                # First hit or coasted: seed / light hold on coasted box only.
                if prev is not None and not obj.observed:
                    pbox, pdist, pclose, _ = prev
                    box = tuple(np.rint(pbox * 0.82 + raw_box * 0.18).astype(int).tolist())
                    dist = float(pdist * 0.90 + obj.distance_m * 0.10)
                    close = float(pclose * 0.85 + obj.closing_mps * 0.15)
                else:
                    box = obj.box
                    dist = float(obj.distance_m)
                    close = float(obj.closing_mps)
                self._state[obj.track_id] = (
                    np.asarray(box, dtype=np.float32), dist, close, now,
                )
                out.append(DetectedObject(
                    obj.track_id, obj.label, obj.confidence, box, dist,
                    obj.bearing_deg, close, obj.image_velocity, obj.mask_polygon,
                    obj.observed, obj.missed_updates, obj.stationary,
                    obj.signal_state, obj.track_quality,
                ))
                continue
            pbox, pdist, pclose, _pt = prev
            # Observed: EMA toward measurement (faster on first few hits via quality).
            a_box = self._box_alpha
            a_rng = self._range_alpha
            a_cl = self._close_alpha
            if obj.track_quality > 0.0 and obj.track_quality < 0.45:
                a_box = min(0.55, a_box + 0.12)
                a_rng = min(0.50, a_rng + 0.10)
            fused_box = pbox * (1.0 - a_box) + raw_box * a_box
            # Reject wild range jumps (occlusion / ID flip).
            if abs(obj.distance_m - pdist) > max(8.0, pdist * 0.55):
                a_rng = min(a_rng, 0.12)
            fused_dist = pdist * (1.0 - a_rng) + float(obj.distance_m) * a_rng
            fused_close = pclose * (1.0 - a_cl) + float(obj.closing_mps) * a_cl
            box_i = tuple(np.rint(fused_box).astype(int).tolist())
            self._state[obj.track_id] = (
                fused_box.astype(np.float32), float(fused_dist), float(fused_close), now,
            )
            out.append(DetectedObject(
                obj.track_id, obj.label, obj.confidence, box_i, float(fused_dist),
                obj.bearing_deg, float(fused_close), obj.image_velocity, obj.mask_polygon,
                obj.observed, obj.missed_updates, obj.stationary,
                obj.signal_state, obj.track_quality,
            ))
        self._prune(now, seen)
        return out

    def _prune(self, now: float, seen: set[int]) -> None:
        dead = [
            tid for tid, state in self._state.items()
            if tid not in seen and (now - state[3]) > self._max_age_s
        ]
        for tid in dead:
            self._state.pop(tid, None)
        if len(self._state) > 48:
            oldest = sorted(self._state.items(), key=lambda kv: kv[1][3])
            for tid, _ in oldest[: len(self._state) - 48]:
                self._state.pop(tid, None)

    def clear(self) -> None:
        self._state.clear()


def estimate_monocular_distance_m(label: str, box: tuple[int, int, int, int],
                                  focal_px: float, frame_height: int,
                                  camera_height_m: float = 1.25,
                                  horizon_ratio: float = 0.52) -> float:
    """Fuse width, height, and flat-ground footpoint range for stable UI depth.

    Ground footpoint is the primary cue for vehicles/people on the road so
    side-by-side traffic does not collapse into one depth plane. Signs/lights
    rely more on apparent size (they do not sit on the road surface).
    """
    x1, y1, x2, y2 = box
    pw = max(1, x2 - x1)
    ph = max(1, y2 - y1)
    width_range = estimate_distance_m(label, pw, focal_px)
    height_range = estimate_height_distance_m(label, ph, focal_px)
    # Geometric mean of width/height ranges resists single-axis box noise.
    size_range = math.sqrt(max(width_range, 1e-3) * max(height_range, 1e-3))
    size_range = float(np.clip(size_range, 1.0, 150.0))

    if frame_height <= 0:
        return size_range

    horizon_y = float(np.clip(horizon_ratio, 0.30, 0.72) * frame_height)
    ground_pixels = float(y2) - horizon_y
    has_ground = ground_pixels >= frame_height * 0.03
    ground_range = (
        float(np.clip(camera_height_m * focal_px / ground_pixels, 1.0, 150.0))
        if has_ground else size_range
    )

    if label in SIGNAL_CLASSES:
        # Elevated furniture: size-led; weak ground only if box is low in frame.
        ground_weight = 0.12 if (has_ground and y2 > frame_height * 0.55) else 0.0
        size_range = 0.55 * width_range + 0.45 * height_range
    elif label not in ROAD_CLASSES:
        return size_range
    elif y2 >= frame_height * 0.985:
        # Hood-clipped vehicle box: don't trust the foot.
        ground_weight = 0.22
    elif label == "person":
        ground_weight = 0.68 if y2 < frame_height * 0.98 else 0.38
    elif y2 >= frame_height * 0.90:
        # Near field: foot is very reliable for lane placement.
        ground_weight = 0.78
    elif y2 >= frame_height * 0.70:
        ground_weight = 0.72
    else:
        # Far field: blend size + ground so distant cars don't float.
        ground_weight = 0.62

    if not has_ground:
        ground_weight = 0.0
    fused_log = ((1.0 - ground_weight) * math.log(max(size_range, 1e-3)) +
                 ground_weight * math.log(max(ground_range, 1e-3)))
    return float(np.clip(math.exp(fused_log), 1.0, 150.0))


def estimate_lanes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Classical lane heuristic. It is intentionally displayed as experimental."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 150)
    mask = np.zeros_like(edges)
    polygon = np.array([[(int(w * 0.06), h), (int(w * 0.42), int(h * 0.56)),
                         (int(w * 0.58), int(h * 0.56)), (int(w * 0.94), h)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    lines = cv2.HoughLinesP(cv2.bitwise_and(edges, mask), 1, np.pi / 180, 35,
                            minLineLength=max(22, w // 18), maxLineGap=30)
    result: list[tuple[int, int, int, int]] = []
    if lines is None:
        return result
    # OpenCV has returned both (N, 1, 4) and (N, 4) from HoughLinesP across releases.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx = x2 - x1
        if dx == 0:
            continue
        slope = (y2 - y1) / dx
        if abs(slope) > 0.35:
            result.append((int(x1), int(y1), int(x2), int(y2)))
    return result


@dataclass
class RoadGeometry:
    """Display-only road estimate inferred from a single forward RGB frame."""

    left_points: np.ndarray | None
    right_points: np.ndarray | None
    center_points: np.ndarray | None
    confidence: float
    topology: str
    intersection_y: int | None = None
    frame_width: int = 0
    frame_height: int = 0
    drivable_mask: np.ndarray | None = None
    lane_probability: np.ndarray | None = None
    source: str = "heuristic"
    # Image row above which the curve is extrapolated beyond real detected
    # evidence rather than directly fit to it. None means "treat it all as
    # extrapolated" (e.g. the drivable-only corridor has no paint evidence).
    confident_top_y: int | None = None


def _sliding_lane_points(binary: np.ndarray, initial_x: int, side: str,
                         nonzero: tuple[np.ndarray, np.ndarray] | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Collect lane-marking pixels in a bird's-eye binary image with sliding windows."""
    height, width = binary.shape
    nonzero_y, nonzero_x = binary.nonzero() if nonzero is None else nonzero
    windows, margin, min_pixels = 9, max(24, width // 15), max(18, width // 55)
    current_x = initial_x
    selected: list[np.ndarray] = []
    for window in range(windows):
        y_low = height - (window + 1) * height // windows
        y_high = height - window * height // windows
        indices = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                   (nonzero_x >= current_x - margin) & (nonzero_x < current_x + margin)).nonzero()[0]
        if len(indices) >= min_pixels:
            selected.append(indices)
            current_x = int(np.mean(nonzero_x[indices]))
    if not selected:
        return np.empty(0), np.empty(0)
    indices = np.concatenate(selected)
    xs, ys = nonzero_x[indices], nonzero_y[indices]
    # Reject pixels that crossed the image centre; this keeps left/right fits separate.
    keep = xs < width // 2 if side == "left" else xs >= width // 2
    return xs[keep], ys[keep]


def _fit_lane_curves(binary: np.ndarray, side: str, limit: int = 2) -> list[tuple[np.ndarray, float, float]]:
    """Return several plausible fits so the ego-lane pair can be selected jointly."""
    height, width = binary.shape
    histogram = np.sum(binary[int(height * 0.55):, :], axis=0).astype(np.float32)
    smoothed = cv2.GaussianBlur(histogram.reshape(1, -1), (0, 0), 4.0).reshape(-1)
    local_maximum = cv2.dilate(smoothed.reshape(1, -1), np.ones((1, 21), dtype=np.float32)).reshape(-1)
    if side == "left":
        start, end = int(width * 0.06), int(width * 0.48)
    else:
        start, end = int(width * 0.52), int(width * 0.94)
    region = smoothed[start:end]
    if region.size == 0 or float(np.max(region)) <= 0.0:
        return []
    peak_indices = np.flatnonzero(
        (region >= local_maximum[start:end] - 1e-3) &
        (region >= float(np.max(region)) * 0.10)
    ) + start
    groups = np.split(peak_indices, np.where(np.diff(peak_indices) > 1)[0] + 1) if len(peak_indices) else []
    candidates = [int(round(float(np.mean(group)))) for group in groups if len(group)]
    candidates.sort(reverse=side == "left")  # Nearest image centre first.
    fallback = int(np.argmax(histogram[start:end]) + start)
    if fallback not in candidates:
        candidates.append(fallback)
    fits: list[tuple[np.ndarray, float, float]] = []
    nonzero = binary.nonzero()
    for initial_x in candidates[:max(1, limit * 2)]:
        xs, ys = _sliding_lane_points(binary, initial_x, side, nonzero)
        if len(xs) < max(60, height // 3):
            continue
        coefficients = np.polyfit(ys, xs, 2)
        residual = np.abs(xs - np.polyval(coefficients, ys))
        inliers = residual < max(10.0, width * 0.028)
        if int(np.count_nonzero(inliers)) < max(48, height // 4):
            continue
        refined = np.polyfit(ys[inliers], xs[inliers], 2)
        test_y = np.linspace(height - 1, int(height * 0.05), 20)
        test_x = np.polyval(refined, test_y)
        valid_projection = (test_x >= -width * 0.15) & (test_x <= width * 1.15)
        valid_projection &= test_x < width * 0.68 if side == "left" else test_x > width * 0.32
        if int(np.count_nonzero(valid_projection)) < 8:
            continue
        support = float(np.clip(np.count_nonzero(inliers) / max(1, height * 2.2), 0.0, 1.0))
        # Smallest inlier y is the farthest row with real detected evidence;
        # rows above it (evaluated later) are an extrapolation of this fit.
        top_y = float(np.min(ys[inliers]))
        fits.append((refined, support, top_y))
        if len(fits) >= limit:
            break
    return fits


def _fit_lane_curve(binary: np.ndarray, side: str) -> np.ndarray | None:
    """Compatibility wrapper returning the strongest centre-nearest fit."""
    fits = _fit_lane_curves(binary, side, 1)
    return fits[0][0] if fits else None


def _curve_image_points(coefficients: np.ndarray | None, inverse_matrix: np.ndarray, width: int, height: int) -> np.ndarray | None:
    if coefficients is None:
        return None
    # A small margin below the ROI's own top edge (y=0), not past it: sampling
    # further into pure extrapolation was tried and reverted -- on a straight
    # or gently curving road the width/monotonicity safety checks in
    # _resample_lane_pair don't trigger for a long distance, so small fit
    # errors compounded over that long unconstrained extrapolation and the
    # corridor visibly drifted into adjacent lanes.
    warped_y = np.linspace(height - 1, int(height * 0.05), 28)
    warped_x = np.polyval(coefficients, warped_y)
    points = np.column_stack((warped_x, warped_y)).astype(np.float32).reshape(-1, 1, 2)
    original = cv2.perspectiveTransform(points, inverse_matrix).reshape(-1, 2)
    valid = ((original[:, 0] >= -width * 0.15) & (original[:, 0] <= width * 1.15) &
             (original[:, 1] >= -height * 0.15) & (original[:, 1] <= height * 1.15))
    return original[valid].astype(np.int32) if np.count_nonzero(valid) >= 8 else None


def _lane_x_at_y(points: np.ndarray, y: float) -> float | None:
    """Interpolate one image-space lane boundary at a requested row."""
    order = np.argsort(points[:, 1])
    ys = points[order, 1].astype(np.float32)
    xs = points[order, 0].astype(np.float32)
    if len(ys) < 2 or y < float(ys[0]) or y > float(ys[-1]):
        return None
    return float(np.interp(y, ys, xs))


def road_lateral_offset_m(image_x: float, image_y: float,
                          road_geometry: RoadGeometry | None,
                          lane_width_m: float = 3.6) -> float | None:
    """Map an image footpoint into metres relative to the detected ego-lane centre."""
    if road_geometry is None or road_geometry.left_points is None or road_geometry.right_points is None:
        return None
    left_x = _lane_x_at_y(road_geometry.left_points, image_y)
    right_x = _lane_x_at_y(road_geometry.right_points, image_y)
    if left_x is None or right_x is None or right_x - left_x < 20.0:
        return None
    return float(np.clip((image_x - (left_x + right_x) * 0.5) /
                         (right_x - left_x) * lane_width_m, -10.8, 10.8))


def _intersection_candidate_y(edges: np.ndarray, paint: np.ndarray,
                              left_points: np.ndarray | None,
                              right_points: np.ndarray | None,
                              confidence: float) -> int | None:
    """Return a conservative stop-line candidate, not a confirmed intersection.

    A generic horizontal edge is not sufficient. Evidence must sit inside the
    fitted road corridor, approximately span the lane, contain marking-colour
    pixels, and produce multiple Hough supports (normally the two edges of a
    painted stop line or crosswalk stripe).
    """
    if left_points is None or right_points is None or confidence < 0.68:
        return None
    height, width = edges.shape
    y_start, y_end = int(height * 0.55), int(height * 0.84)
    x_start, x_end = int(width * 0.08), int(width * 0.92)
    roi = edges[y_start:y_end, x_start:x_end]
    lines = cv2.HoughLinesP(
        roi, 1, np.pi / 180, max(36, width // 16),
        minLineLength=max(64, int(width * 0.16)),
        maxLineGap=max(14, width // 32),
    )
    if lines is None:
        return None
    candidates: list[tuple[int, int, float]] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        if abs(y2 - y1) > max(4, height // 120):
            continue
        global_y = int(round((y1 + y2) * 0.5 + y_start))
        global_x1, global_x2 = sorted((int(x1 + x_start), int(x2 + x_start)))
        left_x = _lane_x_at_y(left_points, global_y)
        right_x = _lane_x_at_y(right_points, global_y)
        if left_x is None or right_x is None:
            continue
        lane_width = right_x - left_x
        line_width = global_x2 - global_x1
        if lane_width < width * 0.12 or not (lane_width * 0.48 <= line_width <= lane_width * 1.38):
            continue
        line_mid = (global_x1 + global_x2) * 0.5
        lane_mid = (left_x + right_x) * 0.5
        if abs(line_mid - lane_mid) > lane_width * 0.24:
            continue
        overlap = max(0.0, min(float(global_x2), right_x) - max(float(global_x1), left_x))
        if overlap < lane_width * 0.46:
            continue
        band_y1, band_y2 = max(0, global_y - 4), min(height, global_y + 5)
        band_x1 = max(0, int(max(global_x1, left_x)))
        band_x2 = min(width, int(min(global_x2, right_x)) + 1)
        if band_x2 <= band_x1:
            continue
        paint_ratio = float(np.count_nonzero(paint[band_y1:band_y2, band_x1:band_x2])) / max(1, (band_y2 - band_y1) * (band_x2 - band_x1))
        if paint_ratio < 0.12:
            continue
        candidates.append((global_y, line_width, paint_ratio))
    if not candidates:
        return None
    # One continuous, strongly painted stop bar is sufficient as a *candidate*;
    # it still requires four temporal confirmations before becoming a junction.
    if len(candidates) == 1:
        return candidates[0][0] if candidates[0][2] >= 0.32 else None
    # Require two independent line supports at approximately the same road row.
    candidates.sort(key=lambda item: item[0])
    best_cluster: list[tuple[int, int, float]] = []
    for candidate in candidates:
        cluster = [item for item in candidates if abs(item[0] - candidate[0]) <= max(10, height // 55)]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    if len(best_cluster) < 2:
        return None
    return int(round(float(np.median([item[0] for item in best_cluster]))))


def _lane_topology(center_points: np.ndarray | None, width: int) -> str:
    if center_points is None or len(center_points) < 2:
        return "unknown"
    bottom_x, top_x = center_points[0, 0], center_points[-1, 0]
    curve_offset = (top_x - bottom_x) / max(1, width)
    return "curve-right" if curve_offset > 0.022 else "curve-left" if curve_offset < -0.022 else "straight"


def _corridor_from_drivable(drivable: np.ndarray,
                            horizon_ratio: float = 0.52
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Infer a conservative ego corridor when paint pixels cannot form a pair.

    The output is deliberately lower confidence: its sides describe an inferred
    corridor inside learned drivable area, not confirmed painted lane markings.
    Width is clamped to a single ego-lane envelope so multi-lane asphalt does
    not become a bloated blue pathway.
    """
    height, width = drivable.shape
    cleaned = cv2.morphologyEx(
        np.asarray(drivable > 0, dtype=np.uint8), cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8), iterations=2,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    if count <= 1:
        return None
    anchor = np.zeros_like(cleaned)
    anchor[int(height * 0.70):int(height * 0.96), int(width * 0.36):int(width * 0.64)] = 1
    best_component, best_score = 0, 0.0
    for component in range(1, count):
        area = float(stats[component, cv2.CC_STAT_AREA])
        # 0.018 (was 0.025): narrow residential/laneless roads fill less of
        # the frame than a highway but are exactly where this fallback matters.
        if area < height * width * 0.018:
            continue
        overlap = float(np.count_nonzero((labels == component) & (anchor > 0)))
        score = area + overlap * 8.0
        if score > best_score:
            best_component, best_score = component, score
    if best_component == 0:
        return None
    component_mask = labels == best_component
    far_limit = float(np.clip(height * max(0.32, horizon_ratio - 0.04), height * 0.30, height * 0.52))
    sample_y = np.linspace(height * 0.96, far_limit, 36)
    image_center = width * 0.5
    seed = image_center
    curb_left: list[tuple[float, float]] = []   # (x, y) of the drivable/road left edge
    curb_right: list[tuple[float, float]] = []
    for y_value in sample_y:
        y = int(np.clip(round(y_value), 0, height - 1))
        y1, y2 = max(0, y - 3), min(height, y + 4)
        columns = np.flatnonzero(np.count_nonzero(component_mask[y1:y2], axis=0) >= 2)
        if not len(columns):
            continue
        groups = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
        runs = [group for group in groups if len(group) >= max(12, width // 80)]
        if not runs:
            continue
        run = min(runs, key=lambda group: 0.0 if group[0] <= seed <= group[-1]
                  else min(abs(seed - group[0]), abs(seed - group[-1])))
        road_left, road_right = float(run[0]), float(run[-1])
        if road_right - road_left < width * 0.048:
            continue
        curb_left.append((road_left, y_value))
        curb_right.append((road_right, y_value))
        seed = (road_left + road_right) * 0.5
    if len(curb_left) < 10:
        return None
    coverage = len(curb_left) / len(sample_y)

    def _fit_curb(samples: list[tuple[float, float]]) -> np.ndarray:
        """Quadratic curb fit with one residual-based outlier rejection pass.

        Fitting the road's actual edges (kerbs / asphalt boundaries) lets the
        corridor inherit real road curvature on unmarked roads instead of a
        centroid-following heuristic that flattened bends.
        """
        arr = np.asarray(samples, dtype=np.float64)
        ys, xs = arr[:, 1], arr[:, 0]
        degree = 2 if len(ys) >= 9 else 1
        coefficients = np.polyfit(ys, xs, degree)
        residual = np.abs(xs - np.polyval(coefficients, ys))
        keep = residual <= max(width * 0.015, float(np.percentile(residual, 78)))
        if int(np.count_nonzero(keep)) >= max(6, len(ys) // 2):
            coefficients = np.polyfit(ys[keep], xs[keep], degree)
        return coefficients

    left_fit = _fit_curb(curb_left)
    right_fit = _fit_curb(curb_right)
    ys = np.asarray([y for _, y in curb_left], dtype=np.float64)
    left_x = np.polyval(left_fit, ys)
    right_x = np.polyval(right_fit, ys)
    valid = right_x - left_x > width * 0.04
    if int(np.count_nonzero(valid)) < 10:
        return None
    ys, left_x, right_x = ys[valid], left_x[valid], right_x[valid]
    road_mid = (left_x + right_x) * 0.5
    # Anchor the corridor under the camera at the near edge, then let it take
    # on the fitted curbs' curvature toward the horizon. The ego camera sits
    # at image centre; the curb midline supplies the *shape* of the road.
    perspective = np.clip((ys / height - 0.32) / 0.64, 0.0, 1.0)
    near_mid = float(road_mid[0])
    centre_x = image_center + (road_mid - near_mid) * (0.55 + 0.45 * (1.0 - perspective))
    centre_x = np.clip(centre_x, left_x + width * 0.02, right_x - width * 0.02)
    # Single-lane envelope, never the full asphalt width.
    expected_half = width * (0.05 + 0.11 * perspective)
    half = np.minimum(expected_half, (right_x - left_x) * 0.28)
    min_half = width * 0.024
    ok = half >= min_half
    if int(np.count_nonzero(ok)) < 10:
        return None
    ys, centre_x, half = ys[ok], centre_x[ok], half[ok]
    center = np.rint(np.column_stack((centre_x, ys))).astype(np.int32)
    left = np.rint(np.column_stack((centre_x - half, ys))).astype(np.int32)
    right = np.rint(np.column_stack((centre_x + half, ys))).astype(np.int32)
    confidence = float(np.clip(0.30 + coverage * 0.24, 0.0, 0.52))
    return left, right, center, confidence


def _resample_lane_pair(left_points: np.ndarray, right_points: np.ndarray,
                        width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Pair both projected boundaries at shared image rows and reject crossings."""
    top_y = max(float(np.min(left_points[:, 1])), float(np.min(right_points[:, 1])))
    bottom_y = min(float(np.max(left_points[:, 1])), float(np.max(right_points[:, 1])))
    if bottom_y - top_y < height * 0.16:
        return None
    sample_y = np.linspace(bottom_y, top_y, 28, dtype=np.float32)

    def interpolate(points: np.ndarray) -> np.ndarray:
        order = np.argsort(points[:, 1])
        ys = points[order, 1].astype(np.float32)
        xs = points[order, 0].astype(np.float32)
        unique_y, unique_indices = np.unique(ys, return_index=True)
        return np.interp(sample_y, unique_y, xs[unique_indices]).astype(np.float32)

    left_x = interpolate(left_points)
    right_x = interpolate(right_points)
    lane_width = right_x - left_x
    valid = (
        (lane_width > width * 0.10) & (lane_width < width * 0.82) &
        (left_x > -width * 0.10) & (right_x < width * 1.10)
    )
    # sample_y runs near-to-far, so real perspective narrows monotonically
    # across this array. Once an extrapolated fit's width stops narrowing and
    # widens again, it has left the plausible regime (the two boundary curves
    # are diverging, not converging toward the vanishing point) -- that row
    # and everything beyond it is discarded rather than treated as reach.
    running_min = np.minimum.accumulate(lane_width)
    still_narrowing = lane_width <= running_min + 1e-3
    first_violation = int(np.argmin(still_narrowing)) if not np.all(still_narrowing) else len(still_narrowing)
    valid[first_violation:] = False
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 10:
        return None
    groups = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)
    group = max(groups, key=len)
    if len(group) < 10:
        return None
    left_x, right_x, sample_y = left_x[group], right_x[group], sample_y[group]
    lane_width = right_x - left_x
    # With a forward camera, the accepted road corridor must narrow toward the
    # horizon and cannot jump laterally between adjacent sampled rows.
    if lane_width[0] <= lane_width[-1] * 1.05:
        return None
    center_x = (left_x + right_x) * 0.5
    reversal_threshold = width * 0.004
    for boundary_x in (left_x, right_x):
        steps = np.diff(boundary_x)
        if (float(np.min(steps)) < -reversal_threshold and
                float(np.max(steps)) > reversal_threshold):
            return None
    if len(center_x) > 2 and float(np.percentile(np.abs(np.diff(center_x)), 95)) > width * 0.09:
        return None
    left = np.column_stack((left_x, sample_y)).round().astype(np.int32)
    right = np.column_stack((right_x, sample_y)).round().astype(np.int32)
    center = np.column_stack((center_x, sample_y)).round().astype(np.int32)
    confidence = float(np.clip(len(group) / 28.0, 0.0, 1.0))
    return left, right, center, confidence


def _birdseye_to_image_y(x_birdseye: float, y_birdseye: float, inverse: np.ndarray) -> float:
    """Map a single birdseye-space point back to its real image-space row."""
    point = np.asarray([[[x_birdseye, y_birdseye]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, inverse)
    return float(transformed[0, 0, 1])


def _select_ego_lane_pair(left_fits: list[tuple[np.ndarray, float, float]],
                          right_fits: list[tuple[np.ndarray, float, float]],
                          inverse: np.ndarray, width: int, height: int
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None:
    """Select the mutually consistent pair most likely to bound the camera's lane."""
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None = None
    best_score = -1.0
    for left_fit, left_support, left_top_birdseye in left_fits:
        left_points = _curve_image_points(left_fit, inverse, width, height)
        if left_points is None:
            continue
        for right_fit, right_support, right_top_birdseye in right_fits:
            right_points = _curve_image_points(right_fit, inverse, width, height)
            if right_points is None:
                continue
            paired = _resample_lane_pair(left_points, right_points, width, height)
            if paired is None:
                continue
            left, right, center, confidence = paired
            bottom_width = float(right[0, 0] - left[0, 0])
            bottom_center = float(center[0, 0])
            width_ratio = bottom_width / max(1.0, width)
            # Ego lane is typically ~0.22-0.42 of frame width at the near
            # edge. Multi-lane spans (adjacent + ego + shoulder) used to win
            # on raw support and bloated the blue corridor across lanes.
            if width_ratio < 0.12 or width_ratio > 0.52:
                continue
            if not (width * 0.18 <= bottom_center <= width * 0.82):
                continue
            centre_score = max(0.0, 1.0 - abs(bottom_center - width * 0.5) / (width * 0.28))
            # Peak score near a single-lane width (~0.30 of image).
            width_score = max(0.0, 1.0 - abs(width_ratio - 0.30) / 0.20)
            support_score = min(1.0, left_support + right_support)
            # Strongly reward being under the camera (ego lane), not a
            # neighbouring lane whose markings happen to fit cleanly.
            score = (confidence * 0.30 + centre_score * 0.40 +
                     width_score * 0.22 + support_score * 0.08)
            if score > best_score:
                best_score = score
                # The pair is only jointly confident as far as its shorter
                # side's real evidence reaches (larger birdseye-y = nearer).
                confident_birdseye_y = max(left_top_birdseye, right_top_birdseye)
                mid_x = float((np.polyval(left_fit, confident_birdseye_y) +
                              np.polyval(right_fit, confident_birdseye_y)) * 0.5)
                confident_image_y = int(np.clip(
                    round(_birdseye_to_image_y(mid_x, confident_birdseye_y, inverse)), 0, height - 1,
                ))
                best = (left, right, center, float(np.clip(score, 0.0, 1.0)), confident_image_y)
    return best


def _extend_lane_to_horizon(left: np.ndarray, right: np.ndarray, center: np.ndarray,
                            width: int, height: int, horizon_ratio: float = 0.52,
                            confident_top_y: int | None = None
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    """Push a stable ego-lane pair further toward the horizon with a poly fit.

    Near rows keep their measured geometry; far rows are extrapolated so the
    corridor does not stop short and look "unconfident" on open highways.
    """
    if len(center) < 8:
        return left, right, center, confident_top_y
    # Conservative tip only -- long free-form extension invented false bends.
    target_top = float(np.clip(height * max(0.30, horizon_ratio - 0.06), 0.0, height - 1))
    current_top = float(min(np.min(left[:, 1]), np.min(right[:, 1]), np.min(center[:, 1])))
    evidence_top = (float(confident_top_y) if confident_top_y is not None
                    else current_top)
    if current_top <= target_top + height * 0.03:
        return left, right, center, int(round(evidence_top))

    def fit_boundary(points: np.ndarray) -> np.ndarray | None:
        order = np.argsort(points[:, 1])
        ys = points[order, 1].astype(np.float64)
        xs = points[order, 0].astype(np.float64)
        unique_y, unique_idx = np.unique(ys, return_index=True)
        if len(unique_y) < 4:
            return None
        xs = xs[unique_idx]
        # Linear extension only: quadratic tips were the main false-curve source.
        return np.polyfit(unique_y, xs, 1)

    left_fit = fit_boundary(left)
    right_fit = fit_boundary(right)
    if left_fit is None or right_fit is None:
        return left, right, center, int(round(evidence_top))

    bottom_y = float(max(np.max(left[:, 1]), np.max(right[:, 1])))
    sample_y = np.linspace(bottom_y, target_top, 36, dtype=np.float64)
    left_x = np.polyval(left_fit, sample_y)
    right_x = np.polyval(right_fit, sample_y)
    lane_w = right_x - left_x
    # Stop extrapolation once the projected lane becomes implausible so the
    # blue path cannot balloon into neighbouring lanes far ahead.
    valid = (
        (lane_w > width * 0.06) & (lane_w < width * 0.55) &
        (left_x > -width * 0.08) & (right_x < width * 1.08) &
        (left_x < right_x - width * 0.04)
    )
    # Perspective should keep narrowing toward the horizon (near â†’ far).
    running_min = np.minimum.accumulate(lane_w)
    still_narrowing = lane_w <= running_min + width * 0.01
    first_bad = int(np.argmin(still_narrowing)) if not np.all(still_narrowing) else len(still_narrowing)
    valid[first_bad:] = False
    # Also reject rows where the corridor centre drifts faster than a real curve.
    center_x = (left_x + right_x) * 0.5
    if len(center_x) > 2:
        step = np.abs(np.diff(center_x))
        jump = np.concatenate([[False], step > width * 0.045])
        if np.any(jump):
            valid[int(np.argmax(jump)):] = False
    keep = np.flatnonzero(valid)
    if len(keep) < 12:
        return left, right, center, int(round(evidence_top))
    # Keep the longest near-to-far contiguous valid run starting at the bottom.
    groups = np.split(keep, np.where(np.diff(keep) > 1)[0] + 1)
    group = groups[0] if groups and groups[0][0] == 0 else max(groups, key=len)
    if len(group) < 12:
        return left, right, center, int(round(evidence_top))
    left_out = np.column_stack((left_x[group], sample_y[group])).round().astype(np.int32)
    right_out = np.column_stack((right_x[group], sample_y[group])).round().astype(np.int32)
    center_out = np.column_stack((center_x[group], sample_y[group])).round().astype(np.int32)
    return left_out, right_out, center_out, int(round(evidence_top))


def _clamp_corridor_width(left: np.ndarray, right: np.ndarray, center: np.ndarray,
                          width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Force the painted corridor to a single-lane envelope around the centre."""
    if len(center) < 2:
        return left, right, center
    left_x = left[:, 0].astype(np.float64)
    right_x = right[:, 0].astype(np.float64)
    center_x = center[:, 0].astype(np.float64)
    ys = center[:, 1].astype(np.float64)
    half = (right_x - left_x) * 0.5
    # Near the camera a full lane is roughly 0.12-0.22 of frame width half-span;
    # far away perspective shrinks it. Cap aggressively to stop multi-lane bleed.
    perspective = np.clip((ys / max(1.0, float(np.max(ys)))) , 0.0, 1.0)
    max_half = width * (0.055 + 0.12 * perspective)
    min_half = width * (0.028 + 0.04 * perspective)
    half = np.clip(half, min_half, max_half)
    # Re-centre on the measured centre so clamping cannot shift into a neighbour.
    left_out = np.column_stack((center_x - half, ys)).round().astype(np.int32)
    right_out = np.column_stack((center_x + half, ys)).round().astype(np.int32)
    center_out = np.column_stack((center_x, ys)).round().astype(np.int32)
    return left_out, right_out, center_out


def _extended_roi_top(horizon_ratio: float, default_top: float) -> float:
    """Raise the perspective-warp ROI's top edge toward the calibrated horizon.

    There is normally no real lane/road evidence between the model's actual
    detection ceiling and the horizon (verified empirically: raw YOLOPv2
    output is exactly zero there). Widening the ROI here doesn't add real
    fitting data; it only widens the birdseye canvas so an already-fitted
    curve can be evaluated (extrapolated) further toward the horizon
    instead of hard-stopping at the old, more conservative edge.
    """
    return float(np.clip(min(default_top, horizon_ratio - 0.015), 0.30, default_top))


def estimate_road_geometry(frame: np.ndarray, horizon_ratio: float = 0.52) -> RoadGeometry:
    """Estimate curved lane boundaries and simple junction evidence from one camera frame.

    This is a visual estimate, not a calibrated road map or driving input.
    """
    height, width = frame.shape[:2]
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    light, saturation, hue = hls[:, :, 1], hls[:, :, 2], hls[:, :, 0]
    white = (light >= 165) & (saturation <= 175)
    yellow = (hue >= 14) & (hue <= 45) & (saturation >= 55) & (light >= 95)
    binary = np.zeros((height, width), dtype=np.uint8)
    binary[white | yellow] = 255
    roi_top = _extended_roi_top(horizon_ratio, 0.56)
    source = np.float32([(width * 0.10, height * 0.98), (width * 0.43, height * roi_top),
                         (width * 0.57, height * roi_top), (width * 0.90, height * 0.98)])
    destination = np.float32([(width * 0.18, height), (width * 0.18, 0),
                              (width * 0.82, 0), (width * 0.82, height)])
    matrix = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)
    birdseye = cv2.warpPerspective(binary, matrix, (width, height), flags=cv2.INTER_LINEAR)
    # Edge evidence preserves faded white paint that does not meet the colour threshold.
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0), 45, 145)
    birdseye = cv2.bitwise_or(birdseye, cv2.warpPerspective(edges, matrix, (width, height)))
    # Stop bars, crosswalk stripes, shadows, and vehicle bumpers can dominate a
    # row histogram and drag a lane fit sideways. Remove only unusually dense
    # horizontal rows from lane fitting; retain the original evidence for the
    # independent intersection-candidate check below.
    lane_evidence = birdseye.copy()
    dense_rows = (np.count_nonzero(lane_evidence, axis=1) > width * 0.14).astype(np.uint8)
    dense_rows = cv2.dilate(dense_rows[:, None], np.ones((9, 1), dtype=np.uint8)).reshape(-1).astype(bool)
    lane_evidence[dense_rows, :] = 0
    left_fits = _fit_lane_curves(lane_evidence, "left")
    right_fits = _fit_lane_curves(lane_evidence, "right")
    paired = _select_ego_lane_pair(left_fits, right_fits, inverse, width, height)
    left_points = right_points = center_points = None
    confidence = 0.0
    confident_top_y = None
    if paired is not None:
        left_points, right_points, center_points, confidence, confident_top_y = paired
        left_points, right_points, center_points = _clamp_corridor_width(
            left_points, right_points, center_points, width,
        )
        left_points, right_points, center_points, confident_top_y = _extend_lane_to_horizon(
            left_points, right_points, center_points, width, height,
            horizon_ratio, confident_top_y,
        )
        left_points, right_points, center_points = _clamp_corridor_width(
            left_points, right_points, center_points, width,
        )
    # This remains an unconfirmed candidate until the temporal tracker sees it
    # repeatedly. Fail closed: ordinary horizontal scenery must not make a road.
    intersection_y = _intersection_candidate_y(edges, binary, left_points, right_points, confidence)
    topology = _lane_topology(center_points, width)
    return RoadGeometry(left_points, right_points, center_points, confidence, topology,
                        intersection_y, width, height, confident_top_y=confident_top_y)


def estimate_road_geometry_from_masks(lane_probability: np.ndarray,
                                      drivable_mask: np.ndarray,
                                      horizon_ratio: float = 0.52) -> RoadGeometry:
    """Fit the ego lane from YOLOPv2 lane pixels and retain its drivable surface."""
    height, width = lane_probability.shape[:2]
    probability = np.clip(lane_probability.astype(np.float32), 0.0, 1.0)
    drivable = np.asarray(drivable_mask > 0, dtype=np.uint8)
    # Keep more mid/far paint: zeroing too much of the upper image was making
    # fitted corridors stop short and look unconfident on open highways.
    paint_cut = int(height * max(0.34, horizon_ratio - 0.12))
    lower_roi = probability[paint_cut:]
    strong_level = float(np.percentile(lower_roi, 99.2)) if lower_roi.size else 0.0
    lane_threshold = float(np.clip(strong_level * 0.48, 0.14, 0.30))
    binary = np.asarray(probability >= lane_threshold, dtype=np.uint8) * 255
    binary[:paint_cut, :] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))
    roi_top = _extended_roi_top(horizon_ratio, 0.52)
    source = np.float32([(width * 0.08, height * 0.99), (width * 0.43, height * roi_top),
                         (width * 0.57, height * roi_top), (width * 0.92, height * 0.99)])
    destination = np.float32([(width * 0.16, height), (width * 0.16, 0),
                              (width * 0.84, 0), (width * 0.84, height)])
    matrix = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)
    birdseye = cv2.warpPerspective(binary, matrix, (width, height), flags=cv2.INTER_NEAREST)
    birdseye = cv2.morphologyEx(birdseye, cv2.MORPH_CLOSE, np.ones((11, 3), np.uint8))
    dense_rows = (np.count_nonzero(birdseye, axis=1) > width * 0.20).astype(np.uint8)
    dense_rows = cv2.dilate(dense_rows[:, None], np.ones((7, 1), dtype=np.uint8)).reshape(-1).astype(bool)
    birdseye[dense_rows, :] = 0
    left_fits = _fit_lane_curves(birdseye, "left", 3)
    right_fits = _fit_lane_curves(birdseye, "right", 3)
    paired = _select_ego_lane_pair(left_fits, right_fits, inverse, width, height)
    inferred_boundary = False
    expected_warped_width = width * 0.28
    if paired is None and left_fits:
        inferred_boundary = True
        inferred_right: list[tuple[np.ndarray, float, float]] = []
        for coefficients, support, top_y in left_fits:
            shifted = coefficients.copy()
            shifted[-1] += expected_warped_width
            inferred_right.append((shifted, support * 0.55, top_y))
        paired = _select_ego_lane_pair(left_fits, inferred_right, inverse, width, height)
    if paired is None and right_fits:
        inferred_boundary = True
        inferred_left: list[tuple[np.ndarray, float, float]] = []
        for coefficients, support, top_y in right_fits:
            shifted = coefficients.copy()
            shifted[-1] -= expected_warped_width
            inferred_left.append((shifted, support * 0.55, top_y))
        paired = _select_ego_lane_pair(inferred_left, right_fits, inverse, width, height)
    left_points = right_points = center_points = None
    confidence = 0.0
    confident_top_y = None
    if paired is not None:
        left_points, right_points, center_points, confidence, confident_top_y = paired
        if inferred_boundary:
            confidence *= 0.62
    source_name = "yolopv2"
    if paired is None:
        inferred_corridor = _corridor_from_drivable(drivable, horizon_ratio)
        if inferred_corridor is not None:
            left_points, right_points, center_points, confidence = inferred_corridor
            source_name = "yolopv2-drivable"
            # No paint evidence supports this corridor at all (it's inferred
            # purely from the drivable-area mask shape); render all of it in
            # the honest "projected" style rather than claiming any of it is
            # a confident direct detection.
            confident_top_y = int(center_points[0, 1]) if center_points is not None else None
    if left_points is not None and right_points is not None and center_points is not None:
        left_points, right_points, center_points = _clamp_corridor_width(
            left_points, right_points, center_points, width,
        )
        left_points, right_points, center_points, confident_top_y = _extend_lane_to_horizon(
            left_points, right_points, center_points, width, height,
            horizon_ratio, confident_top_y,
        )
        # Re-clamp after extension so far-field poly errors cannot re-bloat.
        left_points, right_points, center_points = _clamp_corridor_width(
            left_points, right_points, center_points, width,
        )
    topology = _lane_topology(center_points, width)
    return RoadGeometry(
        left_points, right_points, center_points, confidence, topology,
        None, width, height, drivable, probability, source_name, confident_top_y,
    )


class RoadGeometryTracker:
    """Smooths lane curves across frames and holds a recent good road estimate briefly."""

    def __init__(self, analysis_width: int = 640) -> None:
        self._last: RoadGeometry | None = None
        self._anchor: RoadGeometry | None = None  # last model-backed geometry
        self._misses = 0
        self._intersection_votes: Deque[bool] = deque(maxlen=5)
        self._intersection_rows: Deque[int | None] = deque(maxlen=5)
        self._analysis_width = max(480, analysis_width)
        self._last_update_time = 0.0
        self._intersection_tick = 0
        # update() runs on the async road worker; predict() runs on the render
        # thread between model results. Both touch _last/_anchor.
        self._lock = threading.Lock()

    @staticmethod
    def _resample_curve(points: np.ndarray, sample_y: np.ndarray) -> np.ndarray | None:
        """Interpolate a polyline onto a shared nearâ†’far image-row grid."""
        if points is None or len(points) < 2:
            return None
        order = np.argsort(points[:, 1].astype(np.float32))
        ys = points[order, 1].astype(np.float32)
        xs = points[order, 0].astype(np.float32)
        unique_y, unique_idx = np.unique(ys, return_index=True)
        if len(unique_y) < 2:
            return None
        xs = xs[unique_idx]
        # Clamp query rows to the measured span so we never invent lateral jumps.
        y_lo, y_hi = float(unique_y[0]), float(unique_y[-1])
        query = np.clip(sample_y, y_lo, y_hi)
        return np.column_stack((np.interp(query, unique_y, xs), sample_y)).astype(np.float32)

    @classmethod
    def _blend_points(cls, previous: np.ndarray | None, current: np.ndarray | None,
                      prev_weight: float = 0.38) -> np.ndarray | None:
        """Y-aligned blend so curvature can track without looking static or rubbery.

        Older code stacked the first N point indices (0.76/0.24). When fits
        resampled differently each frame that lagged the corridor in place and
        fought live road motion -- the "stuck overlay" look.
        """
        if current is None:
            return previous
        if previous is None or len(previous) < 6 or len(current) < 6:
            return current
        bottom = float(min(np.max(previous[:, 1]), np.max(current[:, 1])))
        top = float(max(np.min(previous[:, 1]), np.min(current[:, 1])))
        if bottom - top < 12.0:
            return current
        sample_y = np.linspace(bottom, top, 32, dtype=np.float32)
        prev_r = cls._resample_curve(previous, sample_y)
        curr_r = cls._resample_curve(current, sample_y)
        if prev_r is None or curr_r is None:
            return current
        w = float(np.clip(prev_weight, 0.0, 0.85))
        blended = prev_r * w + curr_r * (1.0 - w)
        return np.rint(blended).astype(np.int32)

    def predict(self, dt: float | None = None) -> RoadGeometry | None:
        """Display-only advance from the last model-backed geometry (non-accumulating).

        A dashcam moves forward continuously; without this the path freezes for
        150-300 ms between YOLOPv2 results and reads as a static sticker.
        Warping is always computed from `_anchor` (last real update), never from
        a previously predicted frame, so long gaps cannot compound distortion.
        """
        with self._lock:
            return self._predict_locked(dt)

    def _predict_locked(self, dt: float | None = None) -> RoadGeometry | None:
        base = self._anchor if self._anchor is not None else self._last
        if base is None or base.center_points is None:
            return None
        if dt is None:
            dt = time.perf_counter() - self._last_update_time if self._last_update_time else 0.03
        # Cap so a stalled model cannot stretch the corridor into nonsense.
        dt = float(np.clip(dt, 0.0, 0.22))
        if dt < 0.012:
            return self._last

        def warp(points: np.ndarray | None) -> np.ndarray | None:
            if points is None or len(points) < 4:
                return points
            pts = points.astype(np.float32).copy()
            height = max(1.0, float(base.frame_height))
            # Near rows (larger y) shift down; far rows nearly stationary.
            depth = np.clip((pts[:, 1] / height - 0.35) / 0.60, 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1] + depth * depth * (2.8 * dt / 0.05), 0.0, height - 1.0)
            center_x = float(base.frame_width) * 0.5
            pts[:, 0] = center_x + (pts[:, 0] - center_x) * (1.0 + depth * dt * 0.45)
            return np.rint(pts).astype(np.int32)

        predicted = RoadGeometry(
            warp(base.left_points),
            warp(base.right_points),
            warp(base.center_points),
            base.confidence * float(np.clip(1.0 - dt * 0.15, 0.85, 1.0)),
            base.topology,
            base.intersection_y,
            base.frame_width,
            base.frame_height,
            base.drivable_mask,
            base.lane_probability,
            base.source,
            base.confident_top_y,
        )
        self._last = predicted
        return predicted

    def update(self, frame: np.ndarray, lane_probability: np.ndarray | None = None,
               drivable_mask: np.ndarray | None = None, horizon_ratio: float = 0.52) -> RoadGeometry:
        height, width = frame.shape[:2]
        scale = min(1.0, self._analysis_width / max(1, width))
        analysis_frame = frame
        analysis_lane = lane_probability
        analysis_drivable = drivable_mask
        if scale < 1.0:
            analysis_frame = cv2.resize(
                frame, (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            analysis_size = (analysis_frame.shape[1], analysis_frame.shape[0])
            if lane_probability is not None:
                analysis_lane = cv2.resize(lane_probability, analysis_size, interpolation=cv2.INTER_LINEAR)
            if drivable_mask is not None:
                analysis_drivable = cv2.resize(drivable_mask, analysis_size, interpolation=cv2.INTER_NEAREST)
        if analysis_lane is not None and analysis_drivable is not None:
            current = estimate_road_geometry_from_masks(analysis_lane, analysis_drivable, horizon_ratio)
            # Canny + Hough for stop-lines is expensive; only every 3rd road update.
            self._intersection_tick += 1
            if (current.center_points is not None and current.source == "yolopv2" and
                    self._intersection_tick % 3 == 0):
                hls = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2HLS)
                light, saturation, hue = hls[:, :, 1], hls[:, :, 2], hls[:, :, 0]
                paint = np.asarray(
                    ((light >= 165) & (saturation <= 175)) |
                    ((hue >= 14) & (hue <= 45) & (saturation >= 55) & (light >= 95)),
                    dtype=np.uint8,
                ) * 255
                edges = cv2.Canny(
                    cv2.GaussianBlur(cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                    45, 145,
                )
                candidate_y = _intersection_candidate_y(
                    edges, paint, current.left_points, current.right_points, current.confidence,
                )
                current.intersection_y = candidate_y
        else:
            current = estimate_road_geometry(analysis_frame, horizon_ratio)
        if scale < 1.0:
            def restore(points: np.ndarray | None) -> np.ndarray | None:
                return None if points is None else np.rint(points.astype(np.float32) / scale).astype(np.int32)

            current = RoadGeometry(
                restore(current.left_points),
                restore(current.right_points),
                restore(current.center_points),
                current.confidence,
                current.topology,
                None if current.intersection_y is None else int(round(current.intersection_y / scale)),
                width,
                height,
                None if drivable_mask is None else np.asarray(drivable_mask > 0, dtype=np.uint8),
                lane_probability,
                current.source,
                None if current.confident_top_y is None else int(round(current.confident_top_y / scale)),
            )
        with self._lock:
            if current.center_points is not None:
                self._misses = 0
                if self._last is not None:
                    # Prefer the live fit so curvature and lane shifts track in
                    # real time; light previous weight only kills high-frequency jitter.
                    current = RoadGeometry(
                        self._blend_points(self._last.left_points, current.left_points, 0.34),
                        self._blend_points(self._last.right_points, current.right_points, 0.34),
                        self._blend_points(self._last.center_points, current.center_points, 0.34),
                        current.confidence,
                        current.topology,
                        current.intersection_y,
                        width,
                        height,
                        current.drivable_mask,
                        current.lane_probability,
                        current.source,
                        current.confident_top_y,
                    )
                self._last_update_time = time.perf_counter()
            else:
                self._misses += 1
                hold_limit = 12 if current.source.startswith("yolopv2") else 7
                if self._last is not None and self._misses <= hold_limit:
                    decay = 0.94 if current.source.startswith("yolopv2") else 0.82
                    current = RoadGeometry(
                        self._last.left_points,
                        self._last.right_points,
                        self._last.center_points,
                        self._last.confidence * decay,
                        _lane_topology(self._last.center_points, frame.shape[1]),
                        None,
                        width,
                        height,
                        current.drivable_mask if current.drivable_mask is not None else self._last.drivable_mask,
                        current.lane_probability if current.lane_probability is not None else self._last.lane_probability,
                        current.source,
                        self._last.confident_top_y,
                    )
            candidate = current.center_points is not None and current.intersection_y is not None
            self._intersection_votes.append(candidate)
            self._intersection_rows.append(current.intersection_y if candidate else None)
            confirmed = len(self._intersection_votes) >= 4 and sum(self._intersection_votes) >= 4
            if confirmed:
                rows = [row for row in self._intersection_rows if row is not None]
                current = RoadGeometry(
                    current.left_points, current.right_points, current.center_points,
                    current.confidence, "intersection", int(round(float(np.median(rows)))), width, height,
                    current.drivable_mask, current.lane_probability, current.source, current.confident_top_y,
                )
            else:
                current = RoadGeometry(
                    current.left_points, current.right_points, current.center_points,
                    current.confidence, _lane_topology(current.center_points, frame.shape[1]), None, width, height,
                    current.drivable_mask, current.lane_probability, current.source, current.confident_top_y,
                )
            if current.center_points is not None:
                self._last = current
                # Only model-backed (or held) fits become the prediction anchor --
                # never a pure display warp.
                if self._misses == 0 or self._anchor is None:
                    self._anchor = current
                    self._last_update_time = time.perf_counter()
            return current


def objects_from_yolo_result(result, frame_width: int, focal_px: float,
                             estimator: MotionEstimator, now: float,
                             frame_height: int = 0, camera_height_m: float = 1.25,
                             horizon_ratio: float = 0.52) -> list[DetectedObject]:
    """Convert YOLO detection or segmentation output into the visualizer's tracked world state."""
    objects: list[DetectedObject] = []
    if result.boxes is None or len(result.boxes) == 0:
        return objects
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    # ByteTrack can drop IDs on a frame; still surface detections with synthetic ids
    # so lights/signs/cars don't vanish from the visualizer.
    if result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)
    else:
        ids = np.arange(900000, 900000 + len(boxes), dtype=int)
    polygons = result.masks.xy if result.masks is not None else []
    for index, ((x1, y1, x2, y2), track_id, class_id, confidence) in enumerate(zip(boxes, ids, classes, confidences)):
        label = result.names[int(class_id)]
        if label not in RELEVANT_CLASSES:
            continue
        # Fragile classes (person / lights / signs) keep a lower floor so small
        # or distant detections still enter the tracker.
        if float(confidence) < CLASS_MIN_CONF.get(label, 0.18):
            continue
        # A dashboard-mounted camera often sees the host vehicle's hood. COCO
        # detectors can call that wide, shallow bottom strip a car; it is not a
        # road participant and must not become a one-metre world object.
        if (frame_height > 0 and label in VEHICLE_CLASSES and
                y1 > frame_height * 0.78 and y2 >= frame_height * 0.96 and
                (x2 - x1) > frame_width * 0.36):
            continue
        distance = estimate_monocular_distance_m(
            label, (x1, y1, x2, y2), focal_px, frame_height,
            camera_height_m, horizon_ratio,
        )
        # Bearing from box centre; vehicles also bias slightly toward the foot
        # row for more ego-relative lateral (side of car facing camera).
        cx = (x1 + x2) * 0.5
        if label in ROAD_CLASSES and (y2 - y1) > 8:
            # Lower-third centre better represents ground contact laterally.
            cx = 0.65 * cx + 0.35 * ((x1 + x2) * 0.5)
        bearing = math.degrees(math.atan2(cx - frame_width / 2.0, focal_px))
        center = (cx, (y1 + y2) / 2.0)
        distance, closing, image_velocity = estimator.update(int(track_id), center, distance, now)
        polygon = None
        if index < len(polygons):
            polygon = np.rint(np.asarray(polygons[index])).astype(np.int32)
            if len(polygon) < 3:
                polygon = None
        objects.append(DetectedObject(int(track_id), label, float(confidence), (x1, y1, x2, y2),
                                      distance, bearing, closing, image_velocity, polygon))
    return objects


@dataclass
class _StableTrack:
    obj: DetectedObject
    hits: int
    misses: int
    last_update: float
    still_count: int = 0


# Consecutive observed updates a track must stay near-zero motion for before
# it is treated as stationary/parked rather than ordinary slow-moving traffic.
STATIONARY_HITS_REQUIRED = 5


class TrackStabilizer:
    """Display hysteresis over ByteTrack output to suppress flicker.

    Vehicles that were seen reliably are coasted longer (logical presence hold)
    so a few missed detections do not erase a car that is still in frame.
    """

    # Detection-cycle miss budgets (AsyncObjectPerception updates ~detect_interval).
    # Vehicles with enough hits: ~2s hold at ~7.5 Hz detection (interval 4 @ 30 FPS).
    VEHICLE_MAX_MISSING = 16
    VEHICLE_MAX_VISIBLE_MISS = 14
    VEHICLE_REBIND_MISS = 12
    DEFAULT_MAX_VISIBLE_MISS = 4

    def __init__(self, max_missing_updates: int = 9) -> None:
        self._tracks: dict[int, _StableTrack] = {}
        self._max_missing = max(1, max_missing_updates)
        self._last_prune: float = 0.0

    def _max_missing_for(self, state: _StableTrack) -> int:
        # Explicitly low caps (tests / special modes) stay strict.
        if self._max_missing < 9:
            return self._max_missing
        if state.obj.label in VEHICLE_CLASSES and state.hits >= 2:
            return max(self._max_missing, self.VEHICLE_MAX_MISSING)
        return self._max_missing

    def _max_visible_miss_for(self, state: _StableTrack) -> int:
        if self._max_missing < 9:
            return min(self.DEFAULT_MAX_VISIBLE_MISS, self._max_missing)
        if state.obj.label in VEHICLE_CLASSES and state.hits >= 2:
            return self.VEHICLE_MAX_VISIBLE_MISS
        return self.DEFAULT_MAX_VISIBLE_MISS

    @staticmethod
    def _is_still(obj: DetectedObject) -> bool:
        vx, vy = obj.image_velocity
        # Thresholds sit comfortably above MotionEstimator's own residual
        # noise floor (it already least-squares-smooths both signals), so
        # this reflects real lack of motion, not estimation jitter.
        return abs(obj.closing_mps) < 0.8 and math.hypot(vx, vy) < 15.0

    @staticmethod
    def _blend(previous: DetectedObject, current: DetectedObject) -> DetectedObject:
        old_box = np.asarray(previous.box, dtype=np.float32)
        new_box = np.asarray(current.box, dtype=np.float32)
        box = tuple(np.rint(old_box * 0.68 + new_box * 0.32).astype(int).tolist())
        velocity = (previous.image_velocity[0] * 0.68 + current.image_velocity[0] * 0.32,
                    previous.image_velocity[1] * 0.68 + current.image_velocity[1] * 0.32)
        stable_label = (previous.label if previous.label in VEHICLE_CLASSES and
                        current.label in VEHICLE_CLASSES else current.label)
        return DetectedObject(
            previous.track_id, stable_label,
            previous.confidence * 0.55 + current.confidence * 0.45,
            box,
            previous.distance_m * 0.70 + current.distance_m * 0.30,
            previous.bearing_deg * 0.70 + current.bearing_deg * 0.30,
            previous.closing_mps * 0.66 + current.closing_mps * 0.34,
            velocity, current.mask_polygon, True, 0, previous.stationary,
            current.signal_state or previous.signal_state,
            previous.track_quality,
        )

    @staticmethod
    def _coast_box(box: tuple[int, int, int, int], vx: float, vy: float,
                   dt: float) -> tuple[int, int, int, int]:
        """Shift a box by damped image velocity during a miss (cheap kinematics)."""
        # Cap so a noisy velocity cannot fling the coasted car off-screen.
        dx = int(round(float(np.clip(vx * dt, -28.0, 28.0))))
        dy = int(round(float(np.clip(vy * dt, -18.0, 18.0))))
        if dx == 0 and dy == 0:
            return box
        x1, y1, x2, y2 = box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    @staticmethod
    def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
        second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
        return intersection / max(1.0, first_area + second_area - intersection)

    def _rebind_candidate(self, observation: DetectedObject, excluded: set[int]) -> int | None:
        """Reconnect a recently fragmented ByteTrack ID without merging nearby vehicles."""
        best_id, best_score = None, 0.0
        ox = (observation.box[0] + observation.box[2]) * 0.5
        oy = (observation.box[1] + observation.box[3]) * 0.5
        ow = max(1.0, observation.box[2] - observation.box[0])
        oh = max(1.0, observation.box[3] - observation.box[1])
        diagonal = max(20.0, math.hypot(observation.box[2] - observation.box[0],
                                       observation.box[3] - observation.box[1]))
        is_vehicle = observation.label in VEHICLE_CLASSES
        for track_id, state in self._tracks.items():
            compatible_label = (state.obj.label == observation.label or
                                (state.obj.label in VEHICLE_CLASSES and is_vehicle))
            max_rebind_miss = self.VEHICLE_REBIND_MISS if (
                is_vehicle and state.obj.label in VEHICLE_CLASSES
            ) else 3
            if track_id in excluded or not compatible_label or state.misses > max_rebind_miss:
                continue
            iou = self._box_iou(state.obj.box, observation.box)
            sx = (state.obj.box[0] + state.obj.box[2]) * 0.5
            sy = (state.obj.box[1] + state.obj.box[3]) * 0.5
            sw = max(1.0, state.obj.box[2] - state.obj.box[0])
            sh = max(1.0, state.obj.box[3] - state.obj.box[1])
            centre_similarity = max(0.0, 1.0 - math.hypot(ox - sx, oy - sy) / diagonal)
            range_similarity = max(0.0, 1.0 - abs(state.obj.distance_m - observation.distance_m) /
                                   max(5.0, state.obj.distance_m * 0.45))
            # A genuinely fragmented same-object id keeps roughly its own
            # apparent box size across a brief gap; two distinct objects that
            # happen to overlap in position rarely also share both box
            # dimensions, so this helps avoid merging them.
            size_similarity = (min(sw, ow) / max(sw, ow)) * (min(sh, oh) / max(sh, oh))
            score = (iou * 0.52 + centre_similarity * 0.20 +
                    range_similarity * 0.13 + size_similarity * 0.15)
            # Slightly lower IoU gate for vehicles during long coast rebind.
            iou_gate = 0.18 if (is_vehicle and state.misses >= 3) else 0.26
            if iou >= iou_gate and score > best_score:
                best_id, best_score = track_id, score
        return best_id if best_score >= 0.36 else None

    def update(self, observations: list[DetectedObject], now: float) -> list[DetectedObject]:
        seen: set[int] = set()
        for observation in observations:
            state = self._tracks.get(observation.track_id)
            if state is None:
                old_id = self._rebind_candidate(observation, seen)
                if old_id is not None:
                    state = self._tracks.pop(old_id)
                    self._tracks[observation.track_id] = state
            seen.add(observation.track_id)
            compatible_label = (state is not None and
                                (state.obj.label == observation.label or
                                 (state.obj.label in VEHICLE_CLASSES and observation.label in VEHICLE_CLASSES)))
            if state is None or not compatible_label:
                self._tracks[observation.track_id] = _StableTrack(observation, 1, 0, now)
                continue
            state.obj = self._blend(state.obj, observation)
            state.still_count = state.still_count + 1 if self._is_still(state.obj) else 0
            state.obj.stationary = state.still_count >= STATIONARY_HITS_REQUIRED
            state.hits += 1
            state.misses = 0
            state.last_update = now

        expired: list[int] = []
        for track_id, state in self._tracks.items():
            if track_id in seen:
                continue
            state.misses += 1
            if state.misses > self._max_missing_for(state):
                expired.append(track_id)
                continue
            dt = float(np.clip(now - state.last_update, 0.0, 0.45))
            vx, vy = state.obj.image_velocity
            # Damped coast: keep the car "logically present" while the detector
            # flickers. Vehicles get a slower confidence decay so quality holds.
            is_vehicle = state.obj.label in VEHICLE_CLASSES
            conf_decay = 0.92 if is_vehicle else 0.84
            vel_decay = 0.55 if is_vehicle else 0.35
            coast_box = self._coast_box(state.obj.box, vx, vy, dt) if is_vehicle else state.obj.box
            state.obj = DetectedObject(
                state.obj.track_id, state.obj.label, state.obj.confidence * conf_decay,
                coast_box,
                float(np.clip(state.obj.distance_m - state.obj.closing_mps * dt, 1.0, 150.0)),
                state.obj.bearing_deg, state.obj.closing_mps * 0.94,
                (vx * vel_decay, vy * vel_decay), None, False, state.misses, state.obj.stationary,
                state.obj.signal_state, state.obj.track_quality,
            )
            state.last_update = now
        for track_id in expired:
            del self._tracks[track_id]

        # Hard cap on stabilizer size — ByteTrack ID churn used to leave hundreds
        # of dead coasting tracks after ~30s, which made NMS + quality scoring hitch.
        if len(self._tracks) > 36:
            ranked = sorted(
                self._tracks.items(),
                key=lambda item: (item[1].misses, -item[1].hits, -item[1].last_update),
            )
            for track_id, _state in ranked[36:]:
                del self._tracks[track_id]

        visible = []
        for state in self._tracks.values():
            if state.misses > self._max_visible_miss_for(state):
                continue
            # Fragile classes appear after one solid hit so people/lights/signs
            # are not delayed by the vehicle-oriented two-hit hysteresis.
            fragile = state.obj.label in FRAGILE_CLASSES
            floor = CLASS_MIN_CONF.get(state.obj.label, 0.14)
            if (state.hits >= 2 or state.obj.confidence >= 0.72 or
                    (fragile and state.hits >= 1 and state.obj.confidence >= floor)):
                state.obj.track_quality = track_quality_score(
                    state.obj, hits=state.hits, misses=state.misses,
                )
                visible.append(state.obj)
        # Spatial NMS: one real vehicle must not produce two stacked display
        # tracks (common when ByteTrack renumbers and the prior ID still coasts).
        if len(visible) > 1:
            visible = sorted(
                visible,
                key=lambda item: (0 if item.observed else 1, -item.confidence, item.missed_updates),
            )
            kept: list[DetectedObject] = []
            for candidate in visible:
                duplicate = False
                for existing in kept:
                    compatible = (
                        existing.label == candidate.label or
                        (existing.label in VEHICLE_CLASSES and candidate.label in VEHICLE_CLASSES)
                    )
                    if not compatible:
                        continue
                    if self._box_iou(existing.box, candidate.box) >= 0.32:
                        duplicate = True
                        break
                    ex = (existing.box[0] + existing.box[2]) * 0.5
                    ey = (existing.box[1] + existing.box[3]) * 0.5
                    cx = (candidate.box[0] + candidate.box[2]) * 0.5
                    cy = (candidate.box[1] + candidate.box[3]) * 0.5
                    scale = max(20.0, 0.40 * max(
                        existing.box[2] - existing.box[0], existing.box[3] - existing.box[1],
                        candidate.box[2] - candidate.box[0], candidate.box[3] - candidate.box[1],
                    ))
                    if (math.hypot(ex - cx, ey - cy) < scale and
                            abs(existing.distance_m - candidate.distance_m) <
                            max(3.5, existing.distance_m * 0.18)):
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(candidate)
            visible = kept
        return sorted(visible, key=lambda item: item.distance_m)


def track_quality_score(obj: DetectedObject, *, hits: int | None = None,
                        misses: int | None = None) -> float:
    """Lightweight 0..1 reliability score for mesh selection and lead highlight.

    Combines detector confidence with track age (hits) and coast/miss penalty.
    Cheap — no extra vision work.
    """
    conf = float(np.clip(obj.confidence, 0.0, 1.0))
    hit_n = float(hits if hits is not None else (4 if obj.observed else 1))
    miss_n = float(misses if misses is not None else obj.missed_updates)
    q = 0.40 * conf + 0.30 * min(1.0, hit_n / 6.0)
    if obj.observed:
        q += 0.22
    else:
        q -= 0.10 * min(miss_n, 4.0)
    q -= 0.06 * min(miss_n, 3.0)
    if obj.stationary:
        q += 0.04  # parked stills are usually stable boxes
    return float(np.clip(q, 0.05, 1.0))


def estimate_merge_flag(obj: DetectedObject, lateral_m: float = 0.0) -> bool:
    """True when a vehicle looks like a merge / cut-in (logic only, no new model).

    Cues: side-ish silhouette, adjacent-lane offset, image motion toward ego,
    moderate relative range. Used for HUD / 3D accent — not for mesh yaw.
    """
    if obj.label not in VEHICLE_CLASSES | {"motorcycle"}:
        return False
    if obj.stationary:
        return False
    x1, y1, x2, y2 = obj.box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    aspect = bw / bh
    side_ness = float(np.clip((aspect - 1.05) / 1.55, 0.0, 1.0))
    abs_lat = abs(float(lateral_m))
    # Prefer bearing-based lateral when monocular road lateral is unavailable.
    if abs_lat < 0.15:
        abs_lat = abs(float(obj.distance_m) * math.tan(math.radians(obj.bearing_deg)))
    closing = float(obj.closing_mps)
    vx, _vy = obj.image_velocity
    # Image motion toward optical centre (cut-in).
    toward_ego = (obj.bearing_deg > 4.0 and vx < -12.0) or (obj.bearing_deg < -4.0 and vx > 12.0)
    side_band = 1.25 <= abs_lat <= 5.8
    # Side silhouette or clear lateral offset in adjacent band.
    side_like = side_ness >= 0.42 or abs_lat >= 1.8
    # Not a hard brake-oncoming blast; merges are moderate relative rate.
    mild_close = -1.0 <= closing <= 5.5
    if not (side_band and side_like and mild_close):
        return False
    if toward_ego:
        return True
    # Without clear vx, still flag adjacent side-on traffic that is near.
    return side_ness >= 0.50 and abs_lat >= 1.6 and obj.distance_m < 38.0


def _semantic_color(label: str) -> tuple[int, int, int]:
    """High-contrast BGR colours for the camera-perception overlay."""
    if label in {"car", "truck", "bus", "train"}:
        return (255, 205, 85)
    if label in {"person", "bicycle", "motorcycle"}:
        return (80, 235, 250)
    if label == "traffic light":
        return (80, 80, 255)
    if "sign" in label:
        return (210, 130, 245)
    return (105, 230, 125)


def _smooth_road_polyline(points: np.ndarray, count: int = 40) -> np.ndarray:
    """Resample a vision polyline on shared image rows without inventing wiggles.

    Degree-3 fits previously bowed straight highways. Prefer light smoothing
    (at most quadratic) so the display tracks YOLOPv2 geometry faithfully.
    """
    order = np.argsort(points[:, 1])[::-1]
    source = points[order].astype(np.float32)
    unique_y, indices = np.unique(source[:, 1], return_index=True)
    source = source[indices]
    if len(source) < 3:
        return points.astype(np.int32)
    ys = np.linspace(float(np.max(source[:, 1])), float(np.min(source[:, 1])), count)
    # Linear when nearly straight; quadratic only when residual justifies it.
    degree = 1
    if len(source) >= 6:
        lin = np.polyfit(source[:, 1], source[:, 0], 1)
        residual = float(np.std(source[:, 0] - np.polyval(lin, source[:, 1])))
        span = max(8.0, float(np.ptp(source[:, 0])))
        if residual > span * 0.04:
            degree = 2
    coefficients = np.polyfit(source[:, 1], source[:, 0], min(degree, len(source) - 1))
    xs = np.polyval(coefficients, ys)
    return np.column_stack((xs, ys)).round().astype(np.int32)


def _confidence_split_index(y_values: np.ndarray, confident_top_y: int | None) -> int:
    """Index in a near-to-far ordered curve (`y_values` descending) where
    real detected evidence ends and honest extrapolation begins."""
    if confident_top_y is None:
        return 0
    below = np.flatnonzero(y_values >= confident_top_y)
    return int(below[-1]) if len(below) else 0


def _depth_weight_map(height: int, width: int, horizon_ratio: float = 0.42,
                      hood_ratio: float = 0.93) -> np.ndarray:
    """Perspective falloff so overlays read as lying on the ground plane.

    Bottom of the image (near road) is strongest; weight fades toward the
    horizon and is zero on the dashboard/hood band.
    """
    ys = np.arange(height, dtype=np.float32)[:, None]
    top = height * float(np.clip(horizon_ratio, 0.25, 0.60))
    hood = height * float(np.clip(hood_ratio, 0.85, 0.98))
    weight = (ys - top) / max(1.0, hood - top)
    weight = np.clip(weight, 0.0, 1.0)
    # Ease so far road is soft and near paint is solid.
    weight = weight ** 1.15
    weight[ys[:, 0] >= hood] = 0.0
    return np.repeat(weight, width, axis=1)


def _ground_range_from_image_y(image_y: float, height: int, focal_px: float,
                               camera_height_m: float = 1.25,
                               horizon_ratio: float = 0.52) -> float:
    """Flat-road range (metres) from an image row using the camera pinhole."""
    horizon_y = float(np.clip(horizon_ratio, 0.30, 0.72) * height)
    ground_pixels = float(image_y) - horizon_y
    if ground_pixels < height * 0.02:
        return 85.0
    return float(np.clip(camera_height_m * focal_px / ground_pixels, 2.0, 90.0))


# EMA of ego-lane triangle image corners: (bl_x, bl_y, tip_x, tip_y, br_x, br_y).
_GUIDE_EMA: np.ndarray | None = None
_GUIDE_EMA_SIZE: tuple[int, int] | None = None
# Last committed path style for confidence tint / hold-fade (camera + 3D).
_GUIDE_SOURCE: str = "none"   # lanes | curbs | asphalt | hold | none
_GUIDE_HOLD_MISS: int = 0
_GUIDE_ALPHA: float = 0.0

# Cached road-marking scan (lightweight classical CV on paint).
_MARKING_CACHE: list[RoadMarking] = []
_MARKING_CACHE_AGE: int = 99
_MARKING_CACHE_SHAPE: tuple[int, int] | None = None

# Per-class detector floors (YOLO conf is a global floor; we re-filter here).
# Lights/signs are intentionally lower — they are small and often low-score.
CLASS_MIN_CONF = {
    "person": 0.09,
    "bicycle": 0.11,
    "motorcycle": 0.13,
    "car": 0.15,
    "bus": 0.15,
    "truck": 0.15,
    "train": 0.18,
    "traffic light": 0.06,
    "stop sign": 0.07,
    "parking meter": 0.10,
    "fire hydrant": 0.10,
}


def _project_ground_point(x_m: float, z_m: float, width: int, height: int,
                          focal_px: float, camera_height_m: float,
                          horizon_ratio: float) -> tuple[float, float]:
    """Pinhole project a ground-plane point (X right, Z forward) into the image."""
    z_m = max(0.8, float(z_m))
    u = (width * 0.5) + focal_px * (x_m / z_m)
    horizon_y = float(np.clip(horizon_ratio, 0.30, 0.72) * height)
    v = horizon_y + focal_px * (camera_height_m / z_m)
    return float(u), float(v)


def _vision_lane_x_at(road_geometry: RoadGeometry, y: float, which: str) -> float | None:
    """Sample left/right/center vision polyline at image row y."""
    if which == "left" and road_geometry.left_points is not None:
        return _lane_x_at_y(road_geometry.left_points.astype(np.float32), y)
    if which == "right" and road_geometry.right_points is not None:
        return _lane_x_at_y(road_geometry.right_points.astype(np.float32), y)
    if which == "center":
        if road_geometry.center_points is not None:
            cx = _lane_x_at_y(road_geometry.center_points.astype(np.float32), y)
            if cx is not None:
                return cx
        lx = _vision_lane_x_at(road_geometry, y, "left")
        rx = _vision_lane_x_at(road_geometry, y, "right")
        if lx is not None and rx is not None:
            return 0.5 * (lx + rx)
    return None


def _mask_corridor_at_y(mask: np.ndarray, y: float, width: int,
                        prefer_x: float | None = None
                        ) -> tuple[float, float, float] | None:
    """Left/right curb (or asphalt edge) and centre at image row y from a binary mask."""
    height = mask.shape[0]
    yi = int(np.clip(round(y), 0, height - 1))
    # Thick row band for stability on sparse masks.
    y0, y1 = max(0, yi - 2), min(height, yi + 3)
    band = np.any(mask[y0:y1] > 0, axis=0)
    cols = np.flatnonzero(band)
    if len(cols) < 6:
        return None
    # Prefer the run that contains the optical axis / previous centre (ego road).
    anchor = width * 0.5 if prefer_x is None else float(prefer_x)
    groups = np.split(cols, np.where(np.diff(cols) > 2)[0] + 1)
    runs = [g for g in groups if len(g) >= max(6, width // 80)]
    if not runs:
        return None
    run = min(runs, key=lambda g: 0.0 if g[0] <= anchor <= g[-1]
              else min(abs(anchor - g[0]), abs(anchor - g[-1])))
    left, right = float(run[0]), float(run[-1])
    if right - left < width * 0.06:
        return None
    return left, right, 0.5 * (left + right)


# Cached asphalt mask (full-res) so unmarked-road guidance stays cheap.
_ASPHALT_CACHE_MASK: np.ndarray | None = None
_ASPHALT_CACHE_SHAPE: tuple[int, int] | None = None
_ASPHALT_CACHE_AGE: int = 0
_PATH_HOLD: tuple[np.ndarray, np.ndarray, float | None, float | None, str] | None = None
_PATH_HOLD_MISS: int = 0

_PATH_SAMPLE_ROWS = 7


def _asphalt_mask(frame: np.ndarray) -> np.ndarray | None:
    """Cheap asphalt/road-surface mask when painted lanes are absent.

    Runs on a downscaled FOV (lower half only) then upsamples — fast enough
    for the display path while still finding unmarked roads / faded paint.
    """
    global _ASPHALT_CACHE_MASK, _ASPHALT_CACHE_SHAPE, _ASPHALT_CACHE_AGE
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    # Reuse mask for a few frames — unmarked asphalt rarely changes every tick.
    if (_ASPHALT_CACHE_MASK is not None and _ASPHALT_CACHE_SHAPE == (h, w) and
            _ASPHALT_CACHE_AGE < 4):
        _ASPHALT_CACHE_AGE += 1
        return _ASPHALT_CACHE_MASK

    # Work small: only the lower ~60% of the frame at half width.
    y0 = int(h * 0.40)
    crop = frame[y0:, :]
    scale = min(1.0, 320.0 / max(1, crop.shape[1]))
    if scale < 0.99:
        small = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = crop
    hls = cv2.cvtColor(small, cv2.COLOR_BGR2HLS)
    light, sat, hue = hls[:, :, 1], hls[:, :, 2], hls[:, :, 0]
    # Asphalt: mid-dark, low chroma. Also accept worn/light grey pavement.
    asphalt = ((light >= 22) & (light <= 160) & (sat <= 78))
    # Reject strong green (verge) and strong blue (sky reflections / water).
    asphalt &= ~((hue >= 35) & (hue <= 95) & (sat > 40))
    mask_s = asphalt.astype(np.uint8) * 255
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8), iterations=1)
    mask_s = cv2.morphologyEx(mask_s, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # Largest blob that touches bottom-centre of the small crop.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_s > 0).astype(np.uint8), 8,
    )
    if count <= 1:
        _ASPHALT_CACHE_MASK = None
        _ASPHALT_CACHE_SHAPE = (h, w)
        _ASPHALT_CACHE_AGE = 0
        return None
    sh, sw = mask_s.shape
    best, best_score = 0, -1.0
    for i in range(1, count):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < sh * sw * 0.03:
            continue
        bottom = labels[int(sh * 0.78):, int(sw * 0.28):int(sw * 0.72)]
        touch = float(np.count_nonzero(bottom == i))
        score = area + touch * 50.0
        if score > best_score:
            best, best_score = i, score
    if best == 0:
        return None
    small_out = np.zeros_like(mask_s)
    small_out[labels == best] = 255
    # Upsample into a full-frame mask (zeros above y0).
    full = np.zeros((h, w), dtype=np.uint8)
    up = cv2.resize(small_out, (w, h - y0), interpolation=cv2.INTER_NEAREST)
    full[y0:] = up
    _ASPHALT_CACHE_MASK = full
    _ASPHALT_CACHE_SHAPE = (h, w)
    _ASPHALT_CACHE_AGE = 0
    return full


def _path_is_ego_corridor(xs: np.ndarray, width: int, source: str) -> bool:
    """Reject paths that leave the ego lane / wander to side roads or aisles."""
    if xs is None or len(xs) < 2:
        return False
    mid = width * 0.5
    near = float(xs[0])
    far = float(xs[-1])
    # Near base under camera (ego lane).
    max_near = width * (0.24 if source == "lanes" else 0.20)
    if abs(near - mid) > max_near:
        return False
    # Far tip may bank with real curvature.
    max_far = width * (0.42 if source == "lanes" else 0.34)
    if abs(far - mid) > max_far:
        return False
    # Only reject extreme zig-zag (wrong mask run), not gentle curves.
    if len(xs) >= 3:
        jumps = np.abs(np.diff(xs.astype(np.float64)))
        if float(np.max(jumps)) > width * 0.22:
            return False
    return True


def _accept_or_hold_path(
    candidate: tuple[np.ndarray, np.ndarray, float | None, float | None, str] | None,
    width: int,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None, str] | None:
    """Commit a live ego path; hold only on flicker / impossible jumps."""
    global _PATH_HOLD, _PATH_HOLD_MISS
    if candidate is not None:
        ys, xs, l_near, r_near, source = candidate
        if _path_is_ego_corridor(xs, width, source):
            if _PATH_HOLD is not None:
                prev_xs = _PATH_HOLD[1]
                near_jump = abs(float(xs[0]) - float(prev_xs[0]))
                far_jump = abs(float(xs[-1]) - float(prev_xs[-1]))
                # Only block *teleports* to another corridor — allow live curve follow.
                if near_jump > width * 0.18 or far_jump > width * 0.28:
                    if _PATH_HOLD_MISS < 6:
                        _PATH_HOLD_MISS += 1
                        return _PATH_HOLD
            _PATH_HOLD = (ys, xs, l_near, r_near, source)
            _PATH_HOLD_MISS = 0
            return _PATH_HOLD
    if _PATH_HOLD is not None and _PATH_HOLD_MISS < 8:
        _PATH_HOLD_MISS += 1
        return _PATH_HOLD
    return None


def _ego_path_samples(road_geometry: RoadGeometry | None, width: int, height: int,
                      horizon_ratio: float, frame_bgr: np.ndarray | None
                      ) -> tuple[np.ndarray, np.ndarray, float | None, float | None, str] | None:
    """Sample the ego path centre at several image rows (near -> far).

    Priority:
      1. Painted / fitted lane polylines from YOLOPv2 / geometry tracker
      2. Drivable-area / curb edges from the learned road mask (no paint)
      3. Asphalt colour mask under the camera (unmarked / faded roads)
      4. Hold last good path briefly when support flickers

    Returns (ys, centre_xs, left_near, right_near, source).
    """
    y_near = height * 0.86
    y_far = height * max(0.42, float(np.clip(horizon_ratio, 0.30, 0.65)) + 0.04)
    y_far = min(y_far, y_near - height * 0.14)
    ego_x = width * 0.5
    if _PATH_HOLD is not None:
        ego_x = float(_PATH_HOLD[1][0]) * 0.40 + width * 0.5 * 0.60

    # --- 1) Painted / fitted lane geometry ---------------------------------
    if (road_geometry is not None and road_geometry.center_points is not None and
            len(road_geometry.center_points) >= 4):
        cy = road_geometry.center_points[:, 1].astype(np.float32)
        y_near = float(np.clip(np.percentile(cy, 92), height * 0.68, height * 0.92))
        y_far = float(np.clip(np.percentile(cy, 12), height * 0.34, y_near - height * 0.12))
        ys = np.linspace(y_near, y_far, _PATH_SAMPLE_ROWS)
        xs = [_vision_lane_x_at(road_geometry, float(y), "center") for y in ys]
        valid = [i for i, x in enumerate(xs) if x is not None]
        if len(valid) >= 3 and 0 in valid:
            xs_arr = np.interp(
                ys, ys[valid][::-1],
                np.asarray([xs[i] for i in valid], dtype=np.float64)[::-1],
            )
            # Light ego lock only near the base — far tip follows real road curve.
            t = np.linspace(0.0, 1.0, len(xs_arr))
            xs_arr = xs_arr * (0.88 + 0.12 * t) + (width * 0.5) * (0.12 * (1.0 - t))
            if len(xs_arr) >= 4:
                sm = xs_arr.copy()
                sm[1:-1] = 0.22 * xs_arr[:-2] + 0.56 * xs_arr[1:-1] + 0.22 * xs_arr[2:]
                xs_arr = sm
            l_near = _vision_lane_x_at(road_geometry, y_near, "left")
            r_near = _vision_lane_x_at(road_geometry, y_near, "right")
            accepted = _accept_or_hold_path((ys, xs_arr, l_near, r_near, "lanes"), width)
            if accepted is not None:
                return accepted

    def _sample_mask(mask: np.ndarray, source: str
                     ) -> tuple[np.ndarray, np.ndarray, float | None, float | None, str] | None:
        ys = np.linspace(y_near, y_far, _PATH_SAMPLE_ROWS)
        centres: list[float | None] = []
        l_near = r_near = None
        prefer: float | None = ego_x
        for index, y in enumerate(ys):
            corridor = _mask_corridor_at_y(mask, float(y), width, prefer)
            if corridor is None:
                centres.append(None)
                continue
            left, right, centre = corridor
            if index == 0 and not (left - width * 0.05 <= width * 0.5 <= right + width * 0.05):
                if centre < width * 0.24 or centre > width * 0.76:
                    centres.append(None)
                    continue
            if index == 0:
                l_near, r_near = left, right
            # Mild ego pull near only; far samples keep mask centre (road shape).
            pull = 0.22 if index <= 1 else 0.08
            centre = (1.0 - pull) * centre + pull * (prefer if prefer is not None else width * 0.5)
            centres.append(centre)
            prefer = 0.55 * centre + 0.45 * (prefer if prefer is not None else centre)
        valid = [i for i, c in enumerate(centres) if c is not None]
        if len(valid) < 3 or 0 not in valid:
            return None
        xs_arr = np.interp(
            ys, ys[valid][::-1],
            np.asarray([centres[i] for i in valid], dtype=np.float64)[::-1],
        )
        if len(xs_arr) >= 4:
            sm = xs_arr.copy()
            sm[1:-1] = 0.22 * xs_arr[:-2] + 0.56 * xs_arr[1:-1] + 0.22 * xs_arr[2:]
            xs_arr = sm
        return ys, xs_arr, l_near, r_near, source

    # --- 2) Drivable mask / curb edges when paint is missing ---------------
    if road_geometry is not None and road_geometry.drivable_mask is not None:
        mask = np.asarray(road_geometry.drivable_mask > 0, dtype=np.uint8)
        if mask.shape[0] == height and mask.shape[1] == width:
            if not np.any(mask[int(height * 0.75):int(height * 0.92),
                               int(width * 0.35):int(width * 0.65)]):
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8), iterations=1)
            accepted = _accept_or_hold_path(_sample_mask(mask, "curbs"), width)
            if accepted is not None:
                return accepted

    # --- 3) Asphalt visual fallback (unmarked / faded / dirt-ish roads) ----
    if frame_bgr is not None:
        asphalt = _asphalt_mask(frame_bgr)
        if asphalt is not None and asphalt.shape[0] == height:
            accepted = _accept_or_hold_path(_sample_mask(asphalt, "asphalt"), width)
            if accepted is not None:
                return accepted

    # --- 4) Hold last good path when support flickers -----------------------
    return _accept_or_hold_path(None, width)


def ego_path_style() -> tuple[str, float, tuple[int, int, int], tuple[int, int, int]]:
    """Return (source, alpha, fill_bgr, edge_bgr) for the ego path triangle.

    Confidence tint:
      lanes   — solid gold (high confidence painted corridor)
      curbs   — softer amber (drivable mask)
      asphalt — cooler gold (colour fallback)
      hold    — fading while replaying last path
      none    — invisible / searching
    """
    source = _GUIDE_SOURCE
    alpha = float(np.clip(_GUIDE_ALPHA, 0.0, 1.0))
    if source == "lanes":
        fill, edge = (40, 210, 255), (20, 240, 255)
    elif source == "curbs":
        fill, edge = (55, 185, 235), (40, 210, 245)
    elif source == "asphalt":
        fill, edge = (70, 155, 200), (60, 175, 220)
    elif source == "hold":
        fill, edge = (50, 170, 210), (40, 190, 230)
    else:
        fill, edge = (40, 210, 255), (20, 240, 255)
        alpha = 0.0
    return source, alpha, fill, edge


def build_ego_lane_triangle(
    road_geometry: RoadGeometry | None,
    width: int,
    height: int,
    *,
    horizon_ratio: float = 0.52,
    smooth: float = 0.38,
    frame_bgr: np.ndarray | None = None,
) -> np.ndarray | None:
    """Live yellow guidance triangle pointing along the projected ego path.

    Base sits on the path near the camera; tip tracks the fitted road ahead in
    real time (curves, bends) with light EMA so it is active but not frantic.
    Updates path confidence metadata for tint / hold-fade.
    """
    global _GUIDE_EMA, _GUIDE_EMA_SIZE, _GUIDE_SOURCE, _GUIDE_HOLD_MISS, _GUIDE_ALPHA
    sampled = _ego_path_samples(road_geometry, width, height, horizon_ratio, frame_bgr)
    if sampled is None:
        # Fade out last triangle rather than hard-cut when path support dies.
        if _GUIDE_EMA is not None and _GUIDE_ALPHA > 0.08:
            _GUIDE_SOURCE = "hold"
            _GUIDE_HOLD_MISS = min(20, _GUIDE_HOLD_MISS + 1)
            _GUIDE_ALPHA = max(0.0, _GUIDE_ALPHA * 0.82)
            bl_x, bl_y, tip_x, tip_y, br_x, br_y = (float(v) for v in _GUIDE_EMA)
            return np.array([
                [np.clip(bl_x, 2, width - 3), np.clip(bl_y, 2, height - 3)],
                [np.clip(tip_x, 2, width - 3), np.clip(tip_y, 2, height - 3)],
                [np.clip(br_x, 2, width - 3), np.clip(br_y, 2, height - 3)],
            ], dtype=np.float32)
        _GUIDE_SOURCE = "none"
        _GUIDE_ALPHA = 0.0
        return None
    ys, xs, l_near, r_near, source = sampled
    # Live sample vs held path: hold-fade when replaying last good corridor.
    is_hold = _PATH_HOLD_MISS > 0
    _GUIDE_SOURCE = "hold" if is_hold else source
    _GUIDE_HOLD_MISS = int(_PATH_HOLD_MISS)
    # Slightly softer alpha so a compact triangle does not dominate the image.
    if is_hold:
        _GUIDE_ALPHA = float(np.clip(0.70 - 0.07 * _PATH_HOLD_MISS, 0.14, 0.55))
    elif source == "lanes":
        _GUIDE_ALPHA = 0.42
    elif source == "curbs":
        _GUIDE_ALPHA = 0.36
    else:
        _GUIDE_ALPHA = 0.30

    if l_near is not None and r_near is not None and r_near > l_near + 8:
        span = r_near - l_near
        half = 0.10 * span if source in {"curbs", "asphalt"} else 0.13 * span
    else:
        half = width * 0.022
    # Narrower base so the guide stays compact and less obstructive.
    half = float(np.clip(half, width * 0.010, width * 0.028))

    y_near = float(ys[0])
    y_far_path = float(ys[-1])
    # Cap triangle height (~16% of frame) so the tip does not climb into the
    # mid/horizon view and block traffic / road ahead.
    max_span = height * 0.16
    min_span = height * 0.07
    path_span = max(0.0, y_near - y_far_path)
    span = float(np.clip(path_span, min_span, max_span))
    y_far = y_near - span
    if y_near - y_far < min_span * 0.9:
        return None

    # Quadratic path fit so the tip banks with real road curvature.
    degree = 2 if len(ys) >= 5 else 1
    path_fit = np.polyfit(ys, xs, degree)
    c_near = float(np.polyval(path_fit, y_near))
    c_far = float(np.polyval(path_fit, y_far))
    c_mid = float(np.polyval(path_fit, 0.45 * y_near + 0.55 * y_far))
    tangent = float(np.polyval(np.polyder(path_fit), y_far))  # dx/dy
    # Tip follows the path at the compact far row (still points along trajectory).
    tip_x = 0.55 * c_far + 0.30 * c_mid + 0.15 * (c_far + tangent * (y_far - y_near) * 0.25)
    cone = half * 2.8 + width * 0.04
    tip_x = float(np.clip(tip_x, c_near - cone, c_near + cone))
    # Soft ego lock on base only (allow lane offset when geometry says so).
    c_near = float(np.clip(c_near, width * 0.34, width * 0.66))
    bl_x = c_near - half
    br_x = c_near + half
    if l_near is not None and r_near is not None:
        bl_x = max(bl_x, l_near + 2.0)
        br_x = min(br_x, r_near - 2.0)
        if br_x <= bl_x + 4:
            bl_x, br_x = c_near - half, c_near + half

    raw = np.array([bl_x, y_near, tip_x, y_far, br_x, y_near], dtype=np.float64)
    size = (width, height)
    if (_GUIDE_EMA is None or _GUIDE_EMA_SIZE != size or
            len(_GUIDE_EMA) != len(raw)):
        _GUIDE_EMA = raw.copy()
        _GUIDE_EMA_SIZE = size
    else:
        a = float(np.clip(smooth, 0.05, 0.9))
        # Responsive follow: tip can move several percent of width per frame.
        delta = raw - _GUIDE_EMA
        max_step = np.array([
            width * 0.028, height * 0.020,
            width * 0.040, height * 0.022,
            width * 0.028, height * 0.020,
        ], dtype=np.float64)
        delta = np.clip(delta, -max_step, max_step)
        _GUIDE_EMA = _GUIDE_EMA + a * delta

    bl_x, bl_y, tip_x, tip_y, br_x, br_y = (float(v) for v in _GUIDE_EMA)
    pts = np.array([
        [np.clip(bl_x, 2, width - 3), np.clip(bl_y, 2, height - 3)],
        [np.clip(tip_x, 2, width - 3), np.clip(tip_y, 2, height - 3)],
        [np.clip(br_x, 2, width - 3), np.clip(br_y, 2, height - 3)],
    ], dtype=np.float32)
    area = 0.5 * abs(
        pts[0, 0] * (pts[1, 1] - pts[2, 1]) +
        pts[1, 0] * (pts[2, 1] - pts[0, 1]) +
        pts[2, 0] * (pts[0, 1] - pts[1, 1])
    )
    if area < width * height * 0.0004:
        return None
    return pts


def draw_ego_lane_triangle(canvas: np.ndarray, road_geometry: RoadGeometry | None,
                           *, fov_deg: float = 70.0, camera_height_m: float = 1.25,
                           horizon_ratio: float = 0.52,
                           frame_bgr: np.ndarray | None = None,
                           lite: bool = False) -> bool:
    """Draw a simple yellow ego-path triangle that points along the road trajectory.

    Geometry is one triangle only:
      - base near the camera (ego lane width)
      - tip further ahead on the fitted path (curves with the road)

    ``lite`` uses a cheaper blend for the split-view camera pane.
    """
    # fov_deg / camera_height_m kept for API compatibility with callers.
    _ = (fov_deg, camera_height_m)
    height, width = canvas.shape[:2]
    tri = build_ego_lane_triangle(
        road_geometry, width, height,
        horizon_ratio=horizon_ratio, frame_bgr=frame_bgr,
    )
    if tri is None or len(tri) != 3:
        return False
    source, alpha, fill, edge = ego_path_style()
    if alpha < 0.06:
        return False
    pts = np.rint(tri).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    pad = 2
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + bw + pad)
    y1 = min(height, y + bh + pad)
    if x1 > x0 and y1 > y0:
        roi = canvas[y0:y1, x0:x1]
        overlay = roi.copy()
        local = pts.copy()
        local[:, 0] -= x0
        local[:, 1] -= y0
        line_type = cv2.LINE_8 if lite else cv2.LINE_AA
        cv2.fillConvexPoly(overlay, local, fill, line_type)
        # Slightly lower alpha in lite for cheaper / cleaner look.
        blend = float(np.clip(alpha * (0.85 if lite else 1.0), 0.10, 0.65))
        cv2.addWeighted(overlay, blend, roi, 1.0 - blend, 0, roi)
    edge_w = 1 if lite else (2 if source in {"lanes", "curbs"} else 1)
    cv2.polylines(
        canvas, [pts.reshape(-1, 1, 2)], True, edge, edge_w,
        cv2.LINE_8 if lite else cv2.LINE_AA,
    )
    return True


def estimate_traffic_light_state(frame_bgr: np.ndarray, box: tuple[int, int, int, int]
                                 ) -> str | None:
    """Lightweight lamp colour from a traffic-light ROI (red/yellow/green)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 8:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    # Work on a tiny crop — cheap and stable.
    small = cv2.resize(roi, (24, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bright = (val >= 90) & (sat >= 50)
    if int(np.count_nonzero(bright)) < 8:
        return None
    # OpenCV H: red wraps 0/180, yellow ~15-35, green ~40-90.
    red = bright & ((hue <= 8) | (hue >= 170))
    yellow = bright & (hue >= 12) & (hue <= 38)
    green = bright & (hue >= 40) & (hue <= 95)
    counts = {
        "red": int(np.count_nonzero(red)),
        "yellow": int(np.count_nonzero(yellow)),
        "green": int(np.count_nonzero(green)),
    }
    best = max(counts, key=counts.get)
    if counts[best] < 6:
        return None
    return best


def annotate_signal_states(frame_bgr: np.ndarray, objects: list[DetectedObject]
                           ) -> list[DetectedObject]:
    """Attach traffic-light lamp colour to detected light tracks (in place)."""
    if frame_bgr is None:
        return objects
    for obj in objects:
        if obj.label != "traffic light" or not obj.observed:
            continue
        state = estimate_traffic_light_state(frame_bgr, obj.box)
        if state is not None:
            obj.signal_state = state
    return objects


def scan_road_markings(frame_bgr: np.ndarray | None, *, every_n: int = 4,
                       max_markings: int = 5) -> list[RoadMarking]:
    """Scan road paint for crosswalks / lane paint (3D visualizer only).

    Classical CV on a downscaled lower FOV. Not drawn on the camera pane —
    results feed the 3D world. Stop-line/arrow classes removed (false hits).
    """
    global _MARKING_CACHE, _MARKING_CACHE_AGE, _MARKING_CACHE_SHAPE
    if frame_bgr is None or frame_bgr.size == 0:
        return list(_MARKING_CACHE)
    h, w = frame_bgr.shape[:2]
    shape = (h, w)
    if (_MARKING_CACHE_SHAPE == shape and _MARKING_CACHE_AGE < every_n and
            _MARKING_CACHE):
        _MARKING_CACHE_AGE += 1
        return list(_MARKING_CACHE)

    # Lower FOV only (road surface); small working resolution.
    y0 = int(h * 0.48)
    crop = frame_bgr[y0:h]
    ch, cw = crop.shape[:2]
    scale = 320.0 / max(1, cw)
    sw, sh = 320, max(48, int(round(ch * scale)))
    small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    white = ((val >= 145) & (sat <= 70)).astype(np.uint8) * 255
    yellow = ((hue >= 14) & (hue <= 42) & (sat >= 55) & (val >= 100)).astype(np.uint8) * 255
    paint = cv2.bitwise_or(white, yellow)
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((2, 3), np.uint8), iterations=1)
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8), iterations=1)
    # Focus on centre corridor so sidewalk paint is ignored.
    x_lo, x_hi = int(sw * 0.12), int(sw * 0.88)
    paint[:, :x_lo] = 0
    paint[:, x_hi:] = 0

    contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[RoadMarking] = []
    inv_sx = w / float(sw)
    inv_sy = ch / float(sh)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 40 or area > sw * sh * 0.35:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 8 or bh < 3:
            continue
        aspect = bw / max(1.0, float(bh))
        # Map back to full-frame coords.
        fx1 = int(x * inv_sx)
        fy1 = int(y0 + y * inv_sy)
        fx2 = int((x + bw) * inv_sx)
        fy2 = int(y0 + (y + bh) * inv_sy)
        cx = 0.5 * (fx1 + fx2)
        cy = 0.5 * (fy1 + fy2)
        # Colour sample at contour centroid in small image.
        mask_c = np.zeros((sh, sw), dtype=np.uint8)
        cv2.drawContours(mask_c, [cnt], -1, 255, -1)
        y_pix = yellow[mask_c > 0]
        colour = "yellow" if y_pix.size and float(np.mean(y_pix > 0)) > 0.35 else "white"
        kind = None
        conf = 0.45
        # Stop lines / arrows removed: too many false hits on kerbs, shadows,
        # and random straight edges. Keep only stricter crosswalk + lane paint.
        # Crosswalk: several-stripe proxy — needs width + mid-road placement.
        if (aspect >= 2.4 and bh <= sh * 0.16 and bw >= sw * 0.16
                and y > sh * 0.30 and abs((x + bw * 0.5) - sw * 0.5) < sw * 0.22):
            kind = "crosswalk"
            conf = float(np.clip(0.40 + area / (sw * sh * 0.08), 0.40, 0.88))
        # Long thin strip roughly along depth → lane paint fragment.
        elif aspect <= 0.45 and bh >= sh * 0.16 and bw <= sw * 0.08:
            kind = "lane_paint"
            conf = 0.42
        if kind is None:
            continue
        # Prefer markings near image centre (ego road).
        if abs(cx - w * 0.5) > w * 0.32:
            continue
        found.append(RoadMarking(kind, conf, (fx1, fy1, fx2, fy2), (cx, cy), colour))

    # Keep best few (crosswalks first).
    priority = {"crosswalk": 0, "lane_paint": 1}
    found.sort(key=lambda m: (priority.get(m.kind, 9), -m.confidence, abs(m.centroid[0] - w * 0.5)))
    # De-dupe near-identical boxes.
    kept: list[RoadMarking] = []
    for m in found:
        if any(abs(m.centroid[0] - k.centroid[0]) < 28 and abs(m.centroid[1] - k.centroid[1]) < 22
               for k in kept):
            continue
        kept.append(m)
        if len(kept) >= max_markings:
            break
    _MARKING_CACHE = kept
    _MARKING_CACHE_AGE = 0
    _MARKING_CACHE_SHAPE = shape
    return list(kept)


def latest_road_markings() -> list[RoadMarking]:
    """Return the most recent road-marking scan (may be empty)."""
    return list(_MARKING_CACHE)


def draw_camera_view(frame: np.ndarray, objects: list[DetectedObject], show_lanes: bool,
                     lane_segments: list[tuple[int, int, int, int]] | None = None,
                     fps: float | None = None, detection_fps: float | None = None,
                     road_geometry: RoadGeometry | None = None,
                     horizon_ratio: float = 0.52, camera_height_m: float = 1.25,
                     fov_deg: float = 70.0,
                     stage_ms: dict[str, float] | None = None,
                     annotate_signals: bool = True,
                     env_summary: str | None = None,
                     controlling_light_id: int | None = None,
                     lite: bool = False,
                     lead_track_id: int | None = None) -> np.ndarray:
    """Camera footage with the perception state overlaid; no controls or actuation.

    Guidance is a live yellow ego-lane triangle plus lightweight traffic-sign
    annotations and optional environment-intel summary line.
    Set annotate_signals=False when the caller already ran annotate_signal_states
    (avoids double ROI work on the hot path).
    ``lite=True`` is for the split-view webcam pane: outline path, thin boxes,
    shorter labels — cheap enough to rebuild every display frame.
    Pass ``lead_track_id`` from the sticky same-lane picker when available so the
    camera LEAD tag matches the 3D highlight (no independent thrash).
    """
    # Bind camera args to plain locals up front. Do not `del` parameter names
    # (that historically caused UnboundLocalError on later keyword uses).
    _ = lane_segments  # classical Hough path removed; keep signature
    cam_height = float(camera_height_m)
    cam_fov = float(fov_deg)
    cam_horizon = float(horizon_ratio)
    # Split/lite overlays already own a resized buffer — skip the extra copy.
    if lite and frame.flags["OWNDATA"] and frame.flags["WRITEABLE"]:
        canvas = frame
    else:
        canvas = frame.copy()
    height, width = canvas.shape[:2]
    line_type = cv2.LINE_AA if not lite else cv2.LINE_8

    # Signal colours (skip if main loop already annotated this frame).
    if annotate_signals:
        annotate_signal_states(frame, objects)

    if show_lanes:
        drawn = draw_ego_lane_triangle(
            canvas, road_geometry,
            fov_deg=cam_fov,
            camera_height_m=cam_height,
            horizon_ratio=cam_horizon,
            frame_bgr=None if lite else frame,
            lite=lite,
        )
        if not lite:
            source, alpha, fill, _edge = ego_path_style()
            if drawn and alpha > 0.08:
                tag = {
                    "lanes": "EGO PATH  lanes",
                    "curbs": "EGO PATH  curbs",
                    "asphalt": "EGO PATH  asphalt",
                    "hold": "EGO PATH  hold",
                }.get(source, "EGO PATH")
                cv2.putText(canvas, tag, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, fill, 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "EGO PATH: SEARCHING",
                            (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 210, 255), 1, cv2.LINE_AA)

    # Prefer sticky same-lane LEAD from the 3D picker; fallback is closest observed.
    if lead_track_id is not None and lead_track_id >= 0:
        lead_id = int(lead_track_id)
    else:
        lead_id = -1
        vehicle_objs = [
            o for o in objects
            if o.label in VEHICLE_CLASSES and o.observed and abs(o.bearing_deg) < 12.0
        ]
        if vehicle_objs:
            lead_id = min(vehicle_objs, key=lambda o: o.distance_m).track_id

    merge_count = 0
    font_scale = 0.36 if lite else 0.42
    for obj in objects:
        x1, y1, x2, y2 = obj.box
        color = _semantic_color(obj.label)
        if not obj.observed:
            color = tuple(int(channel * max(0.30, 1.0 - obj.missed_updates * 0.16)) for channel in color)
        is_lead = obj.track_id == lead_id
        is_control_tl = (
            controlling_light_id is not None
            and obj.track_id == controlling_light_id
            and obj.label == "traffic light"
        )
        # Approximate lateral for merge logic when no road map is available.
        lat_approx = obj.distance_m * math.tan(math.radians(obj.bearing_deg))
        if road_geometry is not None and not lite:
            from_offset = road_lateral_offset_m((x1 + x2) * 0.5, float(y2), road_geometry)
            if from_offset is not None:
                lat_approx = from_offset
        is_merge = estimate_merge_flag(obj, lat_approx)
        if is_merge:
            merge_count += 1
            color = (40, 165, 255)  # orange-ish BGR
        if is_control_tl:
            color = (40, 40, 255) if (obj.signal_state == "red") else (
                (40, 200, 255) if obj.signal_state == "yellow" else (40, 220, 80)
            )
        if is_lead:
            color = (40, 190, 255)
        if lite:
            thickness = 2 if (is_lead or is_merge or is_control_tl) else 1
        else:
            thickness = 3 if (is_lead or is_merge or is_control_tl) else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, line_type)
        if lite:
            # Compact labels only for high-priority targets.
            if is_lead or is_merge or is_control_tl or obj.label in SIGNAL_CLASSES:
                tag = obj.label.upper()[:6]
                if is_lead:
                    tag = "LEAD"
                elif is_merge:
                    tag = "MERGE"
                elif is_control_tl:
                    tag = "EGO-TL"
                text = f"{tag} {obj.distance_m:.0f}m"
                cv2.putText(canvas, text, (x1 + 2, max(28, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, line_type)
        else:
            text = f"{obj.label.upper()} {obj.distance_m:.0f}m"
            if abs(obj.closing_mps) >= 0.8 and obj.label in VEHICLE_CLASSES | {"motorcycle"}:
                sign = "+" if obj.closing_mps > 0 else ""
                text += f" {sign}{obj.closing_mps:.0f}m/s"
            if is_lead:
                text = f"LEAD  {text}"
            if is_merge and not is_lead:
                text = f"MERGE  {text}"
            if is_control_tl:
                text = f"EGO-TL  {text}"
            if obj.label == "traffic light" and obj.signal_state:
                text += f" [{obj.signal_state.upper()}]"
            cv2.putText(canvas, text, (x1 + 3, max(40, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    bar_h = 22 if lite else 30
    cv2.rectangle(canvas, (0, 0), (width, bar_h), (9, 10, 14), -1)
    perf = f"{fps:.0f} FPS" if fps is not None else "-- FPS"
    if detection_fps is not None and not lite:
        perf += f"  DET {detection_fps:.0f}"
    extra = f"  M{merge_count}" if merge_count and lite else (f"  MERGE {merge_count}" if merge_count else "")
    header = f"CAM {perf} n={len(objects)}{extra}" if lite else f"CAMERA  |  {perf}  |  n={len(objects)}{extra}"
    cv2.putText(canvas, header, (8, 15 if lite else 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36 if lite else 0.42, (235, 240, 248), 1, line_type)
    # Environment intelligence strip (skip in lite — lives on 3D HUD in split).
    if env_summary and not lite:
        cv2.rectangle(canvas, (0, 30), (width, 52), (12, 14, 18), -1)
        cv2.putText(canvas, env_summary[:88], (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 220, 255), 1, cv2.LINE_AA)
    # Stage budget lives on the OpenGL HUD (top bar) — not redrawn here.
    _ = stage_ms
    return canvas


def _mix(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """Lighten or darken a BGR colour without allocating an image."""
    return tuple(int(np.clip(channel * amount, 0, 255)) for channel in color)


def _poly(panel: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], outline: tuple[int, int, int] | None = None) -> None:
    polygon = np.asarray(points, dtype=np.int32)
    cv2.fillConvexPoly(panel, polygon, color, cv2.LINE_AA)
    if outline is not None:
        cv2.polylines(panel, [polygon], True, outline, 1, cv2.LINE_AA)


def _world_position(obj: DetectedObject, width: int, height: int, horizon: int) -> tuple[int, int, float]:
    """Map coarse polar detections to the perspective scene used only for display."""
    distance = float(np.clip(obj.distance_m, 1.0, 85.0))
    proximity = float(np.clip(1.0 - distance / 88.0, 0.0, 1.0))
    # The exponent makes distant objects cluster near the horizon like the reference view.
    y = int(horizon + (height - horizon - 52) * proximity ** 1.52)
    lateral_m = distance * math.tan(math.radians(obj.bearing_deg))
    perspective = 0.34 + proximity * 0.86
    x = int(width / 2 + lateral_m * (width / 25.0) * perspective)
    return x, y, proximity


def _draw_vehicle(panel: np.ndarray, center: tuple[int, int], scale: float, kind: str, accent: tuple[int, int, int]) -> None:
    """Draw a lightweight, shaded 3D-style semantic model rather than a flat box."""
    x, y = center
    scale = max(4, int(round(scale)))
    neutral = (210, 218, 228) if accent == (216, 224, 236) else accent
    dark = _mix(neutral, 0.42)
    mid = _mix(neutral, 0.76)
    bright = _mix(neutral, 1.12)
    glass = (102, 135, 151)
    outline = (30, 32, 37)
    if kind in {"truck", "bus", "train"}:
        length = int(scale * (2.8 if kind == "truck" else 3.2))
        wide = int(scale * (1.22 if kind == "bus" else 1.05))
        _poly(panel, [(x - wide, y + scale // 2), (x + wide, y + scale // 2), (x + wide - scale // 3, y - length), (x - wide + scale // 3, y - length)], mid, outline)
        _poly(panel, [(x - wide + scale // 3, y - length), (x + wide - scale // 3, y - length), (x + wide - scale // 2, y - length - scale // 2), (x - wide + scale // 2, y - length - scale // 2)], bright, outline)
        if kind == "truck":
            cab_y = y + scale // 3
            _poly(panel, [(x - wide, cab_y), (x + wide, cab_y), (x + wide - scale // 4, cab_y - scale), (x - wide + scale // 4, cab_y - scale)], neutral, outline)
            cv2.line(panel, (x - wide + scale // 4, cab_y - scale), (x + wide - scale // 4, cab_y - scale), glass, max(1, scale // 7), cv2.LINE_AA)
        else:
            for offset in range(-1, 2):
                window_y = int(y - length * (0.32 + (offset + 1) * 0.18))
                cv2.line(panel, (x - wide + scale // 4, window_y), (x + wide - scale // 4, window_y), glass, max(1, scale // 7), cv2.LINE_AA)
        for wheel_x in (x - wide, x + wide):
            cv2.circle(panel, (wheel_x, y + scale // 3), max(2, scale // 4), (20, 20, 22), -1, cv2.LINE_AA)
        return
    if kind == "motorcycle":
        radius = max(3, int(scale * 0.28))
        cv2.circle(panel, (x - int(scale * 0.58), y), radius, (25, 25, 28), -1, cv2.LINE_AA)
        cv2.circle(panel, (x + int(scale * 0.58), y), radius, (25, 25, 28), -1, cv2.LINE_AA)
        cv2.line(panel, (x - int(scale * 0.58), y), (x, y - int(scale * 0.82)), mid, max(2, scale // 5), cv2.LINE_AA)
        cv2.line(panel, (x, y - int(scale * 0.82)), (x + int(scale * 0.58), y), bright, max(2, scale // 5), cv2.LINE_AA)
        return
    # Car, SUV, and generic road-vehicle model: body, roof, glass, lights, wheels.
    half_w = max(6, int(scale * 0.92))
    length = max(10, int(scale * 1.72))
    _poly(panel, [(x - half_w, y + scale // 2), (x + half_w, y + scale // 2),
                  (x + int(half_w * 0.75), y - length), (x - int(half_w * 0.75), y - length)], mid, outline)
    _poly(panel, [(x - int(half_w * 0.58), y - int(length * 0.16)), (x + int(half_w * 0.58), y - int(length * 0.16)),
                  (x + int(half_w * 0.42), y - int(length * 0.76)), (x - int(half_w * 0.42), y - int(length * 0.76))], glass, outline)
    _poly(panel, [(x - int(half_w * 0.72), y - length), (x + int(half_w * 0.72), y - length),
                  (x + int(half_w * 0.44), y - int(length * 1.12)), (x - int(half_w * 0.44), y - int(length * 1.12))], bright, outline)
    wheel_r = max(2, int(scale * 0.19))
    for wheel_x in (x - half_w, x + half_w):
        cv2.circle(panel, (wheel_x, y + int(scale * 0.22)), wheel_r, (18, 18, 20), -1, cv2.LINE_AA)
        cv2.circle(panel, (wheel_x, y + int(scale * 0.22)), max(1, wheel_r // 2), (115, 120, 128), -1, cv2.LINE_AA)
    cv2.line(panel, (x - int(half_w * 0.55), y + int(scale * 0.34)), (x - int(half_w * 0.16), y + int(scale * 0.34)), (45, 60, 255), max(1, scale // 9), cv2.LINE_AA)
    cv2.line(panel, (x + int(half_w * 0.16), y + int(scale * 0.34)), (x + int(half_w * 0.55), y + int(scale * 0.34)), (45, 60, 255), max(1, scale // 9), cv2.LINE_AA)


def _draw_person(panel: np.ndarray, center: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    x, y = center
    scale = max(4, int(round(scale)))
    head = max(2, int(scale * 0.22))
    body = max(7, int(scale * 1.22))
    cv2.circle(panel, (x, y - body), head, _mix(color, 1.12), -1, cv2.LINE_AA)
    cv2.line(panel, (x, y - body + head), (x, y - head), color, max(2, int(scale * 0.22)), cv2.LINE_AA)
    cv2.line(panel, (x, y - int(body * 0.65)), (x - int(scale * 0.42), y - int(body * 0.38)), color, max(1, int(scale * 0.15)), cv2.LINE_AA)
    cv2.line(panel, (x, y - int(body * 0.65)), (x + int(scale * 0.42), y - int(body * 0.38)), color, max(1, int(scale * 0.15)), cv2.LINE_AA)
    cv2.line(panel, (x, y - head), (x - int(scale * 0.35), y), color, max(1, int(scale * 0.16)), cv2.LINE_AA)
    cv2.line(panel, (x, y - head), (x + int(scale * 0.35), y), color, max(1, int(scale * 0.16)), cv2.LINE_AA)


def _draw_bicycle(panel: np.ndarray, center: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    x, y = center
    scale = max(4, int(round(scale)))
    radius = max(2, int(scale * 0.32))
    left, right, top = x - int(scale * 0.58), x + int(scale * 0.58), y - int(scale * 0.72)
    cv2.circle(panel, (left, y), radius, color, 1, cv2.LINE_AA)
    cv2.circle(panel, (right, y), radius, color, 1, cv2.LINE_AA)
    cv2.line(panel, (left, y), (x, top), color, max(1, scale // 8), cv2.LINE_AA)
    cv2.line(panel, (x, top), (right, y), color, max(1, scale // 8), cv2.LINE_AA)
    cv2.line(panel, (left, y), (right - int(scale * 0.22), y - int(scale * 0.12)), color, max(1, scale // 8), cv2.LINE_AA)


def _draw_signal_or_sign(panel: np.ndarray, center: tuple[int, int], scale: float, label: str) -> None:
    x, y = center
    scale = max(4, int(round(scale)))
    pole_top = y - int(scale * 1.85)
    cv2.line(panel, (x, y), (x, pole_top), (145, 150, 158), max(1, scale // 8), cv2.LINE_AA)
    if label == "traffic light":
        cv2.rectangle(panel, (x - int(scale * 0.32), pole_top), (x + int(scale * 0.32), pole_top + int(scale * 0.76)), (34, 36, 40), -1, cv2.LINE_AA)
        cv2.circle(panel, (x, pole_top + int(scale * 0.23)), max(2, scale // 7), (55, 55, 245), -1, cv2.LINE_AA)
        cv2.circle(panel, (x, pole_top + int(scale * 0.52)), max(1, scale // 8), (70, 200, 250), -1, cv2.LINE_AA)
    else:
        _poly(panel, [(x, pole_top - int(scale * 0.28)), (x + int(scale * 0.35), pole_top),
                      (x, pole_top + int(scale * 0.28)), (x - int(scale * 0.35), pole_top)], (70, 130, 240), (225, 225, 235))


def _draw_scene_grid(panel: np.ndarray, horizon: int) -> None:
    height, width = panel.shape[:2]
    road_left, road_right = int(width * 0.06), int(width * 0.94)
    road_top_left, road_top_right = int(width * 0.43), int(width * 0.57)
    cv2.rectangle(panel, (0, 0), (width, height), (10, 11, 14), -1)
    cv2.ellipse(panel, (width // 2, horizon), (int(width * 0.40), int(height * 0.18)), 0, 0, 360, (20, 22, 28), -1, cv2.LINE_AA)
    _poly(panel, [(road_top_left, horizon), (road_top_right, horizon), (road_right, height), (road_left, height)], (24, 27, 32))
    for fraction in np.linspace(0.08, 1.0, 13):
        y = int(horizon + (height - horizon) * fraction ** 1.78)
        left = int(road_top_left + (road_left - road_top_left) * fraction)
        right = int(road_top_right + (road_right - road_top_right) * fraction)
        cv2.line(panel, (left, y), (right, y), (45, 48, 56), 1, cv2.LINE_AA)
    for lane in (-2, -1, 1, 2):
        top_x = int(width * 0.5 + lane * width * 0.035)
        bottom_x = int(width * 0.5 + lane * width * 0.225)
        cv2.line(panel, (top_x, horizon), (bottom_x, height), (125, 130, 142), 1, cv2.LINE_AA)
    # Dashed lane guides establish the wide highway perspective of the reference image.
    for lane in (-1, 1):
        for fraction in np.linspace(0.15, 0.90, 7):
            next_fraction = min(1.0, fraction + 0.055)
            x1 = int(width * 0.5 + lane * width * (0.035 + 0.225 * fraction))
            x2 = int(width * 0.5 + lane * width * (0.035 + 0.225 * next_fraction))
            y1 = int(horizon + (height - horizon) * fraction ** 1.78)
            y2 = int(horizon + (height - horizon) * next_fraction ** 1.78)
            cv2.line(panel, (x1, y1), (x2, y2), (220, 222, 228), 2, cv2.LINE_AA)
    # Blue trajectory corridor, visual only.
    overlay = panel.copy()
    _poly(overlay, [(width // 2 - int(width * 0.025), height - 54), (width // 2 + int(width * 0.025), height - 54),
                    (width // 2 + int(width * 0.008), horizon + int(height * 0.12)), (width // 2 - int(width * 0.008), horizon + int(height * 0.12))], (235, 120, 35))
    cv2.addWeighted(overlay, 0.28, panel, 0.72, 0, panel)


def _draw_ego_vehicle(panel: np.ndarray) -> None:
    height, width = panel.shape[:2]
    x, y, scale = width // 2, height - 34, max(18, width // 38)
    _draw_vehicle(panel, (x, y), scale, "car", (225, 145, 55))
    cv2.ellipse(panel, (x, y + scale // 2), (int(scale * 1.15), max(3, scale // 4)), 0, 0, 360, (15, 15, 17), -1, cv2.LINE_AA)


def draw_world_view(objects: list[DetectedObject], size: tuple[int, int], fps: float, detection_fps: float | None = None) -> np.ndarray:
    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * 0.18)
    _draw_scene_grid(panel, horizon)
    closest_id = min(objects, key=lambda item: item.distance_m).track_id if objects else -1
    for obj in sorted(objects, key=lambda item: item.distance_m, reverse=True):
        x, y, proximity = _world_position(obj, width, height, horizon)
        if x < -120 or x > width + 120 or y < horizon - 40 or y > height + 40:
            continue
        scale = float(np.clip(7 + proximity * 30, 7, 37))
        selected = obj.track_id == closest_id
        accent = (225, 190, 80) if selected else (216, 224, 236)
        shadow_w = max(4, int(scale * 1.04))
        cv2.ellipse(panel, (x, y + int(scale * 0.36)), (shadow_w, max(2, int(scale * 0.23))), 0, 0, 360, (9, 10, 12), -1, cv2.LINE_AA)
        if obj.label == "person":
            _draw_person(panel, (x, y), scale, (238, 238, 242) if selected else (185, 193, 203))
        elif obj.label == "bicycle":
            _draw_bicycle(panel, (x, y), scale, (225, 202, 80) if selected else (185, 195, 210))
        elif obj.label in {"traffic light", "stop sign", "parking meter"}:
            _draw_signal_or_sign(panel, (x, y), scale, obj.label)
        elif obj.label in {"car", "motorcycle", "bus", "truck", "train"}:
            _draw_vehicle(panel, (x, y), scale, obj.label, accent)
        else:
            _poly(panel, [(x - int(scale * 0.6), y), (x + int(scale * 0.6), y),
                          (x + int(scale * 0.38), y - int(scale)), (x - int(scale * 0.38), y - int(scale))], (155, 168, 184), (35, 38, 44))
        label_y = max(horizon + 14, y - int(scale * 1.42))
        label_x = int(np.clip(x - int(scale * 1.4), 4, max(4, width - 145)))
        cv2.putText(panel, f"{obj.label.upper()}  {obj.distance_m:.0f}m", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33 + proximity * 0.10, (220, 226, 235), 1, cv2.LINE_AA)
        vx, _ = obj.image_velocity
        predicted_x = int(x + np.clip(vx * 0.28, -65, 65))
        cv2.line(panel, (x, y - int(scale * 0.45)), (predicted_x, max(horizon, y - int(scale * 2.2))), (130, 185, 235), 1, cv2.LINE_AA)
    _draw_ego_vehicle(panel)
    cv2.putText(panel, "VISIONFSD PILOT", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 241, 246), 2, cv2.LINE_AA)
    cv2.putText(panel, "READ-ONLY PERCEPTION VISUALIZER", (21, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (138, 153, 170), 1, cv2.LINE_AA)
    detection_text = f"DET {detection_fps:.1f} FPS" if detection_fps is not None else "DET --"
    cv2.putText(panel, f"DISPLAY {fps:.1f} FPS   {detection_text}   TRACKS {len(objects)}", (20, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (172, 184, 199), 1, cv2.LINE_AA)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only webcam driving-scene visualizer")
    parser.add_argument("--camera", type=int, default=0, help="Windows webcam index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Ultralytics model name or local .pt file")
    parser.add_argument("--confidence", type=float, default=0.20,
                        help="Detector floor; display hysteresis filters weak one-frame detections")
    parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference resolution; 512 is the CPU-balanced default")
    parser.add_argument("--detect-interval", type=int, default=2, help="Run detection every N captured frames (1 is maximum accuracy)")
    parser.add_argument("--cpu-threads", type=int, default=4, help="CPU threads reserved for PyTorch inference")
    parser.add_argument("--fov", type=float, default=70.0, help="Approximate webcam horizontal field of view")
    parser.add_argument("--camera-height", type=float, default=1.25,
                        help="Camera optical-centre height above road in metres")
    parser.add_argument("--horizon-ratio", type=float, default=0.52,
                        help="Road horizon row divided by image height; calibrate after mounting")
    parser.add_argument("--monitor", type=int, default=0, help="Zero-based Windows monitor for the app window")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--map-only", action="store_true", help="Start with only the synthetic world view")
    return parser.parse_args()


def position_window(window_name: str, monitor_index: int, fullscreen: bool) -> None:
    if get_monitors is None:
        return
    monitors = get_monitors()
    if 0 <= monitor_index < len(monitors):
        monitor = monitors[monitor_index]
        cv2.moveWindow(window_name, monitor.x, monitor.y)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)  # Keeps automatic model downloads and configuration on D:.
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    cpu_threads = int(np.clip(args.cpu_threads, 1, 8))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before some runtimes have initialized.
        pass
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print(f"Unable to open camera {args.camera}. Check the USB webcam and try another --camera index.")
        return 2
    try:
        model = YOLO(args.model)
    except Exception as exc:
        print(f"Unable to load model '{args.model}': {exc}")
        return 3
    window_name = "VisionFSD Pilot - Read Only"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    position_window(window_name, args.monitor, args.fullscreen)
    estimator = MotionEstimator()
    track_stabilizer = TrackStabilizer()
    road_tracker = RoadGeometryTracker()
    show_lanes = True
    view_mode = "world" if args.map_only else "split"
    last_time, displayed_fps = time.perf_counter(), 0.0
    detection_fps = 0.0
    frame_number = 0
    latest_objects: list[DetectedObject] = []
    cached_lanes: list[tuple[int, int, int, int]] = []
    latest_road_geometry: RoadGeometry | None = None
    detection_interval = max(1, args.detect_interval)
    print(f"Running at {args.imgsz}px YOLO input, detection every {detection_interval} frame(s), {cpu_threads} CPU threads. 1 world; 2 camera perception; 3 split; V cycle; L lanes; S screenshot; F fullscreen; Q/Esc quits.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam frame capture failed.")
            break
        now = time.perf_counter()
        instantaneous_fps = 1.0 / max(now - last_time, 1e-3)
        displayed_fps = instantaneous_fps if displayed_fps == 0 else displayed_fps * 0.88 + instantaneous_fps * 0.12
        last_time = now
        fh, fw = frame.shape[:2]
        focal_px = focal_length_px(fw, args.fov)
        frame_number += 1
        if (frame_number - 1) % detection_interval == 0:
            detect_started = time.perf_counter()
            results = model.track(frame, persist=True, verbose=False, conf=args.confidence,
                                  imgsz=args.imgsz, device="cpu",
                                  tracker=str(PROJECT_ROOT / "config" / "visionfsd_bytetrack.yaml"))
            elapsed = max(time.perf_counter() - detect_started, 1e-3)
            instant_detection_fps = 1.0 / elapsed
            detection_fps = instant_detection_fps if detection_fps == 0 else detection_fps * 0.82 + instant_detection_fps * 0.18
            current_objects: list[DetectedObject] = []
            result = results[0]
            current_objects = objects_from_yolo_result(
                result, fw, focal_px, estimator, now, fh,
                args.camera_height, args.horizon_ratio,
            )
            del result, results
            latest_objects = track_stabilizer.update(current_objects, now)
            cached_lanes = []
            latest_road_geometry = road_tracker.update(frame, horizon_ratio=args.horizon_ratio) if show_lanes else None
        camera_view = draw_camera_view(
            frame, latest_objects, show_lanes, cached_lanes, displayed_fps, detection_fps,
            latest_road_geometry, horizon_ratio=args.horizon_ratio,
            camera_height_m=args.camera_height, fov_deg=args.fov,
        )
        world_view = draw_world_view(latest_objects, (fw, fh), displayed_fps, detection_fps)
        if view_mode == "world":
            output = world_view
        elif view_mode == "camera":
            output = camera_view
        else:
            output = np.hstack((camera_view, world_view))
        cv2.imshow(window_name, output)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("1"), ord("m")):
            view_mode = "world"
        elif key == ord("2"):
            view_mode = "camera"
        elif key == ord("3"):
            view_mode = "split"
        elif key == ord("v"):
            view_mode = {"world": "camera", "camera": "split", "split": "world"}[view_mode]
        elif key == ord("l"):
            show_lanes = not show_lanes
        elif key == ord("f"):
            current = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
        elif key == ord("s"):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = SCREENSHOT_DIR / f"visionfsd-{stamp}.png"
            cv2.imwrite(str(path), output)
            print(f"Saved {path}")
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
