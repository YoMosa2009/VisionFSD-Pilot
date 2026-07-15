"""Depth Anything V2 Small (OpenVINO) for range stabilization.

Runs async, shares INFERENCE_LOCK with YOLO / YOLOPv2, and is throttled so the
display loop can stay at ≥25 FPS on Intel iGPU. Relative depth is aligned to
monocular estimates each update (affine scale) then EMA-fused into track range.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from gpu_lock import INFERENCE_LOCK
from visionfsd import (
    DetectedObject,
    ROAD_CLASSES,
    VEHICLE_CLASSES,
    estimate_monocular_distance_m,
)


# Default IR location (created by tools/export_depth_anything_openvino.py).
DEFAULT_DEPTH_MODEL = Path("models/depth_anything_v2_small/openvino_fp16/depth_anything_v2_small.xml")


@dataclass(frozen=True)
class DepthPerceptionResult:
    sequence: int
    depth_map: np.ndarray  # float32 HxW, larger ≈ farther (relative)
    inference_ms: float
    scale: float  # relative→metres scale from last mono alignment


class DepthAnythingEngine:
    """Fixed-shape FP16 OpenVINO inference for Depth Anything V2 Small."""

    # Multiples of 14 (ViT patch). Small enough for iGPU + two other nets.
    INPUT_HEIGHT = 252
    INPUT_WIDTH = 336

    def __init__(self, model_path: Path, device: str = "GPU") -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        core = ov.Core()
        cache_dir = model_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(str(model_path))
        try:
            model.reshape({model.input(0): [1, 3, self.INPUT_HEIGHT, self.INPUT_WIDTH]})
        except Exception:
            pass  # some IRs are already fixed-shape
        self._compiled = core.compile_model(
            model, device,
            {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"},
        )
        self._input = self._compiled.input(0)
        self._output = self._compiled.output(0)
        # ImageNet norm used by Depth Anything / DPT-style models.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self._ema_depth: np.ndarray | None = None

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.INPUT_WIDTH, self.INPUT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        tensor = resized.astype(np.float32) * (1.0 / 255.0)
        tensor = (tensor - self._mean) / self._std
        # NCHW
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])

    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Return full-resolution relative depth (float32) and inference ms."""
        height, width = frame_bgr.shape[:2]
        tensor = self._preprocess(frame_bgr)
        started = time.perf_counter()
        with INFERENCE_LOCK:
            raw = self._compiled({self._input: tensor})[self._output]
        inference_ms = (time.perf_counter() - started) * 1000.0
        depth = np.asarray(raw, dtype=np.float32)
        # Accept NCHW, NHWC, or HW.
        while depth.ndim > 2:
            depth = depth[0]
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected depth output shape: {np.asarray(raw).shape}")
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        # Temporal EMA on the map (cheap, cuts flicker before object fusion).
        if self._ema_depth is None or self._ema_depth.shape != depth.shape:
            self._ema_depth = depth
        else:
            self._ema_depth = self._ema_depth * 0.55 + depth * 0.45
        return np.asarray(self._ema_depth, dtype=np.float32), inference_ms


class AsyncDepthPerception:
    """Newest-frame depth worker; never blocks the render loop."""

    def __init__(self, engine: DepthAnythingEngine) -> None:
        self._engine = engine
        self._condition = threading.Condition()
        self._pending: tuple[int, np.ndarray] | None = None
        self._latest: DepthPerceptionResult | None = None
        self._stopping = False
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, name="visionfsd-depth", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, sequence: int) -> None:
        with self._condition:
            self._pending = (sequence, frame)
            self._condition.notify()

    def latest_after(self, sequence: int) -> DepthPerceptionResult | None:
        with self._condition:
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            return self._latest

    def latest(self) -> DepthPerceptionResult | None:
        with self._condition:
            return self._latest

    @property
    def error(self) -> Exception | None:
        with self._condition:
            return self._error

    def _worker(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        return
                    sequence, frame = self._pending
                    self._pending = None
                try:
                    depth_map, inference_ms = self._engine.infer(frame)
                    result = DepthPerceptionResult(
                        sequence=sequence,
                        depth_map=depth_map,
                        inference_ms=inference_ms,
                        scale=1.0,
                    )
                    with self._condition:
                        self._latest = result
                        self._error = None
                except Exception as exc:
                    with self._condition:
                        self._error = exc
                    time.sleep(0.05)
        except Exception as exc:
            with self._condition:
                self._error = exc

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)


def _sample_box_depth(depth_map: np.ndarray, box: tuple[int, int, int, int]) -> float | None:
    """Median relative depth in the lower third of the box (ground-ish)."""
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    y_lo = y1 + int((y2 - y1) * 0.55)
    roi = depth_map[y_lo:y2, x1:x2]
    if roi.size < 8:
        roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    sample = float(np.median(roi))
    if not np.isfinite(sample) or sample <= 1e-6:
        return None
    return sample


def align_depth_scale(
    depth_map: np.ndarray,
    objects: list[DetectedObject],
    *,
    focal_px: float,
    frame_height: int,
    camera_height_m: float,
    horizon_ratio: float,
) -> float:
    """Estimate relative→metres scale from monocular ranges of solid vehicles."""
    ratios: list[float] = []
    for obj in objects:
        if obj.label not in ROAD_CLASSES:
            continue
        if not obj.observed:
            continue
        d_rel = _sample_box_depth(depth_map, obj.box)
        if d_rel is None or d_rel < 1e-4:
            continue
        mono = estimate_monocular_distance_m(
            obj.label, obj.box, focal_px, frame_height,
            camera_height_m, horizon_ratio,
        )
        if mono < 2.0 or mono > 70.0:
            continue
        ratios.append(mono / d_rel)
    if not ratios:
        return 1.0
    # Robust central tendency — ignores a few bad boxes.
    return float(np.clip(np.median(ratios), 0.05, 80.0))


def fuse_objects_with_depth(
    objects: list[DetectedObject],
    depth_map: np.ndarray | None,
    *,
    focal_px: float,
    frame_height: int,
    camera_height_m: float = 1.25,
    horizon_ratio: float = 0.52,
    depth_weight: float = 0.42,
    prev_scale: float = 1.0,
) -> tuple[list[DetectedObject], float]:
    """Blend monocular track range with depth-aligned range (vehicles/VRUs).

    Returns (updated objects, scale used). Soft weight so a bad depth frame
    cannot dominate; MotionEstimator/stabilizer already smoothed the mono path.
    """
    if depth_map is None or not objects:
        return objects, prev_scale

    scale = align_depth_scale(
        depth_map, objects,
        focal_px=focal_px,
        frame_height=frame_height,
        camera_height_m=camera_height_m,
        horizon_ratio=horizon_ratio,
    )
    # EMA scale across frames so alignment does not thrash.
    if prev_scale > 1e-4 and np.isfinite(prev_scale):
        scale = 0.72 * prev_scale + 0.28 * scale

    w_depth = float(np.clip(depth_weight, 0.0, 0.65))
    w_mono = 1.0 - w_depth
    fused: list[DetectedObject] = []
    for obj in objects:
        if obj.label not in ROAD_CLASSES | VEHICLE_CLASSES | {"person", "bicycle", "motorcycle"}:
            fused.append(obj)
            continue
        d_rel = _sample_box_depth(depth_map, obj.box)
        if d_rel is None:
            fused.append(obj)
            continue
        z_depth = float(np.clip(scale * d_rel, 1.0, 86.0))
        z_mono = float(np.clip(obj.distance_m, 1.0, 86.0))
        # Log-space blend resists one outlier blowing the range.
        z = math.exp(
            w_mono * math.log(max(z_mono, 1e-3)) + w_depth * math.log(max(z_depth, 1e-3))
        )
        # When depth and mono disagree hard, trust mono more (occlusion / sky).
        if abs(z_depth - z_mono) > max(8.0, 0.45 * z_mono):
            z = 0.78 * z_mono + 0.22 * z_depth
        fused.append(DetectedObject(
            obj.track_id, obj.label, obj.confidence, obj.box,
            float(np.clip(z, 1.0, 86.0)),
            obj.bearing_deg, obj.closing_mps, obj.image_velocity,
            obj.mask_polygon, obj.observed, obj.missed_updates, obj.stationary,
            obj.signal_state, obj.track_quality,
        ))
    return fused, scale


def resolve_depth_model_path(path: str | Path, project_root: Path) -> Path:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    return model_path
