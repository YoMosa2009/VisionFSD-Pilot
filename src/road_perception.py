"""YOLOPv2 road/lane segmentation for the read-only visualizer."""

from __future__ import annotations

import threading
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import openvino as ov

from gpu_lock import INFERENCE_LOCK

if TYPE_CHECKING:
    from visionfsd import RoadGeometry, RoadGeometryTracker


@dataclass(frozen=True)
class RoadSegmentation:
    sequence: int
    lane_probability: np.ndarray
    drivable_mask: np.ndarray
    inference_ms: float


@dataclass(frozen=True)
class RoadPerceptionUpdate:
    """One fully post-processed road result, produced off the render thread."""

    sequence: int
    geometry: "RoadGeometry"
    inference_ms: float
    capture_time: float = 0.0
    completed_time: float = 0.0


class YOLOPv2RoadEngine:
    """Fixed-shape FP16 OpenVINO inference for YOLOPv2's road heads."""

    INPUT_WIDTH = 640
    INPUT_HEIGHT = 384

    def __init__(self, model_path: Path, device: str = "GPU") -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        core = ov.Core()
        cache_dir = model_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(model_path)
        model.reshape({model.input(0): [1, 3, self.INPUT_HEIGHT, self.INPUT_WIDTH]})
        self._compiled = core.compile_model(
            model, device,
            {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"},
        )
        self._input = self._compiled.input(0)
        self._ema_lane: np.ndarray | None = None
        self._ema_drivable: np.ndarray | None = None

    @staticmethod
    def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
        shifted = values - np.max(values, axis=axis, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.maximum(np.sum(exponent, axis=axis, keepdims=True), 1e-6)

    def infer(self, frame: np.ndarray, sequence: int) -> RoadSegmentation:
        height, width = frame.shape[:2]
        scale = min(self.INPUT_WIDTH / max(1, width), self.INPUT_HEIGHT / max(1, height))
        content_width = max(1, min(self.INPUT_WIDTH, int(round(width * scale))))
        content_height = max(1, min(self.INPUT_HEIGHT, int(round(height * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (content_width, content_height), interpolation=interpolation)
        pad_left = (self.INPUT_WIDTH - content_width) // 2
        pad_right = self.INPUT_WIDTH - content_width - pad_left
        pad_top = (self.INPUT_HEIGHT - content_height) // 2
        pad_bottom = self.INPUT_HEIGHT - content_height - pad_top
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))[None]
        tensor = tensor.astype(np.float32) / 255.0
        started = time.perf_counter()
        # Share the process GPU lock with YOLO11 so OpenVINO iGPU is never
        # invoked from two threads at once (that hard-crashed the process).
        with INFERENCE_LOCK:
            outputs = list(self._compiled({self._input: tensor}).values())
        inference_ms = (time.perf_counter() - started) * 1000.0
        drivable_logits = next(output for output in outputs if output.shape[1] == 2)
        lane_output = next(output for output in outputs if output.shape[1] == 1)
        drivable_probability = self._softmax(drivable_logits, axis=1)[0, 1]
        lane_probability = lane_output[0, 0]
        if float(np.min(lane_probability)) < 0.0 or float(np.max(lane_probability)) > 1.0:
            lane_probability = 1.0 / (1.0 + np.exp(-lane_probability))
        crop_y = slice(pad_top, pad_top + content_height)
        crop_x = slice(pad_left, pad_left + content_width)
        lane_probability = cv2.resize(
            lane_probability[crop_y, crop_x], (width, height), interpolation=cv2.INTER_LINEAR,
        )
        drivable_probability = cv2.resize(
            drivable_probability[crop_y, crop_x], (width, height), interpolation=cv2.INTER_LINEAR,
        )
        if self._ema_lane is None or self._ema_lane.shape != lane_probability.shape:
            self._ema_lane = lane_probability
            self._ema_drivable = drivable_probability
        else:
            self._ema_lane = self._ema_lane * 0.66 + lane_probability * 0.34
            assert self._ema_drivable is not None
            self._ema_drivable = self._ema_drivable * 0.66 + drivable_probability * 0.34
        lane = np.asarray(self._ema_lane, dtype=np.float32)
        drivable = np.asarray(self._ema_drivable >= 0.52, dtype=np.uint8)
        # Preserve lower-confidence but spatially coherent paint. A fixed 0.48
        # gate erased every lane response on some cameras before geometry fitting.
        lower_roi = lane[int(height * 0.40):]
        strong_level = float(np.percentile(lower_roi, 99.2)) if lower_roi.size else 0.0
        cleanup_threshold = float(np.clip(strong_level * 0.45, 0.12, 0.48))
        binary = np.asarray(lane >= cleanup_threshold, dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        cleaned = np.zeros_like(binary)
        for component in range(1, components):
            if stats[component, cv2.CC_STAT_AREA] >= max(20, height * width // 24000):
                cleaned[labels == component] = 1
        lane *= cleaned.astype(np.float32)
        return RoadSegmentation(sequence, lane, drivable, inference_ms)

    def close(self) -> None:
        self._input = None
        self._compiled = None
        gc.collect()
        time.sleep(0.15)


class AsyncRoadPerception:
    """Runs road inference *and* geometry fitting off the render thread.

    Only the newest submitted frame is kept. Geometry fitting (perspective
    warps, morphology, polynomial fits) used to run on the main render thread
    after each result and was a measurable stutter source; it now runs here,
    on the exact frame the model saw, immediately after inference while the
    GPU is free.
    """

    def __init__(self, engine: YOLOPv2RoadEngine, tracker: "RoadGeometryTracker",
                 horizon_ratio: float = 0.52) -> None:
        self._engine = engine
        self._tracker = tracker
        self._horizon_ratio = float(horizon_ratio)
        self._condition = threading.Condition()
        self._pending: tuple[int, np.ndarray, float] | None = None
        self._latest: RoadPerceptionUpdate | None = None
        self._stopping = False
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, name="visionfsd-road", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, sequence: int,
               capture_time: float | None = None) -> None:
        with self._condition:
            cap_t = float(capture_time) if capture_time is not None else time.perf_counter()
            self._pending = (sequence, frame, cap_t)
            self._condition.notify()

    def latest_after(self, sequence: int, *,
                     max_age_s: float = 0.0) -> RoadPerceptionUpdate | None:
        with self._condition:
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            latest = self._latest
        if max_age_s > 0.0 and latest.capture_time > 0.0:
            if (time.perf_counter() - latest.capture_time) > max_age_s:
                return None
        return latest

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
                    sequence, frame, capture_time = self._pending
                    self._pending = None
                try:
                    result = self._engine.infer(frame, sequence)
                    geometry = self._tracker.update(
                        frame, result.lane_probability, result.drivable_mask,
                        horizon_ratio=self._horizon_ratio,
                    )
                    completed = time.perf_counter()
                    update = RoadPerceptionUpdate(
                        sequence, geometry, result.inference_ms,
                        capture_time=capture_time, completed_time=completed,
                    )
                    with self._condition:
                        self._latest = update
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
        self._thread.join(timeout=3.0)
        self._engine.close()
