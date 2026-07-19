"""Low-cost, read-only VisionFSD visualizer for Raspberry Pi 3B.

This module is intentionally self-contained. It borrows the desktop project's
newest-frame, asynchronous, and sticky-lead principles without importing its
PyTorch/OpenVINO/OpenGL modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
CAR_WIDTH_M = 1.80
WINDOW_TITLE = "VisionFSD Pi 3B - read only"

# The bundled SSD MobileNet labelmap is zero-based.  Keep this intentionally
# small: decoding these classes costs no extra model inference, while avoiding
# clutter and false visual objects from the full 90-class COCO labelmap.
SUPPORTED_COCO_LABELS = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
}
VEHICLE_LABELS = frozenset(("car", "motorcycle", "bus", "truck"))
WORLD_ONLY_LABELS = frozenset(("person", "traffic_light", "stop_sign"))
OBJECT_WIDTH_M = {
    "car": CAR_WIDTH_M,
    "motorcycle": 0.85,
    "bus": 2.55,
    "truck": 2.50,
    "person": 0.55,
    "traffic_light": 0.35,
    "stop_sign": 0.75,
}
DISPLAY_LABELS = {
    "person": "PED",
    "traffic_light": "LIGHT",
    "stop_sign": "STOP",
}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Target:
    track_id: int
    detection: Detection
    distance_m: float
    bearing_deg: float
    observed: bool
    age_s: float


@dataclass(frozen=True)
class DetectorResult:
    sequence: int
    detections: list[Detection]
    inference_ms: float
    capture_time: float
    completed_time: float


@dataclass(frozen=True)
class TouchButton:
    """A large, touchscreen-friendly control in rendered image coordinates."""

    action: str
    label: str
    rect: tuple[int, int, int, int]


def focal_length_px(frame_width: int, fov_deg: float) -> float:
    return (frame_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def estimate_range_m(box: tuple[int, int, int, int], frame_width: int, fov_deg: float,
                     known_width_m: float = CAR_WIDTH_M) -> float:
    """Coarse pinhole range for visualization; never a driving measurement."""
    width = max(1, box[2] - box[0])
    return float(np.clip(known_width_m * focal_length_px(frame_width, fov_deg) / width, 1.0, 120.0))


def estimate_car_range_m(box: tuple[int, int, int, int], frame_width: int, fov_deg: float) -> float:
    """Compatibility wrapper for the lead-car visual estimate."""
    return estimate_range_m(box, frame_width, fov_deg, CAR_WIDTH_M)


def bearing_deg(box: tuple[int, int, int, int], frame_width: int, fov_deg: float) -> float:
    center_x = (box[0] + box[2]) * 0.5
    return float(math.degrees(math.atan2(center_x - frame_width * 0.5, focal_length_px(frame_width, fov_deg))))


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    if inter <= 0:
        return 0.0
    area_a = max(1, a[2] - a[0]) * max(1, a[3] - a[1])
    area_b = max(1, b[2] - b[0]) * max(1, b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def nms(detections: list[Detection], threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    for item in sorted(detections, key=lambda value: value.confidence, reverse=True):
        # Different classes may legitimately overlap (for example a person
        # beside a vehicle), so suppress only duplicate detections of the same
        # semantic class.
        if all(item.label != chosen.label or box_iou(item.box, chosen.box) < threshold for chosen in kept):
            kept.append(item)
    return kept


class TFLiteVehicleDetector:
    """LiteRT scene detector limited to the Pi visualizer's supported classes."""

    def __init__(self, model_path: Path, confidence: float, threads: int) -> None:
        try:
            # Current Google LiteRT package; supports Python 3.13 on ARM64.
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                # Legacy fallback for existing Bookworm/Python <=3.11 installs.
                from tflite_runtime.interpreter import Interpreter
            except ImportError as exc:  # pragma: no cover - requires Pi runtime
                raise RuntimeError(
                    "LiteRT is missing. Run pi3b/install.sh on 64-bit Raspberry Pi OS."
                ) from exc
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        self._interpreter = Interpreter(model_path=str(model_path), num_threads=max(1, threads))
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._outputs = self._interpreter.get_output_details()
        self._confidence = float(confidence)
        shape = self._input["shape"]
        self._input_h, self._input_w = int(shape[1]), int(shape[2])

    @staticmethod
    def _dequantize(array: np.ndarray, detail: dict) -> np.ndarray:
        scale, zero = detail.get("quantization", (0.0, 0))
        if scale and np.issubdtype(array.dtype, np.integer):
            return (array.astype(np.float32) - float(zero)) * float(scale)
        return array.astype(np.float32)

    def _input_tensor(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(frame, (self._input_w, self._input_h)), cv2.COLOR_BGR2RGB)
        dtype = self._input["dtype"]
        scale, zero = self._input.get("quantization", (0.0, 0))
        # The bundled SSD MobileNet input is uint8 with scale 1/128 and
        # zero-point 128. Its graph performs normalization itself, so it needs
        # raw 0..255 RGB bytes. Re-quantizing here clips almost every pixel to
        # 255 and makes the model effectively blind.
        if dtype == np.uint8:
            return rgb[None, ...]
        if np.issubdtype(dtype, np.integer):
            # Signed integer exports generally quantize normalized RGB.
            value = rgb.astype(np.float32) / 255.0
            if scale:
                value = np.rint(value / float(scale) + float(zero))
            return np.clip(value, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)[None, ...]
        return (rgb.astype(np.float32) / 255.0)[None, ...].astype(dtype)

    @staticmethod
    def _to_box(cx: float, cy: float, width: float, height: float, frame_w: int, frame_h: int,
                input_w: int, input_h: int) -> tuple[int, int, int, int] | None:
        # Raw YOLO outputs are in model-input pixels. Convert and clamp once.
        x1 = int(round((cx - width / 2.0) * frame_w / input_w))
        y1 = int(round((cy - height / 2.0) * frame_h / input_h))
        x2 = int(round((cx + width / 2.0) * frame_w / input_w))
        y2 = int(round((cy + height / 2.0) * frame_h / input_h))
        x1, x2 = max(0, x1), min(frame_w - 1, x2)
        y1, y2 = max(0, y1), min(frame_h - 1, y2)
        return (x1, y1, x2, y2) if x2 - x1 >= 3 and y2 - y1 >= 3 else None

    def _decode_detection_postprocess(self, outputs: list[np.ndarray], frame_w: int,
                                      frame_h: int) -> list[Detection] | None:
        # Standard TFLite DetectionPostProcess: boxes, classes, scores, count.
        if len(outputs) != 4:
            return None
        boxes = next((item for item in outputs if item.ndim >= 2 and item.shape[-1] == 4), None)
        scalar = [item for item in outputs if item is not boxes]
        if boxes is None or len(scalar) != 3:
            return None
        vectors = [np.squeeze(item) for item in scalar]
        candidates = [item for item in vectors if item.ndim == 1 and item.size > 1]
        if len(candidates) < 2:
            return None
        # Class ids are integer-like; score values live in [0, 1].
        classes = min(candidates, key=lambda item: float(np.mean(np.abs(item - np.rint(item)))))
        scores = next(item for item in candidates if item is not classes)
        flat_boxes = np.squeeze(boxes)
        decoded: list[Detection] = []
        for raw_box, class_id, score in zip(flat_boxes, classes, scores):
            label = SUPPORTED_COCO_LABELS.get(int(round(float(class_id))))
            if label is None or float(score) < self._confidence:
                continue
            y1, x1, y2, x2 = [float(value) for value in raw_box]
            # DetectionPostProcess normalized boxes are normally [0, 1].
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                x1, x2 = x1 * frame_w, x2 * frame_w
                y1, y2 = y1 * frame_h, y2 * frame_h
            box = (max(0, int(x1)), max(0, int(y1)), min(frame_w - 1, int(x2)), min(frame_h - 1, int(y2)))
            if box[2] - box[0] >= 3 and box[3] - box[1] >= 3:
                decoded.append(Detection(label, float(score), box))
        return nms(decoded)

    def _decode_yolo(self, raw: np.ndarray, frame_w: int, frame_h: int) -> list[Detection]:
        array = np.squeeze(raw)
        if array.ndim != 2:
            raise RuntimeError(f"Unsupported TFLite detector output shape: {raw.shape}")
        # YOLO11 export: [84, candidates]; older YOLO variants: [85, candidates].
        if 6 <= array.shape[0] <= 128 and array.shape[1] > array.shape[0]:
            array = array.T
        largest_class_id = max(SUPPORTED_COCO_LABELS)
        if array.shape[1] < 4 + largest_class_id + 1:
            raise RuntimeError(f"Unsupported TFLite detector channels: {array.shape}")
        has_objectness = array.shape[1] >= 85
        class_start = 5 if has_objectness else 4
        decoded: list[Detection] = []
        for row in array:
            box: tuple[int, int, int, int] | None = None
            for class_id, label in SUPPORTED_COCO_LABELS.items():
                if class_start + class_id >= array.shape[1]:
                    continue
                score = float(row[class_start + class_id])
                if has_objectness:
                    score *= float(row[4])
                if score < self._confidence:
                    continue
                if box is None:
                    box = self._to_box(float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                                       frame_w, frame_h, self._input_w, self._input_h)
                if box is not None:
                    decoded.append(Detection(label, score, box))
        return nms(decoded)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        self._interpreter.set_tensor(self._input["index"], self._input_tensor(frame))
        self._interpreter.invoke()
        outputs = [self._dequantize(self._interpreter.get_tensor(item["index"]), item) for item in self._outputs]
        frame_h, frame_w = frame.shape[:2]
        decoded = self._decode_detection_postprocess(outputs, frame_w, frame_h)
        return decoded if decoded is not None else self._decode_yolo(outputs[0], frame_w, frame_h)


class LatestCamera:
    """V4L2/OpenCV capture that publishes only the newest completed frame."""

    def __init__(self, source: str, width: int, height: int, fps: int) -> None:
        numeric_source: int | str = int(source) if source.isdigit() else source
        self._cap = cv2.VideoCapture(numeric_source, cv2.CAP_V4L2 if isinstance(numeric_source, int) else cv2.CAP_ANY)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera/source: {source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._captured = 0.0
        self._stop = threading.Event()
        self._failed = ""
        self._thread = threading.Thread(target=self._run, name="pi-camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 30:
                    self._failed = "camera repeatedly returned no frames"
                    return
                time.sleep(0.01)
                continue
            failures = 0
            with self._lock:
                self._frame = np.ascontiguousarray(frame)
                self._sequence += 1
                self._captured = time.perf_counter()

    def latest(self) -> tuple[int, np.ndarray | None, float]:
        with self._lock:
            # The capture thread publishes a new frame object rather than
            # mutating the published one. Returning that immutable-by-contract
            # reference avoids an extra 640x480 copy every display tick.
            return self._sequence, self._frame, self._captured

    @property
    def error(self) -> str:
        return self._failed

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


class AsyncDetector:
    """Runs only the most recent pending frame: lag is discarded, never queued."""

    def __init__(self, detector: TFLiteVehicleDetector) -> None:
        self._detector = detector
        self._condition = threading.Condition()
        self._pending: tuple[int, np.ndarray, float] | None = None
        self._latest: DetectorResult | None = None
        self._stop = False
        self._error = ""
        self._thread = threading.Thread(target=self._run, name="pi-detector", daemon=True)
        self._thread.start()

    def submit(self, sequence: int, frame: np.ndarray, captured: float) -> None:
        with self._condition:
            self._pending = (sequence, frame, captured)
            self._condition.notify()

    def latest_after(self, sequence: int) -> DetectorResult | None:
        with self._condition:
            return self._latest if self._latest and self._latest.sequence > sequence else None

    @property
    def error(self) -> str:
        with self._condition:
            return self._error

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                sequence, frame, captured = self._pending
                self._pending = None
            started = time.perf_counter()
            try:
                detections = self._detector.infer(frame)
                result = DetectorResult(sequence, detections, (time.perf_counter() - started) * 1000.0,
                                        captured, time.perf_counter())
                with self._condition:
                    self._latest, self._error = result, ""
            except Exception as exc:
                with self._condition:
                    self._error = str(exc)
                time.sleep(0.05)

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=3.0)


@dataclass
class _Track:
    detection: Detection
    distance_m: float
    bearing: float
    last_seen: float
    hits: int


class TargetSelector:
    """Small, deterministic replacement for desktop ByteTrack + sticky LEAD.

    It tracks only supported vehicle classes, chooses one centred/near lead
    vehicle, and requires a challenger to be materially closer for several
    detector updates.
    """

    def __init__(self, fov_deg: float, hold_s: float = 0.75) -> None:
        self._fov_deg = fov_deg
        self._hold_s = hold_s
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._target_id: int | None = None
        self._challenger_id: int | None = None
        self._challenge_hits = 0

    def _match(self, detection: Detection, reserved: set[int]) -> int | None:
        choices = [
            (box_iou(detection.box, track.detection.box), track_id)
            for track_id, track in self._tracks.items()
            if track_id not in reserved
        ]
        if not choices:
            return None
        score, track_id = max(choices)
        return track_id if score >= 0.25 else None

    def update(self, detections: list[Detection], frame_width: int, now: float) -> Target | None:
        seen: set[int] = set()
        for detection in detections:
            if detection.label not in VEHICLE_LABELS:
                continue
            track_id = self._match(detection, seen)
            distance = estimate_range_m(detection.box, frame_width, self._fov_deg,
                                        OBJECT_WIDTH_M[detection.label])
            heading = bearing_deg(detection.box, frame_width, self._fov_deg)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(detection, distance, heading, now, 1)
            else:
                previous = self._tracks[track_id]
                self._tracks[track_id] = _Track(
                    detection, previous.distance_m * 0.70 + distance * 0.30,
                    previous.bearing * 0.65 + heading * 0.35, now, previous.hits + 1,
                )
            seen.add(track_id)
        self._tracks = {track_id: track for track_id, track in self._tracks.items() if now - track.last_seen <= self._hold_s}
        candidates = [
            (track.distance_m + abs(track.bearing) * 0.18, track_id)
            for track_id, track in self._tracks.items()
            if track_id in seen
        ]
        # Prefer the forward corridor whenever it contains a vehicle. This is
        # only a camera-centre heuristic, not a lane-occupancy claim.
        forward = [
            candidate for candidate in candidates
            if abs(self._tracks[candidate[1]].bearing) <= 18.0
        ]
        if forward:
            candidates = forward
        if not candidates:
            return self.current(now)
        _, best_id = min(candidates)
        if self._target_id not in self._tracks:
            self._target_id, self._challenger_id, self._challenge_hits = best_id, None, 0
        elif best_id != self._target_id:
            incumbent = self._tracks[self._target_id]
            challenger = self._tracks[best_id]
            # A newly visible centre-corridor vehicle should replace a stale
            # side vehicle immediately; other handoffs retain hysteresis.
            if abs(challenger.bearing) <= 18.0 < abs(incumbent.bearing):
                self._target_id, self._challenger_id, self._challenge_hits = best_id, None, 0
            elif challenger.distance_m < incumbent.distance_m - 3.0:
                self._challenge_hits = self._challenge_hits + 1 if self._challenger_id == best_id else 1
                self._challenger_id = best_id
                if self._challenge_hits >= 6:
                    self._target_id, self._challenger_id, self._challenge_hits = best_id, None, 0
            else:
                self._challenger_id, self._challenge_hits = None, 0
        else:
            self._challenger_id, self._challenge_hits = None, 0
        return self.current(now)

    def current(self, now: float) -> Target | None:
        if self._target_id is None or self._target_id not in self._tracks:
            return None
        track = self._tracks[self._target_id]
        age = now - track.last_seen
        if age > self._hold_s:
            self._tracks.pop(self._target_id, None)
            self._target_id = None
            return None
        return Target(self._target_id, track.detection, track.distance_m, track.bearing, age < 0.12, age)


@dataclass(frozen=True)
class SceneObject:
    """A non-vehicle object shown only in the low-cost world view."""

    track_id: int
    detection: Detection
    distance_m: float
    bearing_deg: float
    observed: bool
    age_s: float


@dataclass
class _SceneTrack:
    detection: Detection
    distance_m: float
    bearing: float
    last_seen: float
    hits: int


class SceneObjectTracker:
    """Tiny class-aware tracker for world-only pedestrians, lights, and signs."""

    def __init__(self, fov_deg: float, hold_s: float = 0.80, max_objects: int = 4) -> None:
        self._fov_deg = fov_deg
        self._hold_s = hold_s
        self._max_objects = max_objects
        self._tracks: dict[int, _SceneTrack] = {}
        self._next_id = 1

    def _match(self, detection: Detection, reserved: set[int]) -> int | None:
        choices = [
            (box_iou(detection.box, track.detection.box), track_id)
            for track_id, track in self._tracks.items()
            if track.detection.label == detection.label and track_id not in reserved
        ]
        if not choices:
            return None
        score, track_id = max(choices)
        return track_id if score >= 0.20 else None

    def update(self, detections: list[Detection], frame_width: int, now: float) -> list[SceneObject]:
        seen: set[int] = set()
        for detection in detections:
            if detection.label not in WORLD_ONLY_LABELS:
                continue
            track_id = self._match(detection, seen)
            distance = estimate_range_m(detection.box, frame_width, self._fov_deg,
                                        OBJECT_WIDTH_M[detection.label])
            heading = bearing_deg(detection.box, frame_width, self._fov_deg)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _SceneTrack(detection, distance, heading, now, 1)
            else:
                previous = self._tracks[track_id]
                self._tracks[track_id] = _SceneTrack(
                    detection,
                    previous.distance_m * 0.70 + distance * 0.30,
                    previous.bearing * 0.65 + heading * 0.35,
                    now,
                    previous.hits + 1,
                )
            seen.add(track_id)
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if now - track.last_seen <= self._hold_s
        }
        return self.current(now)

    def current(self, now: float) -> list[SceneObject]:
        candidates = [
            SceneObject(track_id, track.detection, track.distance_m, track.bearing,
                        now - track.last_seen < 0.35, now - track.last_seen)
            for track_id, track in self._tracks.items()
            if now - track.last_seen <= self._hold_s and track.hits >= 2
        ]
        limits = {"person": 2, "traffic_light": 1, "stop_sign": 1}
        chosen: list[SceneObject] = []
        label_counts: dict[str, int] = {}
        for item in sorted(candidates, key=lambda value: (value.distance_m, abs(value.bearing_deg), value.detection.label)):
            if label_counts.get(item.detection.label, 0) >= limits[item.detection.label]:
                continue
            chosen.append(item)
            label_counts[item.detection.label] = label_counts.get(item.detection.label, 0) + 1
            if len(chosen) >= self._max_objects:
                break
        return chosen


@dataclass(frozen=True)
class LaneEstimate:
    """Perspective lane geometry from a low-rate, classical image pass."""

    left_bottom_norm: float
    right_bottom_norm: float
    vanishing_x_norm: float
    observed: bool
    age_s: float


class LowCostLaneDetector:
    """Detect two bright lane boundaries at low rate without a second ML model."""

    def __init__(self, interval_s: float = 0.20, hold_s: float = 0.80, analysis_width: int = 320) -> None:
        self._interval_s = interval_s
        self._hold_s = hold_s
        self._analysis_width = analysis_width
        self._next_analysis = 0.0
        self._last_values: tuple[float, float, float] | None = None
        self._last_seen = 0.0

    def _detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        source_h, source_w = frame.shape[:2]
        if source_w < 80 or source_h < 60:
            return None
        width = min(self._analysis_width, source_w)
        height = max(80, int(round(source_h * width / source_w)))
        reduced = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        top = int(height * 0.52)
        roi = np.zeros_like(gray)
        cv2.fillConvexPoly(roi, np.array([
            (int(width * 0.12), height - 1),
            (int(width * 0.88), height - 1),
            (int(width * 0.60), top),
            (int(width * 0.40), top),
        ], dtype=np.int32), 255)
        bright = cv2.inRange(gray, 150, 255)
        edges = cv2.Canny(gray, 45, 140)
        candidates = cv2.bitwise_and(cv2.bitwise_or(bright, edges), roi)
        lines = cv2.HoughLinesP(candidates, 1, np.pi / 180.0, threshold=18,
                                minLineLength=max(18, height // 12), maxLineGap=max(12, height // 10))
        if lines is None:
            return None
        centre = width * 0.5
        left_points: list[tuple[int, int]] = []
        right_points: list[tuple[int, int]] = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if y1 == y2:
                continue
            if y2 > y1:
                bottom_x, bottom_y, top_x, top_y = x2, y2, x1, y1
            else:
                bottom_x, bottom_y, top_x, top_y = x1, y1, x2, y2
            slope = (bottom_x - top_x) / float(bottom_y - top_y)
            if not 0.14 <= abs(slope) <= 0.95:
                continue
            if bottom_x < centre and slope < 0.0:
                left_points.extend(((top_x, top_y), (bottom_x, bottom_y)))
            elif bottom_x > centre and slope > 0.0:
                right_points.extend(((top_x, top_y), (bottom_x, bottom_y)))
        if len(left_points) < 4 or len(right_points) < 4:
            return None

        def fit(points: list[tuple[int, int]]) -> tuple[float, float, float]:
            y = np.array([point[1] for point in points], dtype=np.float32)
            x = np.array([point[0] for point in points], dtype=np.float32)
            slope, intercept = np.polyfit(y, x, 1)
            return float(np.polyval((slope, intercept), height - 1)), float(np.polyval((slope, intercept), top)), float(slope)

        left_bottom, left_top, left_slope = fit(left_points)
        right_bottom, right_top, right_slope = fit(right_points)
        if left_bottom >= centre - 4 or right_bottom <= centre + 4 or left_top >= right_top:
            return None
        if left_slope >= -0.08 or right_slope <= 0.08:
            return None
        vanish = (left_top + right_top) * 0.5
        return (
            float(np.clip(left_bottom / width, 0.02, 0.49)),
            float(np.clip(right_bottom / width, 0.51, 0.98)),
            float(np.clip(vanish / width, 0.30, 0.70)),
        )

    def update(self, frame: np.ndarray, now: float) -> LaneEstimate:
        if now >= self._next_analysis:
            self._next_analysis = now + self._interval_s
            detected = self._detect(frame)
            if detected is not None:
                if self._last_values is not None:
                    detected = tuple(previous * 0.68 + current * 0.32
                                     for previous, current in zip(self._last_values, detected))
                self._last_values = detected
                self._last_seen = now
        return self.current(now)

    def current(self, now: float) -> LaneEstimate:
        if self._last_values is None or now - self._last_seen > self._hold_s:
            return LaneEstimate(0.25, 0.75, 0.50, False, float("inf"))
        left, right, vanish = self._last_values
        age = now - self._last_seen
        return LaneEstimate(left, right, vanish, age < self._interval_s * 1.75, age)


class Rates:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.display_frames = 0
        self.detections = 0
        self.capture_latency_ms = 0.0
        self.inference_ms = 0.0

    def report(self) -> dict[str, float]:
        elapsed = max(0.001, time.perf_counter() - self.started)
        return {
            "elapsed_s": round(elapsed, 3),
            "display_fps": round(self.display_frames / elapsed, 2),
            "detect_fps": round(self.detections / elapsed, 2),
            "end_to_end_latency_ms": round(self.capture_latency_ms, 1),
            "last_inference_ms": round(self.inference_ms, 1),
        }


def _camera_panel(frame: np.ndarray, target: Target | None, rates: Rates, error: str) -> np.ndarray:
    panel = frame.copy()
    if target:
        x1, y1, x2, y2 = target.detection.box
        colour = (50, 220, 70) if target.observed else (0, 170, 255)
        cv2.rectangle(panel, (x1, y1), (x2, y2), colour, 2)
        state = "LIVE" if target.observed else f"HOLD {target.age_s:.1f}s"
        text = f"TARGET {target.detection.label.upper()} {target.distance_m:.1f}m {target.bearing_deg:+.1f}deg {state}"
        cv2.putText(panel, text, (max(6, x1), max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 2, cv2.LINE_AA)
    stats = rates.report()
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 370), 48), (15, 18, 22), -1)
    cv2.putText(panel, f"DISPLAY {stats['display_fps']:.1f}  DETECT {stats['detect_fps']:.1f}", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 235, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, f"latency {stats['end_to_end_latency_ms']:.0f}ms  infer {stats['last_inference_ms']:.0f}ms", (8, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 190, 205), 1, cv2.LINE_AA)
    if error:
        cv2.putText(panel, error[:75], (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 70, 255), 1, cv2.LINE_AA)
    return panel


_WORLD_BASE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _world_base(size: tuple[int, int]) -> np.ndarray:
    """Cache the unchanging dark background; render only dynamic objects per frame."""
    cached = _WORLD_BASE_CACHE.get(size)
    if cached is not None:
        return cached
    width, height = size
    base = np.empty((height, width, 3), dtype=np.uint8)
    vertical = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    # BGR order: keep the scene cool blue-black rather than warm grey.
    base[:, :, 0] = (25 + 14 * vertical).astype(np.uint8)
    base[:, :, 1] = (18 + 12 * vertical).astype(np.uint8)
    base[:, :, 2] = (10 + 9 * vertical).astype(np.uint8)
    # A restrained horizon glow gives the road depth without a per-frame blur.
    cv2.line(base, (0, int(height * 0.30)), (width, int(height * 0.30)), (47, 38, 31), 1, cv2.LINE_AA)
    _WORLD_BASE_CACHE[size] = base
    return base


def _draw_glowing_lane(panel: np.ndarray, start: tuple[int, int], end: tuple[int, int], active: bool) -> None:
    if active:
        # Three direct strokes give a glow-like halo without full-frame alpha
        # buffers or blur work on the Pi 3B.
        cv2.line(panel, start, end, (130, 155, 178), 12, cv2.LINE_AA)
        cv2.line(panel, start, end, (255, 246, 230), 5, cv2.LINE_AA)
        cv2.line(panel, start, end, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.line(panel, start, end, (108, 96, 82), 2, cv2.LINE_AA)


def _draw_ego_vehicle(panel: np.ndarray, centre_x: int, bottom_y: int, width: int, height: int) -> None:
    """Stylised stationary ego car, inspired by the supplied dark-road references."""
    shadow = (centre_x, bottom_y + 2)
    cv2.ellipse(panel, shadow, (int(width * 0.60), max(4, int(height * 0.14))), 0, 0, 360, (6, 9, 12), -1, cv2.LINE_AA)
    body = np.array([
        (centre_x - int(width * 0.53), bottom_y),
        (centre_x + int(width * 0.53), bottom_y),
        (centre_x + int(width * 0.47), bottom_y - int(height * 0.42)),
        (centre_x + int(width * 0.29), bottom_y - height),
        (centre_x - int(width * 0.29), bottom_y - height),
        (centre_x - int(width * 0.47), bottom_y - int(height * 0.42)),
    ], dtype=np.int32)
    cv2.fillConvexPoly(panel, body, (224, 229, 232))
    cv2.polylines(panel, [body], True, (116, 132, 145), 1, cv2.LINE_AA)
    rear_window = np.array([
        (centre_x - int(width * 0.27), bottom_y - int(height * 0.47)),
        (centre_x + int(width * 0.27), bottom_y - int(height * 0.47)),
        (centre_x + int(width * 0.19), bottom_y - int(height * 0.83)),
        (centre_x - int(width * 0.19), bottom_y - int(height * 0.83)),
    ], dtype=np.int32)
    cv2.fillConvexPoly(panel, rear_window, (25, 37, 48))
    cv2.polylines(panel, [rear_window], True, (74, 94, 110), 1, cv2.LINE_AA)
    light_y = bottom_y - int(height * 0.30)
    for sign in (-1, 1):
        x1 = centre_x + sign * int(width * 0.43)
        x2 = centre_x + sign * int(width * 0.12)
        cv2.line(panel, (x1, light_y), (x2, light_y), (35, 35, 235), max(2, int(height * 0.06)), cv2.LINE_AA)
    cv2.line(panel, (centre_x - int(width * 0.43), bottom_y - 3),
             (centre_x + int(width * 0.43), bottom_y - 3), (55, 68, 78), 2, cv2.LINE_AA)


def _world_position(distance_m: float, bearing: float, width: int, horizon: int, ego_top: int) -> tuple[int, int, float]:
    closeness = 1.0 - float(np.clip((distance_m - 2.0) / 80.0, 0.0, 1.0))
    y = int(horizon + (ego_top - horizon) * (0.16 + 0.80 * closeness))
    lateral = np.clip(bearing / 35.0, -1.0, 1.0)
    x = int(width * 0.5 + lateral * width * 0.36 * (0.35 + 0.65 * closeness))
    return x, y, closeness


def _draw_lead_vehicle(panel: np.ndarray, target: Target, width: int, horizon: int, ego_top: int) -> None:
    x, y, closeness = _world_position(target.distance_m, target.bearing_deg, width, horizon, ego_top)
    scale = max(9, int(10 + 29 * closeness))
    colour = (64, 228, 95) if target.observed else (0, 170, 255)
    body = np.array([
        (x - scale, y), (x + scale, y),
        (x + int(scale * .72), y - scale), (x - int(scale * .72), y - scale),
    ], dtype=np.int32)
    roof = np.array([
        (x - int(scale * .48), y - scale), (x + int(scale * .48), y - scale),
        (x + int(scale * .26), y - int(scale * 1.52)), (x - int(scale * .26), y - int(scale * 1.52)),
    ], dtype=np.int32)
    cv2.fillConvexPoly(panel, body, colour)
    cv2.fillConvexPoly(panel, roof, tuple(int(value * 0.70) for value in colour))


def _draw_scene_object(panel: np.ndarray, item: SceneObject, width: int, horizon: int, ego_top: int) -> None:
    x, y, closeness = _world_position(item.distance_m, item.bearing_deg, width, horizon, ego_top)
    scale = max(6, int(7 + 20 * closeness))
    label = item.detection.label
    if label == "person":
        colour = (65, 210, 255) if item.observed else (60, 150, 190)
        cv2.circle(panel, (x, y - scale), max(2, scale // 4), colour, -1, cv2.LINE_AA)
        cv2.line(panel, (x, y - int(scale * 0.75)), (x, y), colour, max(2, scale // 5), cv2.LINE_AA)
        cv2.line(panel, (x - scale // 2, y - scale // 2), (x + scale // 2, y - scale // 2), colour, 1, cv2.LINE_AA)
    elif label == "traffic_light":
        cv2.line(panel, (x, y), (x, y - int(scale * 1.7)), (120, 130, 140), max(1, scale // 5), cv2.LINE_AA)
        cv2.rectangle(panel, (x - scale // 3, y - int(scale * 2.2)), (x + scale // 3, y - int(scale * 1.45)), (42, 45, 48), -1)
        # The model identifies a traffic light, not its signal colour.
        for index, colour in enumerate(((45, 45, 220), (35, 190, 235), (55, 220, 70))):
            cv2.circle(panel, (x, y - int(scale * (2.05 - 0.26 * index))), max(1, scale // 9), colour, -1, cv2.LINE_AA)
    elif label == "stop_sign":
        points = []
        for index in range(8):
            angle = math.radians(22.5 + index * 45.0)
            points.append((int(x + math.cos(angle) * scale * .60), int(y - scale + math.sin(angle) * scale * .60)))
        cv2.fillConvexPoly(panel, np.array(points, dtype=np.int32), (35, 45, 220))
        cv2.putText(panel, "STOP", (x - scale // 2, y - scale + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (245, 245, 245), 1, cv2.LINE_AA)


def _world_panel(size: tuple[int, int], target: Target | None, scene_objects: list[SceneObject],
                 lane: LaneEstimate, rates: Rates) -> np.ndarray:
    width, height = size
    panel = _world_base(size).copy()
    control_top = touch_buttons(width, height)[0].rect[1]
    horizon = int(height * 0.27)
    vanishing = (int(width * lane.vanishing_x_norm), horizon)
    left_bottom = (int(width * lane.left_bottom_norm), control_top)
    right_bottom = (int(width * lane.right_bottom_norm), control_top)
    road = np.array([left_bottom, right_bottom, vanishing], dtype=np.int32)
    cv2.fillConvexPoly(panel, road, (47, 38, 30))
    _draw_glowing_lane(panel, left_bottom, vanishing, lane.observed)
    _draw_glowing_lane(panel, right_bottom, vanishing, lane.observed)
    lane_state = "LANES LIVE" if lane.observed else ("LANES HOLD" if lane.age_s < 1.0 else "LANES SEARCH")
    cv2.putText(panel, f"TARGET WORLD  |  {lane_state}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (220, 230, 238), 1, cv2.LINE_AA)
    ego_height = max(62, int(height * 0.19))
    ego_width = max(88, int(width * 0.20))
    ego_bottom = control_top - 9
    ego_top = ego_bottom - ego_height
    if target is not None:
        _draw_lead_vehicle(panel, target, width, horizon, ego_top)
    for item in scene_objects:
        _draw_scene_object(panel, item, width, horizon, ego_top)
    _draw_ego_vehicle(panel, width // 2, ego_bottom, ego_width, ego_height)
    stats = rates.report()
    cv2.putText(panel, f"DISPLAY {stats['display_fps']:.1f}  DETECT {stats['detect_fps']:.1f}", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (155, 175, 190), 1, cv2.LINE_AA)
    if target is not None:
        cv2.putText(panel, f"LEAD {target.detection.label.upper()} {target.distance_m:.0f}m", (8, 61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 225, 195), 1, cv2.LINE_AA)
    if scene_objects:
        scene_text = "  |  ".join(
            f"{DISPLAY_LABELS[item.detection.label]} {item.distance_m:.0f}m"
            for item in scene_objects
        )
        cv2.putText(panel, scene_text, (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (190, 205, 218), 1, cv2.LINE_AA)
    return panel


def compose_view(frame: np.ndarray, target: Target | None, scene_objects: list[SceneObject], lane: LaneEstimate,
                 rates: Rates, view: str, error: str) -> np.ndarray:
    if view == "camera":
        return _camera_panel(frame, target, rates, error)
    world = _world_panel((frame.shape[1], frame.shape[0]), target, scene_objects, lane, rates)
    if view == "world":
        return world
    return np.hstack((world, _camera_panel(frame, target, rates, error)))


def touch_buttons(frame_width: int, frame_height: int) -> list[TouchButton]:
    """Lay out the permanent touch controls across the bottom of any view."""
    margin = max(6, min(14, frame_height // 35))
    gap = max(5, margin // 2)
    button_height = max(42, min(68, frame_height // 7))
    available_width = frame_width - (2 * margin) - (3 * gap)
    button_width = max(1, available_width // 4)
    y1 = frame_height - margin - button_height
    specs = (
        ("quit", "QUIT"),
        ("world", "SCREEN 1"),
        ("camera", "SCREEN 2"),
        ("split", "SCREEN 3"),
    )
    buttons: list[TouchButton] = []
    for index, (action, label) in enumerate(specs):
        x1 = margin + index * (button_width + gap)
        x2 = frame_width - margin if index == len(specs) - 1 else x1 + button_width
        buttons.append(TouchButton(action, label, (x1, y1, x2, frame_height - margin)))
    return buttons


def touch_action_at(buttons: list[TouchButton], x: int, y: int) -> str | None:
    """Return the action for a touchscreen click, if it landed on a button."""
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            return button.action
    return None


def draw_touch_controls(panel: np.ndarray) -> list[TouchButton]:
    """Render high-contrast buttons after the camera/world panels are composed."""
    buttons = touch_buttons(panel.shape[1], panel.shape[0])
    font_scale = 0.48 if panel.shape[0] < 600 else 0.62
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        colour = (45, 45, 185) if button.action == "quit" else (42, 105, 42)
        cv2.rectangle(panel, (x1, y1), (x2, y2), colour, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (230, 235, 240), 2)
        text_size, _ = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = x1 + max(4, (x2 - x1 - text_size[0]) // 2)
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(panel, button.label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return buttons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only, single-target Pi 3B vehicle visualizer")
    parser.add_argument("--camera", default="0", help="V4L2 camera index or local video path")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/vehicle_ssd_mobilenet_v1.tflite")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, choices=(25, 30), default=25)
    parser.add_argument("--imgsz", type=int, default=320, help="Documented model input size; model metadata is authoritative")
    parser.add_argument("--threads", type=int, default=2, help="LiteRT CPU threads; 2 leaves capture/display headroom on Pi 3B")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--view", choices=("world", "camera", "split"), default="split")
    parser.add_argument("--test-seconds", type=float, default=0.0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--benchmark-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width < 160 or args.height < 120:
        raise ValueError("Camera dimensions must be at least 160x120")
    # Reserve CPU for LiteRT on the 4-core Pi 3B. OpenCV's resize/draw work is
    # tiny at 640x480, but unrestricted worker pools can still starve inference.
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
    detector = TFLiteVehicleDetector(args.model, args.confidence, args.threads)
    camera = LatestCamera(args.camera, args.width, args.height, args.fps)
    worker = AsyncDetector(detector)
    selector = TargetSelector(args.fov)
    scene_tracker = SceneObjectTracker(args.fov)
    lane_detector = LowCostLaneDetector()
    rates = Rates()
    last_camera_sequence = 0
    last_result_sequence = 0
    last_frame: np.ndarray | None = None
    target: Target | None = None
    scene_objects: list[SceneObject] = []
    lane = lane_detector.current(time.perf_counter())
    view = args.view
    deadline = rates.started + args.test_seconds if args.test_seconds > 0 else None
    next_tick = time.perf_counter()
    controls: list[TouchButton] = []
    pending_touch_action: str | None = None

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal pending_touch_action
        if event == cv2.EVENT_LBUTTONUP:
            pending_touch_action = touch_action_at(controls, x, y)

    if not args.no_display:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_TITLE, on_mouse)
    try:
        while True:
            sequence, frame, captured = camera.latest()
            if frame is not None:
                last_frame = frame
                if sequence > last_camera_sequence:
                    worker.submit(sequence, frame, captured)
                    last_camera_sequence = sequence
            result = worker.latest_after(last_result_sequence)
            if result is not None and last_frame is not None:
                target = selector.update(result.detections, last_frame.shape[1], result.completed_time)
                scene_objects = scene_tracker.update(result.detections, last_frame.shape[1], result.completed_time)
                last_result_sequence = result.sequence
                rates.detections += 1
                rates.inference_ms = result.inference_ms
                rates.capture_latency_ms = (result.completed_time - result.capture_time) * 1000.0
            else:
                now = time.perf_counter()
                target = selector.current(now)
                scene_objects = scene_tracker.current(now)
            if last_frame is None:
                if camera.error:
                    raise RuntimeError(camera.error)
                time.sleep(0.01)
                continue
            if not args.no_display and view != "camera":
                lane = lane_detector.update(last_frame, time.perf_counter())
            else:
                lane = lane_detector.current(time.perf_counter())
            rendered = compose_view(last_frame, target, scene_objects, lane, rates, view, worker.error or camera.error)
            if not args.no_display:
                controls = draw_touch_controls(rendered)
            rates.display_frames += 1
            if not args.no_display:
                cv2.imshow(WINDOW_TITLE, rendered)
                key = cv2.waitKey(1) & 0xFF
                action = pending_touch_action
                pending_touch_action = None
                if key in (27, ord("q"), ord("Q")) or action == "quit":
                    break
                if key == ord("1") or action == "world":
                    view = "world"
                elif key == ord("2") or action == "camera":
                    view = "camera"
                elif key == ord("3") or action == "split":
                    view = "split"
                elif key in (ord("s"), ord("S")):
                    out = PROJECT_ROOT / "logs" / f"pi3b-{int(time.time())}.jpg"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), rendered)
            if deadline is not None and time.perf_counter() >= deadline:
                break
            next_tick += 1.0 / args.fps
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.5:
                next_tick = time.perf_counter()
    finally:
        worker.close()
        camera.close()
        cv2.destroyAllWindows()
    report = rates.report()
    print(json.dumps(report, indent=2))
    if args.benchmark_report:
        args.benchmark_report.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
