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
CAR_CLASS_ID = 2  # COCO car. The Pi renderer intentionally shows this class only.
CAR_WIDTH_M = 1.80


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


def focal_length_px(frame_width: int, fov_deg: float) -> float:
    return (frame_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def estimate_car_range_m(box: tuple[int, int, int, int], frame_width: int, fov_deg: float) -> float:
    """Coarse pinhole range for visualization; never a driving measurement."""
    width = max(1, box[2] - box[0])
    return float(np.clip(CAR_WIDTH_M * focal_length_px(frame_width, fov_deg) / width, 1.0, 120.0))


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
        if all(box_iou(item.box, chosen.box) < threshold for chosen in kept):
            kept.append(item)
    return kept


class TFLiteVehicleDetector:
    """TFLite/LiteRT detector that decodes common YOLO and DetectionPostProcess outputs."""

    def __init__(self, model_path: Path, confidence: float, threads: int) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as exc:  # pragma: no cover - requires Pi LiteRT runtime
            raise RuntimeError("tflite-runtime is missing; run pi3b/install.sh") from exc
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
        if np.issubdtype(dtype, np.integer):
            value = rgb.astype(np.float32)
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
            if int(round(float(class_id))) != CAR_CLASS_ID or float(score) < self._confidence:
                continue
            y1, x1, y2, x2 = [float(value) for value in raw_box]
            # DetectionPostProcess normalized boxes are normally [0, 1].
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
                x1, x2 = x1 * frame_w, x2 * frame_w
                y1, y2 = y1 * frame_h, y2 * frame_h
            box = (max(0, int(x1)), max(0, int(y1)), min(frame_w - 1, int(x2)), min(frame_h - 1, int(y2)))
            if box[2] - box[0] >= 3 and box[3] - box[1] >= 3:
                decoded.append(Detection("car", float(score), box))
        return nms(decoded)

    def _decode_yolo(self, raw: np.ndarray, frame_w: int, frame_h: int) -> list[Detection]:
        array = np.squeeze(raw)
        if array.ndim != 2:
            raise RuntimeError(f"Unsupported TFLite detector output shape: {raw.shape}")
        # YOLO11 export: [84, candidates]; older YOLO variants: [85, candidates].
        if 6 <= array.shape[0] <= 128 and array.shape[1] > array.shape[0]:
            array = array.T
        if array.shape[1] < 4 + CAR_CLASS_ID + 1:
            raise RuntimeError(f"Unsupported TFLite detector channels: {array.shape}")
        has_objectness = array.shape[1] >= 85
        class_start = 5 if has_objectness else 4
        decoded: list[Detection] = []
        for row in array:
            score = float(row[class_start + CAR_CLASS_ID])
            if has_objectness:
                score *= float(row[4])
            if score < self._confidence:
                continue
            box = self._to_box(float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                               frame_w, frame_h, self._input_w, self._input_h)
            if box is not None:
                decoded.append(Detection("car", score, box))
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
            return self._sequence, None if self._frame is None else self._frame.copy(), self._captured

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

    It tracks only cars, chooses one centred/near vehicle, and requires a
    challenger to be materially closer for several detector updates.
    """

    def __init__(self, fov_deg: float, hold_s: float = 0.75) -> None:
        self._fov_deg = fov_deg
        self._hold_s = hold_s
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._target_id: int | None = None
        self._challenger_id: int | None = None
        self._challenge_hits = 0

    def _match(self, detection: Detection) -> int | None:
        choices = [(box_iou(detection.box, track.detection.box), track_id) for track_id, track in self._tracks.items()]
        if not choices:
            return None
        score, track_id = max(choices)
        return track_id if score >= 0.25 else None

    def update(self, detections: list[Detection], frame_width: int, now: float) -> Target | None:
        seen: set[int] = set()
        for detection in detections:
            track_id = self._match(detection)
            distance = estimate_car_range_m(detection.box, frame_width, self._fov_deg)
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
        candidates = [(track.distance_m + abs(track.bearing) * 0.08, track_id) for track_id, track in self._tracks.items() if track_id in seen]
        if not candidates:
            return self.current(now)
        _, best_id = min(candidates)
        if self._target_id not in self._tracks:
            self._target_id, self._challenger_id, self._challenge_hits = best_id, None, 0
        elif best_id != self._target_id:
            incumbent = self._tracks[self._target_id]
            challenger = self._tracks[best_id]
            if challenger.distance_m < incumbent.distance_m - 3.0:
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
        text = f"TARGET CAR {target.distance_m:.1f}m {target.bearing_deg:+.1f}deg {state}"
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


def _world_panel(size: tuple[int, int], target: Target | None, rates: Rates) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), (23, 27, 34), dtype=np.uint8)
    horizon = int(height * 0.28)
    vanishing = (width // 2, horizon)
    road = np.array([(int(width * 0.08), height), (int(width * 0.92), height), vanishing], np.int32)
    cv2.fillConvexPoly(panel, road, (49, 53, 60))
    for lateral in (-0.30, 0.30):
        bottom_x = int(width * (0.5 + lateral))
        cv2.line(panel, (bottom_x, height), vanishing, (125, 135, 145), 1, cv2.LINE_AA)
    for fraction in (0.16, 0.29, 0.45, 0.65, 0.88):
        y = int(horizon + (height - horizon) * fraction)
        half = int((width * 0.44) * fraction)
        cv2.line(panel, (width // 2 - half, y), (width // 2 + half, y), (68, 73, 80), 1, cv2.LINE_AA)
    cv2.putText(panel, "TARGET WORLD (visual estimate)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 228, 235), 1, cv2.LINE_AA)
    if target:
        closeness = 1.0 - float(np.clip((target.distance_m - 2.0) / 80.0, 0.0, 1.0))
        y = int(horizon + (height - horizon - 35) * (0.18 + 0.82 * closeness))
        x = int(width / 2 + np.clip(target.bearing_deg / 35.0, -1.0, 1.0) * width * 0.36 * (0.35 + 0.65 * closeness))
        scale = int(12 + 34 * closeness)
        body = np.array([(x - scale, y), (x + scale, y), (x + int(scale * .75), y - scale),
                         (x - int(scale * .75), y - scale)], np.int32)
        roof = np.array([(x - int(scale * .52), y - scale), (x + int(scale * .52), y - scale),
                         (x + int(scale * .28), y - int(scale * 1.55)), (x - int(scale * .28), y - int(scale * 1.55))], np.int32)
        colour = (52, 220, 80) if target.observed else (0, 170, 255)
        cv2.fillConvexPoly(panel, body, colour)
        cv2.fillConvexPoly(panel, roof, tuple(int(value * 0.72) for value in colour))
        cv2.putText(panel, f"CAR {target.distance_m:.1f}m", (x - scale, y - int(scale * 1.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
    stats = rates.report()
    cv2.putText(panel, f"D {stats['display_fps']:.1f} FPS / I {stats['detect_fps']:.1f} FPS", (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 190, 205), 1, cv2.LINE_AA)
    return panel


def compose_view(frame: np.ndarray, target: Target | None, rates: Rates, view: str, error: str) -> np.ndarray:
    camera = _camera_panel(frame, target, rates, error)
    world = _world_panel((frame.shape[1], frame.shape[0]), target, rates)
    if view == "camera":
        return camera
    if view == "world":
        return world
    return np.hstack((world, camera))


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
    detector = TFLiteVehicleDetector(args.model, args.confidence, args.threads)
    camera = LatestCamera(args.camera, args.width, args.height, args.fps)
    worker = AsyncDetector(detector)
    selector = TargetSelector(args.fov)
    rates = Rates()
    last_camera_sequence = 0
    last_result_sequence = 0
    last_frame: np.ndarray | None = None
    target: Target | None = None
    view = args.view
    deadline = rates.started + args.test_seconds if args.test_seconds > 0 else None
    next_tick = time.perf_counter()
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
                last_result_sequence = result.sequence
                rates.detections += 1
                rates.inference_ms = result.inference_ms
                rates.capture_latency_ms = (result.completed_time - result.capture_time) * 1000.0
            else:
                target = selector.current(time.perf_counter())
            if last_frame is None:
                if camera.error:
                    raise RuntimeError(camera.error)
                time.sleep(0.01)
                continue
            rendered = compose_view(last_frame, target, rates, view, worker.error or camera.error)
            rates.display_frames += 1
            if not args.no_display:
                cv2.imshow("VisionFSD Pi 3B - read only", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == ord("1"):
                    view = "world"
                elif key == ord("2"):
                    view = "camera"
                elif key == ord("3"):
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
