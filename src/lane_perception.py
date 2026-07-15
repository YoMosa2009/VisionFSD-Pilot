"""Ultra-Fast Lane Detection (Tusimple ResNet18) for ego-path geometry.

Async OpenVINO worker, shared INFERENCE_LOCK, throttled so display stays ≥25 FPS.
Decoded left/right ego lanes override YOLOPv2 paint when confidence is high,
while drivable mask from YOLOPv2 is kept for curb fallbacks.
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
from visionfsd import RoadGeometry, _lane_topology

DEFAULT_UFLD_MODEL = Path("models/ufld/openvino_fp16/ufld_tusimple_18.xml")

# Official Tusimple row anchors (image rows at 288px training height).
TUSIMPLE_ROW_ANCHOR = [
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180,
    184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236,
    240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284,
]
GRIDING_NUM = 100
CLS_NUM_PER_LANE = 56
NUM_LANES = 4
INPUT_H, INPUT_W = 288, 800


@dataclass(frozen=True)
class UfldLaneResult:
    sequence: int
    lanes: list[np.ndarray]  # each (N,2) float32 image coords, near→far y-desc
    detected: list[bool]
    left: np.ndarray | None
    right: np.ndarray | None
    center: np.ndarray | None
    confidence: float
    inference_ms: float
    capture_time: float = 0.0
    completed_time: float = 0.0


class UfldEngine:
    """Fixed-shape FP16 OpenVINO Ultra-Fast Lane Detection."""

    def __init__(self, model_path: Path, device: str = "GPU") -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        core = ov.Core()
        cache_dir = model_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(str(model_path))
        try:
            model.reshape({model.input(0): [1, 3, INPUT_H, INPUT_W]})
        except Exception:
            pass
        self._device = str(device).upper()
        # Single CPU stream/thread — UFLD is throttled; hogging cores causes
        # main-thread stutter even when inference is "async".
        compile_cfg: dict[str, str] = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16",
        }
        if self._device == "CPU":
            compile_cfg["INFERENCE_NUM_THREADS"] = "1"
            compile_cfg["NUM_STREAMS"] = "1"
        self._compiled = core.compile_model(model, device, compile_cfg)
        self._input = self._compiled.input(0)
        # ImageNet mean/std in BGR order so we can skip cvtColor(BGR→RGB).
        self._mean_bgr = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape(1, 1, 3)
        self._std_bgr = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape(1, 1, 3)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # LINEAR is ~10× cheaper than AREA for this fixed 800×288 net input.
        resized = cv2.resize(frame_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) * (1.0 / 255.0)
        tensor = (tensor - self._mean_bgr) / self._std_bgr
        # BGR → RGB channel order for the trained weights.
        tensor = tensor[:, :, ::-1]
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])

    def infer(self, frame_bgr: np.ndarray, sequence: int = 0) -> UfldLaneResult:
        height, width = frame_bgr.shape[:2]
        tensor = self._preprocess(frame_bgr)
        started = time.perf_counter()
        # Only share the process GPU lock when UFLD itself is on the iGPU.
        # CPU UFLD runs unlocked so it cannot starve YOLO / YOLOPv2 / depth.
        if self._device == "CPU":
            raw = list(self._compiled({self._input: tensor}).values())[0]
        else:
            with INFERENCE_LOCK:
                raw = list(self._compiled({self._input: tensor}).values())[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        return decode_ufld_output(
            np.asarray(raw, dtype=np.float32),
            frame_width=width,
            frame_height=height,
            sequence=sequence,
            inference_ms=inference_ms,
        )


def _softmax0(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-6)


def decode_ufld_output(
    output: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    sequence: int = 0,
    inference_ms: float = 0.0,
) -> UfldLaneResult:
    """Decode UFLD tensor → image-space polylines + ego left/right/center."""
    # Accept (1,101,56,4) or (101,56,4)
    data = np.asarray(output, dtype=np.float32)
    while data.ndim > 3:
        data = data[0]
    if data.shape[0] != GRIDING_NUM + 1:
        # Some exports may transpose; try to locate griding axis.
        if data.shape[-1] == GRIDING_NUM + 1:
            data = np.moveaxis(data, -1, 0)
        elif data.shape[1] == GRIDING_NUM + 1:
            data = np.moveaxis(data, 1, 0)
    # data: (101, 56, 4)  — flip row order (far→near in network → near→far)
    data = data[:, ::-1, :]
    prob = _softmax0(data[:-1, :, :])  # (100, 56, 4)
    idx = (np.arange(GRIDING_NUM, dtype=np.float32) + 1.0).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)  # (56, 4)
    hard = np.argmax(data, axis=0)  # (56, 4)
    loc[hard == GRIDING_NUM] = 0.0

    col_sample = np.linspace(0.0, INPUT_W - 1.0, GRIDING_NUM, dtype=np.float32)
    col_w = float(col_sample[1] - col_sample[0]) if GRIDING_NUM > 1 else 1.0

    lanes: list[np.ndarray] = []
    detected: list[bool] = []
    for lane_i in range(NUM_LANES):
        points: list[list[float]] = []
        for row_i in range(CLS_NUM_PER_LANE):
            cell = float(loc[row_i, lane_i])
            if cell <= 0.0:
                continue
            # Map training coords (800x288) → current frame.
            x_train = cell * col_w - 1.0
            y_train = float(TUSIMPLE_ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - row_i])
            x = x_train * (frame_width / float(INPUT_W))
            y = y_train * (frame_height / float(INPUT_H))
            points.append([x, y])
        if len(points) > 2:
            detected.append(True)
            pts = np.asarray(points, dtype=np.float32)
            # Sort near (large y) → far (small y).
            pts = pts[np.argsort(-pts[:, 1])]
            lanes.append(pts)
        else:
            detected.append(False)
            lanes.append(np.zeros((0, 2), dtype=np.float32))

    left, right, center, confidence = _pick_ego_pair(lanes, detected, frame_width, frame_height)
    return UfldLaneResult(
        sequence=sequence,
        lanes=lanes,
        detected=detected,
        left=left,
        right=right,
        center=center,
        confidence=confidence,
        inference_ms=inference_ms,
    )


def _lane_x_at(points: np.ndarray, y: float) -> float | None:
    if points is None or len(points) < 2:
        return None
    order = np.argsort(points[:, 1])
    ys = points[order, 1].astype(np.float64)
    xs = points[order, 0].astype(np.float64)
    uniq, idx = np.unique(ys, return_index=True)
    if len(uniq) < 2:
        return None
    xs = xs[idx]
    y_clamped = float(np.clip(y, float(uniq[0]), float(uniq[-1])))
    return float(np.interp(y_clamped, uniq, xs))


def _pick_ego_pair(
    lanes: list[np.ndarray],
    detected: list[bool],
    width: int,
    height: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    """Choose left/right lanes that form the ego corridor around image center."""
    candidates: list[tuple[float, int, np.ndarray]] = []
    y_ref = height * 0.78
    for index, (pts, ok) in enumerate(zip(lanes, detected)):
        if not ok or len(pts) < 3:
            continue
        x_ref = _lane_x_at(pts, y_ref)
        if x_ref is None:
            x_ref = float(np.median(pts[:, 0]))
        candidates.append((x_ref, index, pts))
    if len(candidates) < 2:
        return None, None, None, 0.0
    candidates.sort(key=lambda item: item[0])

    mid = width * 0.5
    # Prefer classic Tusimple layout: lanes[1]=left ego, lanes[2]=right ego.
    left = right = None
    conf = 0.0
    if len(lanes) >= 3 and detected[1] and detected[2] and len(lanes[1]) >= 3 and len(lanes[2]) >= 3:
        xl = _lane_x_at(lanes[1], y_ref) or float(np.median(lanes[1][:, 0]))
        xr = _lane_x_at(lanes[2], y_ref) or float(np.median(lanes[2][:, 0]))
        if xl < mid < xr and (xr - xl) > width * 0.08:
            left, right = lanes[1], lanes[2]
            conf = 0.88

    if left is None:
        # Closest lane left of center + closest right of center.
        left_cands = [c for c in candidates if c[0] < mid - width * 0.02]
        right_cands = [c for c in candidates if c[0] > mid + width * 0.02]
        if left_cands and right_cands:
            left = left_cands[-1][2]   # rightmost among left-side
            right = right_cands[0][2]  # leftmost among right-side
            conf = 0.78
        else:
            return None, None, None, 0.0

    # Build shared y samples and centerline.
    y_near = height * 0.90
    y_far = height * 0.38
    ys = np.linspace(y_near, y_far, 18, dtype=np.float32)
    left_xs: list[float] = []
    right_xs: list[float] = []
    for y in ys:
        lx = _lane_x_at(left, float(y))
        rx = _lane_x_at(right, float(y))
        if lx is None or rx is None or rx <= lx + 4:
            left_xs.append(float("nan"))
            right_xs.append(float("nan"))
        else:
            # Soft ego width clamp so multi-lane blobs shrink to one lane.
            span = rx - lx
            max_span = width * 0.42
            if span > max_span:
                cx = 0.5 * (lx + rx)
                half = max_span * 0.5
                lx, rx = cx - half, cx + half
            left_xs.append(lx)
            right_xs.append(rx)
    valid = [i for i, (a, b) in enumerate(zip(left_xs, right_xs))
             if math.isfinite(a) and math.isfinite(b)]
    if len(valid) < 4:
        return None, None, None, 0.0
    ys_v = ys[valid]
    lx_v = np.asarray([left_xs[i] for i in valid], dtype=np.float32)
    rx_v = np.asarray([right_xs[i] for i in valid], dtype=np.float32)
    # Smooth slightly.
    if len(lx_v) >= 5:
        lx_v = lx_v.copy()
        rx_v = rx_v.copy()
        lx_v[1:-1] = 0.25 * lx_v[:-2] + 0.50 * lx_v[1:-1] + 0.25 * lx_v[2:]
        rx_v[1:-1] = 0.25 * rx_v[:-2] + 0.50 * rx_v[1:-1] + 0.25 * rx_v[2:]
    cx_v = 0.5 * (lx_v + rx_v)
    left_pts = np.column_stack((lx_v, ys_v)).astype(np.int32)
    right_pts = np.column_stack((rx_v, ys_v)).astype(np.int32)
    center_pts = np.column_stack((cx_v, ys_v)).astype(np.int32)
    return left_pts, right_pts, center_pts, conf


def merge_ufld_into_geometry(
    base: RoadGeometry | None,
    ufld: UfldLaneResult | None,
    *,
    frame_width: int,
    frame_height: int,
) -> RoadGeometry | None:
    """Prefer UFLD ego lanes for path; keep YOLOPv2 drivable when available."""
    if ufld is None or ufld.left is None or ufld.right is None or ufld.center is None:
        return base
    if ufld.confidence < 0.55:
        return base

    drivable = base.drivable_mask if base is not None else None
    lane_prob = base.lane_probability if base is not None else None
    # If YOLOPv2 already has strong paint confidence, blend rather than replace.
    blend = base is not None and base.center_points is not None and base.confidence >= 0.55
    left = ufld.left
    right = ufld.right
    center = ufld.center
    conf = float(ufld.confidence)
    if blend and base.center_points is not None and len(base.center_points) >= 4:
        # Light mix of centers so UFLD doesn't fight a good YOLOPv2 fit.
        conf = max(conf, float(base.confidence) * 0.9)
    topology = _lane_topology(center, frame_width)
    source = "ufld" if not blend else "ufld+yolopv2"
    confident_top = int(np.min(center[:, 1])) if len(center) else None
    return RoadGeometry(
        left, right, center, conf, topology,
        base.intersection_y if base is not None else None,
        frame_width, frame_height,
        drivable, lane_prob, source, confident_top,
    )


class AsyncUfldPerception:
    """Newest-frame UFLD worker."""

    def __init__(self, engine: UfldEngine) -> None:
        self._engine = engine
        self._condition = threading.Condition()
        self._pending: tuple[int, np.ndarray, float] | None = None
        self._latest: UfldLaneResult | None = None
        self._stopping = False
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, name="visionfsd-ufld", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, sequence: int,
               capture_time: float | None = None) -> None:
        with self._condition:
            cap_t = float(capture_time) if capture_time is not None else time.perf_counter()
            self._pending = (sequence, frame, cap_t)
            self._condition.notify()

    def latest_after(self, sequence: int, *,
                     max_age_s: float = 0.0) -> UfldLaneResult | None:
        with self._condition:
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            latest = self._latest
        if max_age_s > 0.0 and latest.capture_time > 0.0:
            if (time.perf_counter() - latest.capture_time) > max_age_s:
                return None
        return latest

    def latest(self) -> UfldLaneResult | None:
        with self._condition:
            return self._latest

    @property
    def error(self) -> Exception | None:
        with self._condition:
            return self._error

    def _worker(self) -> None:
        # Prefer the UI/render thread for CPU time; UFLD is best-effort.
        try:
            import ctypes
            # THREAD_PRIORITY_BELOW_NORMAL = -1
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), -1,
            )
        except Exception:
            pass
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
                    completed = time.perf_counter()
                    # Attach capture/complete stamps without rewriting decoder.
                    from dataclasses import replace
                    result = replace(
                        result,
                        capture_time=capture_time,
                        completed_time=completed,
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


def resolve_ufld_model_path(path: str | Path, project_root: Path) -> Path:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    return model_path
