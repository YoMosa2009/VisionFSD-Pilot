"""Asynchronous YOLO object perception for a non-blocking visualization loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from gpu_lock import INFERENCE_LOCK
from visionfsd import (
    DetectedObject,
    MotionEstimator,
    PROJECT_ROOT,
    TemporalTrackFusion,
    TrackStabilizer,
    focal_length_px,
    objects_from_yolo_result,
)


@dataclass(frozen=True)
class ObjectPerceptionResult:
    sequence: int
    objects: list[DetectedObject]
    pipeline_ms: float
    capture_time: float = 0.0   # perf_counter when frame was submitted
    completed_time: float = 0.0  # perf_counter when result became ready


class AsyncObjectPerception:
    """Runs tracking on the newest submitted frame without blocking rendering."""

    def __init__(self, model, confidence: float, imgsz: int, device: str,
                 fov: float, camera_height: float, horizon_ratio: float) -> None:
        self._model = model
        self._confidence = confidence
        self._imgsz = imgsz
        self._device = device
        self._fov = fov
        self._camera_height = camera_height
        self._horizon_ratio = horizon_ratio
        self._motion = MotionEstimator()
        self._stabilizer = TrackStabilizer()
        self._fusion = TemporalTrackFusion()
        self._condition = threading.Condition()
        # pending: (sequence, frame, capture_time)
        self._pending: tuple[int, np.ndarray, float] | None = None
        self._latest: ObjectPerceptionResult | None = None
        self._stopping = False
        self._error: Exception | None = None
        self._force_housekeep = False
        self._detect_passes = 0
        self._busy = False
        self._thread = threading.Thread(target=self._worker, name="visionfsd-objects", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, sequence: int,
               capture_time: float | None = None) -> None:
        with self._condition:
            # Capture/video-reader arrays are immutable after publication; a
            # shared reference avoids allocating another 2.6 MB every frame.
            # Always keep only the newest pending frame (drop backlog → lower lag).
            cap_t = float(capture_time) if capture_time is not None else time.perf_counter()
            self._pending = (sequence, frame, cap_t)
            self._condition.notify()

    def idle(self) -> bool:
        """True when the worker can accept work immediately (no backlog)."""
        with self._condition:
            return (not self._busy) and self._pending is None and not self._stopping

    def latest_after(self, sequence: int, *,
                     min_capture_time: float = 0.0,
                     max_age_s: float = 0.0) -> ObjectPerceptionResult | None:
        """Return a newer result, optionally dropping stale capture ages.

        ``max_age_s`` > 0: reject results older than that many seconds vs now.
        ``min_capture_time`` > 0: reject results captured before that stamp.
        """
        with self._condition:
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            latest = self._latest
        if min_capture_time > 0.0 and latest.capture_time > 0.0:
            if latest.capture_time < min_capture_time:
                return None
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
                    self._busy = True
                # If this frame is already ancient (UI moved on) and a newer
                # pending already arrived, skip to the fresh one — cuts lag when
                # GPU lock made us wait.
                if capture_time > 0.0 and (time.perf_counter() - capture_time) > 0.45:
                    with self._condition:
                        if self._pending is not None:
                            sequence, frame, capture_time = self._pending
                            self._pending = None
                        elif (time.perf_counter() - capture_time) > 0.80:
                            # Too old and nothing newer — drop rather than
                            # publish a multi-second-late 3D update.
                            self._busy = False
                            continue
                started = time.perf_counter()
                try:
                    # Serialize with YOLOPv2 when both use the Intel iGPU.
                    with INFERENCE_LOCK:
                        # COCO ids: person, bicycle, car, motorcycle, bus, train,
                        # truck, traffic light, fire hydrant, stop sign, parking meter.
                        # conf floor is low so lights/signs survive into ByteTrack's
                        # second association stage (per-class floors re-filter later).
                        # max_det keeps NMS/postprocess bounded for busy multi-lane
                        # scenes without hiding the nearest relevant traffic.
                        result = self._model.track(
                            frame, persist=True, verbose=False,
                            conf=min(0.08, self._confidence),
                            imgsz=self._imgsz, device=self._device,
                            classes=[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12],
                            max_det=40,
                            tracker=str(PROJECT_ROOT / "config" / "visionfsd_bytetrack.yaml"),
                        )[0]
                    now = time.perf_counter()
                    height, width = frame.shape[:2]
                    focal_px = focal_length_px(width, self._fov)
                    observations = objects_from_yolo_result(
                        result, width, focal_px, self._motion, now, height,
                        self._camera_height, self._horizon_ratio,
                    )
                    # Drop Ultralytics Results ASAP — holds large tensor refs.
                    del result
                    objects = self._stabilizer.update(observations, now)
                    # Temporal EMA fusion after stabilizer (anti-flicker, cheap).
                    objects = self._fusion.update(objects, now)
                    # All tracker/motion/stabilizer maintenance stays on THIS
                    # worker thread (never the render thread). ByteTrack
                    # removed_stracks growth was a classic ~1–2 min stutter cliff.
                    self._detect_passes += 1
                    force = self._force_housekeep
                    if force:
                        self._force_housekeep = False
                    # Housekeep every pass (cheap list trims); aggressive more often.
                    self._housekeep_tracker(
                        now,
                        aggressive=force or (self._detect_passes % 8 == 0),
                    )
                    # Drop large Ultralytics/OpenVINO leftovers that pin RAM.
                    try:
                        predictor = getattr(self._model, "predictor", None)
                        if predictor is not None:
                            if hasattr(predictor, "results"):
                                predictor.results = None
                            # Batch tensors from the last forward.
                            for attr in ("im", "im0", "imgs", "batch"):
                                if hasattr(predictor, attr):
                                    try:
                                        setattr(predictor, attr, None)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    pipeline_ms = (time.perf_counter() - started) * 1000.0
                    completed = time.perf_counter()
                    with self._condition:
                        self._latest = ObjectPerceptionResult(
                            sequence, objects, pipeline_ms,
                            capture_time=capture_time, completed_time=completed,
                        )
                        self._error = None
                        self._busy = False
                except Exception as exc:
                    # Keep the worker alive after a single bad frame.
                    with self._condition:
                        self._error = exc
                        self._busy = False
                    time.sleep(0.05)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._busy = False

    def _housekeep_tracker(self, now: float, *, aggressive: bool = False) -> None:
        """Bound ByteTrack / motion / stabilizer state (worker-thread only)."""
        self._motion.prune(now, max_age_s=2.5 if aggressive else 3.5)
        try:
            self._fusion._prune(now, set())
        except Exception:
            pass
        try:
            tracks = getattr(self._stabilizer, "_tracks", None)
            if isinstance(tracks, dict) and tracks:
                stale = [
                    tid for tid, state in tracks.items()
                    if (now - float(getattr(state, "last_update", now))) > 2.5
                    or int(getattr(state, "misses", 0)) > 16
                ]
                for tid in stale:
                    tracks.pop(tid, None)
                cap = 20 if aggressive else 28
                if len(tracks) > cap:
                    ranked = sorted(
                        tracks.items(),
                        key=lambda item: (
                            int(getattr(item[1], "misses", 0)),
                            -int(getattr(item[1], "hits", 0)),
                            -float(getattr(item[1], "last_update", 0.0)),
                        ),
                    )
                    for tid, _ in ranked[cap:]:
                        tracks.pop(tid, None)
        except Exception:
            pass
        # Tight caps: long sessions otherwise accumulate hundreds of STrack
        # objects and hitch every few seconds when GC finally runs.
        removed_cap = 16 if aggressive else 24
        lost_cap = 8 if aggressive else 12
        try:
            predictor = getattr(self._model, "predictor", None)
            if predictor is None:
                return
            trackers = getattr(predictor, "trackers", None)
            if not trackers:
                return
            for tracker in trackers:
                removed = getattr(tracker, "removed_stracks", None)
                if isinstance(removed, list) and len(removed) > removed_cap:
                    del removed[:-removed_cap]
                lost = getattr(tracker, "lost_stracks", None)
                if isinstance(lost, list) and len(lost) > lost_cap:
                    del lost[:-lost_cap]
                tracked = getattr(tracker, "tracked_stracks", None)
                if isinstance(tracked, list) and len(tracked) > 40:
                    del tracked[:-32]
                # Some ultralytics builds keep frame_id / score history on tracks.
                if aggressive:
                    for attr_name in ("removed_stracks", "lost_stracks"):
                        bucket = getattr(tracker, attr_name, None)
                        if isinstance(bucket, list) and len(bucket) > removed_cap:
                            bucket.clear()
        except Exception:
            pass

    def maintenance(self) -> None:
        """Render-thread hint only — real work runs on the object worker."""
        # Flag the worker to do an aggressive pass soon without touching
        # tracker lists from the UI thread (avoids races + render hitches).
        self._force_housekeep = True

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
