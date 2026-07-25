"""Reliable live webcam capture for VisionFSD Pilot.

Design goals (learned the hard way):
  * Never freeze the UI on the first frame.
  * Always expose only the *newest* frame (drop backlog).
  * Prefer a simple in-process OpenCV grabber thread over fragile ffmpeg pipes.
  * Release the GIL during camera I/O so YOLO/OpenGL do not stop capture.

The capture thread continuously grabs; the main/render thread only samples the
latest complete frame via ``read()`` (never blocks on USB).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2
import numpy as np

# Used only for clip in the capture pacer (avoid importing elsewhere).
def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _fourcc_name(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


class LiveWebcamCapture:
    """Drop-to-newest USB webcam. Safe for the VisionFSD render loop.

    Parameters
    ----------
    camera_index:
        DirectShow / OpenCV camera index (1 = icspring on this laptop).
    width, height, fps:
        Requested mode. If the device cannot sustain it, we fall back to
        640x480 which this hardware reliably runs at ~30 FPS.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        device_name: str | None = None,
    ) -> None:
        self.camera_index = int(camera_index)
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.requested_fps = max(1, int(fps))
        self.device_name = device_name
        self.width = self.requested_width
        self.height = self.requested_height
        self.fps = self.requested_fps

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._capture_time = 0.0
        self._stop = threading.Event()
        self._failed = False
        self._error: str | None = None
        self._backend = "opencv-dshow"
        self._thread: threading.Thread | None = None

        self._open_device()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="visionfsd-live-webcam",
            daemon=True,
        )
        self._thread.start()

        # Wait for the first real frame (hard fail if the device is dead).
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frame is not None and self._sequence > 0:
                    break
                if self._failed:
                    break
            time.sleep(0.02)
        with self._lock:
            ok = self._frame is not None and self._sequence > 0
        if not ok:
            self.release()
            raise RuntimeError(
                self._error
                or f"Camera index {self.camera_index} produced no frames"
            )

    def _open_device(self) -> None:
        # Try requested resolution with MJPEG first (USB webcams need MJPEG for 720p@30).
        attempts: list[tuple[int, int, bool]] = [
            (self.requested_width, self.requested_height, True),
            (1280, 720, True),
            (800, 600, True),
            (640, 480, True),
            (640, 480, False),
        ]
        # Deduplicate while preserving order.
        seen: set[tuple[int, int, bool]] = set()
        ordered: list[tuple[int, int, bool]] = []
        for item in attempts:
            if item not in seen:
                seen.add(item)
                ordered.append(item)

        last_err = "open failed"
        for w, h, want_mjpg in ordered:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                last_err = "VideoCapture open failed"
                continue
            # Order matters on some UVC drivers: size first, then FOURCC, then FPS.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if want_mjpg:
                try:
                    # MJPG = 0x47504A4D little-endian 'MJPG'
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FOURCC, 0x47504A4D)
                except Exception:
                    pass
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # Warm a few frames and measure short-term rate.
            ok_count = 0
            t0 = time.perf_counter()
            last = None
            for _ in range(12):
                ok, frame = cap.read()
                if ok and frame is not None:
                    ok_count += 1
                    last = frame
            dt = max(1e-3, time.perf_counter() - t0)
            rate = ok_count / dt
            if last is None or ok_count < 3:
                cap.release()
                last_err = f"no frames at {w}x{h}"
                continue
            # Accept if we got a usable stream. Prefer >= 12 FPS for live feel.
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or last.shape[1])
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or last.shape[0])
            if rate < 8.0 and (w > 640 or h > 480):
                # Too slow at this mode — try a smaller mode.
                cap.release()
                last_err = f"only {rate:.1f} FPS at {aw}x{ah}"
                continue
            self._cap = cap
            self.width = aw
            self.height = ah
            self.fps = max(1, int(round(rate))) if rate < self.requested_fps else self.requested_fps
            fcc = _fourcc_name(cap)
            self._backend = f"opencv-dshow-{fcc.strip() or 'raw'}-{aw}x{ah}"
            # Seed first frame so the UI never starts blank.
            with self._lock:
                self._frame = last.copy()
                self._sequence = 1
                self._capture_time = time.perf_counter()
            return
        raise RuntimeError(last_err)

    def _capture_loop(self) -> None:
        """Tight grab loop: always keep only the newest frame."""
        # Raise priority a bit so capture is not starved by the UI thread.
        if os.name == "nt":
            try:
                import ctypes
                k32 = ctypes.windll.kernel32
                k32.SetThreadPriority(k32.GetCurrentThread(), 1)  # ABOVE_NORMAL
            except Exception:
                pass

        cap = self._cap
        if cap is None:
            self._failed = True
            return
        consecutive_fail = 0
        # Pace near camera rate. A free-spin loop was allocating 100+ full frames
        # per second → GC thrash and stutter after ~1–2 minutes.
        period = 1.0 / _clip(float(self.fps if self.fps > 1 else 30.0), 15.0, 30.0)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                # Peel one stale buffered frame (BUFFERSIZE=1 is unreliable on
                # Windows DSHOW), then retrieve the latest.
                cap.grab()
                ok, frame = cap.read()
            except Exception as exc:
                self._error = str(exc)
                consecutive_fail += 1
                if consecutive_fail >= 60:
                    self._failed = True
                    return
                time.sleep(0.01)
                continue

            if not ok or frame is None:
                consecutive_fail += 1
                if consecutive_fail >= 60:
                    self._failed = True
                    self._error = "camera read failed repeatedly"
                    return
                time.sleep(0.005)
                continue

            consecutive_fail = 0
            # Contiguous buffer; render thread always copies again in read().
            frame = np.ascontiguousarray(frame)
            now = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._sequence += 1
                self._capture_time = now
            # Sleep until next camera slot (keeps CPU/GC load bounded).
            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0005:
                time.sleep(sleep_s)
            elif sleep_s < -period:
                # Fell far behind — resync clock so we don't busy-spin catch-up.
                next_t = time.perf_counter()

    def isOpened(self) -> bool:
        return (
            self._cap is not None
            and not self._failed
            and not self._stop.is_set()
            and (self._thread is None or self._thread.is_alive())
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return a *copy* of the newest frame (never blocks on USB)."""
        with self._lock:
            if self._frame is None:
                return False, None
            frame = self._frame
            seq = self._sequence
            # Shallow metadata for callers that inspect _last_seq.
            self._last_seq = seq  # type: ignore[attr-defined]
            # Return a copy so the capture thread can replace _frame freely and
            # so overlay drawing cannot mutate the shared latest frame.
            out = frame.copy()
        return True, out

    def get(self, prop: int) -> float:
        if prop in (3, getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)):
            return float(self.width)
        if prop in (4, getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)):
            return float(self.height)
        if prop in (5, getattr(cv2, "CAP_PROP_FPS", 5)):
            return float(self.fps)
        if prop in (6, getattr(cv2, "CAP_PROP_FOURCC", 6)):
            if self._cap is not None:
                return float(self._cap.get(cv2.CAP_PROP_FOURCC) or 0)
            return 0.0
        if self._cap is not None:
            try:
                return float(self._cap.get(prop) or 0.0)
            except Exception:
                return 0.0
        return 0.0

    def set(self, _prop: int, _value: float) -> bool:
        return False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def sequence(self) -> int:
        with self._lock:
            return int(self._sequence)

    def release(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        with self._lock:
            self._frame = None


# Back-compat aliases used by visionfsd_3d imports / older call sites.
ProcessWebcamCapture = LiveWebcamCapture  # type: ignore[misc]


def capture_process_main(*_args: Any, **_kwargs: Any) -> None:
    """Deprecated multiprocessing entrypoint (kept so old spawns fail soft)."""
    return
