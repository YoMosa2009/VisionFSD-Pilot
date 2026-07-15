"""GPU-rendered, read-only 3D world visualizer for VisionFSD Pilot.

This module renders procedural low-poly meshes from perception tracks. It has
no vehicle-control, CAN-bus, steering, braking, or actuator code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Deque

import cv2
import numpy as np
import pygame
import torch
import OpenGL

# PyOpenGL wraps every GL call with glGetError checking by default. This app
# issues thousands of immediate-mode calls per frame, so that safety net is a
# top *CPU* cost of the render stage (measured 46-84 ms/frame in split view).
# Disable it for release running; re-enable locally when debugging GL issues.
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_12, glutBitmapCharacter, glutInit
from ultralytics import YOLO

from depth_perception import (
    DEFAULT_DEPTH_MODEL,
    AsyncDepthPerception,
    DepthAnythingEngine,
    fuse_objects_with_depth,
    resolve_depth_model_path,
)
from lane_perception import (
    DEFAULT_UFLD_MODEL,
    AsyncUfldPerception,
    UfldEngine,
    UfldLaneResult,
    merge_ufld_into_geometry,
    resolve_ufld_model_path,
)
from object_perception import AsyncObjectPerception
from road_perception import AsyncRoadPerception, YOLOPv2RoadEngine

from environment_intel import (
    FS_BLOCKED,
    FS_FREE,
    FS_UNKNOWN,
    latest_environment,
    prune_environment_caches,
    update_environment_intel,
)
from visionfsd import (
    PROJECT_ROOT,
    SCREENSHOT_DIR,
    ROAD_CLASSES,
    SIGNAL_CLASSES,
    VEHICLE_CLASSES,
    DetectedObject,
    RoadGeometry,
    RoadGeometryTracker,
    annotate_signal_states,
    apply_road_range_prior,
    draw_camera_view,
    ego_path_style,
    estimate_merge_flag,
    estimate_monocular_distance_m,
    focal_length_px,
    get_monitors,
    latest_road_markings,
    road_lateral_offset_m,
    scan_road_markings,
    track_quality_score,
)

_GLUT_READY = False


ROAD_COLOUR = (0.075, 0.085, 0.105)
MESH_NEUTRAL = (0.74, 0.79, 0.86)
MESH_SELECTED = (0.18, 0.62, 0.95)
MESH_LISTS: dict[tuple[str, str], int] = {}
CAMERA_TEXTURE_SIZE: tuple[int, int] | None = None
# Live 3D ego-path triangle EMA: (base_x, tip_x, tip_z).
_WORLD_GUIDE_EMA: tuple[float, float, float] | None = None
# Road-relative pose state: (lateral_m, distance_m, heading_deg, timestamp).
# Reprojected onto the *current* road centre each frame so meshes stay locked
# to the lane map instead of slowly drifting in absolute world XZ.
WORLD_POSES: dict[int, tuple[float, float, float, float]] = {}
# Raw (unsmoothed) road-relative history per track for velocity / turn gating.
WORLD_TRAJECTORY: dict[int, Deque[tuple[float, float, float]]] = {}
# Consecutive frames a track has shown strong lateral world motion; used to
# gate turn headings so monocular range noise cannot spin meshes on highways.
WORLD_TURN_HITS: dict[int, int] = {}
# Sticky oncoming latch: once a vehicle is classified as facing us, keep its
# mesh hard-locked at 180° (front toward camera). Counter only decays when
# the car clearly recedes — prevents 0°↔180° flip-spin and motion-yaw wobble.
WORLD_ONCOMING_LOCK: dict[int, int] = {}
# Per-track EMA of road-tangent cruise yaw so noisy centre-polyline curvature
# cannot spin same-direction meshes that are simply driving straight.
WORLD_CRUISE_YAW: dict[int, float] = {}
# Sticky 3D lane slot per track: "ego" | "left" | "right" | "far_left" | "far_right".
# Keeps adjacent cars from drifting onto the grey ego-lane ribbon.
WORLD_LANE_SLOT: dict[int, str] = {}
# Mesh local front = -Z; yaw 180° points headlights at the ego camera (+Z).
ONCOMING_HEADING_DEG = 180.0
# Grey ego-lane ribbon half-width (must match `_draw_road_world` half_lane).
EGO_ROAD_HALF_M = 1.85
# Car mesh half-width ≈ 1.0 m (see `_draw_vehicle_local`).
VEHICLE_HALF_WIDTH_M = 1.00
# Adjacent-lane display centers (metres from ego-lane centre).
ADJ_LANE_CENTER_M = 3.70
FAR_LANE_CENTER_M = 7.40
# Minimum |lateral| so an adjacent car's hull clears the ego ribbon + gap.
ADJ_LANE_CLEAR_M = EGO_ROAD_HALF_M + VEHICLE_HALF_WIDTH_M + 0.30  # ~3.15 m
# Classification thresholds (metres from ego-lane centre) with hysteresis.
LANE_EGO_ENTER_M = 1.55
LANE_EGO_EXIT_M = 2.05
LANE_FAR_ENTER_M = 5.40
# Same-dir free-driving meshes stay near road tangent; larger |yaw| is reserved
# for intentional orientation (parked / angled / turning).
CRUISE_HEADING_ABS_CAP_DEG = 28.0
# Display-slot grounding: sticky track IDs so the top-N set does not thrash
# every frame when scores are nearly equal (confidence / flicker fix).
DISPLAY_SLOT_IDS: dict[str, list[int]] = {
    "vehicle": [],
    "vru": [],
    "signal": [],
}
# Challenger scores must beat incumbents this many consecutive frames.
DISPLAY_SLOT_CHALLENGE: dict[str, dict[int, int]] = {
    "vehicle": {},
    "vru": {},
    "signal": {},
}
# Incumbent grace: keep a track even when briefly unobserved / lower score.
DISPLAY_SLOT_GRACE: dict[int, float] = {}
# Logical vehicle presence: cars that were reliably detected recently stay
# "alive" for a couple of seconds even when YOLO briefly drops them.
# Keyed by track_id → (last_obj, last_observed_t, hits, last_update_t).
_VEHICLE_PRESENCE: dict[int, tuple[DetectedObject, float, int, float]] = {}
PRESENCE_HOLD_S = 2.15          # wall-clock hold after last real observation
PRESENCE_MIN_HITS = 3           # need a few solid hits before reasoning in ghosts
PRESENCE_MAX_GHOSTS = 3         # hard cap so display stays cheap
PRESENCE_MAX_TRACKS = 24        # bound map size for long sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only 3D driving-scene visualizer")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", help="Local video path, direct stream, or YouTube URL; overrides --camera")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Seek a video source before processing")
    parser.add_argument(
        "--cookies-from-browser",
        default="",
        help=(
            "Browser profile for yt-dlp YouTube auth (chrome, edge, firefox, brave, …). "
            "Required when YouTube returns 'Sign in to confirm you’re not a bot'."
        ),
    )
    parser.add_argument(
        "--cookies",
        default="",
        help="Path to a Netscape cookies.txt exported for YouTube (alternative to --cookies-from-browser)",
    )
    parser.add_argument("--realtime-video", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip source frames when necessary to keep video playback near real time")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--model-task", choices=("detect", "segment"), default="detect")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or intel:gpu")
    parser.add_argument("--confidence", type=float, default=0.14,
                        help="Detector floor; per-class floors in CLASS_MIN_CONF refine this further")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Must match the exported OpenVINO model (yolo11n_openvino_model is 512)")
    parser.add_argument("--detect-interval", type=int, default=3,
                        help="Run object tracking every N display frames (3 targets ~20+ display FPS)")
    parser.add_argument("--road-interval", type=int, default=3,
                        help="Submit a road-inference frame every N display frames (async keeps newest)")
    parser.add_argument("--learned-road", action=argparse.BooleanOptionalAction, default=True,
                        help="Use YOLOPv2 learned drivable-area and lane segmentation")
    parser.add_argument("--road-model", default="models/yolopv2/openvino_fp16/yolopv2_road.xml")
    parser.add_argument("--road-device", default="GPU", help="OpenVINO device for YOLOPv2 road inference")
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True,
                        help="Depth Anything V2 Small for range stabilization (OpenVINO, throttled)")
    parser.add_argument("--depth-model", default=str(DEFAULT_DEPTH_MODEL),
                        help="OpenVINO IR for Depth Anything V2 Small")
    parser.add_argument("--depth-device", default="GPU",
                        help="OpenVINO device for depth (same iGPU as road/detect)")
    parser.add_argument("--depth-interval", type=int, default=6,
                        help="Submit a depth frame every N display frames (6 keeps ≥25 FPS on iGPU)")
    parser.add_argument("--ufld", action=argparse.BooleanOptionalAction, default=True,
                        help="Ultra-Fast Lane Detection for ego-path triangle (OpenVINO, throttled)")
    parser.add_argument("--ufld-model", default=str(DEFAULT_UFLD_MODEL),
                        help="OpenVINO IR for UFLD Tusimple ResNet18")
    parser.add_argument("--ufld-device", default="CPU",
                        help="OpenVINO device for UFLD (CPU default avoids iGPU lock thrash)")
    parser.add_argument("--ufld-interval", type=int, default=20,
                        help="Submit a UFLD frame every N display frames (20 keeps ≥25 FPS)")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--camera-height", type=float, default=1.25,
                        help="Camera optical-centre height above road in metres")
    parser.add_argument("--horizon-ratio", type=float, default=0.52,
                        help="Road horizon row divided by image height; calibrate after mounting")
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--window-width", type=int, default=0, help="Optional windowed output width")
    parser.add_argument("--window-height", type=int, default=0, help="Optional windowed output height")
    parser.add_argument("--view", choices=("world", "camera", "split"), default="world")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--test-frames", type=int, default=0, help="Render this many frames then exit; used for validation")
    parser.add_argument("--test-seconds", type=float, default=0.0, help="Render for this many warmed-up seconds then exit")
    parser.add_argument("--test-screenshot", action="store_true", help="Save the final self-test frame to logs")
    parser.add_argument("--test-report", help="Write a JSON perception summary, relative paths are stored under the project")
    parser.add_argument("--demo-scene", action="store_true", help="Show fixed sample tracks for renderer validation")
    parser.add_argument("--demo-road", action="store_true", help="Show a fixed curved road estimate for renderer validation")
    return parser.parse_args()


def _safe_console_text(value: object) -> str:
    """Avoid Windows legacy-console failures on emoji or other Unicode titles."""
    return str(value).encode("ascii", errors="replace").decode("ascii")


def _repair_metadata_text(value: object) -> str:
    """Repair the common UTF-8-through-Windows-codepage YouTube title artifact."""
    text = str(value)
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class LatestVideoFrameReader:
    """Decode video frames independently and expose only the newest one.

    This prevents network decoding or deliberate source-frame dropping from
    blocking detection/rendering in the main thread.
    """

    def __init__(self, cap: cv2.VideoCapture, source_fps: float, realtime: bool) -> None:
        self.cap = cap
        self.source_fps = max(1.0, source_fps)
        self.realtime = realtime
        self._condition = threading.Condition()
        self._stop = False
        self._ended = False
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._position_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._thread = threading.Thread(target=self._run, name="visionfsd-video-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        next_frame_time = time.perf_counter()
        while True:
            with self._condition:
                if self._stop:
                    break
            if self.realtime:
                delay = next_frame_time - time.perf_counter()
                if delay > 0:
                    with self._condition:
                        self._condition.wait(timeout=delay)
                        if self._stop:
                            break
                next_frame_time = max(next_frame_time + 1.0 / self.source_fps, time.perf_counter())
            ok, frame = self.cap.read()
            if not ok:
                with self._condition:
                    self._ended = True
                    self._condition.notify_all()
                break
            position = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._position_seconds = position
                self._condition.notify_all()

    def latest(self, after_sequence: int, timeout: float = 0.04) -> tuple[bool, np.ndarray | None, int]:
        """Return the newest decoded frame with low latency.

        - If a newer frame is already available, return it immediately (skip backlog).
        - Wait at most ``timeout`` for the next frame (default ~1 source frame).
        - On timeout, re-present the last frame so the UI never blocks for seconds
          on a stream hiccup (previous 2.0 s wait was a major latency stall).
        """
        with self._condition:
            if self._sequence > after_sequence and self._frame is not None:
                return True, self._frame, self._sequence
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._ended or self._stop,
                timeout=timeout,
            )
            if self._frame is not None and (self._sequence > after_sequence or not self._ended):
                # Prefer the absolute newest frame even if we timed out mid-wait.
                return True, self._frame, self._sequence
            return False, None, after_sequence

    @property
    def position_seconds(self) -> float:
        with self._condition:
            return self._position_seconds

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=3.0)


def _open_input(
    args: argparse.Namespace,
    status_cb: Callable[[str], None] | None = None,
) -> tuple[cv2.VideoCapture, bool, dict[str, object]]:
    """Open a webcam, local video, direct stream, or a resolved YouTube stream."""

    def _status(msg: str) -> None:
        print(msg, flush=True)
        if status_cb is not None:
            try:
                status_cb(msg)
            except Exception:
                pass

    if not args.source:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        # Keep a 1-frame buffer so we always grab the freshest webcam image.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, False, {"kind": "camera", "camera": args.camera}

    original_source = args.source
    resolved_source = original_source
    metadata: dict[str, object] = {"kind": "video", "source": original_source}
    lowered = original_source.lower()
    if "youtube.com/" in lowered or "youtu.be/" in lowered:
        _status("Resolving YouTube (local clip cache — not full video)...")
        from yt_dlp import YoutubeDL

        options: dict[str, object] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 20,
            # Prefer an OpenCV-friendly progressive/stream under 720p.
            # (Modern YouTube often serves AV1/VP9; OpenCV/ffmpeg still handles many.)
            "format": (
                "bestvideo[height<=720][vcodec^=avc1]/"
                "bestvideo[height<=720]/"
                "best[height<=720]/"
                "best"
            ),
            # Node is required for YouTube n/sig JS challenges (yt-dlp EJS).
            # Without it only storyboard images may appear and resolution fails.
            "js_runtimes": {"node": {}},
            "remote_components": ["ejs:github"],
        }
        cookies_browser = str(getattr(args, "cookies_from_browser", "") or "").strip()
        cookies_file = str(getattr(args, "cookies", "") or "").strip()
        cookie_path: Path | None = None
        if cookies_file:
            cookie_path = Path(cookies_file)
            if not cookie_path.is_absolute():
                cookie_path = PROJECT_ROOT / cookie_path
            if not cookie_path.is_file():
                raise FileNotFoundError(
                    f"YouTube cookies file not found: {cookie_path}\n"
                    "Export a Netscape cookies.txt from Edge (see logs/YOUTUBE_COOKIES.md) "
                    "or omit --cookies and use --cookies-from-browser."
                )
            options["cookiefile"] = str(cookie_path)
            _status(f"Using YouTube cookies file: {cookie_path}")
        elif cookies_browser:
            # yt-dlp accepts ("chrome",), ("edge",), ("firefox",), …
            # NOTE: Edge/Chrome on modern Windows often fail with DPAPI /
            # app-bound encryption. Prefer a cookies.txt export when that happens.
            browser = cookies_browser.lower().split(":")[0].strip()
            options["cookiesfrombrowser"] = (browser,)
            _status(f"Using YouTube cookies from browser: {browser}")
        else:
            _status(
                "Note: no YouTube cookies set. If resolution fails with a bot check, "
                "export cookies to logs\\youtube-cookies.txt (recommended on Edge) "
                "or use --cookies-from-browser firefox."
            )

        def _youtube_help_for_error(msg: str) -> str:
            low = msg.lower()
            if "dpapi" in low or "decrypt" in low:
                return (
                    "Edge/Chrome cookie decryption failed (Windows DPAPI / app-bound encryption).\n"
                    "yt-dlp often cannot read Edge cookies directly anymore.\n\n"
                    "Recommended fix (Edge):\n"
                    "  1) In Edge, install extension: \"Get cookies.txt LOCALLY\"\n"
                    "     (or \"cookies.txt\" — export Netscape format)\n"
                    "  2) Open https://www.youtube.com and export cookies\n"
                    "  3) Save as:  D:\\VisionFSD-Pilot\\logs\\youtube-cookies.txt\n"
                    "  4) Re-run run_youtube_test.bat  (it auto-uses that file)\n\n"
                    "Alternatives:\n"
                    "  - Use Firefox (logged into YouTube) + --cookies-from-browser firefox\n"
                    "  - See logs\\YOUTUBE_COOKIES.md\n"
                    f"Original error: {msg}"
                )
            if "not a bot" in low or "sign in" in low or "cookies" in low:
                return (
                    "YouTube blocked access (bot check / auth).\n"
                    "Fix (Edge-friendly):\n"
                    "  1) Export cookies to logs\\youtube-cookies.txt  (see logs\\YOUTUBE_COOKIES.md)\n"
                    "  2) Or try: --cookies-from-browser firefox\n"
                    f"Original error: {msg}"
                )
            return msg

        # Auto-export Edge cookies when none were provided.
        default_cookie = PROJECT_ROOT / "logs" / "youtube-cookies.txt"
        if cookie_path is None and not cookies_browser:
            if not default_cookie.is_file():
                _status("No cookies file — exporting from Edge automatically...")
                export_script = PROJECT_ROOT / "tools" / "export_edge_youtube_cookies.py"
                if export_script.is_file():
                    import subprocess
                    import sys as _sys
                    code = subprocess.call([_sys.executable, str(export_script)])
                    if code != 0:
                        raise RuntimeError(
                            "Auto cookie export from Edge failed.\n"
                            "Open Edge, sign into YouTube, visit youtube.com, then re-run.\n"
                            "Or see logs/YOUTUBE_COOKIES.md"
                        )
            if default_cookie.is_file():
                options["cookiefile"] = str(default_cookie)
                cookie_path = default_cookie
                _status(f"Using YouTube cookies file: {cookie_path}")

        # OpenCV cannot open googlevideo signed URLs reliably on Windows.
        # Download a local window around --start-seconds (stream-ish, not full video).
        from yt_dlp.utils import download_range_func

        video_id = str(
            (original_source.split("v=")[-1].split("&")[0]
             if "v=" in original_source else original_source.rstrip("/").split("/")[-1])
        )
        start_s = max(0.0, float(getattr(args, "start_seconds", 0.0) or 0.0))
        # ~8 minutes of footage is enough for pilot sessions; shorter = faster first cache.
        window_s = 8.0 * 60.0
        end_s = start_s + window_s
        cache_dir = PROJECT_ROOT / "logs" / "youtube_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{video_id}_{int(start_s)}s_{int(window_s)}s.mp4"

        # Drop empty leftover .part files from crashed prior runs (they block reuse).
        for stale in cache_dir.glob(f"{video_id}_{int(start_s)}s_*.part"):
            try:
                if stale.stat().st_size < 50_000:
                    stale.unlink(missing_ok=True)
            except OSError:
                pass

        # Reuse any ready cache for this start time (old window lengths included).
        if not (cache_path.is_file() and cache_path.stat().st_size > 1_000_000):
            for cand in sorted(
                cache_dir.glob(f"{video_id}_{int(start_s)}s_*.mp4"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            ):
                if cand.suffix.lower() == ".mp4" and cand.stat().st_size > 1_000_000:
                    cache_path = cand
                    break

        if cache_path.is_file() and cache_path.stat().st_size > 1_000_000:
            _status(
                f"Using cached clip: {cache_path.name} "
                f"({cache_path.stat().st_size / 1e6:.1f} MB)"
            )
            info = {
                "title": cache_path.stem,
                "duration": window_s,
                "format_id": "cache",
                "width": None,
                "height": None,
                "fps": None,
                "vcodec": None,
            }
        else:
            _status(
                f"Downloading clip {start_s:.0f}s–{end_s:.0f}s "
                f"(~{window_s/60:.0f} min window, first run only)..."
            )
            # Prefer progressive H.264 for OpenCV friendliness.
            options["format"] = (
                "best[ext=mp4][protocol=https][height<=720]/"
                "best[ext=mp4][height<=720]/"
                "best[height<=720]/best"
            )
            # Keep .mp4 in the template so OpenCV can open the result.
            options["outtmpl"] = str(cache_path)
            options["merge_output_format"] = "mp4"
            options["download_ranges"] = download_range_func(None, [(start_s, end_s)])
            # Stream-copy when possible (much faster than re-encode). Keyframe
            # edges may be slightly soft; fine for pilot testing.
            options["force_keyframes_at_cuts"] = False
            options["quiet"] = False
            options["no_warnings"] = False

            def _yt_progress(d: dict) -> None:
                status = d.get("status")
                if status == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    done = d.get("downloaded_bytes") or 0
                    if total:
                        pct = 100.0 * float(done) / float(total)
                        _status(f"Downloading clip… {pct:.0f}% ({done/1e6:.1f}/{total/1e6:.1f} MB)")
                    else:
                        _status(f"Downloading clip… {done/1e6:.1f} MB")
                elif status == "finished":
                    _status("Download finished — finalizing clip…")

            options["progress_hooks"] = [_yt_progress]
            try:
                with YoutubeDL(options) as ydl:
                    info = ydl.extract_info(original_source, download=True)
            except Exception as exc:
                msg = str(exc)
                if (
                    cookie_path is None
                    and ("dpapi" in msg.lower() or "decrypt" in msg.lower())
                    and default_cookie.is_file()
                ):
                    _status(f"Browser cookies failed (DPAPI). Retrying with {default_cookie} ...")
                    options.pop("cookiesfrombrowser", None)
                    options["cookiefile"] = str(default_cookie)
                    with YoutubeDL(options) as ydl:
                        info = ydl.extract_info(original_source, download=True)
                elif (
                    "not a bot" in msg.lower()
                    or "sign in" in msg.lower()
                    or "cookies" in msg.lower()
                    or "dpapi" in msg.lower()
                    or "decrypt" in msg.lower()
                ):
                    raise RuntimeError(_youtube_help_for_error(msg)) from exc
                else:
                    raise RuntimeError(_youtube_help_for_error(msg)) from exc
            # yt-dlp may omit/alter the extension (e.g. no suffix, .part then final).
            if not cache_path.is_file() or cache_path.stat().st_size < 1_000_000:
                stem = f"{video_id}_{int(start_s)}s_{int(window_s)}s"
                candidates = sorted(
                    [p for p in cache_dir.glob(stem + "*") if p.is_file()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                # Prefer real media files, skip incomplete .part
                for cand in candidates:
                    if cand.suffix.lower() == ".part":
                        continue
                    if cand.stat().st_size > 1_000_000:
                        # Normalize to .mp4 name for OpenCV.
                        target = cache_path if cache_path.suffix else cache_path.with_suffix(".mp4")
                        if cand != target:
                            try:
                                if target.exists():
                                    target.unlink()
                                cand.rename(target)
                                cache_path = target
                            except OSError:
                                cache_path = cand
                        else:
                            cache_path = cand
                        break
            if (not cache_path.is_file()) or cache_path.stat().st_size < 1_000_000:
                raise RuntimeError(
                    f"YouTube download finished but cache file missing/too small in {cache_dir}"
                )
            # Ensure .mp4 suffix when file has no extension (yt-dlp section mode).
            if cache_path.suffix == "" and cache_path.is_file():
                target = cache_path.with_suffix(".mp4")
                try:
                    if target.exists():
                        target.unlink()
                    cache_path.rename(target)
                    cache_path = target
                except OSError:
                    pass
            _status(f"Cached clip ready: {cache_path.name} ({cache_path.stat().st_size / 1e6:.1f} MB)")

        resolved_source = str(cache_path)
        if not isinstance(info, dict):
            info = {}
        title = _repair_metadata_text(info.get("title", cache_path.stem))
        metadata.update({
            "kind": "youtube-cache",
            "title": title,
            "duration_seconds": info.get("duration", window_s),
            "format_id": info.get("format_id", "cache"),
            "source_width": info.get("width"),
            "source_height": info.get("height"),
            "source_fps": info.get("fps"),
            "video_codec": info.get("vcodec"),
            "cache_path": str(cache_path),
            "requested_start_seconds": start_s,
        })
        _status(
            f"YouTube source: {_safe_console_text(title)} | "
            f"local cache @ {start_s:.0f}s for ~{window_s/60:.0f} min"
        )
        # Clip already starts at the requested time window — do not seek again.
        seek_seconds = 0.0
    else:
        seek_seconds = float(getattr(args, "start_seconds", 0.0) or 0.0)

    cap = cv2.VideoCapture(resolved_source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(resolved_source)
    if cap.isOpened() and seek_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000.0)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video source: {resolved_source}")
    return cap, True, metadata


_COLOUR_DIM_FACTOR = 1.0


def _colour(rgb: tuple[float, float, float]) -> None:
    if _COLOUR_DIM_FACTOR != 1.0:
        rgb = tuple(channel * _COLOUR_DIM_FACTOR for channel in rgb)
    glColor3f(*rgb)
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (*rgb, 1.0))


def _cuboid(center: tuple[float, float, float], size: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    """Render a solid six-faced box (CCW outward winding for GL_CULL_FACE)."""
    cx, cy, cz = center
    sx, sy, sz = (value / 2.0 for value in size)
    # 0-3: z=-sz (front), 4-7: z=+sz (back). Each face listed CCW when viewed
    # from outside so back-face culling never punches holes in vehicles.
    vertices = (
        (cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz),
    )
    faces = (
        (0, 1, 2, 3),  # -Z
        (5, 4, 7, 6),  # +Z  (CCW from +Z outside)
        (4, 0, 3, 7),  # -X
        (1, 5, 6, 2),  # +X
        (3, 2, 6, 7),  # +Y
        (4, 5, 1, 0),  # -Y
    )
    mesh_center = np.asarray(center, dtype=np.float32)
    _colour(color)
    glBegin(GL_QUADS)
    for face in faces:
        indices = list(face)
        coordinates = np.asarray([vertices[index] for index in indices], dtype=np.float32)
        normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
        # Guarantee outward normals even if a face listing is wrong.
        if float(np.dot(normal, np.mean(coordinates, axis=0) - mesh_center)) < 0.0:
            indices.reverse()
            coordinates = np.asarray([vertices[index] for index in indices], dtype=np.float32)
            normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
        norm = max(1e-6, float(np.linalg.norm(normal)))
        glNormal3f(float(normal[0] / norm), float(normal[1] / norm), float(normal[2] / norm))
        for index in indices:
            glVertex3f(*vertices[index])
    glEnd()


def _tapered_prism(center: tuple[float, float, float], base_size: tuple[float, float], top_size: tuple[float, float], height: float, color: tuple[float, float, float]) -> None:
    """Closed eight-vertex cabin mesh with outward winding and face normals."""
    cx, cy, cz = center
    base_x, base_z = base_size[0] / 2.0, base_size[1] / 2.0
    top_x, top_z = top_size[0] / 2.0, top_size[1] / 2.0
    bottom_y, top_y = cy - height / 2.0, cy + height / 2.0
    vertices = ((cx - base_x, bottom_y, cz - base_z), (cx + base_x, bottom_y, cz - base_z),
                (cx + base_x, bottom_y, cz + base_z), (cx - base_x, bottom_y, cz + base_z),
                (cx - top_x, top_y, cz - top_z), (cx + top_x, top_y, cz - top_z),
                (cx + top_x, top_y, cz + top_z), (cx - top_x, top_y, cz + top_z))
    faces = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
             (3, 0, 4, 7), (4, 5, 6, 7), (3, 2, 1, 0))
    mesh_center = np.asarray(center, dtype=np.float32)
    _colour(color)
    glBegin(GL_QUADS)
    for face in faces:
        indices = list(face)
        coordinates = np.asarray([vertices[index] for index in indices], dtype=np.float32)
        normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
        if float(np.dot(normal, np.mean(coordinates, axis=0) - mesh_center)) < 0.0:
            indices.reverse()
            coordinates = np.asarray([vertices[index] for index in indices], dtype=np.float32)
            normal = np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
        normal /= max(1e-6, float(np.linalg.norm(normal)))
        glNormal3f(float(normal[0]), float(normal[1]), float(normal[2]))
        for index in indices:
            glVertex3f(*vertices[index])
    glEnd()


def _disc(center: tuple[float, float, float], radius: float, color: tuple[float, float, float], segments: int = 16) -> None:
    _colour(color)
    glNormal3f(0, 1, 0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(*center)
    for index in range(segments + 1):
        angle = math.tau * index / segments
        glVertex3f(center[0] + math.cos(angle) * radius, center[1], center[2] + math.sin(angle) * radius)
    glEnd()


def _ellipsoid(center: tuple[float, float, float],
               radii: tuple[float, float, float],
               color: tuple[float, float, float],
               *, slices: int = 12, stacks: int = 8) -> None:
    """Solid curvy mold: unit sphere scaled into an oval (X/Y/Z radii in metres)."""
    quadric = gluNewQuadric()
    _colour(color)
    glPushMatrix()
    glTranslatef(*center)
    glScalef(*radii)
    # Low segment count keeps display-list compile + draw cheap on iGPU.
    gluSphere(quadric, 1.0, slices, stacks)
    glPopMatrix()
    gluDeleteQuadric(quadric)


def _oval_shadow(center: tuple[float, float, float],
                 radius_x: float, radius_z: float,
                 color: tuple[float, float, float],
                 segments: int = 14) -> None:
    """Flat elliptical ground contact under the vehicle mold."""
    _colour(color)
    glNormal3f(0, 1, 0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(*center)
    for index in range(segments + 1):
        angle = math.tau * index / segments
        glVertex3f(
            center[0] + math.cos(angle) * radius_x,
            center[1],
            center[2] + math.sin(angle) * radius_z,
        )
    glEnd()


def _wheel(x: float, z: float, radius: float = 0.31) -> None:
    """Low-poly tyre (kept for motorcycle / non-mold meshes only)."""
    quadric = gluNewQuadric()
    _colour((0.045, 0.05, 0.06))
    glPushMatrix()
    glTranslatef(x - 0.18, radius, z)
    glRotatef(90, 0, 1, 0)
    gluCylinder(quadric, radius, radius, 0.36, 12, 1)
    gluDisk(quadric, radius * 0.48, radius, 12, 1)
    _colour((0.46, 0.49, 0.53))
    gluDisk(quadric, 0.0, radius * 0.46, 10, 1)
    glTranslatef(0.0, 0.0, 0.36)
    _colour((0.045, 0.05, 0.06))
    gluDisk(quadric, radius * 0.48, radius, 12, 1)
    _colour((0.46, 0.49, 0.53))
    gluDisk(quadric, 0.0, radius * 0.46, 10, 1)
    glPopMatrix()
    gluDeleteQuadric(quadric)


def _limb(start: tuple[float, float, float], end: tuple[float, float, float],
          radius: float, color: tuple[float, float, float], sides: int = 8) -> None:
    """Draw a small cylinder between two local-space points."""
    direction = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    length = float(np.linalg.norm(direction))
    if length < 1e-4:
        return
    direction /= length
    angle = math.degrees(math.acos(float(np.clip(direction[2], -1.0, 1.0))))
    axis = (-float(direction[1]), float(direction[0]), 0.0)
    quadric = gluNewQuadric()
    _colour(color)
    glPushMatrix()
    glTranslatef(*start)
    if abs(angle) > 1e-3:
        glRotatef(angle, *axis)
    glPushMatrix()
    glRotatef(180.0, 1.0, 0.0, 0.0)
    gluDisk(quadric, 0.0, radius, sides, 1)
    glPopMatrix()
    gluCylinder(quadric, radius, radius * 0.92, length, sides, 1)
    glTranslatef(0.0, 0.0, length)
    gluDisk(quadric, 0.0, radius * 0.92, sides, 1)
    glPopMatrix()
    gluDeleteQuadric(quadric)


def _draw_vehicle_local(kind: str, selected: bool) -> None:
    """Grey rect/oval hybrid mold for cars/trucks/buses (no panels or trim).

    Rectangular mid-body + smooth oval nose/tail caps — reads as a car-length
    capsule without extra surface detail.
    """
    shadow = (0.015, 0.018, 0.024)
    if kind == "motorcycle":
        body = (0.52, 0.58, 0.64) if selected else (0.56, 0.58, 0.62)
        _oval_shadow((0.0, 0.012, 0.0), 0.48, 1.25, shadow, segments=16)
        # Compact stadium: short rect core + oval ends.
        _cuboid((0.0, 0.46, 0.0), (0.58, 0.58, 0.95), body)
        _ellipsoid((0.0, 0.46, -0.48), (0.30, 0.30, 0.42), body, slices=16, stacks=10)
        _ellipsoid((0.0, 0.46, 0.48), (0.30, 0.30, 0.42), body, slices=16, stacks=10)
        return

    # Cooler blue-grey when selected so lead still stands out; otherwise neutral grey.
    hull = (0.50, 0.62, 0.72) if selected else (0.58, 0.60, 0.64)

    # Full extents ≈ width 2.0 m, height 1.24 m, length ~3.2 m (half previous).
    half_w, half_h = 1.00, 0.62
    # Rectangular core covers most of the length; oval caps add rounded ends.
    core_half_z = 1.075         # ~2.15 m straight body
    cap_rz = 0.525              # each end ~0.525 m of oval → total ~3.2 m
    cy = half_h

    _oval_shadow((0.0, 0.012, 0.0), half_w * 1.05, core_half_z + cap_rz * 0.95, shadow, segments=18)

    # Mid section: solid rectangle (slightly inset so oval caps dominate the tips).
    _cuboid((0.0, cy, 0.0), (half_w * 1.92, half_h * 1.92, core_half_z * 2.0), hull)
    # Nose / tail oval caps — same radii as body so the join reads as one mold.
    _ellipsoid((0.0, cy, -core_half_z), (half_w, half_h, cap_rz), hull, slices=22, stacks=14)
    _ellipsoid((0.0, cy, core_half_z), (half_w, half_h, cap_rz), hull, slices=22, stacks=14)


def _draw_person_local(selected: bool) -> None:
    colour = (0.28, 0.72, 0.98) if selected else (0.84, 0.86, 0.91)
    _disc((0.0, 0.012, 0.0), 0.46, (0.015, 0.018, 0.024))
    quadric = gluNewQuadric()
    _colour(colour)
    glPushMatrix()
    glTranslatef(0.0, 1.77, 0.0)
    gluSphere(quadric, 0.22, 12, 9)
    glPopMatrix()
    gluDeleteQuadric(quadric)
    _tapered_prism((0.0, 1.18, 0.0), (0.42, 0.30), (0.34, 0.24), 0.86, colour)
    _limb((-0.13, 0.78, 0.0), (-0.17, 0.10, 0.04), 0.075, colour)
    _limb((0.13, 0.78, 0.0), (0.18, 0.10, -0.04), 0.075, colour)
    _limb((-0.20, 1.45, 0.0), (-0.46, 0.88, 0.02), 0.065, colour)
    _limb((0.20, 1.45, 0.0), (0.47, 0.98, -0.02), 0.065, colour)


def _bike_wheel(z: float, y: float = 0.38, radius: float = 0.36) -> None:
    _colour((0.055, 0.06, 0.07))
    glLineWidth(3.0)
    glBegin(GL_LINE_LOOP)
    for index in range(20):
        angle = math.tau * index / 20
        glVertex3f(0.0, y + math.sin(angle) * radius, z + math.cos(angle) * radius)
    glEnd()
    _colour((0.42, 0.45, 0.49))
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for index in range(0, 20, 4):
        angle = math.tau * index / 20
        glVertex3f(0.0, y, z)
        glVertex3f(0.0, y + math.sin(angle) * radius, z + math.cos(angle) * radius)
    glEnd()


def _draw_bicycle_local(selected: bool) -> None:
    colour = (0.25, 0.74, 0.98) if selected else (0.75, 0.78, 0.84)
    _disc((0.0, 0.012, 0.0), 0.78, (0.015, 0.018, 0.024))
    _bike_wheel(-0.62)
    _bike_wheel(0.62)
    _limb((0.0, 0.38, -0.62), (0.0, 0.72, 0.0), 0.035, colour)
    _limb((0.0, 0.72, 0.0), (0.0, 0.38, 0.62), 0.035, colour)
    _limb((0.0, 0.38, 0.62), (0.0, 0.38, -0.62), 0.035, colour)
    _limb((0.0, 0.38, -0.62), (0.0, 0.38, 0.10), 0.035, colour)
    _limb((0.0, 0.38, 0.10), (0.0, 0.82, 0.18), 0.035, colour)
    _limb((-0.22, 0.84, 0.18), (0.22, 0.84, 0.18), 0.025, colour)
    _cuboid((0.0, 0.76, -0.10), (0.16, 0.06, 0.34), (0.12, 0.13, 0.15))


def _draw_signal_local(label: str) -> None:
    """Lightweight traffic furniture: lights, stop signs, hydrants, meters."""
    _disc((0.0, 0.012, 0.0), 0.38, (0.015, 0.018, 0.024))
    if label == "traffic light":
        _cuboid((0.0, 1.55, 0.0), (0.12, 3.05, 0.12), (0.38, 0.40, 0.44))
        _cuboid((0.0, 3.20, 0.0), (0.58, 1.28, 0.36), (0.08, 0.09, 0.11))
        # Unlit lamps (lit state overdrawn per-instance in render_world).
        _cuboid((0.0, 3.55, 0.20), (0.22, 0.20, 0.05), (0.35, 0.08, 0.06))
        _cuboid((0.0, 3.20, 0.20), (0.22, 0.20, 0.05), (0.35, 0.28, 0.05))
        _cuboid((0.0, 2.85, 0.20), (0.22, 0.20, 0.05), (0.05, 0.28, 0.10))
        return
    if label == "stop sign":
        _cuboid((0.0, 1.35, 0.0), (0.10, 2.65, 0.10), (0.40, 0.42, 0.46))
        # Octagon-ish red face (two stacked quads + side bevels).
        _cuboid((0.0, 2.95, 0.0), (0.92, 0.92, 0.07), (0.88, 0.12, 0.10))
        _cuboid((0.0, 2.95, 0.0), (0.72, 1.08, 0.06), (0.88, 0.12, 0.10))
        _cuboid((0.0, 2.95, 0.0), (1.08, 0.72, 0.06), (0.88, 0.12, 0.10))
        _cuboid((0.0, 2.95, 0.05), (0.50, 0.12, 0.03), (0.95, 0.92, 0.88))
        return
    if label == "fire hydrant":
        _cuboid((0.0, 0.22, 0.0), (0.48, 0.28, 0.48), (0.72, 0.12, 0.10))
        _cuboid((0.0, 0.70, 0.0), (0.38, 0.85, 0.38), (0.86, 0.16, 0.12))
        _cuboid((0.0, 1.22, 0.0), (0.46, 0.22, 0.46), (0.90, 0.20, 0.14))
        _cuboid((0.0, 0.78, 0.28), (0.18, 0.18, 0.22), (0.75, 0.14, 0.10))
        _cuboid((0.0, 0.78, -0.28), (0.18, 0.18, 0.22), (0.75, 0.14, 0.10))
        return
    if label == "parking meter":
        _cuboid((0.0, 0.85, 0.0), (0.09, 1.65, 0.09), (0.42, 0.44, 0.48))
        _cuboid((0.0, 1.85, 0.0), (0.32, 0.48, 0.22), (0.55, 0.58, 0.62))
        _cuboid((0.0, 1.92, 0.12), (0.22, 0.28, 0.04), (0.15, 0.55, 0.95))
        return
    # Generic sign face fallback.
    _cuboid((0.0, 1.7, 0.0), (0.11, 3.35, 0.11), (0.42, 0.45, 0.49))
    _cuboid((0.0, 2.85, 0.0), (0.82, 0.82, 0.08), (0.90, 0.22, 0.20))
    _cuboid((0.0, 2.85, 0.05), (0.54, 0.10, 0.04), (0.95, 0.90, 0.82))


_BACKGROUND_DIM_FACTOR = 0.42


def _compile_variant(builder, *builder_args, dimmed: bool = False) -> int:
    """Compile one display list, optionally with every colour in it darkened.

    Dimming goes through `_colour` (every mesh primitive routes color through
    it), so this needs no changes to the drawing functions themselves.
    """
    global _COLOUR_DIM_FACTOR
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    _COLOUR_DIM_FACTOR = _BACKGROUND_DIM_FACTOR if dimmed else 1.0
    builder(*builder_args)
    _COLOUR_DIM_FACTOR = 1.0
    glEndList()
    return list_id


def initialize_mesh_cache() -> None:
    """Compile detailed static geometry once; instances only transform/call it.

    Each kind gets three precompiled variants: "neutral" (normal traffic),
    "selected" (the nearest/highlighted track), and "background" (off-corridor
    traffic rendered dim to signal lower position confidence -- see
    `_corridor_membership`). Instances only ever pick one of these three by
    key; no per-instance colour is set at draw time, which is what keeps
    `glCallList` cheap.
    """
    if MESH_LISTS:
        return
    # Road vehicles (car/suv/truck/bus) share one oval mold; aliases all point
    # at the "car" lists so we only compile the geometry once.
    for kind in ("car", "motorcycle"):
        MESH_LISTS[(kind, "neutral")] = _compile_variant(_draw_vehicle_local, kind, False)
        MESH_LISTS[(kind, "selected")] = _compile_variant(_draw_vehicle_local, kind, True)
        MESH_LISTS[(kind, "background")] = _compile_variant(_draw_vehicle_local, kind, False, dimmed=True)
    for alias in ("suv", "truck", "bus"):
        for variant in ("neutral", "selected", "background"):
            MESH_LISTS[(alias, variant)] = MESH_LISTS[("car", variant)]
    for kind, builder in (("person", _draw_person_local), ("bicycle", _draw_bicycle_local)):
        MESH_LISTS[(kind, "neutral")] = _compile_variant(builder, False)
        MESH_LISTS[(kind, "selected")] = _compile_variant(builder, True)
        MESH_LISTS[(kind, "background")] = _compile_variant(builder, False, dimmed=True)
    for label in ("traffic light", "stop sign", "parking meter", "fire hydrant"):
        MESH_LISTS[(label, "neutral")] = _compile_variant(_draw_signal_local, label)
        MESH_LISTS[(label, "background")] = _compile_variant(_draw_signal_local, label, dimmed=True)


def delete_mesh_cache() -> None:
    # Aliases (truck/bus/suv → car) share list IDs; delete each GL list once.
    for list_id in set(MESH_LISTS.values()):
        glDeleteLists(list_id, 1)
    MESH_LISTS.clear()


def _cached_mesh(kind: str, variant: str, x: float, z: float, heading: float = 0.0) -> None:
    list_id = MESH_LISTS.get((kind, variant), MESH_LISTS.get((kind, "neutral")))
    if list_id is None:
        return
    glPushMatrix()
    glTranslatef(x, 0.0, z)
    glRotatef(heading, 0.0, 1.0, 0.0)
    glCallList(list_id)
    glPopMatrix()


# Per-frame cache for road-centre samples (cleared at start of render_world).
_ROAD_CENTER_FRAME_CACHE: dict[int, float] = {}


def _road_center_x(z: float, road_geometry: RoadGeometry | None) -> float:
    """Convert the image-space centre curve into a conservative visual road offset."""
    if (road_geometry is None or road_geometry.center_points is None or
            road_geometry.left_points is None or road_geometry.right_points is None):
        return 0.0
    # Quantize z so many meshes share one sample (cuts CPU + lateral thrash).
    cache_key = int(round(float(z) * 2.0))
    cached = _ROAD_CENTER_FRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached
    points = road_geometry.center_points.astype(np.float32)
    index = float(np.clip((-z) / 88.0, 0.0, 1.0)) * (len(points) - 1)
    xs = np.interp(index, np.arange(len(points)), points[:, 0])
    base_x = points[0, 0]
    lane_width_px = float(np.linalg.norm(
        road_geometry.right_points[0].astype(float) - road_geometry.left_points[0].astype(float),
    ))
    value = float(np.clip((xs - base_x) / max(lane_width_px, 40.0) * 3.6, -8.5, 8.5))
    _ROAD_CENTER_FRAME_CACHE[cache_key] = value
    return value


def _road_heading(z: float, road_geometry: RoadGeometry | None) -> float:
    """Road-tangent yaw at depth z; long baseline rejects centre-curve jitter."""
    before = _road_center_x(z - 4.0, road_geometry)
    after = _road_center_x(z + 4.0, road_geometry)
    heading = math.degrees(math.atan2(after - before, 8.0))
    return float(np.clip(heading, -32.0, 32.0))


def _ego_lateral_position(road_geometry: RoadGeometry | None) -> float:
    """Locate the camera/ego car within the detected lane instead of assuming centre."""
    if (road_geometry is None or road_geometry.frame_width <= 0 or
            road_geometry.left_points is None or road_geometry.right_points is None):
        return 0.0
    top = max(float(np.min(road_geometry.left_points[:, 1])),
              float(np.min(road_geometry.right_points[:, 1])))
    bottom = min(float(np.max(road_geometry.left_points[:, 1])),
                 float(np.max(road_geometry.right_points[:, 1])))
    sample_y = float(np.clip(road_geometry.frame_height * 0.94, top, bottom))
    offset = road_lateral_offset_m(road_geometry.frame_width * 0.5, sample_y, road_geometry)
    return 0.0 if offset is None else float(np.clip(offset, -1.45, 1.45))


def _curved_lane_segment(offset: float, z: float, length: float, colour: tuple[float, float, float], road_geometry: RoadGeometry | None) -> None:
    center_z = z - length / 2.0
    center_x = _road_center_x(center_z, road_geometry) + offset
    glPushMatrix()
    glTranslatef(center_x, 0.014, center_z)
    glRotatef(_road_heading(center_z, road_geometry), 0, 1, 0)
    _cuboid((0.0, 0.0, 0.0), (0.16, 0.035, length), colour)
    glPopMatrix()


def _ensure_glut() -> bool:
    """One-shot GLUT init for bitmap distance badges (optional, fail-soft)."""
    global _GLUT_READY
    if _GLUT_READY:
        return True
    try:
        glutInit()
        _GLUT_READY = True
        return True
    except Exception:
        return False


def _draw_world_text(x: float, y: float, z: float, text: str,
                     color: tuple[float, float, float] = (0.92, 0.94, 0.98)) -> None:
    """Tiny bitmap label at a world point (GLUT). No-op if GLUT unavailable."""
    if not text or not _ensure_glut():
        return
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor3f(*color)
    glRasterPos3f(x, y, z)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


def _draw_range_rings(road_geometry: RoadGeometry | None, ego_lateral: float) -> None:
    """Lightweight 10 / 20 / 30 m depth bars on the ego road strip (no GLUT text)."""
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.5)
    glNormal3f(0, 1, 0)
    glBegin(GL_LINES)
    for dist_m, alpha in ((10.0, 0.50), (20.0, 0.38), (30.0, 0.28)):
        z = -dist_m
        cx = _road_center_x(z, road_geometry) + _ego_lateral_fade(z, ego_lateral)
        glColor4f(0.55, 0.72, 0.88, alpha)
        glVertex3f(cx - 2.0, 0.04, z)
        glVertex3f(cx + 2.0, 0.04, z)
    glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)


def _draw_free_space_world(road_geometry: RoadGeometry | None, ego_lateral: float) -> None:
    """Draw L/C/R free-space chips from environment intel (green/red/gray)."""
    env = latest_environment()
    if not env.free_space:
        return
    status_rgb = {
        FS_FREE: (0.15, 0.85, 0.35),
        FS_BLOCKED: (0.95, 0.22, 0.18),
        FS_UNKNOWN: (0.45, 0.48, 0.52),
    }
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glNormal3f(0, 1, 0)
    for band in env.free_space:
        z = -float(band.distance_m)
        cx = _road_center_x(z, road_geometry) + _ego_lateral_fade(z, ego_lateral) * 0.85
        # Three small plates: left / center / right of the ego ribbon.
        for status, x_off in (
            (band.left, -1.35),
            (band.center, 0.0),
            (band.right, 1.35),
        ):
            r, g, b = status_rgb.get(status, status_rgb[FS_UNKNOWN])
            glColor4f(r, g, b, 0.55)
            hx, hz = 0.38, 0.22
            x = cx + x_off
            y = 0.06
            glBegin(GL_QUADS)
            glVertex3f(x - hx, y, z - hz)
            glVertex3f(x + hx, y, z - hz)
            glVertex3f(x + hx, y, z + hz)
            glVertex3f(x - hx, y, z + hz)
            glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)


def _draw_vehicle_badge(x: float, z: float, obj: DetectedObject, *,
                        selected: bool = False, merge: bool = False) -> None:
    """Floating distance badge; GLUT text only for LEAD/MERGE to protect FPS."""
    # Rect/oval mold apex (~1.24 m); pin sits just above the surface.
    roof_y = 1.28
    pin_y = roof_y + 0.40
    dist = float(np.clip(obj.distance_m, 0.0, 99.0))
    if dist < 12.0:
        col = (1.0, 0.55, 0.25) if not merge else (1.0, 0.45, 0.12)
    elif dist < 25.0:
        col = (1.0, 0.88, 0.30) if not merge else (1.0, 0.55, 0.18)
    else:
        col = (0.55, 0.85, 1.0) if not merge else (1.0, 0.60, 0.22)
    if selected:
        col = (0.25, 0.75, 1.0)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.5)
    glColor4f(col[0], col[1], col[2], 0.85)
    glBegin(GL_LINES)
    glVertex3f(x, roof_y, z)
    glVertex3f(x, pin_y, z)
    glEnd()
    hw = 0.35 + min(0.35, dist * 0.012)
    hd = 0.10
    glColor4f(col[0], col[1], col[2], 0.80)
    glBegin(GL_QUADS)
    glVertex3f(x - hw, pin_y, z - hd)
    glVertex3f(x + hw, pin_y, z - hd)
    glVertex3f(x + hw, pin_y, z + hd)
    glVertex3f(x - hw, pin_y, z + hd)
    glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)
    if selected or merge:
        tag = "LEAD" if selected else "MERGE"
        _draw_world_text(x - 0.40, pin_y + 0.12, z, f"{tag} {dist:.0f}m", col)


def _ego_lateral_fade(z: float, ego_lateral: float) -> float:
    """Same near→far ego bias used by the grey ego-lane ribbon."""
    return float(ego_lateral) * float(np.clip((float(z) + 55.0) / 58.0, 0.0, 1.0))


def _draw_road_world(road_geometry: RoadGeometry | None = None, ego_lateral: float = 0.0,
                     lite: bool = False) -> None:
    """Ultra-light ego-only road strip + range rings + path triangle + paint.

    ``lite=True`` (split view) skips free-space chips and road paint for FPS.
    The grey ribbon is **our lane only** (~3.7 m wide).
    """
    global _WORLD_GUIDE_EMA
    depth_samples = np.linspace(3.0, -55.0, 4 if lite else 7)
    half_lane = EGO_ROAD_HALF_M
    glDisable(GL_LIGHTING)
    glDisable(GL_BLEND)
    _colour(ROAD_COLOUR)
    glNormal3f(0, 1, 0)
    glBegin(GL_QUAD_STRIP)
    for z in depth_samples:
        center = _road_center_x(float(z), road_geometry) + _ego_lateral_fade(
            float(z), ego_lateral,
        )
        glVertex3f(center - half_lane, -0.08, float(z))
        glVertex3f(center + half_lane, -0.08, float(z))
    glEnd()

    # Range rings only in full world view (split lite skips for FPS).
    if not lite:
        _draw_range_rings(road_geometry, ego_lateral)
        _draw_free_space_world(road_geometry, ego_lateral)

    # Live ego path triangle — confidence-tinted (lanes/curbs/asphalt/hold fade).
    source, path_alpha, _fill_bgr, _edge_bgr = ego_path_style()
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(-2.0, -2.0)
    if source == "lanes":
        tri_rgb = (1.0, 0.92, 0.08)
        edge_rgb = (1.0, 0.98, 0.35)
    elif source == "curbs":
        tri_rgb = (0.95, 0.78, 0.12)
        edge_rgb = (1.0, 0.90, 0.30)
    elif source == "asphalt":
        tri_rgb = (0.78, 0.68, 0.18)
        edge_rgb = (0.90, 0.82, 0.28)
    elif source == "hold":
        tri_rgb = (0.82, 0.72, 0.14)
        edge_rgb = (0.92, 0.85, 0.28)
    else:
        tri_rgb = (1.0, 0.92, 0.08)
        edge_rgb = (1.0, 0.98, 0.35)
    tri_a = float(np.clip(path_alpha if path_alpha > 0.05 else 0.0, 0.0, 0.95))
    if tri_a > 0.06:
        glColor4f(tri_rgb[0], tri_rgb[1], tri_rgb[2], tri_a)
        glNormal3f(0, 1, 0)
        base_z = -8.5
        tip_z_target = -22.0
        half = 0.42
        base_c = _road_center_x(base_z, road_geometry) + ego_lateral * 0.90
        z_mid, z_far = -14.0, -24.0
        c_mid = _road_center_x(z_mid, road_geometry) + ego_lateral * 0.55
        c_far = _road_center_x(z_far, road_geometry) + ego_lateral * 0.30
        tip_raw = 0.35 * c_mid + 0.65 * c_far
        tip_raw = tip_raw + (c_far - c_mid) * 0.45
        tip_raw = float(np.clip(tip_raw, base_c - 3.2, base_c + 3.2))
        if _WORLD_GUIDE_EMA is None:
            _WORLD_GUIDE_EMA = (base_c, tip_raw, tip_z_target)
        else:
            pb, pt, pz = _WORLD_GUIDE_EMA
            base_c = pb * 0.55 + base_c * 0.45
            tip_c = pt * 0.52 + tip_raw * 0.48
            tip_z = pz * 0.70 + tip_z_target * 0.30
            _WORLD_GUIDE_EMA = (base_c, tip_c, tip_z)
        base_c, tip_c, tip_z = _WORLD_GUIDE_EMA
        y = 0.10
        glBegin(GL_TRIANGLES)
        glVertex3f(base_c - half, y, base_z)
        glVertex3f(tip_c, y, tip_z)
        glVertex3f(base_c + half, y, base_z)
        glEnd()
        glLineWidth(2.0 if source in {"lanes", "curbs"} else 1.0)
        glColor4f(edge_rgb[0], edge_rgb[1], edge_rgb[2], min(1.0, tri_a + 0.15))
        glBegin(GL_LINE_LOOP)
        glVertex3f(base_c - half, y + 0.01, base_z)
        glVertex3f(tip_c, y + 0.01, tip_z)
        glVertex3f(base_c + half, y + 0.01, base_z)
        glEnd()

    # Road-paint symbols only in full quality (skip in split lite path).
    if not lite:
        _draw_road_markings_world(road_geometry, ego_lateral)

    glDisable(GL_POLYGON_OFFSET_FILL)
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)


def _marking_world_pose(centroid: tuple[float, float],
                        road_geometry: RoadGeometry | None,
                        ego_lateral: float) -> tuple[float, float]:
    """Project an image marking centroid onto the ground path (x, z)."""
    fw = float(road_geometry.frame_width) if road_geometry and road_geometry.frame_width else 1280.0
    fh = float(road_geometry.frame_height) if road_geometry and road_geometry.frame_height else 720.0
    u, v = float(centroid[0]), float(centroid[1])
    # Lower image = nearer; clamp into a useful display band.
    rel = float(np.clip((v / max(1.0, fh) - 0.42) / 0.50, 0.0, 1.0))
    dist = float(np.clip(8.0 + (1.0 - rel) * 32.0, 6.0, 42.0))
    z = -dist
    # Lateral from image x vs centre, mixed with road centre at that depth.
    lateral_img = (u - fw * 0.5) / max(40.0, fw * 0.22)
    road_c = _road_center_x(z, road_geometry) + ego_lateral * 0.4
    x = road_c + float(np.clip(lateral_img * 2.4, -4.5, 4.5))
    return x, z


def _draw_road_markings_world(road_geometry: RoadGeometry | None,
                              ego_lateral: float = 0.0) -> None:
    """Draw scanned stop lines / crosswalks / arrows as flat ground symbols."""
    markings = latest_road_markings()
    if not markings:
        return
    glNormal3f(0, 1, 0)
    y = 0.06
    for mark in markings[:2]:  # hard cap — paint is decorative, keep GL light
        x, z = _marking_world_pose(mark.centroid, road_geometry, ego_lateral)
        if mark.colour == "yellow":
            glColor4f(0.98, 0.86, 0.12, 0.90)
        else:
            glColor4f(0.92, 0.93, 0.95, 0.88)
        if mark.kind == "crosswalk":
            for dx in (-1.2, -0.4, 0.4, 1.2):
                glBegin(GL_QUADS)
                glVertex3f(x + dx - 0.22, y, z - 1.1)
                glVertex3f(x + dx + 0.22, y, z - 1.1)
                glVertex3f(x + dx + 0.22, y, z + 1.1)
                glVertex3f(x + dx - 0.22, y, z + 1.1)
                glEnd()
        else:  # lane_paint (stop_line / arrow removed — too many false hits)
            glBegin(GL_QUADS)
            glVertex3f(x - 0.08, y, z - 1.4)
            glVertex3f(x + 0.08, y, z - 1.4)
            glVertex3f(x + 0.08, y, z + 1.4)
            glVertex3f(x - 0.08, y, z + 1.4)
            glEnd()


def _road_world(road_geometry: RoadGeometry | None = None, ego_lateral: float = 0.0,
                lite: bool = False) -> None:
    """Draw the small road strip directly every frame.

    This used to compile into a display list keyed on geometry identity, but
    the road tracker's display-motion `predict()` returns a *new* RoadGeometry
    object whenever the model is briefly stale, so the list was being deleted
    and recompiled almost every frame. The strip is only ~34 vertices;
    immediate drawing is cheaper than that driver churn.
    """
    _draw_road_world(road_geometry, ego_lateral, lite=lite)


def _corridor_membership(obj: DetectedObject, road_geometry: RoadGeometry | None) -> bool | None:
    """Whether the object's footpoint sits on the mapped drivable surface.

    Returns None when that image row is beyond the road model's effective
    range (empirically it shares the same range ceiling as lane detection --
    see the Phase A investigation): there is no signal there to rule
    anything in or out, so the object is treated as ordinary/on-corridor
    rather than penalised just for being far away. Returns False only when
    the model actively saw drivable surface at that row and this footpoint
    isn't on it -- e.g. traffic on the far side of a highway median.
    """
    if road_geometry is None or road_geometry.drivable_mask is None:
        return None
    mask = road_geometry.drivable_mask
    height, width = mask.shape[:2]
    if height != road_geometry.frame_height or width != road_geometry.frame_width:
        return None
    x1, _y1, x2, y2 = obj.box
    foot_x = int(np.clip((x1 + x2) // 2, 0, width - 1))
    foot_y = int(np.clip(y2, 0, height - 1))
    row_low, row_high = max(0, foot_y - 10), min(height, foot_y + 11)
    row_band = mask[row_low:row_high, :]
    if not np.any(row_band):
        return None
    col_low, col_high = max(0, foot_x - 14), min(width, foot_x + 15)
    return bool(np.any(row_band[:, col_low:col_high]))


def _object_road_coords(obj: DetectedObject, road_geometry: RoadGeometry | None = None,
                        off_corridor: bool = False) -> tuple[float, float]:
    """Return (lateral_m from ego-lane centre, distance_m ahead).

    Vision measurement only (no display snap). Positive = right of ego lane.
      - distance: re-fuse monocular box with flat-ground foot y when possible
        so lead/adjacent cars separate in depth instead of stacking.
      - lateral: prefer road-corridor offset (metres from ego-lane centre);
        blend pinhole image lateral when road geometry is weak/far.
    """
    distance = float(np.clip(obj.distance_m, 1.0, 86.0))
    x1, y1, x2, y2 = obj.box
    foot_x = (x1 + x2) * 0.5
    foot_y = float(y2)
    fw = float(road_geometry.frame_width) if road_geometry and road_geometry.frame_width > 0 else 0.0
    fh = float(road_geometry.frame_height) if road_geometry and road_geometry.frame_height > 0 else 0.0

    # Soft refine depth from image foot — light blend only so 3D pose grounding
    # is not undone by a full re-estimate every frame (major flicker source).
    if fh >= 100.0 and obj.label in ROAD_CLASSES | VEHICLE_CLASSES:
        focal = focal_length_px(int(fw) if fw >= 100 else 1280, 70.0)
        refined = estimate_monocular_distance_m(
            obj.label, obj.box, focal, int(fh), 1.25, 0.52,
        )
        distance = float(np.clip(0.78 * distance + 0.22 * refined, 1.0, 86.0))

    # Primary lateral: pinhole from foot x (not noisy bearing alone).
    if fw >= 100.0:
        focal = focal_length_px(int(fw), 70.0)
        bearing_foot = math.degrees(math.atan2(foot_x - fw * 0.5, focal))
        lateral_img = distance * math.tan(math.radians(bearing_foot))
    else:
        lateral_img = distance * math.tan(math.radians(obj.bearing_deg))
    lateral = lateral_img

    valid_footprint = (x2 - x1) > 4 and (y2 - y1) > 4
    if (valid_footprint and road_geometry is not None and road_geometry.left_points is not None and
            road_geometry.right_points is not None):
        top = max(float(np.min(road_geometry.left_points[:, 1])),
                  float(np.min(road_geometry.right_points[:, 1])))
        bottom = min(float(np.max(road_geometry.left_points[:, 1])),
                     float(np.max(road_geometry.right_points[:, 1])))
        if top <= foot_y <= bottom:
            road_lat = road_lateral_offset_m(foot_x, foot_y, road_geometry)
            if road_lat is not None:
                # Prefer ego-lane-centre offset so lane slotting is meaningful.
                # Image pinhole is a soft prior only (noise / far rows).
                if abs(road_lat) <= LANE_EGO_EXIT_M:
                    lateral = 0.72 * road_lat + 0.28 * lateral_img
                elif abs(road_lat) <= ADJ_LANE_CENTER_M + 1.0:
                    lateral = 0.62 * road_lat + 0.38 * lateral_img
                else:
                    lateral = 0.48 * road_lat + 0.52 * lateral_img

    # Signals sit roadside — allow wider lateral envelope.
    if obj.label in SIGNAL_CLASSES:
        bound = 18.0
    else:
        bound = 22.0 if off_corridor else 14.0
    return float(np.clip(lateral, -bound, bound)), float(np.clip(distance, 1.0, 86.0))


def _lane_slot_from_lateral(lane_lat_m: float, previous: str | None = None) -> str:
    """Classify a measured lateral into ego / left / right (with hysteresis)."""
    lat = float(lane_lat_m)
    abs_lat = abs(lat)
    prev = previous or ""

    # Hold current slot through the hysteresis band so noise cannot thrash.
    if prev == "ego" and abs_lat <= LANE_EGO_EXIT_M:
        return "ego"
    if prev == "left" and lat <= -LANE_EGO_ENTER_M and abs_lat < LANE_FAR_ENTER_M + 0.35:
        return "left"
    if prev == "right" and lat >= LANE_EGO_ENTER_M and abs_lat < LANE_FAR_ENTER_M + 0.35:
        return "right"
    if prev == "far_left" and lat <= -LANE_FAR_ENTER_M + 0.55:
        return "far_left"
    if prev == "far_right" and lat >= LANE_FAR_ENTER_M - 0.55:
        return "far_right"

    if abs_lat <= LANE_EGO_ENTER_M:
        return "ego"
    if abs_lat >= LANE_FAR_ENTER_M:
        return "far_left" if lat < 0.0 else "far_right"
    return "left" if lat < 0.0 else "right"


def _update_lane_slot(track_id: int, lane_lat_m: float) -> str:
    """Sticky lane slot for a track (used only by the 3D display)."""
    prev = WORLD_LANE_SLOT.get(track_id)
    slot = _lane_slot_from_lateral(lane_lat_m, prev)
    WORLD_LANE_SLOT[track_id] = slot
    return slot


def _lane_display_lateral(slot: str, lane_lat_m: float) -> float:
    """Snap measured lateral into a stable display position for the grey ribbon.

    - ego: stay on our grey road (inside the ribbon, mild residual OK)
    - left/right: sit in the adjacent lane, hull fully clear of our ribbon
    - far_*: second lane over (or median / opposite shoulder)
    """
    lat = float(lane_lat_m)
    if slot == "ego":
        # Soft center bias so same-lane cars read as "on our road", not straddling.
        centered = lat * 0.55
        return float(np.clip(centered, -(EGO_ROAD_HALF_M - VEHICLE_HALF_WIDTH_M),
                             (EGO_ROAD_HALF_M - VEHICLE_HALF_WIDTH_M)))
    if slot == "left":
        residual = float(np.clip(lat + ADJ_LANE_CENTER_M, -0.90, 0.90))
        return float(min(-ADJ_LANE_CENTER_M + residual, -ADJ_LANE_CLEAR_M))
    if slot == "right":
        residual = float(np.clip(lat - ADJ_LANE_CENTER_M, -0.90, 0.90))
        return float(max(ADJ_LANE_CENTER_M + residual, ADJ_LANE_CLEAR_M))
    if slot == "far_left":
        residual = float(np.clip(lat + FAR_LANE_CENTER_M, -1.0, 1.0))
        return float(min(-FAR_LANE_CENTER_M + residual, -(ADJ_LANE_CLEAR_M + 3.2)))
    if slot == "far_right":
        residual = float(np.clip(lat - FAR_LANE_CENTER_M, -1.0, 1.0))
        return float(max(FAR_LANE_CENTER_M + residual, ADJ_LANE_CLEAR_M + 3.2))
    return float(np.clip(lat, -12.0, 12.0))


def _display_lateral_for_object(
    obj: DetectedObject,
    measured_lateral_m: float,
    *,
    off_corridor: bool = False,
) -> float:
    """Map vision lateral → 3D display lateral (lane-slotted for vehicles)."""
    if obj.label not in RELEVANT_VEHICLE_LABELS | {"motorcycle"}:
        # VRUs / signals: light clamp only — no hard lane snap.
        return float(np.clip(measured_lateral_m, -12.0, 12.0))
    # Off-corridor (e.g. opposite carriageway): force far side, never ego ribbon.
    if off_corridor and abs(measured_lateral_m) > 1.2:
        slot = "far_left" if measured_lateral_m < 0.0 else "far_right"
        WORLD_LANE_SLOT[obj.track_id] = slot
        return _lane_display_lateral(slot, measured_lateral_m)
    slot = _update_lane_slot(obj.track_id, measured_lateral_m)
    return _lane_display_lateral(slot, measured_lateral_m)


def _project_road_coords(lateral_m: float, distance_m: float,
                         road_geometry: RoadGeometry | None,
                         ego_lateral: float = 0.0) -> tuple[float, float]:
    """Map road-relative (lateral, range) onto current world XZ for rendering.

    Lateral is metres from **ego-lane centre**. Near-field ego bias matches the
    grey ribbon so cars and road share one frame.
    """
    distance = float(np.clip(distance_m, 1.0, 86.0))
    z = -distance
    x = (
        _road_center_x(z, road_geometry)
        + _ego_lateral_fade(z, ego_lateral)
        + float(lateral_m)
    )
    return x, z


def _object_position(obj: DetectedObject, road_geometry: RoadGeometry | None = None,
                     off_corridor: bool = False,
                     ego_lateral: float = 0.0) -> tuple[float, float]:
    """Absolute world XZ for one detection (convenience / debug)."""
    measured, distance = _object_road_coords(obj, road_geometry, off_corridor)
    lateral = _display_lateral_for_object(obj, measured, off_corridor=off_corridor)
    return _project_road_coords(lateral, distance, road_geometry, ego_lateral=ego_lateral)


def _draw_predicted_motion(x: float, z: float, obj: DetectedObject) -> None:
    vx, _ = obj.image_velocity
    future_x = x + float(np.clip(vx * 0.012, -3.0, 3.0))
    # Positive closing speed means the object is approaching the ego camera,
    # which is positive Z in this display coordinate system.
    future_z = z + float(np.clip(obj.closing_mps * 1.15, -7.0, 7.0))
    if abs(future_z - z) < 0.35:
        future_z = z - 2.6
    glDisable(GL_LIGHTING)
    glLineWidth(2.0)
    glColor3f(0.25, 0.74, 1.0)
    glBegin(GL_LINES)
    glVertex3f(x, 0.10, z)
    glVertex3f(future_x, 0.10, future_z)
    glEnd()
    glEnable(GL_LIGHTING)


def _wrap_angle_deg(angle: float) -> float:
    """Map degrees into (-180, 180] so heading blends do not jump across ±180."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _blend_heading_deg(previous: float, target: float, alpha: float,
                       max_step_deg: float = 10.0) -> float:
    """Slerp-like short-arc blend with a hard per-update rotation cap."""
    delta = _wrap_angle_deg(target - previous)
    delta = float(np.clip(delta, -max_step_deg, max_step_deg))
    return _wrap_angle_deg(previous + float(np.clip(alpha, 0.0, 1.0)) * delta)


def _trajectory_velocity(track_id: int, raw_lateral: float, raw_distance: float, now: float,
                         sample: bool) -> tuple[float, float] | None:
    """Least-squares road-relative velocity from genuinely new samples only.

    Returns (v_lateral m/s, v_distance m/s) where positive v_distance means the
    target range is increasing (pulling away). Converted to world XZ velocity
    only when deciding turn headings.
    """
    history = WORLD_TRAJECTORY.setdefault(track_id, deque(maxlen=10))
    if sample and (not history or abs(raw_lateral - history[-1][1]) > 1e-3 or
                   abs(raw_distance - history[-1][2]) > 1e-3):
        history.append((now, raw_lateral, raw_distance))
    if len(history) < 4:
        return None
    times = np.asarray([entry[0] for entry in history], dtype=np.float64)
    times -= times[0]
    if times[-1] < 0.55:
        return None
    laterals = np.asarray([entry[1] for entry in history], dtype=np.float64)
    distances = np.asarray([entry[2] for entry in history], dtype=np.float64)
    if math.hypot(float(laterals[-1] - laterals[0]), float(distances[-1] - distances[0])) < 0.85:
        return None
    v_lat = float(np.polyfit(times, laterals, 1)[0])
    v_dist = float(np.polyfit(times, distances, 1)[0])
    return v_lat, v_dist


def _heading_from_road_velocity(v_lat: float, v_dist: float) -> float:
    """Lateral road-relative velocity → mesh yaw for genuine turns only.

    Relative range rate is *not* vehicle heading (catching up to slower traffic
    shrinks range without flipping the car). Always treat longitudinal motion
    as down-road and only bank yaw from sustained lateral slip.
    """
    along = max(2.0, abs(v_dist))
    return math.degrees(math.atan2(v_lat, along))


def _is_face_on_heading(heading_deg: float, tol_deg: float = 25.0) -> bool:
    """True when yaw is near ±180° (mesh front toward the ego camera)."""
    return abs(abs(_wrap_angle_deg(heading_deg)) - 180.0) < tol_deg


def _binary_vehicle_heading(track_id: int, want_oncoming: bool) -> float:
    """Vehicle meshes only ever use 0° (with traffic) or 180° (oncoming).

    Hysteresis prevents 0↔180 flip-spin when the latch is near the threshold.
    """
    lock = WORLD_ONCOMING_LOCK.get(track_id, 0)
    previous = WORLD_POSES.get(track_id)
    was_oncoming = previous is not None and _is_face_on_heading(previous[2], 40.0)
    # Acquire: need a solid latch. Hold: keep face-on until lock fully drains.
    if was_oncoming:
        oncoming = lock >= 1 or want_oncoming
    else:
        oncoming = lock >= 5 or (want_oncoming and lock >= 3)
    return ONCOMING_HEADING_DEG if oncoming else 0.0


def _stable_world_pose(track_id: int, raw_lateral: float, raw_distance: float,
                       raw_heading: float, road_geometry: RoadGeometry | None = None,
                       off_corridor: bool = False, observed: bool = True,
                       stationary: bool = False,
                       closing_mps: float = 0.0,
                       kind: str = "vehicle",
                       ego_lateral: float = 0.0,
                       lane_changed: bool = False) -> tuple[float, float, float]:
    """Smooth road-relative placement, then reproject onto the live road map.

    Grounding strength depends on object class:
      - vehicle: heavy EMA (anti-flicker for traffic)
      - vru (person/bike): same heavy grounding
      - signal (lights/signs): near-static hold — monocular size thrash is worst

    Heading is binary 0°/180° only for vehicles; others stay at 0°.
    ``raw_lateral`` should already be the *display* lateral (lane-slotted for
    vehicles) so EMA does not pull adjacent cars back onto the ego ribbon.
    """
    del stationary
    del off_corridor
    now = time.perf_counter()
    previous = WORLD_POSES.get(track_id)
    raw_lateral = float(raw_lateral)
    raw_distance = float(np.clip(raw_distance, 1.0, 86.0))
    if kind == "vehicle":
        want_oncoming = (
            _is_face_on_heading(raw_heading, 20.0)
            or WORLD_ONCOMING_LOCK.get(track_id, 0) >= 5
        )
        heading = _binary_vehicle_heading(track_id, want_oncoming)
    else:
        heading = 0.0

    # Class-specific grounding (signals nearly frozen; people/vehicles eased).
    if kind == "signal":
        hold_ttl = 2.4
        lat_alpha, dist_alpha = 0.08, 0.10
        max_dist_step_n, max_lat_step_n = 0.7, 0.45
        lat_dead, dist_dead = 0.45, 0.90
        coast_closing = 0.08
    elif kind == "vru":
        hold_ttl = 2.0
        lat_alpha, dist_alpha = 0.12, 0.14
        max_dist_step_n, max_lat_step_n = 1.1, 0.65
        lat_dead, dist_dead = 0.35, 0.70
        coast_closing = 0.15
    else:
        hold_ttl = 1.80
        lat_alpha, dist_alpha = 0.16, 0.18
        max_dist_step_n, max_lat_step_n = 1.6, 0.85
        lat_dead, dist_dead = 0.28, 0.55
        coast_closing = 0.22

    # Lane-slot transitions (adjacent ↔ ego) need a faster lateral settle so the
    # mesh does not linger on the grey ribbon for half a second of EMA lag.
    if lane_changed and kind == "vehicle":
        lat_alpha = max(lat_alpha, 0.48)
        max_lat_step_n = max(max_lat_step_n, 2.8)
        lat_dead = 0.0

    if previous is None or now - previous[3] > hold_ttl:
        WORLD_TRAJECTORY.pop(track_id, None)
        WORLD_TURN_HITS.pop(track_id, None)
        WORLD_CRUISE_YAW.pop(track_id, None)
        lateral, distance = raw_lateral, raw_distance
    else:
        dt = float(np.clip(now - previous[3], 1e-3, 0.12))
        prev_lat, prev_dist = previous[0], previous[1]
        if observed:
            max_dist_step = max_dist_step_n * (dt / 0.05)
            max_lat_step = max_lat_step_n * (dt / 0.05)
            target_dist = float(np.clip(
                raw_distance, prev_dist - max_dist_step, prev_dist + max_dist_step,
            ))
            target_lat = float(np.clip(
                raw_lateral, prev_lat - max_lat_step, prev_lat + max_lat_step,
            ))
            if abs(target_lat - prev_lat) < lat_dead:
                target_lat = prev_lat
            if abs(target_dist - prev_dist) < dist_dead:
                target_dist = prev_dist
            lateral = prev_lat * (1.0 - lat_alpha) + target_lat * lat_alpha
            distance = prev_dist * (1.0 - dist_alpha) + target_dist * dist_alpha
            if kind == "vehicle":
                _trajectory_velocity(track_id, raw_lateral, raw_distance, now, True)
        else:
            lateral = prev_lat
            distance = float(np.clip(prev_dist - closing_mps * dt * coast_closing, 1.0, 86.0))
        WORLD_TURN_HITS[track_id] = 0

    # Belt-and-braces: never leave a non-ego vehicle mesh overlapping our ribbon.
    slot = WORLD_LANE_SLOT.get(track_id)
    if kind == "vehicle" and slot in {"left", "far_left"}:
        lateral = min(lateral, -ADJ_LANE_CLEAR_M)
    elif kind == "vehicle" and slot in {"right", "far_right"}:
        lateral = max(lateral, ADJ_LANE_CLEAR_M)
    elif kind == "vehicle" and slot == "ego":
        lateral = float(np.clip(
            lateral,
            -(EGO_ROAD_HALF_M - VEHICLE_HALF_WIDTH_M),
            (EGO_ROAD_HALF_M - VEHICLE_HALF_WIDTH_M),
        ))

    WORLD_POSES[track_id] = (float(lateral), float(distance), float(heading), now)
    if len(WORLD_POSES) > 96:
        expired = [key for key, value in WORLD_POSES.items() if now - value[3] > 2.8]
        for key in expired:
            del WORLD_POSES[key]
            WORLD_TRAJECTORY.pop(key, None)
            WORLD_TURN_HITS.pop(key, None)
            WORLD_ONCOMING_LOCK.pop(key, None)
            WORLD_CRUISE_YAW.pop(key, None)
            WORLD_LANE_SLOT.pop(key, None)
            DISPLAY_SLOT_GRACE.pop(key, None)
    x, z = _project_road_coords(lateral, distance, road_geometry, ego_lateral=ego_lateral)
    return x, z, heading


def _dedup_world_objects(objects: list[DetectedObject]) -> list[DetectedObject]:
    """Drop ghost/fragment tracks that stack on the same real vehicle.

    ByteTrack ID flips plus brief coasted estimates previously drew two (or
    more) meshes at nearly the same world pose -- the "multiple cars overlaid
    while spinning" artefact on highway tests.
    """
    if len(objects) <= 1:
        return objects
    ordered = sorted(
        objects,
        key=lambda item: (
            0 if item.observed else 1,
            -item.confidence,
            item.missed_updates,
            item.distance_m,
        ),
    )
    kept: list[DetectedObject] = []
    for candidate in ordered:
        cx = (candidate.box[0] + candidate.box[2]) * 0.5
        cy = (candidate.box[1] + candidate.box[3]) * 0.5
        cw = max(1.0, candidate.box[2] - candidate.box[0])
        ch = max(1.0, candidate.box[3] - candidate.box[1])
        duplicate = False
        for existing in kept:
            compatible = (
                existing.label == candidate.label or
                (existing.label in {"car", "truck", "bus", "train"} and
                 candidate.label in {"car", "truck", "bus", "train"})
            )
            if not compatible:
                continue
            ex = (existing.box[0] + existing.box[2]) * 0.5
            ey = (existing.box[1] + existing.box[3]) * 0.5
            ew = max(1.0, existing.box[2] - existing.box[0])
            eh = max(1.0, existing.box[3] - existing.box[1])
            centre_dist = math.hypot(cx - ex, cy - ey)
            box_scale = max(24.0, 0.45 * max(cw, ch, ew, eh))
            range_delta = abs(existing.distance_m - candidate.distance_m)
            ix1 = max(existing.box[0], candidate.box[0])
            iy1 = max(existing.box[1], candidate.box[1])
            ix2 = min(existing.box[2], candidate.box[2])
            iy2 = min(existing.box[3], candidate.box[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = cw * ch + ew * eh - inter
            iou = inter / max(1.0, union)
            # Dense traffic: slightly stricter IoU so neighbouring cars survive;
            # still collapse true ghost duplicates (high IoU or near-identical feet).
            if iou >= 0.35 or (centre_dist < box_scale * 0.85 and range_delta < max(3.0, existing.distance_m * 0.16) and iou >= 0.18):
                if existing.observed and not candidate.observed:
                    duplicate = True
                    break
                if (not existing.observed) and candidate.observed:
                    kept[kept.index(existing)] = candidate
                    duplicate = True
                    break
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


# Relevance envelope for display meshes / camera labels (metres).
# Hard cap: at most 2 vehicle meshes/labels (camera + 3D share this filter).
RELEVANT_SAME_LANE_M = 1.90
RELEVANT_ADJACENT_LANE_M = 5.40
RELEVANT_MAX_RANGE_M = 45.0
RELEVANT_MIN_RANGE_M = 2.5
RELEVANT_MAX_VEHICLES = 2
RELEVANT_MAX_VRU = 3          # person / bicycle / motorcycle
RELEVANT_MAX_SIGNALS = 6      # traffic light / stop sign / hydrant / meter
RELEVANT_VEHICLE_LABELS = {"car", "truck", "bus"}
RELEVANT_VRU_LABELS = {"person", "bicycle", "motorcycle"}
RELEVANT_SIGNAL_LABELS = {"traffic light", "stop sign", "parking meter", "fire hydrant"}
RELEVANT_MESH_LABELS = RELEVANT_VEHICLE_LABELS | RELEVANT_VRU_LABELS | RELEVANT_SIGNAL_LABELS
# Prefer true traffic control devices when the signal mesh budget is tight.
_SIGNAL_PRIORITY = {
    "traffic light": 0,
    "stop sign": 1,
    "parking meter": 3,
    "fire hydrant": 4,
}


def _object_quality(obj: DetectedObject) -> float:
    """Prefer stored track_quality; fall back to lightweight estimator."""
    if obj.track_quality > 0.0:
        return float(np.clip(obj.track_quality, 0.05, 1.0))
    return track_quality_score(obj)


def _predict_presence_ghost(obj: DetectedObject, age_s: float) -> DetectedObject:
    """Cheap kinematic coast of a remembered vehicle (no vision re-run)."""
    dt = float(np.clip(age_s, 0.0, PRESENCE_HOLD_S))
    vx, vy = obj.image_velocity
    dx = int(round(float(np.clip(vx * dt * 0.55, -40.0, 40.0))))
    dy = int(round(float(np.clip(vy * dt * 0.55, -24.0, 24.0))))
    x1, y1, x2, y2 = obj.box
    box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    distance = float(np.clip(obj.distance_m - obj.closing_mps * dt * 0.85, 1.0, 86.0))
    misses = max(1, int(round(dt * 8.0)))
    conf = float(max(0.12, obj.confidence * (0.94 ** min(misses, 12))))
    return DetectedObject(
        obj.track_id, obj.label, conf, box, distance, obj.bearing_deg,
        obj.closing_mps * 0.90, (vx * 0.45, vy * 0.45), None, False, misses,
        obj.stationary, obj.signal_state, max(0.15, obj.track_quality * 0.88),
    )


def merge_vehicle_presence(objects: list[DetectedObject]) -> list[DetectedObject]:
    """Logical presence: reinject recently-solid cars that the detector dropped.

    If a vehicle had several consecutive hits then vanished for < ~2s, keep a
    coasted ghost so it does not pop out while still in view. Dict ops only.
    """
    now = time.perf_counter()
    live_ids: set[int] = set()
    for obj in objects:
        if obj.label not in RELEVANT_VEHICLE_LABELS:
            continue
        live_ids.add(obj.track_id)
        prev = _VEHICLE_PRESENCE.get(obj.track_id)
        if obj.observed:
            hits = (prev[2] + 1) if prev is not None else 1
            _VEHICLE_PRESENCE[obj.track_id] = (obj, now, hits, now)
        elif prev is not None:
            _VEHICLE_PRESENCE[obj.track_id] = (obj, prev[1], prev[2], now)
        else:
            _VEHICLE_PRESENCE[obj.track_id] = (obj, now - 0.05, PRESENCE_MIN_HITS, now)

    ghosts: list[DetectedObject] = []
    for tid, (last_obj, last_obs, hits, _last_up) in list(_VEHICLE_PRESENCE.items()):
        age = now - last_obs
        if age > PRESENCE_HOLD_S:
            _VEHICLE_PRESENCE.pop(tid, None)
            continue
        if tid in live_ids:
            continue
        if hits < PRESENCE_MIN_HITS:
            if age > 0.35:
                _VEHICLE_PRESENCE.pop(tid, None)
            continue
        ghosts.append(_predict_presence_ghost(last_obj, age))
        if len(ghosts) >= PRESENCE_MAX_GHOSTS:
            break
    # Bound map growth when ByteTrack issues new IDs every few frames.
    if len(_VEHICLE_PRESENCE) > PRESENCE_MAX_TRACKS:
        ordered = sorted(
            _VEHICLE_PRESENCE.items(),
            key=lambda item: item[1][1],  # last_observed
        )
        for tid, _mem in ordered[: len(_VEHICLE_PRESENCE) - PRESENCE_MAX_TRACKS]:
            _VEHICLE_PRESENCE.pop(tid, None)
    if not ghosts:
        return objects
    ghost_ids = {g.track_id for g in ghosts}
    return [o for o in objects if o.track_id not in ghost_ids] + ghosts


def _sticky_top_n(
    ranked: list[tuple[float, DetectedObject]],
    slot_key: str,
    max_n: int,
    *,
    challenge_frames: int = 8,
    score_margin: float = 4.5,
    ghost_by_id: dict[int, DetectedObject] | None = None,
) -> list[DetectedObject]:
    """Keep display slots sticky so top-N IDs do not thrash every frame.

    A challenger must beat an incumbent by ``score_margin`` for
    ``challenge_frames`` consecutive calls before it steals the slot.
    Lower score = better (same convention as filter_relevant_traffic).
    Grace slots can materialize from ``ghost_by_id`` (presence memory).
    """
    ghosts = ghost_by_id or {}
    if max_n <= 0:
        DISPLAY_SLOT_IDS[slot_key] = []
        return []
    if not ranked and not ghosts:
        DISPLAY_SLOT_IDS[slot_key] = []
        return []
    now = time.perf_counter()
    by_id = {obj.track_id: (score, obj) for score, obj in ranked}
    incumbents = list(DISPLAY_SLOT_IDS.get(slot_key, []))
    challenges = DISPLAY_SLOT_CHALLENGE.setdefault(slot_key, {})
    grace_s = 2.05 if slot_key == "vehicle" else 0.90

    live: list[int] = []
    for tid in incumbents:
        if tid in by_id or tid in ghosts:
            live.append(tid)
            DISPLAY_SLOT_GRACE[tid] = now
        elif now - DISPLAY_SLOT_GRACE.get(tid, 0.0) < grace_s:
            if tid in WORLD_POSES or tid in _VEHICLE_PRESENCE:
                live.append(tid)
        else:
            DISPLAY_SLOT_GRACE.pop(tid, None)
            challenges.pop(tid, None)

    held = set(live)
    for score, obj in ranked:
        if len(live) >= max_n:
            break
        if obj.track_id not in held:
            live.append(obj.track_id)
            held.add(obj.track_id)
            DISPLAY_SLOT_GRACE[obj.track_id] = now

    if len(live) >= max_n and ranked:
        worst_tid = None
        worst_score = -1e9
        for tid in live:
            if tid in by_id:
                sc = by_id[tid][0]
            else:
                sc = 50.0
            if sc > worst_score:
                worst_score = sc
                worst_tid = tid
        for score, obj in ranked:
            if obj.track_id in held:
                continue
            if score < worst_score - score_margin:
                challenges[obj.track_id] = challenges.get(obj.track_id, 0) + 1
            else:
                challenges[obj.track_id] = 0
            if challenges.get(obj.track_id, 0) >= challenge_frames and worst_tid is not None:
                live = [tid for tid in live if tid != worst_tid]
                live.append(obj.track_id)
                held = set(live)
                challenges[obj.track_id] = 0
                DISPLAY_SLOT_GRACE[obj.track_id] = now
                DISPLAY_SLOT_GRACE.pop(worst_tid, None)
                break
        for cid in list(challenges.keys()):
            if cid not in held and cid not in by_id:
                challenges.pop(cid, None)

    DISPLAY_SLOT_IDS[slot_key] = live[:max_n]
    out: list[DetectedObject] = []
    for tid in DISPLAY_SLOT_IDS[slot_key]:
        if tid in by_id:
            out.append(by_id[tid][1])
        elif tid in ghosts:
            out.append(ghosts[tid])
    return out


def filter_relevant_traffic(
    objects: list[DetectedObject],
    road_geometry: RoadGeometry | None = None,
    *,
    max_range_m: float = RELEVANT_MAX_RANGE_M,
) -> list[DetectedObject]:
    """Keep nearby vehicles (max 2) plus people/signs/lights for the display.

    Sticky slots + vehicle presence memory reduce pop-out when the detector
    flickers on a car that was just there.
    """
    objects = merge_vehicle_presence(objects)
    if not objects:
        return []
    vehicles: list[tuple[float, DetectedObject]] = []
    vrus: list[tuple[float, DetectedObject]] = []
    signals: list[tuple[float, DetectedObject]] = []
    vehicle_ghosts: dict[int, DetectedObject] = {}
    for obj in objects:
        if obj.label not in RELEVANT_MESH_LABELS:
            continue
        if obj.label in RELEVANT_VEHICLE_LABELS:
            if not obj.observed and obj.missed_updates >= 14:
                continue
        elif not obj.observed and obj.missed_updates >= 5:
            continue
        quality = _object_quality(obj)
        if (obj.label not in RELEVANT_SIGNAL_LABELS and
                obj.label not in RELEVANT_VEHICLE_LABELS and
                quality < 0.18 and obj.missed_updates >= 3):
            continue
        if (obj.label in RELEVANT_VEHICLE_LABELS and not obj.observed and
                quality < 0.12 and obj.missed_updates >= 10):
            continue
        off = _corridor_membership(obj, road_geometry) is False
        lateral, distance = _object_road_coords(obj, road_geometry, off)
        if obj.label in RELEVANT_SIGNAL_LABELS:
            max_sig_range = 72.0 if obj.label in {"traffic light", "stop sign"} else 55.0
            max_sig_lat = 16.0 if obj.label in {"traffic light", "stop sign"} else 12.0
            if distance > max_sig_range or abs(lateral) > max_sig_lat:
                continue
            if quality < 0.10 and obj.missed_updates >= 4:
                continue
            pri = _SIGNAL_PRIORITY.get(obj.label, 5)
            score = distance + pri * 4.0 - obj.confidence * 6.0 - quality * 4.0
            if not obj.observed:
                score += 2.0
            if obj.track_id in DISPLAY_SLOT_IDS.get("signal", []):
                score -= 3.0
            signals.append((score, obj))
            continue
        if distance < RELEVANT_MIN_RANGE_M or distance > max_range_m:
            continue
        if abs(lateral) > RELEVANT_ADJACENT_LANE_M + (1.5 if obj.label in RELEVANT_VRU_LABELS else 0.0):
            continue
        frame_h = road_geometry.frame_height if road_geometry is not None else 0
        if (frame_h > 0 and obj.label in RELEVANT_VEHICLE_LABELS and
                obj.box[3] >= frame_h * 0.97 and distance < 7.0):
            continue
        lane_cost = 0.0 if abs(lateral) <= RELEVANT_SAME_LANE_M else 6.0 + abs(lateral)
        score = distance + lane_cost - obj.confidence * 2.5 - quality * 8.0
        if not obj.observed:
            score += 1.8 + (1.0 - quality) * 1.5
        if obj.label in RELEVANT_VEHICLE_LABELS:
            if obj.track_id in DISPLAY_SLOT_IDS.get("vehicle", []):
                score -= 4.5
            if not obj.observed:
                vehicle_ghosts[obj.track_id] = obj
            vehicles.append((score, obj))
        else:
            if obj.track_id in DISPLAY_SLOT_IDS.get("vru", []):
                score -= 3.5
            vrus.append((score, obj))

    vehicles.sort(key=lambda item: item[0])
    vrus.sort(key=lambda item: item[0])
    signals.sort(key=lambda item: item[0])
    kept = (
        _sticky_top_n(vehicles, "vehicle", RELEVANT_MAX_VEHICLES,
                      challenge_frames=12, score_margin=5.5,
                      ghost_by_id=vehicle_ghosts) +
        _sticky_top_n(vrus, "vru", RELEVANT_MAX_VRU,
                      challenge_frames=12, score_margin=5.5) +
        _sticky_top_n(signals, "signal", RELEVANT_MAX_SIGNALS,
                      challenge_frames=14, score_margin=6.0)
    )
    return sorted(kept, key=lambda item: item.distance_m)


# Sticky lead highlight: closest same-lane vehicle ahead (not quality thrash).
_LEAD_ID: int = -1
_LEAD_HITS: int = 0
_LEAD_CHALLENGER: int = -1
_LEAD_CHALLENGE: int = 0
_LEAD_DIST: float = 1e9
# Half-width of ego lane for LEAD eligibility (metres from path centre).
LEAD_SAME_LANE_M = 1.55
# Challenger must be this much closer (m) for N consecutive frames to steal LEAD.
LEAD_SWAP_MARGIN_M = 4.0
LEAD_SWAP_FRAMES = 18
LEAD_HOLD_MISS = 55  # ~2s @30 FPS — hold LEAD through multi-second detector gaps


def pick_lead_vehicle_id(
    objects: list[DetectedObject],
    road_geometry: RoadGeometry | None = None,
) -> int:
    """Sticky LEAD = closest vehicle directly ahead in the ego lane.

    Rules (strict, in order):
      1. Vehicle class only (car / truck / bus).
      2. Same lane: |lateral| ≤ LEAD_SAME_LANE_M (road-relative when available).
      3. Ahead of ego, not oncoming.
      4. Rank by distance (closest wins); tiny lateral tie-break only.
      5. Sticky: challenger must stay clearly closer for many frames.
      6. Incumbent LEAD may coast as an unobserved ghost for ~2s.

    Quality is *not* mixed into the rank — that was the main thrash source.
    """
    global _LEAD_ID, _LEAD_HITS, _LEAD_CHALLENGER, _LEAD_CHALLENGE, _LEAD_DIST

    # (rank, distance, track_id) — lower rank wins.
    candidates: list[tuple[float, float, int]] = []
    by_id: dict[int, tuple[float, float]] = {}  # id -> (lateral, distance)

    for obj in objects:
        if obj.label not in RELEVANT_VEHICLE_LABELS:
            continue
        # Incumbent LEAD may coast longer; other cars need a fresher observation.
        if not obj.observed:
            max_miss = 14 if obj.track_id == _LEAD_ID else 8
            if obj.missed_updates >= max_miss:
                continue
        # Oncoming latch: never LEAD (opposite direction traffic).
        if WORLD_ONCOMING_LOCK.get(obj.track_id, 0) >= 5:
            continue
        off = _corridor_membership(obj, road_geometry) is False
        if off:
            continue
        lateral, distance = _object_road_coords(obj, road_geometry, False)
        # Directly ahead in ego lane only — no adjacent-lane "almost lead".
        if abs(lateral) > LEAD_SAME_LANE_M:
            continue
        # Fast closing + wide bearing is head-on, not a lead car.
        if float(obj.closing_mps) > 6.0 and abs(obj.bearing_deg) < 14.0 and abs(lateral) > 0.9:
            continue
        if distance < RELEVANT_MIN_RANGE_M or distance > RELEVANT_MAX_RANGE_M:
            continue
        by_id[obj.track_id] = (lateral, distance)
        # Primary: range. Secondary: how centered in lane (directly in front).
        # Incumbent gets a soft distance credit so monocular noise does not flip.
        rank = distance + abs(lateral) * 0.35
        if obj.track_id == _LEAD_ID:
            rank -= LEAD_SWAP_MARGIN_M * 0.55
        if not obj.observed:
            rank += 1.2  # slight preference for live detections, not a hard ban
        candidates.append((rank, distance, obj.track_id))

    if not candidates:
        # Grace hold: keep last LEAD a few frames when it drops from the filter
        # (box flicker) instead of instantly hopping to another car or clearing.
        _LEAD_HITS = max(0, _LEAD_HITS - 1)
        if _LEAD_HITS <= 0:
            _LEAD_ID = -1
            _LEAD_DIST = 1e9
            _LEAD_CHALLENGER = -1
            _LEAD_CHALLENGE = 0
        return _LEAD_ID

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_rank, best_dist, best_id = candidates[0]

    if _LEAD_ID < 0:
        _LEAD_ID = best_id
        _LEAD_DIST = best_dist
        _LEAD_HITS = LEAD_HOLD_MISS
        _LEAD_CHALLENGER = -1
        _LEAD_CHALLENGE = 0
        return _LEAD_ID

    # Incumbent still a valid same-lane candidate — refresh hold.
    if _LEAD_ID in by_id:
        _LEAD_DIST = by_id[_LEAD_ID][1]
        _LEAD_HITS = LEAD_HOLD_MISS
        if best_id == _LEAD_ID:
            _LEAD_CHALLENGER = -1
            _LEAD_CHALLENGE = 0
            return _LEAD_ID
        # Challenger must be clearly closer, not just a quality/rank wobble.
        closer_by = _LEAD_DIST - best_dist
        if closer_by < LEAD_SWAP_MARGIN_M:
            _LEAD_CHALLENGER = -1
            _LEAD_CHALLENGE = 0
            return _LEAD_ID
        if _LEAD_CHALLENGER == best_id:
            _LEAD_CHALLENGE += 1
        else:
            _LEAD_CHALLENGER = best_id
            _LEAD_CHALLENGE = 1
        if _LEAD_CHALLENGE >= LEAD_SWAP_FRAMES:
            _LEAD_ID = best_id
            _LEAD_DIST = best_dist
            _LEAD_HITS = LEAD_HOLD_MISS
            _LEAD_CHALLENGE = 0
            _LEAD_CHALLENGER = -1
        return _LEAD_ID

    # Incumbent missing this frame but still in grace — do not hop yet.
    _LEAD_HITS = max(0, _LEAD_HITS - 1)
    if _LEAD_HITS > 0:
        return _LEAD_ID

    # Grace expired: adopt the current closest same-lane car.
    _LEAD_ID = best_id
    _LEAD_DIST = best_dist
    _LEAD_HITS = LEAD_HOLD_MISS
    _LEAD_CHALLENGER = -1
    _LEAD_CHALLENGE = 0
    return _LEAD_ID


def _oncoming_score(obj: DetectedObject, off_corridor: bool, lateral_m: float,
                    side_ness: float) -> int:
    """Vote count for true opposite-direction traffic (not merges / co-speed)."""
    score = 0
    closing = float(obj.closing_mps)
    abs_lat = abs(lateral_m)
    # Merging cars: side-on silhouette + moderate close — not oncoming.
    if side_ness > 0.55 and closing < 5.0:
        score -= 3
    # Opposite carriageway is the only strong lateral prior (not "any offset").
    if off_corridor and closing > 2.0:
        score += 2
    if off_corridor and closing > 4.0:
        score += 1
    # Fast relative close (closing speeds sum) — true head-on.
    if closing > 4.0:
        score += 1
    if closing > 6.5:
        score += 2
    if closing > 9.0:
        score += 1
    # Frontal + fast close (not used alone — merges can look frontal briefly).
    if side_ness < 0.35 and closing > 4.5:
        score += 1
    if side_ness < 0.28 and closing > 6.0:
        score += 1
    # Undivided head-on: dead-ahead + very fast close.
    if abs(obj.bearing_deg) < 12.0 and closing > 5.5 and side_ness < 0.48:
        score += 2
    # Same-lane / mild offset closer = lead / merge, not oncoming.
    if abs_lat < 2.2 and not off_corridor and closing < 5.0:
        score -= 3
    if abs_lat < 3.5 and side_ness > 0.50 and closing < 4.0:
        score -= 2  # classic merge / cut-in side view
    if obj.stationary:
        score -= 5
    if closing < -0.5:
        score -= 4
    if closing < 1.5:
        score -= 2
    return score


def _update_oncoming_lock(track_id: int, votes: int, closing_mps: float,
                          side_ness: float) -> int:
    """Sticky latch: slow acquire, very slow release (prevents flip-spin)."""
    lock = WORLD_ONCOMING_LOCK.get(track_id, 0)
    closing = float(closing_mps)
    # Need strong multi-cue evidence — do not latch on merge noise.
    if votes >= 6 and closing > 4.0:
        lock = min(30, lock + 3)
    elif votes >= 5 and closing > 3.5:
        lock = min(30, lock + 2)
    elif votes >= 4 and closing > 5.0 and side_ness < 0.45:
        lock = min(30, lock + 1)
    elif lock >= 5:
        # Fully latched: only clear when clearly receding for a while.
        if closing < -1.5:
            lock = max(0, lock - 2)
        elif closing < -0.6 and votes <= 0:
            lock = max(0, lock - 1)
        elif votes <= -2:
            lock = max(0, lock - 1)
        # else hold — never decay on mid-range vote noise
    else:
        # Not latched — kill false starts quickly (merges / co-speed).
        if votes <= 2 or closing < 2.0 or side_ness > 0.50:
            lock = max(0, lock - 2)
        else:
            lock = max(0, lock - 1)
    WORLD_ONCOMING_LOCK[track_id] = lock
    return lock


def _visual_vehicle_heading(obj: DetectedObject, road_heading: float,
                            off_corridor: bool, lateral_m: float = 0.0,
                            track_id: int | None = None) -> float:
    """Binary vehicle yaw only: 0° (with traffic) or 180° (oncoming).

    Silhouette / parked / road-tangent banks are intentionally disabled — they
    spun meshes on merges, curves, and co-speed traffic when aspect/bearing
    jittered. Road_heading is ignored for the same reason.
    """
    del road_heading  # never drive mesh yaw from noisy centre-polyline
    x1, y1, x2, y2 = obj.box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    aspect = bw / bh
    side_ness = float(np.clip((aspect - 1.05) / 1.55, 0.0, 1.0))

    votes = _oncoming_score(obj, off_corridor, lateral_m, side_ness)
    if track_id is not None:
        lock = _update_oncoming_lock(track_id, votes, obj.closing_mps, side_ness)
    else:
        lock = 8 if votes >= 6 else 0

    # Only strong latch → face-on. No intermediate angles. No parked banks.
    want = lock >= 5 or votes >= 6
    if track_id is not None:
        return _binary_vehicle_heading(track_id, want)
    return ONCOMING_HEADING_DEG if want else 0.0


def _resolve_dense_world_slots(
    slots: list[tuple[DetectedObject, float, float, bool]],
) -> list[tuple[DetectedObject, float, float, bool]]:
    """Separate or collapse dense placements so meshes do not stack in one cell.

    Two different cars with nearly equal monocular range were landing on top of
    each other. We (1) drop true duplicates in world space, (2) enforce a
    minimum range gap ordered by image foot y (lower in image = closer).
    """
    if len(slots) <= 1:
        return slots
    # Prefer observed + confident first when collapsing duplicates.
    ranked = sorted(
        slots,
        key=lambda item: (
            0 if item[0].observed else 1,
            -item[0].confidence,
            item[0].missed_updates,
        ),
    )
    accepted: list[tuple[DetectedObject, float, float, bool]] = []
    for obj, lat, dist, off in ranked:
        duplicate = False
        for i, (other, olat, odist, _) in enumerate(accepted):
            d_lat = abs(lat - olat)
            d_dist = abs(dist - odist)
            # Same world cell → keep the better track only.
            if d_lat < 1.35 and d_dist < 3.2:
                # If image feet clearly differ, they are distinct cars: separate.
                foot_a = obj.box[3]
                foot_b = other.box[3]
                if abs(foot_a - foot_b) > 18 or abs(
                    (obj.box[0] + obj.box[2]) * 0.5 - (other.box[0] + other.box[2]) * 0.5
                ) > 28:
                    # Push the farther image foot farther in range.
                    if foot_a < foot_b and dist <= odist + 2.5:
                        dist = odist + 3.4
                    elif foot_b < foot_a and odist <= dist + 2.5:
                        accepted[i] = (other, olat, dist + 3.4, accepted[i][3])
                    else:
                        lat = olat + (1.8 if lat >= olat else -1.8)
                    break
                duplicate = True
                break
        if not duplicate:
            accepted.append((obj, lat, dist, off))

    # Soft depth-order preference vs image foot (closer foot → nearer range).
    # Mild gap only when clearly inverted — large forced gaps every frame looked
    # like position thrash.
    ordered = sorted(accepted, key=lambda item: -item[0].box[3])
    min_gap = 1.6
    last_dist = 0.0
    fixed: list[tuple[DetectedObject, float, float, bool]] = []
    for obj, lat, dist, off in ordered:
        if fixed and dist < last_dist + min_gap:
            # Blend toward the gap instead of hard snap.
            dist = 0.55 * dist + 0.45 * (last_dist + min_gap)
        dist = float(np.clip(dist, 1.0, 86.0))
        fixed.append((obj, lat, dist, off))
        last_dist = dist
    return fixed


def render_world(objects: list[DetectedObject], viewport: tuple[int, int, int, int],
                 road_geometry: RoadGeometry | None = None,
                 *, prefiltered: bool = False, lite: bool = False,
                 lead_id: int | None = None) -> None:
    """Render 3D world. ``lite=True`` (split view) skips costly extras for FPS.

    Pass ``lead_id`` from the once-per-frame sticky picker so challenge counters
    are not advanced twice (overlay + world) and camera/3D stay in sync.
    """
    global _ROAD_CENTER_FRAME_CACHE
    _ROAD_CENTER_FRAME_CACHE = {}  # clear per-frame road-centre sample cache
    x, y, width, height = viewport
    glViewport(x, y, width, height)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_TEXTURE_2D)
    glClear(GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(46.0, width / max(1, height), 0.1, 160.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0.0, 10.5, 14.5, 0.0, 0.9, -24.0, 0.0, 1.0, 0.0)
    # Flat shading in lite split is much cheaper and still readable.
    if lite:
        glDisable(GL_LIGHTING)
    else:
        glEnable(GL_LIGHTING)
    ego_lateral = _ego_lateral_position(road_geometry)
    _road_world(road_geometry, ego_lateral, lite=lite)
    # Caller may already have filtered (shared with camera overlay).
    if prefiltered:
        objects = _dedup_world_objects(objects)
    else:
        objects = filter_relevant_traffic(_dedup_world_objects(objects), road_geometry)
    # Build road-relative slots, resolve dense stacking, then smooth/draw.
    # measured_lateral = vision (lane-centre); display_lateral = lane-slotted for 3D.
    raw_slots: list[tuple[DetectedObject, float, float, bool, float, bool]] = []
    for obj in objects:
        off_corridor = _corridor_membership(obj, road_geometry) is False
        measured_lateral, raw_distance = _object_road_coords(obj, road_geometry, off_corridor)
        # Signals may sit roadside; vehicles stay in the adjacent-lane envelope.
        if obj.label not in RELEVANT_SIGNAL_LABELS and abs(measured_lateral) > RELEVANT_ADJACENT_LANE_M + 0.6:
            continue
        prev_slot = WORLD_LANE_SLOT.get(obj.track_id)
        display_lateral = _display_lateral_for_object(
            obj, measured_lateral, off_corridor=off_corridor,
        )
        lane_changed = (
            prev_slot is not None
            and WORLD_LANE_SLOT.get(obj.track_id) is not None
            and prev_slot != WORLD_LANE_SLOT.get(obj.track_id)
        )
        raw_slots.append(
            (obj, display_lateral, raw_distance, off_corridor, measured_lateral, lane_changed)
        )
    # Dense-slot separation uses display laterals so adjacent stays off ego road.
    dense_in = [(o, lat, dist, off) for o, lat, dist, off, _m, _c in raw_slots]
    dense_out = _resolve_dense_world_slots(dense_in)
    # Re-attach measured lateral + lane_changed by track id.
    meta = {
        o.track_id: (meas, changed)
        for o, _d, _r, _off, meas, changed in raw_slots
    }
    slots: list[tuple[DetectedObject, float, float, bool, float, bool]] = []
    for obj, lat, dist, off in dense_out:
        meas, changed = meta.get(obj.track_id, (lat, False))
        slots.append((obj, lat, dist, off, meas, changed))
    # Lead = closest same-lane vehicle ahead (sticky). Use precomputed id when given.
    if lead_id is None:
        lead_id = pick_lead_vehicle_id([item[0] for item in slots], road_geometry)
    active_ids: set[int] = set()
    for obj, display_lateral, raw_distance, off_corridor, measured_lateral, lane_changed in sorted(
            slots, key=lambda item: item[2], reverse=True):
        selected = (
            obj.track_id == lead_id
            and obj.label in RELEVANT_VEHICLE_LABELS
            and not off_corridor
            and WORLD_LANE_SLOT.get(obj.track_id, "ego") == "ego"
        )
        if obj.label in RELEVANT_VEHICLE_LABELS:
            pose_kind = "vehicle"
            heading = _visual_vehicle_heading(
                obj, 0.0, off_corridor, lateral_m=measured_lateral,
                track_id=obj.track_id,
            )
        elif obj.label in RELEVANT_SIGNAL_LABELS:
            pose_kind = "signal"
            heading = 0.0
        else:
            pose_kind = "vru"
            heading = 0.0
        object_x, object_z, heading = _stable_world_pose(
            obj.track_id, display_lateral, raw_distance, heading, road_geometry,
            off_corridor, obj.observed, obj.stationary, obj.closing_mps,
            kind=pose_kind,
            ego_lateral=ego_lateral,
            lane_changed=lane_changed,
        )
        # Final belt-and-braces: mesh yaw is only ever 0° or 180°.
        if obj.label in RELEVANT_VEHICLE_LABELS | {"motorcycle"}:
            heading = ONCOMING_HEADING_DEG if _is_face_on_heading(heading, 40.0) else 0.0
        else:
            heading = 0.0
        if abs(object_x) > 22.0:
            continue
        active_ids.add(obj.track_id)
        # Coasted / low-quality tracks render dim so live lead stays obvious.
        quality = _object_quality(obj)
        if selected:
            variant = "selected"
        elif off_corridor or (not obj.observed) or quality < 0.35:
            variant = "background"
        else:
            variant = "neutral"
        if obj.label in {"car", "truck", "bus", "motorcycle"}:
            # Cars / trucks / buses share one grey oval mold list ("car").
            # Motorcycle keeps a smaller oval pod so it does not read as a car.
            mesh_kind = "motorcycle" if obj.label == "motorcycle" else "car"
            _cached_mesh(mesh_kind, variant, object_x, object_z, heading)
            # Badges only for LEAD/MERGE; skip entirely in lite split path.
            if not lite:
                is_merge = estimate_merge_flag(obj, measured_lateral)
                if selected or is_merge:
                    _draw_vehicle_badge(
                        object_x, object_z, obj, selected=selected, merge=is_merge,
                    )
        elif obj.label == "person":
            _cached_mesh("person", variant, object_x, object_z, 0.0)
        elif obj.label == "bicycle":
            _cached_mesh("bicycle", variant, object_x, object_z, 0.0)
        elif obj.label in {"traffic light", "stop sign", "parking meter", "fire hydrant"}:
            env = latest_environment()
            is_ego_tl = (
                env.controlling_light is not None
                and obj.track_id == env.controlling_light.track_id
                and obj.label == "traffic light"
            )
            # Signals use neutral/background only (no selected mesh variant).
            _cached_mesh(
                obj.label,
                "background" if off_corridor else "neutral",
                object_x, object_z,
            )
            # Live lamp colour for traffic lights (from vision ROI scan).
            if obj.label == "traffic light" and obj.signal_state:
                lamp = {
                    "red": ((0.0, 3.55, 0.22), (0.98, 0.08, 0.05)),
                    "yellow": ((0.0, 3.20, 0.22), (0.98, 0.82, 0.08)),
                    "green": ((0.0, 2.85, 0.22), (0.08, 0.92, 0.22)),
                }.get(obj.signal_state)
                if lamp is not None:
                    (lx, ly, lz), col = lamp
                    glDisable(GL_LIGHTING)
                    glPushMatrix()
                    glTranslatef(object_x, 0.0, object_z)
                    size = (0.36, 0.32, 0.10) if is_ego_tl else (0.28, 0.26, 0.08)
                    _cuboid((lx, ly, lz), size, col)
                    glPopMatrix()
                    glEnable(GL_LIGHTING)
                if is_ego_tl and not lite:
                    _draw_world_text(
                        object_x - 0.5, 3.95, object_z,
                        f"EGO-TL {(obj.signal_state or '?').upper()}",
                        (1.0, 0.95, 0.4),
                    )
        else:
            color = (0.56, 0.62, 0.72)
            if off_corridor:
                color = tuple(channel * _BACKGROUND_DIM_FACTOR for channel in color)
            _cuboid((object_x, 0.55, object_z), (1.15, 1.10, 1.15), color)
        # Motion arrows disabled in the hot path (extra GL work; low value at 20 FPS).
    # Always prune pose state (even when the frame has zero active tracks).
    # Previously cleanup was gated on `if active_ids`, so empty frames left
    # stale IDs forever — a slow leak that kicked in after ~30s of driving.
    now_pose = time.perf_counter()
    stale = [
        key for key, value in list(WORLD_POSES.items())
        if (key not in active_ids and now_pose - value[3] > 1.35) or (now_pose - value[3] > 3.5)
    ]
    for key in stale:
        WORLD_POSES.pop(key, None)
        WORLD_TRAJECTORY.pop(key, None)
        WORLD_TURN_HITS.pop(key, None)
        WORLD_ONCOMING_LOCK.pop(key, None)
        WORLD_CRUISE_YAW.pop(key, None)
        WORLD_LANE_SLOT.pop(key, None)
        DISPLAY_SLOT_GRACE.pop(key, None)
    # Ego mesh last; fixed down-road heading (no road-noise spin).
    # Same frame as the grey ribbon (lane centre + near-field ego bias).
    ego_x = _road_center_x(2.1, road_geometry) + _ego_lateral_fade(2.1, ego_lateral)
    _cached_mesh("car", "selected", ego_x, 2.1, 0.0)


def _draw_stage_hud(viewport: tuple[int, int, int, int],
                    stage_ms: dict[str, float],
                    display_fps: float,
                    detection_fps: float) -> None:
    """Full-window 2D strip with smoothed stage timings (always-on budget HUD)."""
    vx, vy, vw, vh = viewport
    if vw < 80 or vh < 40:
        return
    glViewport(vx, vy, vw, vh)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, vw, vh, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    bar_h = 22.0
    glColor4f(0.03, 0.035, 0.05, 0.78)
    glBegin(GL_QUADS)
    glVertex2f(0.0, 0.0)
    glVertex2f(float(vw), 0.0)
    glVertex2f(float(vw), bar_h)
    glVertex2f(0.0, bar_h)
    glEnd()
    total = sum(float(stage_ms.get(k, 0.0)) for k in ("src", "ovl", "gl", "flip"))
    line = (
        f"STAGE ms  src {stage_ms.get('src', 0.0):.1f}  "
        f"ovl {stage_ms.get('ovl', 0.0):.1f}  "
        f"gl {stage_ms.get('gl', 0.0):.1f}  "
        f"flip {stage_ms.get('flip', 0.0):.1f}  "
        f"road {stage_ms.get('road', 0.0):.1f}  "
        f"| sum {total:.0f}  |  {display_fps:.0f} FPS  det {detection_fps:.0f}"
    )
    if _ensure_glut():
        glColor3f(0.78, 0.86, 0.94)
        glRasterPos2f(8.0, 15.0)
        for ch in line:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
    glDisable(GL_BLEND)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)


def render_camera_texture(texture_id: int, image_bgr: np.ndarray,
                          viewport: tuple[int, int, int, int],
                          *, upload: bool = True) -> None:
    """Draw the camera pane. Upload native overlay size; GL stretches to the viewport.

    Avoids a full-pane CPU resize every frame (was the main split webcam hitch).
    """
    global CAMERA_TEXTURE_SIZE
    x, y, vp_w, vp_h = viewport
    glViewport(x, y, vp_w, vp_h)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    img_h, img_w = image_bgr.shape[:2]
    if upload or CAMERA_TEXTURE_SIZE != (img_w, img_h):
        # Avoid an extra CPU copy when the overlay buffer is already contiguous.
        pixels = image_bgr if image_bgr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(image_bgr)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        if CAMERA_TEXTURE_SIZE != (img_w, img_h):
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, img_w, img_h, 0,
                GL_BGR, GL_UNSIGNED_BYTE, pixels,
            )
            CAMERA_TEXTURE_SIZE = (img_w, img_h)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0, img_w, img_h,
                GL_BGR, GL_UNSIGNED_BYTE, pixels,
            )
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(1, 0)
    glTexCoord2f(1, 0)
    glVertex2f(1, 1)
    glTexCoord2f(0, 0)
    glVertex2f(0, 1)
    glEnd()
    glDisable(GL_TEXTURE_2D)


# Cache last scaled road for the camera pane when geometry identity is unchanged.
_OVERLAY_ROAD_CACHE_KEY: int | None = None
_OVERLAY_ROAD_CACHE: RoadGeometry | None = None
_OVERLAY_ROAD_SCALE: float = 0.0


def _camera_overlay_inputs(frame: np.ndarray, objects: list[DetectedObject],
                           road_geometry: RoadGeometry | None, max_width: int
                           ) -> tuple[np.ndarray, list[DetectedObject], RoadGeometry | None]:
    """Scale camera-overlay work to its visible viewport instead of full source resolution."""
    global _OVERLAY_ROAD_CACHE_KEY, _OVERLAY_ROAD_CACHE, _OVERLAY_ROAD_SCALE
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, objects, road_geometry
    scale = max_width / width
    output_size = (max_width, max(1, int(round(height * scale))))
    # LINEAR is much cheaper than AREA at this size and looks fine on the pane.
    scaled_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
    scaled_objects: list[DetectedObject] = []
    for obj in objects:
        box = tuple(int(round(value * scale)) for value in obj.box)
        # Skip mask polygons on the overlay path — boxes are enough and cheap.
        scaled_objects.append(DetectedObject(
            obj.track_id, obj.label, obj.confidence, box, obj.distance_m,
            obj.bearing_deg, obj.closing_mps,
            (obj.image_velocity[0] * scale, obj.image_velocity[1] * scale), None,
            obj.observed, obj.missed_updates, obj.stationary,
            obj.signal_state, obj.track_quality,
        ))
    scaled_road = None
    if road_geometry is not None:
        # Reuse scaled polylines while the same RoadGeometry object is live
        # (fresh UFLD/road updates allocate a new geometry → natural bust).
        road_key = id(road_geometry)
        if (
            _OVERLAY_ROAD_CACHE is not None
            and _OVERLAY_ROAD_CACHE_KEY == road_key
            and abs(_OVERLAY_ROAD_SCALE - scale) < 1e-6
            and _OVERLAY_ROAD_CACHE.frame_width == output_size[0]
        ):
            return scaled_frame, scaled_objects, _OVERLAY_ROAD_CACHE

        def points(value: np.ndarray | None) -> np.ndarray | None:
            return None if value is None else np.rint(value.astype(np.float32) * scale).astype(np.int32)

        # Overlay only needs polylines for the ego triangle. Drivable mask is
        # only required when no centreline exists (rare with UFLD).
        needs_mask = (road_geometry.center_points is None or
                      len(road_geometry.center_points) < 4)
        scaled_road = RoadGeometry(
            points(road_geometry.left_points), points(road_geometry.right_points),
            points(road_geometry.center_points), road_geometry.confidence,
            road_geometry.topology,
            None if road_geometry.intersection_y is None else int(round(road_geometry.intersection_y * scale)),
            output_size[0], output_size[1],
            None if road_geometry.drivable_mask is None or not needs_mask else cv2.resize(
                road_geometry.drivable_mask, output_size, interpolation=cv2.INTER_NEAREST,
            ),
            None,
            road_geometry.source,
            None if road_geometry.confident_top_y is None else int(round(road_geometry.confident_top_y * scale)),
        )
        _OVERLAY_ROAD_CACHE_KEY = road_key
        _OVERLAY_ROAD_CACHE = scaled_road
        _OVERLAY_ROAD_SCALE = scale
    return scaled_frame, scaled_objects, scaled_road


def _window_size(args: argparse.Namespace) -> tuple[int, int, tuple[int, int] | None]:
    monitors = get_monitors() if get_monitors is not None else []
    if 0 <= args.monitor < len(monitors):
        monitor = monitors[args.monitor]
        if args.window_width > 0 and args.window_height > 0 and not args.fullscreen:
            width = min(args.window_width, monitor.width)
            height = min(args.window_height, monitor.height)
            position = (
                monitor.x + max(0, (monitor.width - width) // 2),
                monitor.y + max(0, (monitor.height - height) // 2),
            )
            return width, height, position
        return monitor.width, monitor.height, (monitor.x, monitor.y)
    return 1024, 600, None


def _draw_loading_screen(
    surface: pygame.Surface,
    headline: str,
    detail: str = "",
    *,
    phase: int = 1,
    phases: int = 3,
) -> None:
    """Simple 2D loading UI so the window is visible during YouTube/model load."""
    w, h = surface.get_size()
    surface.fill((10, 12, 18))
    # Accent bar
    pygame.draw.rect(surface, (28, 110, 190), pygame.Rect(0, 0, w, 4))
    try:
        title_font = pygame.font.SysFont("segoeui", 32, bold=True)
        body_font = pygame.font.SysFont("segoeui", 20)
        small_font = pygame.font.SysFont("segoeui", 16)
    except Exception:
        title_font = pygame.font.Font(None, 40)
        body_font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)

    title = title_font.render("VisionFSD Pilot", True, (230, 236, 245))
    surface.blit(title, (w // 2 - title.get_width() // 2, h // 2 - 90))

    sub = body_font.render(headline[:120] if headline else "Loading…", True, (180, 200, 220))
    surface.blit(sub, (w // 2 - sub.get_width() // 2, h // 2 - 30))

    if detail:
        det = small_font.render(detail[:140], True, (120, 140, 160))
        surface.blit(det, (w // 2 - det.get_width() // 2, h // 2 + 8))

    # Progress track
    bar_w = min(420, w - 80)
    bar_h = 10
    bx = (w - bar_w) // 2
    by = h // 2 + 50
    pygame.draw.rect(surface, (40, 48, 60), pygame.Rect(bx, by, bar_w, bar_h), border_radius=4)
    fill = int(bar_w * max(0.05, min(1.0, phase / max(1, phases))))
    # Pulse a slice while waiting on long I/O
    pulse = int((time.perf_counter() * 1.6) % 1.0 * (bar_w - fill * 0.35))
    pygame.draw.rect(
        surface, (40, 140, 230),
        pygame.Rect(bx + min(pulse, bar_w - max(fill, 40)), by, max(fill, 40), bar_h),
        border_radius=4,
    )

    hint = small_font.render(
        "Window stays responsive — Esc cancels · first YouTube start may take a few minutes",
        True, (90, 105, 125),
    )
    surface.blit(hint, (w // 2 - hint.get_width() // 2, h - 48))


def _pump_loading(
    surface: pygame.Surface,
    headline: str,
    detail: str = "",
    *,
    phase: int = 1,
    phases: int = 3,
) -> bool:
    """Draw loading UI and pump events. Returns False if user wants to quit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    _draw_loading_screen(surface, headline, detail, phase=phase, phases=phases)
    pygame.display.flip()
    return True


def _run_while_loading(
    surface: pygame.Surface,
    headline: str,
    detail: str,
    work: Callable[[], object],
    *,
    phase: int = 2,
    phases: int = 4,
) -> tuple[bool, object | None, BaseException | None]:
    """Run ``work`` on a daemon thread while keeping the loading window alive.

    Returns (ok_not_cancelled, result, error).
    """
    box: dict[str, object] = {"done": False, "result": None, "error": None}

    def _runner() -> None:
        try:
            box["result"] = work()
        except BaseException as exc:  # noqa: BLE001 — surface to caller
            box["error"] = exc
        finally:
            box["done"] = True

    t = threading.Thread(target=_runner, name="visionfsd-load-step", daemon=True)
    t.start()
    clock = pygame.time.Clock()
    while not bool(box["done"]):
        if not _pump_loading(surface, headline, detail, phase=phase, phases=phases):
            return False, None, None
        clock.tick(30)
    t.join(timeout=1.0)
    err = box["error"]
    if isinstance(err, BaseException):
        return True, None, err
    return True, box["result"], None


def _save_frame(path: Path, width: int, height: int) -> None:
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    glReadBuffer(GL_BACK)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    cv2.imwrite(str(path), cv2.cvtColor(np.flipud(image), cv2.COLOR_RGB2BGR))


def _demo_objects() -> list[DetectedObject]:
    specs = (("car", 11.0, -5.0), ("truck", 25.0, 7.0), ("bus", 35.0, -13.0),
             ("person", 15.0, 15.0), ("bicycle", 21.0, -20.0),
             ("traffic light", 42.0, 2.0), ("stop sign", 29.0, 24.0), ("motorcycle", 18.0, -9.0))
    return [DetectedObject(index + 1, label, 0.92, (0, 0, 1, 1), distance, bearing, 0.0, (0.0, 0.0))
            for index, (label, distance, bearing) in enumerate(specs)]


def _demo_road_geometry() -> RoadGeometry:
    depth = np.linspace(0.0, 1.0, 28)
    center_x = 640.0 - 145.0 * depth ** 1.45
    center_y = 704.0 - 330.0 * depth
    width = 430.0 - 225.0 * depth
    center = np.column_stack((center_x, center_y)).astype(np.int32)
    left = np.column_stack((center_x - width / 2.0, center_y)).astype(np.int32)
    right = np.column_stack((center_x + width / 2.0, center_y)).astype(np.int32)
    # Fully confident for demo purposes: the whole curve should render solid.
    return RoadGeometry(left, right, center, 1.0, "curve-left", None, 1280, 720,
                        confident_top_y=int(center[-1, 1]))


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
    # Leave spare cores for OpenVINO CPU UFLD + the render loop.
    cpu_budget = int(np.clip(args.cpu_threads, 1, 8))
    if args.ufld and str(args.ufld_device).upper() == "CPU":
        cpu_budget = min(cpu_budget, 2)
    torch.set_num_threads(cpu_budget)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    # Cap OpenCV internal threads so resize/draw do not fight the main loop.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Show a 2D loading window IMMEDIATELY so the user always sees a screen.
    # Heavy I/O (YouTube clip cache) runs on a worker thread; models load
    # after, then we switch the same window to OpenGL for the real UI.
    # ------------------------------------------------------------------
    target_width, target_height, position = _window_size(args)
    if position is not None:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{position[0]},{position[1]}"
    os.environ.setdefault("SDL_GL_SWAP_CONTROL", "0")
    os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
    pygame.init()
    try:
        pygame.font.init()
    except Exception:
        pass
    load_display = pygame.display.set_mode((target_width, target_height))
    pygame.display.set_caption("VisionFSD Pilot - Loading…")
    if not _pump_loading(load_display, "Starting…", "Opening window", phase=0, phases=4):
        pygame.quit()
        return 0

    print("VisionFSD Pilot — loading (window is open)...")
    print("  1/3 Input source...")
    load_status: dict[str, object] = {
        "msg": "1/3 Opening input source…",
        "detail": "YouTube first-run may download a short local clip",
        "done": False,
        "error": None,
        "result": None,
    }

    def _on_input_status(msg: str) -> None:
        load_status["msg"] = "1/3 Input source"
        load_status["detail"] = msg[:140]

    def _input_worker() -> None:
        try:
            load_status["result"] = _open_input(args, status_cb=_on_input_status)
        except Exception as exc:
            load_status["error"] = exc
        finally:
            load_status["done"] = True

    worker = threading.Thread(target=_input_worker, name="visionfsd-open-input", daemon=True)
    worker.start()
    load_clock = pygame.time.Clock()
    while not bool(load_status["done"]):
        if not _pump_loading(
            load_display,
            str(load_status.get("msg") or "Loading…"),
            str(load_status.get("detail") or ""),
            phase=1,
            phases=4,
        ):
            print("Cancelled during input load.")
            pygame.quit()
            return 0
        load_clock.tick(30)
    worker.join(timeout=1.0)

    if load_status["error"] is not None:
        print(f"Unable to resolve input source: {load_status['error']}")
        _pump_loading(load_display, "Input failed", str(load_status["error"])[:120], phase=1, phases=4)
        time.sleep(2.0)
        pygame.quit()
        return 2
    cap, video_mode, source_metadata = load_status["result"]  # type: ignore[misc]
    if not cap.isOpened():
        print(f"Unable to open input source: {args.source if args.source else args.camera}.")
        pygame.quit()
        return 2
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) if video_mode else 0.0
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = float(args.fps)
    if video_mode:
        actual_start = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        print(f"  Video ready at {actual_start:.2f}s; source rate {source_fps:.2f} FPS.")

    print("  2/3 Object model...")
    ok, model, model_err = _run_while_loading(
        load_display,
        "2/4 Loading object model…",
        str(args.model),
        lambda: YOLO(args.model, task=args.model_task),
        phase=2,
        phases=4,
    )
    if not ok:
        cap.release()
        pygame.quit()
        return 0
    if model_err is not None:
        cap.release()
        print(f"Unable to load model '{args.model}': {model_err}")
        pygame.quit()
        return 3
    object_perception = AsyncObjectPerception(
        model, args.confidence, args.imgsz, args.device, args.fov,
        args.camera_height, args.horizon_ratio,
    )
    road_tracker = RoadGeometryTracker()
    road_perception: AsyncRoadPerception | None = None
    if args.learned_road:
        road_model_path = Path(args.road_model)
        if not road_model_path.is_absolute():
            road_model_path = PROJECT_ROOT / road_model_path
        print(f"  Loading YOLOPv2 road model on {args.road_device}...")

        def _load_road() -> AsyncRoadPerception:
            return AsyncRoadPerception(
                YOLOPv2RoadEngine(road_model_path, args.road_device),
                road_tracker, args.horizon_ratio,
            )

        ok, road_perception, road_err = _run_while_loading(
            load_display,
            "2/4 Loading YOLOPv2 road model…",
            f"device={args.road_device}",
            _load_road,
            phase=2,
            phases=4,
        )
        if not ok:
            cap.release()
            pygame.quit()
            return 0
        if road_err is not None:
            print(f"  YOLOPv2 road model unavailable; classical fallback: {road_err}")
            road_perception = None
        else:
            print("  YOLOPv2 road/lane perception ready.")
    depth_perception: AsyncDepthPerception | None = None
    if args.depth:
        depth_model_path = resolve_depth_model_path(args.depth_model, PROJECT_ROOT)
        print(f"  Loading Depth Anything V2 Small on {args.depth_device}...")

        def _load_depth() -> AsyncDepthPerception:
            return AsyncDepthPerception(
                DepthAnythingEngine(depth_model_path, args.depth_device),
            )

        ok, depth_perception, depth_err = _run_while_loading(
            load_display,
            "2/4 Loading depth model…",
            f"device={args.depth_device}",
            _load_depth,
            phase=2,
            phases=4,
        )
        if not ok:
            cap.release()
            pygame.quit()
            return 0
        if depth_err is not None:
            print(f"  Depth model unavailable; monocular range only: {depth_err}")
            print("  Export with: .venv\\Scripts\\python.exe tools\\export_depth_anything_openvino.py")
            depth_perception = None
        else:
            print("  Depth range stabilization ready.")
    ufld_perception: AsyncUfldPerception | None = None
    if args.ufld:
        ufld_model_path = resolve_ufld_model_path(args.ufld_model, PROJECT_ROOT)
        print(f"  Loading Ultra-Fast Lane Detection on {args.ufld_device}...")

        def _load_ufld() -> AsyncUfldPerception:
            return AsyncUfldPerception(
                UfldEngine(ufld_model_path, args.ufld_device),
            )

        ok, ufld_perception, ufld_err = _run_while_loading(
            load_display,
            "2/4 Loading UFLD lane model…",
            f"device={args.ufld_device}",
            _load_ufld,
            phase=2,
            phases=4,
        )
        if not ok:
            cap.release()
            pygame.quit()
            return 0
        if ufld_err is not None:
            print(f"  UFLD unavailable; YOLOPv2/classical path only: {ufld_err}")
            print("  Export with: .venv\\Scripts\\python.exe tools\\export_ufld_openvino.py")
            ufld_perception = None
        else:
            print("  UFLD ego-path lanes ready.")

    if not _pump_loading(load_display, "3/4 Starting 3D renderer…", "OpenGL init", phase=3, phases=4):
        cap.release()
        pygame.quit()
        return 0
    print("  3/3 Opening OpenGL display...")
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    try:
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)
    except Exception:
        pass
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    try:
        display = pygame.display.set_mode((target_width, target_height), flags, vsync=0)
    except TypeError:
        display = pygame.display.set_mode((target_width, target_height), flags)
    pygame.display.set_caption("VisionFSD Pilot - 3D Read-Only Visualizer")
    glClearColor(0.012, 0.014, 0.020, 1.0)
    glDisable(GL_DITHER)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glLightfv(GL_LIGHT0, GL_POSITION, (7.0, 18.0, 12.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.34, 0.36, 0.43, 1.0))
    initialize_mesh_cache()
    texture_id = glGenTextures(1)
    # Immediate clear so the first visible frame is not garbage / stuck black.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    pygame.display.flip()
    pygame.event.pump()

    latest_objects: list[DetectedObject] = []
    cached_lanes: list[tuple[int, int, int, int]] = []
    latest_road_geometry: RoadGeometry | None = None
    show_lanes = True
    view_mode = args.view
    frame_number = 0
    detection_interval = max(1, args.detect_interval)
    road_interval = max(1, args.road_interval)
    depth_interval = max(2, int(args.depth_interval))
    ufld_interval = max(3, int(args.ufld_interval))
    # When UFLD owns the ego path, YOLOPv2 can run less often (drivable still
    # updates, but paint is secondary) — frees iGPU lock time for detect.
    # Floor 6 (was 8) keeps road a bit fresher without stacking both nets.
    if ufld_perception is not None and road_interval < 6:
        road_interval = max(road_interval, 6)
    detection_fps, display_fps = 0.0, 0.0
    detection_passes = 0
    object_result_sequence = -1
    last_detect_submit = -999
    object_observations = 0
    class_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    road_result_sequence = -1
    road_updates = 0
    road_inference_ms = 0.0
    depth_result_sequence = -1
    depth_updates = 0
    depth_inference_ms = 0.0
    depth_scale = 1.0
    latest_depth_map: np.ndarray | None = None
    ufld_result_sequence = -1
    ufld_updates = 0
    ufld_inference_ms = 0.0
    latest_ufld: UfldLaneResult | None = None
    _cached_ufld_geometry: RoadGeometry | None = None
    _cached_filtered_objects: list[DetectedObject] | None = None
    _cached_frame_lead_id: int = -1
    _road_defer_submit = False
    _last_maintenance = time.perf_counter()
    _maint_phase = 0
    _last_gc1 = 0.0
    video_reader: LatestVideoFrameReader | None = None
    initial_video_frame: np.ndarray | None = None
    last_display_frame: np.ndarray | None = None
    video_sequence = 0
    test_started: float | None = None
    test_frame_start = 0
    last_time = time.perf_counter()
    timing_samples = 0
    source_wait_total = 0.0
    overlay_total = 0.0
    render_total = 0.0
    flip_total = 0.0
    tick_total = 0.0
    # Live EMA stage budget (ms) — always on for the STAGE HUD strip.
    stage_ms: dict[str, float] = {
        "src": 0.0, "ovl": 0.0, "gl": 0.0, "flip": 0.0, "road": 0.0,
        "depth": 0.0, "ufld": 0.0, "tick": 0.0,
    }
    clock = pygame.time.Clock()
    running = True
    cached_camera_overlay: np.ndarray | None = None
    print("Ready. 1 world; 2 camera; 3 split; V cycle; L lanes; S screenshot; F fullscreen; Q/Esc quits.")
    print("Stage HUD: src/ovl/gl/flip/road ms (top bar).")
    while running:
        measure_stages = test_started is not None
        # Always pump the event queue first so Windows never marks us hung.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_1, pygame.K_m):
                    view_mode = "world"
                elif event.key == pygame.K_2:
                    view_mode = "camera"
                elif event.key == pygame.K_3:
                    view_mode = "split"
                elif event.key == pygame.K_v:
                    view_mode = {"world": "camera", "camera": "split", "split": "world"}[view_mode]
                elif event.key == pygame.K_l:
                    show_lanes = not show_lanes
                elif event.key == pygame.K_f:
                    try:
                        pygame.display.toggle_fullscreen()
                    except pygame.error:
                        pass
                elif event.key == pygame.K_s:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    _save_frame(SCREENSHOT_DIR / f"visionfsd-3d-{stamp}.png", target_width, target_height)
        if not running:
            break
        source_started = time.perf_counter()
        if video_reader is not None:
            # Short wait (~1 source frame). On timeout re-show last decoded frame
            # instead of stalling the whole UI (was up to 2.0 s).
            ok, frame, video_sequence = video_reader.latest(video_sequence, timeout=0.038)
            if ok and frame is not None:
                last_display_frame = frame
            elif last_display_frame is not None:
                ok, frame = True, last_display_frame
        elif video_mode and args.realtime_video and initial_video_frame is not None:
            ok, frame = True, initial_video_frame
        else:
            # Webcam / direct capture: drop stale buffered frames for freshest image.
            if not video_mode:
                # BUFFERSIZE=1 still occasionally queues; one grab peels a stale frame.
                try:
                    cap.grab()
                except Exception:
                    pass
            ok, frame = cap.read()
            if ok and video_mode and args.realtime_video and initial_video_frame is None:
                initial_video_frame = frame
            if ok and frame is not None:
                last_display_frame = frame
        if not ok:
            print("Video stream ended or failed." if video_mode else "Webcam frame capture failed.")
            break
        source_wait = time.perf_counter() - source_started
        assert frame is not None
        now = time.perf_counter()
        display_instant = 1.0 / max(now - last_time, 1e-3)
        display_fps = display_instant if display_fps == 0 else display_fps * 0.88 + display_instant * 0.12
        last_time = now
        frame_number += 1
        frame_height, frame_width = frame.shape[:2]
        objects_refreshed = False
        capture_stamp = now  # frame capture time for stale-result rejection
        # --- Adaptive intervals: snappy when FPS high; protect ≥25 floor ---
        # Priority when stressed: stretch UFLD first, then road, never kill detect.
        if display_fps < 1.0 or frame_number < 30:
            effective_detect = detection_interval
            effective_road = road_interval
            effective_ufld_interval = ufld_interval
        elif display_fps >= 29.0:
            effective_detect = max(2, detection_interval - 1)
            effective_road = max(5, road_interval - 1) if ufld_perception else max(3, road_interval - 1)
            effective_ufld_interval = max(12, ufld_interval - 4)
        elif display_fps >= 27.0:
            effective_detect = detection_interval
            effective_road = road_interval
            effective_ufld_interval = ufld_interval
        elif display_fps >= 25.2:
            effective_detect = max(detection_interval, 3)
            effective_road = max(road_interval, 8)
            effective_ufld_interval = max(ufld_interval, 28)
        else:
            # Below target: shed secondary nets first; keep detect alive.
            effective_detect = max(detection_interval, 4)
            effective_road = max(road_interval, 12)
            effective_ufld_interval = max(ufld_interval, 40)

        detect_phase = (frame_number - 1) % max(1, effective_detect)
        # Submit on schedule, or fill pipeline when worker is idle.
        if detect_phase == 0 or (
                object_perception.idle()
                and (frame_number - last_detect_submit) >= max(2, effective_detect // 2)
                and display_fps >= 25.0
        ):
            object_perception.submit(frame, frame_number, capture_time=capture_stamp)
            last_detect_submit = frame_number
        # Latest-only + stale reject: drop results older than ~1.25s or lagging
        # more than ~48 display frames behind the current sequence.
        object_result = object_perception.latest_after(
            object_result_sequence, max_age_s=1.25,
        )
        if object_result is not None:
            lag = frame_number - int(object_result.sequence)
            if lag <= 48:
                object_result_sequence = object_result.sequence
                latest_objects = object_result.objects
                objects_refreshed = True
                instant_detection_fps = 1000.0 / max(object_result.pipeline_ms, 1.0)
                detection_fps = (instant_detection_fps if detection_fps == 0.0 else
                                 detection_fps * 0.82 + instant_detection_fps * 0.18)
                detection_passes += 1
                object_observations += len(latest_objects)
                class_counts.update(obj.label for obj in latest_objects)
                cached_lanes = []
            # else: stale late result — keep previous latest_objects
        # Transient worker errors must not kill the whole app (they often clear
        # after the shared GPU lock recovers). Log once and keep rendering.
        object_err = object_perception.error
        if object_err is not None and frame_number % 60 == 0:
            print(f"Object perception warning (continuing): {object_err}")
        # Road perception runs on its own cadence (not gated on object results)
        # so the camera path/lanes track live road curvature in real time.
        # Geometry fitting happens inside the worker thread on the exact frame
        # the model saw; the render thread only picks up finished results.
        road_got_fresh = False
        if show_lanes:
            if road_perception is not None:
                # Mid-cycle road submit. Prefer not to share a frame with detect.
                road_due = (frame_number - 1) % max(1, effective_road) == (
                    max(1, effective_road) // 2
                )
                if road_due:
                    if detect_phase != 0:
                        road_perception.submit(frame, frame_number, capture_time=capture_stamp)
                    else:
                        _road_defer_submit = True
                if _road_defer_submit and detect_phase != 0:
                    road_perception.submit(frame, frame_number, capture_time=capture_stamp)
                    _road_defer_submit = False
                road_update = road_perception.latest_after(
                    road_result_sequence, max_age_s=2.0,
                )
                if road_update is not None and (frame_number - int(road_update.sequence)) <= 60:
                    road_result_sequence = road_update.sequence
                    latest_road_geometry = road_update.geometry
                    road_inference_ms = (road_update.inference_ms if road_inference_ms == 0.0 else
                                         road_inference_ms * 0.75 + road_update.inference_ms * 0.25)
                    road_updates += 1
                    topology_counts[latest_road_geometry.topology] += 1
                    road_got_fresh = True
                road_err = road_perception.error
                if road_err is not None and frame_number % 60 == 0:
                    print(f"YOLOPv2 road warning (retaining last path): {road_err}")
            elif (frame_number - 1) % max(1, effective_road) == 0:
                latest_road_geometry = road_tracker.update(frame, horizon_ratio=args.horizon_ratio)
                road_updates += 1
                topology_counts[latest_road_geometry.topology] += 1
                road_got_fresh = True
            # Skip predict() when UFLD is driving the path (saves CPU every frame).
            if (ufld_perception is None and not road_got_fresh and
                    latest_road_geometry is not None and
                    (time.perf_counter() - getattr(road_tracker, "_last_update_time", 0.0)) > 0.12):
                predicted = road_tracker.predict()
                if predicted is not None:
                    latest_road_geometry = predicted
        # Depth Anything V2 Small — throttled async; fuse into track ranges.
        depth_refreshed = False
        if depth_perception is not None:
            # Stagger vs road submits so the iGPU lock is not triple-hitched.
            # Stretch depth first when FPS is near the floor.
            eff_depth = depth_interval
            if 1.0 <= display_fps < 25.5:
                eff_depth = max(depth_interval, 10)
            depth_phase = (frame_number + eff_depth // 2) % eff_depth
            if depth_phase == 0:
                depth_perception.submit(frame, frame_number)
            depth_update = depth_perception.latest_after(depth_result_sequence)
            if depth_update is not None:
                depth_result_sequence = depth_update.sequence
                latest_depth_map = depth_update.depth_map
                depth_inference_ms = (
                    depth_update.inference_ms if depth_inference_ms == 0.0 else
                    depth_inference_ms * 0.75 + depth_update.inference_ms * 0.25
                )
                depth_updates += 1
                depth_refreshed = True
            depth_err = depth_perception.error
            if depth_err is not None and frame_number % 60 == 0:
                print(f"Depth warning (monocular fallback): {depth_err}")
            # Fuse only when mono tracks or depth map changed (avoid double-blend).
            if latest_depth_map is not None and latest_objects and (
                    objects_refreshed or depth_refreshed
            ):
                latest_objects, depth_scale = fuse_objects_with_depth(
                    latest_objects, latest_depth_map,
                    focal_px=focal_length_px(frame_width, args.fov),
                    frame_height=frame_height,
                    camera_height_m=args.camera_height,
                    horizon_ratio=args.horizon_ratio,
                    depth_weight=0.40,
                    prev_scale=depth_scale,
                )
        display_road_geometry = latest_road_geometry if show_lanes else None
        # UFLD — throttled; ego left/right polylines feed the yellow path triangle.
        ufld_merged_dirty = road_got_fresh  # remarge when YOLOPv2 road refreshes
        if ufld_perception is not None and show_lanes:
            # Offset phase; never collide with a detect submit (CPU/GPU calm).
            # Adaptive: stretch UFLD first when FPS sags; never permanently disable.
            ufld_phase = (frame_number + 2) % max(1, effective_ufld_interval)
            fps_ok_for_ufld = display_fps >= 25.0 or display_fps < 1.0 or frame_number < 45
            if ufld_phase == 0 and detect_phase != 0 and fps_ok_for_ufld:
                ufld_perception.submit(frame, frame_number, capture_time=capture_stamp)
            ufld_update = ufld_perception.latest_after(
                ufld_result_sequence, max_age_s=2.5,
            )
            if ufld_update is not None and (frame_number - int(ufld_update.sequence)) <= 90:
                ufld_result_sequence = ufld_update.sequence
                latest_ufld = ufld_update
                ufld_inference_ms = (
                    ufld_update.inference_ms if ufld_inference_ms == 0.0 else
                    ufld_inference_ms * 0.75 + ufld_update.inference_ms * 0.25
                )
                ufld_updates += 1
                ufld_merged_dirty = True
            ufld_err = ufld_perception.error
            if ufld_err is not None and frame_number % 60 == 0:
                print(f"UFLD warning (path fallback): {ufld_err}")
            # Only re-merge when UFLD or road geometry actually changed.
            if latest_ufld is not None and ufld_merged_dirty:
                display_road_geometry = merge_ufld_into_geometry(
                    display_road_geometry, latest_ufld,
                    frame_width=frame_width, frame_height=frame_height,
                )
                _cached_ufld_geometry = display_road_geometry
            elif latest_ufld is not None and _cached_ufld_geometry is not None:
                display_road_geometry = _cached_ufld_geometry
        # Road-foot range prior (no depth NN) when tracks or road geometry refresh.
        if latest_objects and (objects_refreshed or road_got_fresh or ufld_merged_dirty):
            latest_objects = apply_road_range_prior(
                latest_objects, display_road_geometry,
                frame_width=frame_width, frame_height=frame_height,
                fov_deg=args.fov,
                camera_height_m=args.camera_height,
                horizon_ratio=args.horizon_ratio,
            )
        if args.demo_scene:
            latest_objects = _demo_objects()
        if args.demo_road:
            display_road_geometry = _demo_road_geometry()
        # Stagger heavy work across frames so split view does not hitch on one
        # "super frame" (root of dual-pane stutter).
        split_mode = view_mode == "split"
        # Road-paint: skip in split entirely (3D lite); world-only every 8 frames.
        if view_mode == "world" and frame_number % 8 == 0:
            scan_road_markings(frame, every_n=1, max_markings=2)
        # Filter + LEAD only when perception state changed (saves main-thread CPU).
        perception_dirty = (
            objects_refreshed or road_got_fresh or ufld_merged_dirty or
            frame_number <= 2 or _cached_filtered_objects is None
        )
        if perception_dirty:
            filtered_objects = filter_relevant_traffic(latest_objects, display_road_geometry)
            frame_lead_id = pick_lead_vehicle_id(filtered_objects, display_road_geometry)
            _cached_filtered_objects = filtered_objects
            _cached_frame_lead_id = frame_lead_id
        else:
            filtered_objects = _cached_filtered_objects or []
            frame_lead_id = _cached_frame_lead_id
        # Environment intel — less often in split (8) than solo world (4).
        env_period = 8 if split_mode else 4
        if frame_number % env_period == 0 or frame_number <= 2:
            oncoming_active = any(
                WORLD_ONCOMING_LOCK.get(obj.track_id, 0) >= 5
                for obj in filtered_objects
                if obj.label in RELEVANT_VEHICLE_LABELS
            )
            update_environment_intel(
                filtered_objects, display_road_geometry,
                oncoming_active=oncoming_active,
                horizon_ratio=args.horizon_ratio,
                camera_height_m=args.camera_height,
                fov_deg=args.fov,
            )
        env_snap = latest_environment()
        control_tl_id = (
            env_snap.controlling_light.track_id
            if env_snap.controlling_light is not None else None
        )
        camera_overlay = None
        overlay_dirty = False
        overlay_started = time.perf_counter()
        if view_mode != "world":
            # Split pane: lite overlay, native GL stretch. Slightly leaner widths
            # cut resize+upload cost while keeping the video pane live every frame.
            split_cam = view_mode == "split"
            overlay_width = min(frame_width, 416 if view_mode == "camera" else 288)
            # Always rebuild on split (live video). Camera-only: every frame when
            # perception moved, else every other frame to save CPU.
            rebuild_overlay = (
                split_cam or
                perception_dirty or
                frame_number % 2 == 0 or
                cached_camera_overlay is None
            )
            if rebuild_overlay or cached_camera_overlay is None:
                # TL colour — rare (not a hitch source).
                if frame_number % (12 if split_cam else 6) == 0:
                    annotate_signal_states(frame, filtered_objects)
                overlay_frame, overlay_objects, overlay_road = _camera_overlay_inputs(
                    frame, filtered_objects, display_road_geometry, overlay_width,
                )
                camera_overlay = draw_camera_view(
                    overlay_frame, overlay_objects, show_lanes, cached_lanes,
                    display_fps, detection_fps, overlay_road,
                    horizon_ratio=args.horizon_ratio,
                    camera_height_m=args.camera_height,
                    fov_deg=args.fov,
                    stage_ms=None,
                    annotate_signals=False,
                    env_summary=None if split_cam else env_snap.summary,
                    controlling_light_id=control_tl_id,
                    lite=split_cam or view_mode == "camera",
                    lead_track_id=frame_lead_id,
                )
                cached_camera_overlay = camera_overlay
                overlay_dirty = True
            else:
                camera_overlay = cached_camera_overlay
        overlay_elapsed = time.perf_counter() - overlay_started
        render_started = time.perf_counter()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if view_mode == "world":
            render_world(
                filtered_objects, (0, 0, target_width, target_height),
                display_road_geometry, prefiltered=True, lite=False,
                lead_id=frame_lead_id,
            )
        elif view_mode == "camera":
            assert camera_overlay is not None
            render_camera_texture(
                texture_id, camera_overlay, (0, 0, target_width, target_height),
                upload=overlay_dirty,
            )
        else:
            assert camera_overlay is not None
            left_width = target_width // 2
            # Split: lite 3D (no free-space chips / paint / most badges) + cached cam.
            render_world(
                filtered_objects, (0, 0, left_width, target_height),
                display_road_geometry, prefiltered=True, lite=True,
                lead_id=frame_lead_id,
            )
            render_camera_texture(
                texture_id, camera_overlay,
                (left_width, 0, target_width - left_width, target_height),
                upload=overlay_dirty,
            )
        # Stage HUD sparsely (GLUT strings are pure main-thread latency).
        if (not split_mode and frame_number % 3 == 0) or (split_mode and frame_number % 8 == 0):
            _draw_stage_hud((0, 0, target_width, target_height), stage_ms, display_fps, detection_fps)
        render_elapsed = time.perf_counter() - render_started
        flip_started = time.perf_counter()
        pygame.display.flip()
        flip_elapsed = time.perf_counter() - flip_started
        # Periodic housekeeping: prune long-lived maps / soft GC so FPS does not
        # cliff after ~25–40s (ByteTrack removed_stracks + dict + gen1 GC).
        # Stagger work across ticks so one frame never pays the full bill.
        if now - _last_maintenance >= 1.5:
            _last_maintenance = now
            phase = _maint_phase % 3
            _maint_phase += 1
            if phase == 0:
                object_perception.maintenance()
            elif phase == 1:
                prune_environment_caches()
                if len(_VEHICLE_PRESENCE) > 20:
                    ordered = sorted(
                        _VEHICLE_PRESENCE.items(),
                        key=lambda item: item[1][3],  # last_up
                    )
                    for tid, _mem in ordered[: max(0, len(_VEHICLE_PRESENCE) - 14)]:
                        _VEHICLE_PRESENCE.pop(tid, None)
                # Orphan pose keys that never reappear (belt-and-braces).
                if len(WORLD_POSES) > 24:
                    now_p = time.perf_counter()
                    for key, value in list(WORLD_POSES.items()):
                        if now_p - value[3] > 2.5:
                            WORLD_POSES.pop(key, None)
                            WORLD_TRAJECTORY.pop(key, None)
                            WORLD_ONCOMING_LOCK.pop(key, None)
                            WORLD_TURN_HITS.pop(key, None)
                            WORLD_CRUISE_YAW.pop(key, None)
                            DISPLAY_SLOT_GRACE.pop(key, None)
            else:
                # gen0 often; gen1 every ~8s. Avoid gen2 on the render thread —
                # a full collection is a multi-frame hitch that looks like the
                # "sudden stutter kick" users report around 25–30s.
                try:
                    import gc
                    if now - _last_gc1 >= 8.0:
                        gc.collect(1)
                        _last_gc1 = now
                    else:
                        gc.collect(0)
                except Exception:
                    pass
        # Live EMA of stage timings (ms) for the HUD — cheap and always updated.
        def _ema_ms(key: str, sample_ms: float, alpha: float = 0.18) -> None:
            sample = max(0.0, float(sample_ms))
            prev = stage_ms.get(key, 0.0)
            stage_ms[key] = sample if prev <= 1e-6 else prev * (1.0 - alpha) + sample * alpha
        _ema_ms("src", source_wait * 1000.0)
        _ema_ms("ovl", overlay_elapsed * 1000.0)
        _ema_ms("gl", render_elapsed * 1000.0)
        _ema_ms("flip", flip_elapsed * 1000.0)
        # road_inference_ms is already a smoothed ms estimate from the worker.
        if road_inference_ms > 0.0:
            _ema_ms("road", road_inference_ms, alpha=0.12)
        if depth_inference_ms > 0.0:
            _ema_ms("depth", depth_inference_ms, alpha=0.12)
        if ufld_inference_ms > 0.0:
            _ema_ms("ufld", ufld_inference_ms, alpha=0.12)
        if (video_mode and args.realtime_video and video_reader is None and
                detection_passes > 0):
            # Start realtime decode as soon as object net is alive — do not wait
            # for road/UFLD warmup (that delayed live video by many seconds).
            video_reader = LatestVideoFrameReader(cap, source_fps, realtime=True)
            video_reader.start()
            initial_video_frame = None
        ready_for_measurement = detection_passes > 0 and (
            road_perception is None or road_updates > 0 or not show_lanes
        )
        # Start the FPS clock only after realtime decode is running so the
        # freeze-on-first-frame warmup does not inflate / distort metrics.
        realtime_ready = (
            not video_mode or not args.realtime_video or video_reader is not None
        )
        if test_started is None and ready_for_measurement and realtime_ready:
            test_started = time.perf_counter()
            test_frame_start = frame_number
        elapsed_test = 0.0 if test_started is None else time.perf_counter() - test_started
        frame_limit_reached = bool(args.test_frames and frame_number >= args.test_frames)
        time_limit_reached = bool(test_started is not None and args.test_seconds and elapsed_test >= args.test_seconds)
        if frame_limit_reached or time_limit_reached:
            if args.test_screenshot:
                filename = "youtube-test-preview.png" if video_mode else "three-d-renderer-preview.png"
                _save_frame(PROJECT_ROOT / "logs" / filename, target_width, target_height)
            print(
                f"Self-test: display={display_fps:.1f} FPS, inference={detection_fps:.1f} FPS, "
                f"frames={frame_number}, detection passes={detection_passes}"
            )
            running = False
        tick_started = time.perf_counter()
        # Cap at 60; do not force a low ceiling that fights the 25+ FPS goal.
        clock.tick(60)
        tick_elapsed = time.perf_counter() - tick_started
        stage_ms["tick"] = (
            tick_elapsed * 1000.0 if stage_ms.get("tick", 0.0) <= 1e-6
            else stage_ms["tick"] * 0.82 + tick_elapsed * 1000.0 * 0.18
        )
        if measure_stages:
            timing_samples += 1
            source_wait_total += source_wait
            overlay_total += overlay_elapsed
            render_total += render_elapsed
            flip_total += flip_elapsed
            tick_total += tick_elapsed
    test_elapsed = 0.0 if test_started is None else time.perf_counter() - test_started
    if video_reader is not None:
        video_reader.stop()
        final_position_seconds = video_reader.position_seconds
    else:
        final_position_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if video_mode else None
    if road_perception is not None:
        road_perception.close()
    if depth_perception is not None:
        depth_perception.close()
    if ufld_perception is not None:
        ufld_perception.close()
    object_perception.close()
    report = {
        "source": source_metadata,
        "requested_start_seconds": (
            source_metadata.get("requested_start_seconds", args.start_seconds)
            if video_mode else None
        ),
        "final_source_position_seconds": final_position_seconds,
        "elapsed_processing_seconds": round(test_elapsed, 3),
        "rendered_frames": frame_number,
        "measured_rendered_frames": max(0, frame_number - test_frame_start),
        "measured_display_fps": round(max(0, frame_number - test_frame_start) / max(test_elapsed, 1e-3), 3),
        "detection_passes": detection_passes,
        "object_observations": object_observations,
        "class_observations": dict(class_counts.most_common()),
        "road_topology_observations": dict(topology_counts.most_common()),
        "road_model": "yolopv2-openvino" if road_perception is not None else "classical-fallback",
        "road_updates": road_updates,
        "smoothed_road_inference_ms": round(road_inference_ms, 3),
        "depth_model": "depth-anything-v2-small-openvino" if depth_perception is not None else "disabled",
        "depth_updates": depth_updates,
        "smoothed_depth_inference_ms": round(depth_inference_ms, 3),
        "depth_scale": round(depth_scale, 4),
        "ufld_model": "ufld-tusimple-resnet18-openvino" if ufld_perception is not None else "disabled",
        "ufld_updates": ufld_updates,
        "smoothed_ufld_inference_ms": round(ufld_inference_ms, 3),
        "smoothed_display_fps": round(display_fps, 3),
        "smoothed_inference_fps": round(detection_fps, 3),
        "stage_timing_ms": {
            "source_wait": round(source_wait_total * 1000.0 / max(1, timing_samples), 3),
            "camera_overlay": round(overlay_total * 1000.0 / max(1, timing_samples), 3),
            "opengl_render": round(render_total * 1000.0 / max(1, timing_samples), 3),
            "buffer_swap": round(flip_total * 1000.0 / max(1, timing_samples), 3),
            "frame_limiter": round(tick_total * 1000.0 / max(1, timing_samples), 3),
            "depth_inference": round(depth_inference_ms, 3),
            "ufld_inference": round(ufld_inference_ms, 3),
        },
        "realtime_frame_skipping": bool(video_mode and args.realtime_video),
    }
    if args.test_report:
        report_path = Path(args.test_report)
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Test report written to {report_path}")
    if detection_passes:
        print(f"Observed classes: {dict(class_counts.most_common())}")
        print(f"Road topology: {dict(topology_counts.most_common())}")
    cap.release()
    delete_mesh_cache()
    glDeleteTextures([texture_id])
    pygame.quit()
    return 0


if __name__ == "__main__":
    import traceback
    from pathlib import Path as _Path

    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        # Native GPU aborts still cannot be caught, but pure-Python failures
        # land here so the batch window is not an unexplained close.
        crash_path = PROJECT_ROOT / "logs" / "crash.log"
        crash_path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        crash_path.write_text(text, encoding="utf-8")
        print(f"\nVisionFSD crashed: {exc}")
        print(f"Full traceback written to {crash_path}")
        raise SystemExit(1) from exc
