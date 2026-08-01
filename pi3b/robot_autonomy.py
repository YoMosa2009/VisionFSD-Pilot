#!/usr/bin/env python3
"""Conservative Pi 3B robot runtime for LD19, camera, Uno and optional IMU.

The Pi is the high-level planner.  The Uno is the real-time motor and
front-ultrasonic safety controller.  A missing serial link, stale LiDAR data,
or a close obstacle therefore always results in STOP rather than a guessed
movement command.

This is deliberately a low-speed indoor demonstrator.  A supported IMU improves
short-term yaw prediction, but the chassis still has no wheel encoders or
absolute heading sensor.  Its local map remains approximate and is not a claim
of metric SLAM.
"""

from __future__ import annotations

import argparse
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import serial
from serial.tools import list_ports

from lidar_visualizer import LD19Parser, LivePolarMap
from robot_explorer import ExplorationState, FrontierExplorer
from robot_imu import AutoIMULink, IMUState
from robot_slam_lite import LidarSlamLite, SlamLiteState
from visionfsd_pi import (
    LatestCamera,
)


WINDOW_TITLE = "VisionFSD Pi Robot - standby"
UNO_BAUD = 115200
UNO_HEARTBEAT_S = 0.09
UNO_CONTROL_LEASE_S = 0.25
# The L298N bridge on this shield drops roughly 2 V, so a 7.9 V pack puts at
# most about 5.6 V across a motor at full duty.  Capping PWM at 105 meant 41%
# of that, near 2.3 V: enough to spin a free wheel on blocks, not enough to
# move the loaded chassis on a floor, where the motor just sits buzzing.
MAX_PWM = 255
# Lowest PWM that reliably turns a *loaded* wheel.  A deadband measured with
# the wheels in the air reads far lower than the real one.
# This is the lowest direct PWM measured to turn a loaded wheel.  Do not use
# on/off torque pulses below it: a small robot has too little inertia to make
# that feel smooth, and the pulses are audible as stop-start motion.
MIN_MOVE_PWM = 105
DEFAULT_CRUISE_PWM = 118
# Largest direct-PWM rise per planner decision after the initial non-stalling
# floor.  The Uno applies its own 20 ms output ramp as the final authority.
MAX_PWM_STEP = 4
# Straight cruise stays low, but a useful forward arc needs more than the old
# eight-PWM split.  Bounded outer-wheel headroom supplies yaw while the inside
# wheel remains at the loaded floor; neither wheel counter-rotates.
MAX_GENTLE_HEADING_DEG = 40.0
MAX_TURN_SPLIT_PWM = 28
MAX_STEERING_STEP_DEG = 2.5
CORRIDOR_STEERING_GAIN = 1.6
ESCAPE_TURN_MARGIN_PWM = 24
ESCAPE_REVERSE_SECONDS = 0.70
ESCAPE_REAR_CLEARANCE_M = 0.24
ESCAPE_REVERSE_SEARCH_SECONDS = 0.45
ESCAPE_FRONT_RELEASE_M = 0.68
ESCAPE_COMMIT_SECONDS = 1.00
ESCAPE_MIN_TURN_DEG = 28.0
ESCAPE_TARGET_TURN_DEG = 58.0
ESCAPE_MAX_TURN_DEG = 88.0
ESCAPE_TURN_TIMEOUT_S = 2.40
ESCAPE_TURN_SIDE_CLEARANCE_M = 0.18
ESCAPE_DIRECT_TURN_CLEARANCE_M = 0.34
ESCAPE_SIDE_HARD_CLEARANCE_M = 0.16
FORWARD_PREFERENCE_CLEARANCE_M = 1.10
CLOSE_LIDAR_M = 0.28
CLOSE_ULTRASONIC_CM = 22.0
MIN_NAV_CORRIDOR_M = 0.30
DISPLAY_PERIOD_S = 0.10
TELEMETRY_PERIOD_S = 1.0
CAPS_RETRY_S = 0.50
CAMERA_RETRY_S = 1.0
CAMERA_START_TIMEOUT_S = 2.0
CAMERA_AUTO_INDEX_LIMIT = 8
CAMERA_CAPTURE_WIDTH = 320
CAMERA_CAPTURE_HEIGHT = 240
CAMERA_CAPTURE_FPS = 15
IMU_SOFT_YAW_RATE_DPS = 38.0
IMU_HARD_YAW_RATE_DPS = 55.0

# Measured chassis, in metres.  The planner needs its own width because a
# rectangle fits through a gap that a point always would: this is what lets it
# steer around an object rather than treat one sector as blocked.
ROBOT_WIDTH_M = 0.14
ROBOT_LENGTH_M = 0.15
SAFETY_MARGIN_M = 0.06
CORRIDOR_HALF_WIDTH_M = ROBOT_WIDTH_M / 2.0 + SAFETY_MARGIN_M
FRONT_OVERHANG_M = ROBOT_LENGTH_M / 2.0
PLANNING_HORIZON_M = 3.0
# Candidate headings for the corridor sweep, symmetric so straight ahead is
# itself an option rather than falling between two near-tied neighbours.
STEER_HEADINGS = np.arange(-72.0, 72.1, 1.0, dtype=np.float32)
_CLEARANCE_HEADINGS = np.concatenate((STEER_HEADINGS, np.array([180.0], dtype=np.float32)))
LD19_BAUD = 230400


@dataclass(frozen=True)
class ArduinoStatus:
    front_cm: float | None
    motion: str
    received_at: float
    blocked: bool = False
    left_pwm: int = 0
    right_pwm: int = 0


@dataclass(frozen=True)
class SectorClearance:
    front_m: float | None
    left_m: float | None
    right_m: float | None
    fresh: bool
    # These narrower sectors let the planner steer before an obstacle reaches
    # the centre of the robot.  The wider left/right values remain useful when
    # choosing a close-range escape direction.
    front_left_m: float | None = None
    front_right_m: float | None = None
    # Body-inflated clear travel for each heading in STEER_HEADINGS, metres
    # ahead of the front bumper.  None when no scan was available.
    profile: np.ndarray | None = None
    # Rear range is used only before a short reverse recovery.  The static Uno
    # ultrasonic sensor faces forward, so reverse motion must be LD19-gated.
    rear_m: float | None = None
    scan_at: float | None = None

    def limit_at(self, heading_deg: float) -> float | None:
        if self.profile is None:
            return None
        index = int(np.argmin(np.abs(STEER_HEADINGS - heading_deg)))
        return float(self.profile[index])


@dataclass(frozen=True)
class CameraMotionState:
    fresh: bool = False
    confidence: float = 0.0
    tracked_features: int = 0
    yaw_rate_dps: float | None = None
    translation_scale: float = 1.0
    motion_observed: bool = False
    captured_at: float = 0.0


def _signed_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _sector_clearance(points: list[tuple[int, object]], centre_deg: float, half_width_deg: float) -> float | None:
    """Return a conservative range only when adjacent LD19 returns agree.

    A raw nearest-point rule reacts to one bad LiDAR return.  Requiring a
    small angular cluster preserves small real obstacles while suppressing a
    lone speckle, which otherwise makes a lightweight robot twitch or spin.
    """
    samples = sorted(
        (
            float(point.angle_deg),
            point.distance_mm / 1000.0,
            int(getattr(point, "confidence", 0)),
        )
        for _index, point in points
        if abs(_signed_angle(point.angle_deg - centre_deg)) <= half_width_deg
        # Keep the useful indoor portion of the LD19's range.  The live map
        # itself rejects anything beyond 6 m; the old 4.5 m cut-off made an
        # open room look like an unseen path and caused unnecessary stops.
        and 0.08 <= point.distance_mm / 1000.0 <= 5.8
    )
    clustered: list[float] = []
    for angle, distance, confidence in samples:
        neighbours = [
            other_distance
            for other_angle, other_distance, _other_confidence in samples
            if abs(_signed_angle(other_angle - angle)) <= 3.0
            and abs(other_distance - distance) <= max(0.14, distance * 0.22)
        ]
        # A thin chair leg can occupy one angular bin.  Preserve a strong close
        # return even without a neighbour; farther isolated returns remain
        # rejected so a distant speckle cannot steer the chassis.
        if len(neighbours) >= 2 or (distance <= 0.75 and confidence >= 180):
            clustered.append(float(np.median(neighbours)))
    return min(clustered) if clustered else None


def corridor_profile(
    points: list[tuple[int, object]],
    candidate_headings: np.ndarray = STEER_HEADINGS,
) -> np.ndarray:
    """Clear travel ahead of the bumper for every candidate heading.

    For each heading the robot is swept forward as a rectangle of its own
    width, and any return entering that corridor limits how far it can go.
    A five-sector summary cannot express "there is a 40 cm gap 12 degrees to
    the left", which is exactly the information needed to drive around an
    object instead of stopping in front of it or swerving past the whole side.
    """
    if not points:
        return np.zeros(candidate_headings.size, dtype=np.float32)
    # Build robust one-degree bins first.  The old corridor sweep accepted every
    # isolated return, so one LD19 speckle could make a clear path disappear for
    # one control cycle and jerk the wheel command.  A physical obstacle normally
    # occupies adjacent angular bins; require that local support here just as the
    # fixed-sector clearance calculation does.
    raw_angles = np.fromiter(
        (float(point.angle_deg) for _index, point in points),
        dtype=np.float32,
        count=len(points),
    )
    raw_ranges = np.fromiter(
        (float(point.distance_mm) / 1000.0 for _index, point in points),
        dtype=np.float32,
        count=len(points),
    )
    raw_confidences = np.fromiter(
        (float(getattr(point, "confidence", 0)) for _index, point in points),
        dtype=np.float32,
        count=len(points),
    )
    valid = (raw_ranges >= 0.08) & (raw_ranges <= 5.8)
    binned = np.full(360, np.inf, dtype=np.float32)
    if np.any(valid):
        bins = np.rint(raw_angles[valid]).astype(np.int16) % 360
        np.minimum.at(binned, bins, raw_ranges[valid])
    binned[~np.isfinite(binned)] = np.nan
    finite = np.isfinite(binned)
    tolerance = np.maximum(0.14, binned * 0.22)
    supported = np.zeros(360, dtype=bool)
    for offset in (-3, -2, -1, 1, 2, 3):
        neighbour = np.roll(binned, -offset)
        supported |= finite & np.isfinite(neighbour) & (np.abs(neighbour - binned) <= tolerance)
    # Do not erase a high-confidence close return merely because the object is
    # narrower than the LD19's adjacent angular samples.
    close_confirmed = valid & (raw_ranges <= 0.75) & (raw_confidences >= 180)
    if np.any(close_confirmed):
        close_bins = np.rint(raw_angles[close_confirmed]).astype(np.int16) % 360
        supported[close_bins] = True
    usable_indices = np.flatnonzero(supported)
    if usable_indices.size == 0:
        return np.full(candidate_headings.size, PLANNING_HORIZON_M, dtype=np.float32)
    angles = usable_indices.astype(np.float32)
    ranges = binned[usable_indices]

    delta = np.radians(
        ((angles[None, :] - candidate_headings[:, None]) + 180.0) % 360.0 - 180.0
    )
    cos, sin = np.cos(delta), np.sin(delta)
    lateral = sin * ranges[None, :]
    along = cos * ranges[None, :]
    # cos > 0 keeps only returns actually ahead of the candidate heading;
    # without it, obstacles behind the robot produce negative travel limits.
    inside = (cos > 0.02) & (np.abs(lateral) <= CORRIDOR_HALF_WIDTH_M)
    limits = np.where(inside, along, np.inf).min(axis=1) - FRONT_OVERHANG_M
    return np.clip(limits, 0.0, PLANNING_HORIZON_M).astype(np.float32)


def discover_arduino_port() -> str | None:
    """Pick an Uno-compatible USB serial device without choosing the LD19."""
    matches: list[str] = []
    for item in list_ports.comports():
        text = f"{item.device} {item.description} {item.manufacturer or ''}".upper()
        if any(name in text for name in ("ARDUINO", "UNO", "CH340", "CH341", "ACM")):
            matches.append(item.device)
    return matches[0] if len(matches) == 1 else None


def discover_ld19_port(exclude: str | None = None) -> str | None:
    matches: list[str] = []
    for item in list_ports.comports():
        if item.device == exclude:
            continue
        text = f"{item.description} {item.manufacturer or ''}".upper()
        if any(name in text for name in ("CP210", "SILICON LABS", "USB SERIAL", "UART", "FTDI")):
            matches.append(item.device)
    return matches[0] if len(matches) == 1 else None


class ArduinoLink:
    """Bounded serial link.  The Uno independently times out motion commands."""

    def __init__(self, port: str) -> None:
        self._serial = serial.Serial(port, UNO_BAUD, timeout=0.05, write_timeout=0.2)
        # Opening an Uno serial port resets it; wait for the sketch banner.
        time.sleep(2.1)
        self._lines: queue.Queue[str] = queue.Queue()
        self._running = True
        self._write_lock = threading.Lock()
        self._drive_lock = threading.Lock()
        self._drive_command = "STOP"
        self._drive_lease_until = 0.0
        self._drive_last_write = 0.0
        self._drive_expired = True
        self._status = ArduinoStatus(None, "S", 0.0)
        self._supports_differential = False
        self._last_caps_sent_at = float("-inf")
        self._reader = threading.Thread(target=self._read_loop, name="uno-status", daemon=True)
        self._reader.start()
        self.send("STOP")
        self.poll_capabilities(time.monotonic())
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="uno-drive-heartbeat", daemon=True
        )
        self._heartbeat.start()

    @staticmethod
    def _parse_status(line: str, received_at: float) -> ArduinoStatus:
        fields = dict(part.split("=", 1) for part in line[7:].split() if "=" in part)
        try:
            front = float(fields["front_cm"]) if fields.get("front_cm") not in (None, "NO_ECHO") else None
        except ValueError:
            front = None
        try:
            left_pwm = int(fields.get("left_pwm", "0"))
        except ValueError:
            left_pwm = 0
        try:
            right_pwm = int(fields.get("right_pwm", "0"))
        except ValueError:
            right_pwm = 0
        return ArduinoStatus(
            front,
            fields.get("motion", "S"),
            received_at,
            fields.get("blocked", "0") == "1",
            left_pwm,
            right_pwm,
        )

    def _read_loop(self) -> None:
        while self._running:
            try:
                line = self._serial.readline().decode("ascii", "replace").strip()
            except (serial.SerialException, OSError, TypeError):
                break
            if not line:
                continue
            if line.startswith("STATUS "):
                self._status = self._parse_status(line, time.monotonic())
            elif line == "CAPS DRIVE":
                self._supports_differential = True
            self._lines.put(line)

    def _write(self, command: str) -> None:
        with self._write_lock:
            if self._running and self._serial.is_open:
                self._serial.write((command + "\n").encode("ascii"))

    def send(self, command: str) -> None:
        self._write(command)

    def publish_drive(self, left_pwm: int, right_pwm: int) -> None:
        """Publish a leased command and refresh it independently of planning.

        The short lease preserves fail-safe STOP behavior if the main loop
        hangs.  While the loop remains healthy, this thread prevents display,
        camera, or route-planning jitter from tripping the Uno dead-man timer.
        """
        now = time.monotonic()
        command = f"DRIVE {left_pwm} {right_pwm}"
        with self._drive_lock:
            changed = command != self._drive_command
            self._drive_command = command
            self._drive_lease_until = now + UNO_CONTROL_LEASE_S
            self._drive_expired = False
            due = changed or now - self._drive_last_write >= UNO_HEARTBEAT_S
            if due:
                self._drive_last_write = now
        if due:
            self._write(command)

    def _heartbeat_loop(self) -> None:
        while self._running:
            command = self._heartbeat_command(time.monotonic())
            if command is not None:
                self._write(command)
            time.sleep(UNO_HEARTBEAT_S / 3.0)

    def _heartbeat_command(self, now: float) -> str | None:
        command: str | None = None
        with self._drive_lock:
            if now <= self._drive_lease_until:
                if now - self._drive_last_write >= UNO_HEARTBEAT_S:
                    command = self._drive_command
                    self._drive_last_write = now
            elif not self._drive_expired:
                command = "STOP"
                self._drive_expired = True
                self._drive_last_write = now
        return command

    def status(self) -> ArduinoStatus:
        return self._status

    def poll_capabilities(self, now: float) -> None:
        """Retry the Uno capability handshake until differential drive is confirmed."""
        if not self._supports_differential and now - self._last_caps_sent_at >= CAPS_RETRY_S:
            self.send("CAPS")
            self._last_caps_sent_at = now

    @property
    def differential_ready(self) -> bool:
        return self._supports_differential

    def close(self) -> None:
        self._running = False
        try:
            with self._write_lock:
                if self._serial.is_open:
                    self._serial.write(b"STOP\n")
        except (serial.SerialException, OSError, TypeError):
            pass
        self._reader.join(timeout=0.5)
        self._heartbeat.join(timeout=0.5)
        try:
            self._serial.close()
        except (serial.SerialException, OSError, TypeError):
            pass


class LD19Link:
    """Newest-only LD19 reader; no motor/configuration packets are ever sent."""

    def __init__(self, port: str, front_offset_deg: float) -> None:
        self._serial = serial.Serial(port, LD19_BAUD, timeout=0.02)
        self._parser = LD19Parser()
        self._map = LivePolarMap()
        self._offset = front_offset_deg
        self._lock = threading.Lock()
        self._running = True
        self._last_packet_at = 0.0
        self._clearance_stamp = -1.0
        self._clearance_cache: SectorClearance | None = None
        self._thread = threading.Thread(target=self._read_loop, name="ld19-reader", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._serial.read(max(1, self._serial.in_waiting))
            except (serial.SerialException, OSError, TypeError):
                break
            if not raw:
                continue
            now = time.monotonic()
            points = self._parser.feed(raw, now)
            if not points:
                continue
            # Convert the physical mounting orientation once at input.
            rotated = [
                type(point)((point.angle_deg + self._offset) % 360.0, point.distance_mm,
                            point.confidence, point.captured_at)
                for point in points
            ]
            with self._lock:
                self._map.update(rotated, min_confidence=8, min_range_mm=80, max_range_mm=6000)
                self._last_packet_at = now

    def snapshot(self) -> tuple[list[tuple[int, object]], bool]:
        now = time.monotonic()
        with self._lock:
            # Keep roughly two LD19 revolutions.  Longer point persistence made
            # obstacles remain in a corridor after the chassis had already
            # changed angle, causing repeated stop/reverse decisions.
            points = self._map.fresh(now, 0.22)
            fresh = now - self._last_packet_at <= 0.45
        return points, fresh

    def clearance(self) -> SectorClearance:
        points, fresh = self.snapshot()
        if not fresh:
            return SectorClearance(None, None, None, False)
        scan_at = max((float(point.captured_at) for _index, point in points), default=0.0)
        if self._clearance_cache is not None and scan_at <= self._clearance_stamp:
            return self._clearance_cache
        profiles = corridor_profile(points, _CLEARANCE_HEADINGS)
        rear_evidence = _sector_clearance(points, 180.0, 60.0)
        clearance = SectorClearance(
            _sector_clearance(points, 0.0, 20.0),
            _sector_clearance(points, -75.0, 35.0),
            _sector_clearance(points, 75.0, 35.0),
            True,
            _sector_clearance(points, -35.0, 20.0),
            _sector_clearance(points, 35.0, 20.0),
            profiles[:-1],
            None if rear_evidence is None else float(profiles[-1]),
            scan_at,
        )
        self._clearance_stamp = scan_at
        self._clearance_cache = clearance
        return clearance

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=0.5)
        try:
            self._serial.close()
        except (serial.SerialException, OSError, TypeError):
            pass


class LocalLidarMap:
    """Small display-only local occupancy map with explicitly approximate pose."""

    def __init__(self, cells: int = 100, metres: float = 5.0) -> None:
        self.cells = cells
        self.metres = metres
        self.grid = np.zeros((cells, cells), dtype=np.uint8)
        self.x = metres / 2.0
        self.y = metres / 2.0
        self.heading = 0.0
        self._last_motion_at = time.monotonic()

    def integrate_motion(self, left_pwm: int, right_pwm: int, now: float) -> None:
        elapsed = min(0.20, max(0.0, now - self._last_motion_at))
        self._last_motion_at = now
        # Conservative commanded-motion dead reckoning.  It makes the map
        # follow gradual differential turns, but deliberately never feeds the
        # movement planner: this chassis has no encoders or IMU.
        linear = ((left_pwm + right_pwm) * 0.5 / MAX_PWM) * 0.30
        turn_rate = ((left_pwm - right_pwm) / MAX_PWM) * 140.0
        self.heading = (self.heading + turn_rate * elapsed) % 360.0
        self.x += math.sin(math.radians(self.heading)) * linear * elapsed
        self.y -= math.cos(math.radians(self.heading)) * linear * elapsed

    def integrate_points(self, points: list[tuple[int, object]]) -> None:
        scale = self.cells / self.metres
        for _index, point in points[::3]:
            distance = point.distance_mm / 1000.0
            if not 0.10 <= distance <= self.metres / 1.5:
                continue
            angle = math.radians(point.angle_deg + self.heading)
            x = self.x + math.sin(angle) * distance
            y = self.y - math.cos(angle) * distance
            col, row = int(x * scale), int(y * scale)
            if 0 <= row < self.cells and 0 <= col < self.cells:
                self.grid[row, col] = min(255, int(self.grid[row, col]) + 32)
        self.grid = (self.grid.astype(np.float32) * 0.992).astype(np.uint8)

    def render(self, size: int = 500) -> np.ndarray:
        image = cv2.resize(self.grid, (size, size), interpolation=cv2.INTER_NEAREST)
        panel = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        px = int(np.clip(self.x / self.metres * size, 0, size - 1))
        py = int(np.clip(self.y / self.metres * size, 0, size - 1))
        radians = math.radians(self.heading)
        tip = (int(px + math.sin(radians) * 20), int(py - math.cos(radians) * 20))
        cv2.circle(panel, (px, py), 8, (80, 240, 100), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, (px, py), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(panel, "LOCAL LIDAR MAP - POSE APPROXIMATE", (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (235, 245, 250), 1, cv2.LINE_AA)
        return panel


class CameraSafety:
    """Camera health and optical flow support non-IMU pose prediction."""

    def __init__(self, camera: str, fov: float) -> None:
        self._fov = fov
        self._last_camera_sequence = 0
        self._last_frame_at = 0.0
        self.frame: np.ndarray | None = None
        self.camera: LatestCamera | None = None
        self.camera_source = "none"
        self._camera_sources = self._candidate_sources(camera)
        self._next_camera_source = 0
        self._next_camera_retry_at = 0.0
        self._camera_opened_at = 0.0
        self._last_camera_error = ""
        self._flow_gray: np.ndarray | None = None
        self._flow_at = 0.0
        self.motion = CameraMotionState()
        self._open_next_camera(time.monotonic())

    @staticmethod
    def _candidate_sources(requested: str) -> list[str]:
        sources: list[str] = []
        if requested != "auto":
            sources.append(requested)
        if requested == "auto" or requested.isdigit():
            by_id = Path("/dev/v4l/by-id")
            if by_id.is_dir():
                sources.extend(str(path) for path in sorted(by_id.glob("*-video-index0")))
            sources.extend(str(index) for index in range(CAMERA_AUTO_INDEX_LIMIT))
        return list(dict.fromkeys(sources))

    def _open_next_camera(self, now: float) -> None:
        errors: list[str] = []
        for _attempt in range(len(self._camera_sources)):
            source = self._camera_sources[self._next_camera_source % len(self._camera_sources)]
            self._next_camera_source += 1
            try:
                self.camera = LatestCamera(
                    source,
                    CAMERA_CAPTURE_WIDTH,
                    CAMERA_CAPTURE_HEIGHT,
                    CAMERA_CAPTURE_FPS,
                )
            except Exception as exc:
                errors.append(f"{source}: {exc}")
                continue
            self.camera_source = source
            self._camera_opened_at = now
            self._last_camera_error = ""
            print(f"Camera candidate opened: {source}")
            return
        self.camera = None
        self.camera_source = "none"
        self._next_camera_retry_at = now + CAMERA_RETRY_S
        error = "; ".join(errors) if errors else "no camera candidates"
        if error != self._last_camera_error:
            print(f"Camera unavailable; safe STOP while retrying: {error}", file=sys.stderr)
            self._last_camera_error = error

    def _drop_camera(self, now: float, reason: str) -> None:
        if self.camera is not None:
            self.camera.close()
        self.camera = None
        self.camera_source = "none"
        self.frame = None
        self._last_frame_at = 0.0
        self._last_camera_sequence = 0
        self._flow_gray = None
        self._flow_at = 0.0
        self.motion = CameraMotionState()
        self._next_camera_retry_at = now + CAMERA_RETRY_S
        print(f"Camera lost; safe STOP while retrying: {reason}", file=sys.stderr)

    def _update_motion(
        self, frame: np.ndarray, captured: float, left_pwm: int, right_pwm: int
    ) -> None:
        gray = cv2.cvtColor(
            cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        previous = self._flow_gray
        elapsed = captured - self._flow_at
        self._flow_gray = gray
        self._flow_at = captured
        if previous is None or not 0.035 <= elapsed <= 0.35:
            self.motion = CameraMotionState(captured_at=captured)
            return
        features = cv2.goodFeaturesToTrack(
            previous, maxCorners=80, qualityLevel=0.025, minDistance=7, blockSize=7
        )
        if features is None or len(features) < 10:
            self.motion = CameraMotionState(captured_at=captured)
            return
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            features,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )
        if next_points is None or status is None:
            self.motion = CameraMotionState(captured_at=captured)
            return
        valid = status.reshape(-1) == 1
        if errors is not None:
            valid &= errors.reshape(-1) < 24.0
        old = features.reshape(-1, 2)[valid]
        new = next_points.reshape(-1, 2)[valid]
        if old.shape[0] < 8:
            self.motion = CameraMotionState(captured_at=captured)
            return
        flow = new - old
        median = np.median(flow, axis=0)
        deviation = np.median(np.linalg.norm(flow - median, axis=1))
        confidence = float(np.clip((old.shape[0] / 55.0) * (1.0 - deviation / 5.0), 0.0, 1.0))
        flow_magnitude = float(np.median(np.linalg.norm(flow, axis=1)))
        commanded_turn = abs(left_pwm - right_pwm) >= 18
        yaw_rate = (
            float(-median[0] / 160.0 * self._fov / elapsed)
            if commanded_turn and confidence >= 0.25
            else None
        )
        commanded_forward = left_pwm > 0 and right_pwm > 0
        motion_observed = flow_magnitude >= 0.35
        translation_scale = (
            (1.0 if motion_observed else 0.20)
            if commanded_forward and confidence >= 0.35
            else 1.0
        )
        self.motion = CameraMotionState(
            fresh=True,
            confidence=confidence,
            tracked_features=int(old.shape[0]),
            yaw_rate_dps=yaw_rate,
            translation_scale=translation_scale,
            motion_observed=motion_observed,
            captured_at=captured,
        )

    def tick(self, left_pwm: int = 0, right_pwm: int = 0) -> None:
        now = time.monotonic()
        if self.camera is None:
            if now >= self._next_camera_retry_at:
                self._open_next_camera(now)
            return
        sequence, frame, captured = self.camera.latest()
        timed_out = (
            (self.frame is None and now - self._camera_opened_at >= CAMERA_START_TIMEOUT_S)
            or (self._last_frame_at > 0.0 and now - self._last_frame_at >= CAMERA_START_TIMEOUT_S)
        )
        if self.camera.error or timed_out:
            self._drop_camera(now, self.camera.error or "no live frames")
            return
        if frame is not None:
            self.frame = frame
            if sequence > self._last_camera_sequence:
                self._update_motion(frame, captured, left_pwm, right_pwm)
                self._last_camera_sequence = sequence
                self._last_frame_at = captured

    def ready(self, now: float) -> bool:
        """Do not drive blind if the webcam has stopped delivering frames."""
        return (
            self.camera is not None
            and self.frame is not None
            and not self.camera.error
            and now - self._last_frame_at <= 1.0
        )

    def annotated_frame(self) -> np.ndarray | None:
        return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()


class AutonomousPolicy:
    """Sensor-fused, low-speed differential-drive planner.

    The LD19 supplies geometry, the webcam supplies optical-flow pose cues,
    and the Uno's ultrasonic sensor remains the final near-field guard.
    This is reactive local navigation, not a claim of room-scale SLAM.
    """

    def __init__(self, standby_s: float, speed: int, min_move_pwm: int = MIN_MOVE_PWM) -> None:
        self.started_at = time.monotonic()
        self.standby_s = standby_s
        self.speed = max(min_move_pwm, speed)
        self.min_move_pwm = min_move_pwm
        self._heading_index: int | None = None
        self.cruise_pwm = 0
        self.turn_command = "L"
        self._direction_lock_until = 0.0
        self._escape_phase = "IDLE"
        self._escape_phase_until = 0.0
        self._escape_turn_start_yaw_deg: float | None = None
        self._escape_attempt = 0
        self._arc_active = False
        self._guidance_bias = 0.0
        self._steering_deg = 0.0
        self.drive_confidence = 0.0
        self.last_command = "STOP"
        self.left_pwm = 0
        self.right_pwm = 0
        self._last_output = (0, 0)
        self.last_sent_at = 0.0
        self.reason = "BOOT_STANDBY"
        self.imu_yaw_rate_dps: float | None = None
        self.imu_yaw_deg: float | None = None
        self.imu_limited = False
        self.pose_heading_deg: float | None = None
        self.pose_yaw_source = "COMMAND+LD19"
        self._escape_turn_yaw_source = "TIME"
        self.exploration = ExplorationState()

    def observe_imu(self, state: IMUState) -> None:
        ready = state.connected and state.calibrated and state.fresh
        self.imu_yaw_rate_dps = state.gyro_z_dps if ready else None
        self.imu_yaw_deg = state.yaw_deg if ready else None

    def observe_exploration(self, state: ExplorationState) -> None:
        self.exploration = state

    def observe_pose(self, state: SlamLiteState) -> None:
        self.pose_heading_deg = (
            state.heading_deg if state.map_updates >= 2 else None
        )
        self.pose_yaw_source = state.yaw_source

    def _turn_reference_yaw_deg(self) -> tuple[float | None, str]:
        if self.imu_yaw_deg is not None:
            return self.imu_yaw_deg, "IMU"
        if self.pose_heading_deg is not None:
            return self.pose_heading_deg, "LD19+COMMAND"
        return None, "TIME"

    def _yaw_for_source(self, source: str) -> float | None:
        if source == "IMU":
            return self.imu_yaw_deg
        if source == "LD19+COMMAND":
            return self.pose_heading_deg
        return None

    def _limit_turn_split(self, requested_split: int) -> int:
        """Reduce differential steering when measured yaw is already too fast."""
        self.imu_limited = False
        if requested_split <= 0 or self.imu_yaw_rate_dps is None:
            return requested_split
        yaw_rate = abs(self.imu_yaw_rate_dps)
        if yaw_rate <= IMU_SOFT_YAW_RATE_DPS:
            return requested_split
        scale = float(np.clip(
            (IMU_HARD_YAW_RATE_DPS - yaw_rate)
            / (IMU_HARD_YAW_RATE_DPS - IMU_SOFT_YAW_RATE_DPS),
            0.0,
            1.0,
        ))
        self.imu_limited = True
        return int(round(requested_split * scale))

    @staticmethod
    def _side_score(primary: float | None, outer: float | None) -> float | None:
        values = [value for value in (primary, outer) if value is not None]
        return min(values) if values else None

    def _ramp(self, current: int, target: int) -> int:
        """Use a direct, non-buzzing motor band with a controlled rise.

        A direct brushed-motor command below MIN_MOVE_PWM only buzzes under the
        chassis.  Starting at that floor, then rising in small steps, is smooth
        without the 50 Hz on/off gating that made the prior experiment pulse.
        Slowing and stopping remain immediate for safety.
        """
        if target == 0:
            return 0
        direction = 1 if target > 0 else -1
        target = direction * max(self.min_move_pwm, abs(target))
        if current == 0:
            return direction * self.min_move_pwm
        if (current > 0) != (target > 0):
            # Brake to zero before reversing; the following decision begins
            # the other direction at the non-stalling floor.
            return 0
        if abs(target) < abs(current):
            magnitude = max(abs(target), abs(current) - MAX_PWM_STEP)
            return direction * magnitude
        return direction * min(abs(target), abs(current) + MAX_PWM_STEP)

    def _set_output(self, label: str, left_pwm: int, right_pwm: int) -> str:
        self.left_pwm = self._ramp(self.left_pwm, int(np.clip(left_pwm, -MAX_PWM, MAX_PWM)))
        self.right_pwm = self._ramp(self.right_pwm, int(np.clip(right_pwm, -MAX_PWM, MAX_PWM)))
        return label

    def _raw_turn_side_score(
        self, lidar: SectorClearance, direction: str
    ) -> float | None:
        return (
            self._side_score(lidar.front_left_m, lidar.left_m)
            if direction == "L"
            else self._side_score(lidar.front_right_m, lidar.right_m)
        )

    def _turn_side_score(self, lidar: SectorClearance, direction: str) -> float | None:
        """Use the full corridor sweep without ignoring a close side return."""
        sector_score = self._raw_turn_side_score(lidar, direction)
        if lidar.profile is None:
            return sector_score
        mask = STEER_HEADINGS <= -20.0 if direction == "L" else STEER_HEADINGS >= 20.0
        corridor_score = float(np.max(lidar.profile[mask]))
        if not np.isfinite(corridor_score):
            return sector_score
        if (
            sector_score is not None
            and sector_score < ESCAPE_TURN_SIDE_CLEARANCE_M
        ):
            return sector_score
        return corridor_score

    def _choose_turn(self, lidar: SectorClearance, now: float) -> tuple[str, float] | None:
        left = self._turn_side_score(lidar, "L")
        right = self._turn_side_score(lidar, "R")
        if left is None and right is None:
            return None
        scores = {"L": left, "R": right}
        locked = scores.get(self.turn_command)
        exploration_turn: str | None = None
        if self.exploration.active and abs(self.exploration.heading_error_deg) >= 12.0:
            exploration_turn = (
                "L" if self.exploration.heading_error_deg < 0.0 else "R"
            )
        if (
            exploration_turn is not None
            and scores.get(exploration_turn) is not None
            and left is not None
            and right is not None
            and abs(left - right) < 0.30
        ):
            candidate = exploration_turn
        else:
            candidate = (
                "L"
                if right is None or (left is not None and left >= right)
                else "R"
            )
        candidate_score = scores[candidate] or 0.0
        # Keep an already-selected escape/arc side unless it became unsafe or
        # the other side is materially clearer.  This stops scan-to-scan
        # flicker from making the robot weave left/right.
        if (now < self._direction_lock_until and locked is not None and locked >= 0.30
                and candidate_score < locked + 0.24):
            return self.turn_command, locked
        self.turn_command = candidate
        self._direction_lock_until = now + 1.20
        return candidate, candidate_score

    def _reverse_arc(self, direction: str) -> str:
        """Back away in a shallow curve, with both tracks driven.

        A one-wheel reverse is still a pivot on this short wheelbase.  Keeping
        both wheels in their loaded movement band gives the LiDAR time to gain
        a little front clearance without a spin or a brake/reverse pulse.
        """
        turn_margin = self._limit_turn_split(ESCAPE_TURN_MARGIN_PWM)
        outer = min(MAX_PWM, self.min_move_pwm + turn_margin)
        inner = self.min_move_pwm
        if direction == "L":
            return self._set_output("L", -outer, -inner)
        return self._set_output("R", -inner, -outer)

    def _reverse_straight(self) -> str:
        return self._set_output("B", -self.min_move_pwm, -self.min_move_pwm)

    def _pivot_crawl(self, direction: str) -> str:
        """Rotate slowly around one stopped wheel, never as an endless pivot."""
        if direction == "L":
            return self._set_output("L", -self.min_move_pwm, 0)
        return self._set_output("R", 0, -self.min_move_pwm)

    @staticmethod
    def _yaw_delta_deg(current: float, start: float) -> float:
        return (current - start + 180.0) % 360.0 - 180.0

    def _reset_escape(self) -> None:
        self._escape_phase = "IDLE"
        self._escape_phase_until = 0.0
        self._escape_turn_start_yaw_deg = None
        self._escape_turn_yaw_source = "TIME"
        self._escape_attempt = 0

    def _mark_escape_blocked(self, _lidar: SectorClearance, now: float) -> str:
        self._escape_phase = "BLOCKED"
        self._escape_phase_until = now + 0.50
        self._escape_turn_start_yaw_deg = None
        return self._hold_stop("STOP:BOXED_IN")

    def _can_turn_without_reverse(
        self,
        lidar: SectorClearance,
        direction: str,
        score: float | None = None,
    ) -> bool:
        if score is None:
            score = self._turn_side_score(lidar, direction)
        raw_score = self._raw_turn_side_score(lidar, direction)
        return (
            score is not None
            and score >= ESCAPE_DIRECT_TURN_CLEARANCE_M
            and (raw_score is None or raw_score >= ESCAPE_SIDE_HARD_CLEARANCE_M)
        )

    def _start_escape(self, lidar: SectorClearance, now: float, source: str) -> str:
        choice = self._choose_turn(lidar, now)
        if choice is None or choice[1] < MIN_NAV_CORRIDOR_M:
            if lidar.rear_m is not None and lidar.rear_m >= ESCAPE_REAR_CLEARANCE_M:
                self._escape_phase = "REVERSE_SEARCH"
                self._escape_phase_until = now + ESCAPE_REVERSE_SEARCH_SECONDS
                self._escape_attempt = 1
                self.reason = f"ESCAPE_REVERSE_SEARCH_{source}"
                self.drive_confidence = 0.20
                return self._reverse_straight()
            return self._mark_escape_blocked(lidar, now)
        if self._can_turn_without_reverse(lidar, choice[0], choice[1]):
            self.turn_command = choice[0]
            self._escape_attempt = 1
            self._start_escape_turn(now)
            self.reason = f"ESCAPE_DIRECT_TURN_{source}:{self.turn_command}"
            self.drive_confidence = 0.30
            return self._pivot_crawl(self.turn_command)
        if lidar.rear_m is None or lidar.rear_m < ESCAPE_REAR_CLEARANCE_M:
            return self._hold_stop("STOP:ESCAPE_REAR_BLOCKED")
        self.turn_command = choice[0]
        self._escape_phase = "REVERSE"
        self._escape_phase_until = now + ESCAPE_REVERSE_SECONDS
        self._escape_turn_start_yaw_deg = None
        self._escape_attempt = 1
        self.reason = f"ESCAPE_REVERSE_{source}:{self.turn_command}"
        self.drive_confidence = 0.35
        return self._reverse_arc(self.turn_command)

    def _start_escape_turn(self, now: float) -> None:
        self._escape_phase = "TURN"
        self._escape_phase_until = now + ESCAPE_TURN_TIMEOUT_S
        (
            self._escape_turn_start_yaw_deg,
            self._escape_turn_yaw_source,
        ) = self._turn_reference_yaw_deg()

    def _start_escape_commit(self, now: float) -> None:
        self._escape_phase = "COMMIT"
        self._escape_phase_until = now + ESCAPE_COMMIT_SECONDS
        self._direction_lock_until = self._escape_phase_until

    def _turn_progress_deg(self) -> float | None:
        current_yaw = self._yaw_for_source(self._escape_turn_yaw_source)
        if (
            current_yaw is None
            or self._escape_turn_start_yaw_deg is None
        ):
            return None
        return abs(self._yaw_delta_deg(
            current_yaw, self._escape_turn_start_yaw_deg
        ))

    def _retry_opposite_escape(self, lidar: SectorClearance, now: float) -> bool:
        if self._escape_attempt >= 2:
            return False
        opposite = "R" if self.turn_command == "L" else "L"
        score = self._turn_side_score(lidar, opposite)
        if score is None or score < MIN_NAV_CORRIDOR_M:
            return False
        if lidar.rear_m is None or lidar.rear_m < ESCAPE_REAR_CLEARANCE_M:
            return False
        self.turn_command = opposite
        self._direction_lock_until = now + 1.20
        self._escape_phase = "REVERSE"
        self._escape_phase_until = now + ESCAPE_REVERSE_SECONDS
        self._escape_turn_start_yaw_deg = None
        self._escape_attempt += 1
        return True

    def _continue_escape(
        self,
        lidar: SectorClearance,
        arduino: ArduinoStatus,
        straight_clearance: float,
        now: float,
    ) -> str | None:
        close_ultrasonic = (
            arduino.front_cm is not None and arduino.front_cm < CLOSE_ULTRASONIC_CM
        )

        if self._escape_phase == "BLOCKED":
            if not close_ultrasonic and straight_clearance >= ESCAPE_FRONT_RELEASE_M:
                self._reset_escape()
                return None
            choice = self._choose_turn(lidar, now)
            raw_score = (
                None
                if choice is None
                else self._raw_turn_side_score(lidar, choice[0])
            )
            geometry_open = (
                choice is not None
                and choice[1] >= MIN_NAV_CORRIDOR_M
                and (raw_score is None or raw_score >= ESCAPE_SIDE_HARD_CLEARANCE_M)
            )
            rear_clear = (
                lidar.rear_m is not None
                and lidar.rear_m >= ESCAPE_REAR_CLEARANCE_M
            )
            if (
                choice is not None
                and now >= self._escape_phase_until
                and geometry_open
            ):
                self.turn_command = choice[0]
                self._escape_attempt = 1
                if self._can_turn_without_reverse(lidar, choice[0], choice[1]):
                    self._start_escape_turn(now)
                    self.reason = f"ESCAPE_OPENING_TURN:{self.turn_command}"
                    self.drive_confidence = 0.25
                    return self._pivot_crawl(self.turn_command)
                if rear_clear:
                    self._escape_phase = "REVERSE"
                    self._escape_phase_until = now + ESCAPE_REVERSE_SECONDS
                    self.reason = f"ESCAPE_GEOMETRY_CHANGED:{self.turn_command}"
                    self.drive_confidence = 0.25
                    return self._reverse_arc(self.turn_command)
            if (
                (choice is None or choice[1] < MIN_NAV_CORRIDOR_M)
                and rear_clear
                and now >= self._escape_phase_until
            ):
                self._escape_phase = "REVERSE_SEARCH"
                self._escape_phase_until = now + ESCAPE_REVERSE_SEARCH_SECONDS
                self.reason = "ESCAPE_REVERSE_SEARCH"
                self.drive_confidence = 0.20
                return self._reverse_straight()
            return self._hold_stop("STOP:BOXED_IN")

        if self._escape_phase == "REVERSE_SEARCH":
            if lidar.rear_m is None or lidar.rear_m < ESCAPE_REAR_CLEARANCE_M:
                return self._hold_stop("STOP:ESCAPE_REAR_BLOCKED")
            if now < self._escape_phase_until:
                self.reason = "ESCAPE_REVERSE_SEARCH"
                self.drive_confidence = 0.20
                return self._reverse_straight()
            choice = self._choose_turn(lidar, now)
            if choice is None or choice[1] < MIN_NAV_CORRIDOR_M:
                return self._mark_escape_blocked(lidar, now)
            self.turn_command = choice[0]
            self._start_escape_turn(now)

        if self._escape_phase == "REVERSE":
            if lidar.rear_m is None or lidar.rear_m < ESCAPE_REAR_CLEARANCE_M:
                return self._hold_stop("STOP:ESCAPE_REAR_BLOCKED")
            if (
                now < self._escape_phase_until
                and straight_clearance < ESCAPE_FRONT_RELEASE_M
            ):
                self.reason = f"ESCAPE_REVERSE:{self.turn_command}"
                self.drive_confidence = 0.35
                return self._reverse_arc(self.turn_command)
            self._start_escape_turn(now)

        if self._escape_phase == "TURN":
            selected_side = self._turn_side_score(lidar, self.turn_command)
            progress = self._turn_progress_deg()
            turn_clear = not close_ultrasonic and straight_clearance >= ESCAPE_FRONT_RELEASE_M
            minimum_turn_complete = progress is None or progress >= ESCAPE_MIN_TURN_DEG
            target_turn_complete = progress is not None and progress >= ESCAPE_TARGET_TURN_DEG
            if turn_clear and minimum_turn_complete:
                self._start_escape_commit(now)
            elif target_turn_complete and not close_ultrasonic and straight_clearance >= CLOSE_LIDAR_M:
                self._start_escape_commit(now)
            elif (
                (progress is not None and progress >= ESCAPE_MAX_TURN_DEG)
                or now >= self._escape_phase_until
                or selected_side is None
                or selected_side < ESCAPE_TURN_SIDE_CLEARANCE_M
            ):
                if self._retry_opposite_escape(lidar, now):
                    self.reason = f"ESCAPE_RETRY:{self.turn_command}"
                    self.drive_confidence = 0.25
                    return self._reverse_arc(self.turn_command)
                return self._mark_escape_blocked(lidar, now)
            else:
                progress_label = (
                    "TIME"
                    if progress is None
                    else f"{self._escape_turn_yaw_source} {progress:.0f}deg"
                )
                self.reason = f"ESCAPE_TURN:{self.turn_command} {progress_label}"
                self.drive_confidence = 0.30
                return self._pivot_crawl(self.turn_command)

        if self._escape_phase == "COMMIT":
            if close_ultrasonic or straight_clearance < CLOSE_LIDAR_M:
                self._start_escape_turn(now)
                self.reason = f"ESCAPE_REPLAN:{self.turn_command}"
                self.drive_confidence = 0.25
                return self._pivot_crawl(self.turn_command)
            if now < self._escape_phase_until:
                heading = (
                    -MAX_GENTLE_HEADING_DEG
                    if self.turn_command == "L"
                    else MAX_GENTLE_HEADING_DEG
                )
                speed = self._cruise_speed(straight_clearance)
                self.reason = f"ESCAPE_COMMIT:{self.turn_command}"
                self.drive_confidence = 0.50
                return self._differential(speed, heading)
            self._reset_escape()

        return None

    def _cruise_speed(self, limit_m: float | None) -> int:
        """Scale speed with how far the body can actually travel.

        Open-loop brushed motors have a narrow usable band: below roughly
        MIN_MOVE_PWM nothing turns under load, and speed rises steeply above it
        because only the voltage *above* the stall threshold does any work.  So
        the floor is set just above the deadband and the ceiling is the
        configured cruise, giving a real several-fold speed range in between.
        """
        floor = min(self.speed, max(self.min_move_pwm, int(self.speed * 0.90)))
        if limit_m is None:
            return floor
        span = float(np.clip((limit_m - 0.45) / 1.45, 0.0, 1.0))
        return int(round(floor + span * max(0, self.speed - floor)))

    def _differential(self, speed: int, heading_deg: float) -> str:
        """Apply a smooth but useful forward arc with both wheels powered."""
        target_heading = float(
            np.clip(heading_deg, -MAX_GENTLE_HEADING_DEG, MAX_GENTLE_HEADING_DEG)
        )
        steering_delta = float(np.clip(
            target_heading - self._steering_deg,
            -MAX_STEERING_STEP_DEG,
            MAX_STEERING_STEP_DEG,
        ))
        self._steering_deg += steering_delta
        heading = self._steering_deg
        turn_fraction = abs(heading) / MAX_GENTLE_HEADING_DEG
        requested_split = self._limit_turn_split(
            int(round(MAX_TURN_SPLIT_PWM * turn_fraction))
        )
        # The old eight-PWM split produced a turn radius too large to avoid an
        # obstacle detected one metre ahead.  Add only the headroom needed for
        # steering, keeping the inside wheel above its measured loaded floor.
        outer = min(MAX_PWM, max(speed, self.min_move_pwm + requested_split))
        inner = max(self.min_move_pwm, outer - requested_split)
        if heading < 0.0:
            left, right = inner, outer
        else:
            left, right = outer, inner
        self.cruise_pwm = outer
        return self._set_output("F", left, right)

    def _heading_from_profile(self, profile: np.ndarray, now: float) -> tuple[float, float] | None:
        """Pick the best heading, preferring straight and resisting flicker."""
        # Clear distance stops being worth anything once there is a decent run
        # ahead; without saturation the robot always turns toward whichever
        # direction is roomiest and curls instead of crossing the room.
        # The straight-line cost has to be large relative to the saturated
        # clearance, or a flat wall ahead always makes some oblique heading
        # look better and the robot veers off instead of closing on it.
        score = np.minimum(profile, 1.5) - np.abs(STEER_HEADINGS) / 90.0 * 0.65
        exploration_heading: float | None = None
        if self.exploration.active:
            exploration_heading = float(np.clip(
                self.exploration.heading_error_deg,
                float(STEER_HEADINGS[0]),
                float(STEER_HEADINGS[-1]),
            ))
            goal_error = np.abs(STEER_HEADINGS - exploration_heading)
            score += np.clip(1.0 - goal_error / 90.0, 0.0, 1.0) * 0.82
        usable = profile >= MIN_NAV_CORRIDOR_M
        if not np.any(usable):
            return None
        straight = int(np.argmin(np.abs(STEER_HEADINGS)))
        # If the body already has a full metre straight ahead, keep making
        # progress.  Previously a slightly longer side corridor won every scan,
        # causing the robot to orbit local objects instead of crossing open floor.
        prefer_straight = (
            profile[straight] >= FORWARD_PREFERENCE_CLEARANCE_M
            and (
                exploration_heading is None
                or abs(exploration_heading) < 10.0
            )
        )
        if prefer_straight:
            best = straight
        else:
            best = int(np.argmax(np.where(usable, score, -np.inf)))
        previous = self._heading_index
        if not prefer_straight and previous is not None and usable[previous] and best != previous:
            # Only switch for a materially better option, so scan noise cannot
            # make the chassis weave between two near-tied headings.
            if score[previous] + 0.14 >= score[best]:
                best = previous
            # Do not flip across the centre while the currently chosen side
            # still contains a viable corridor.  This is the common multi-object
            # oscillation that looked like indecisive left/right spinning.
            elif (STEER_HEADINGS[previous] * STEER_HEADINGS[best] < 0.0
                  and now < self._direction_lock_until):
                best = previous
        if previous is None or STEER_HEADINGS[previous] * STEER_HEADINGS[best] < 0.0:
            self._direction_lock_until = now + 1.20
        self._heading_index = best
        return float(STEER_HEADINGS[best]), float(profile[best])

    def _hold_stop(self, reason: str) -> str:
        """Stop without clearing the close-obstacle recovery latch."""
        self.reason = reason
        self.imu_limited = False
        self.drive_confidence = 0.0
        self.cruise_pwm = 0
        self._steering_deg = 0.0
        return self._set_output("STOP", 0, 0)

    def _set_stop(self, reason: str) -> str:
        self.reason = reason
        self.imu_limited = False
        self._arc_active = False
        self.drive_confidence = 0.0
        self.cruise_pwm = 0
        self._steering_deg = 0.0
        self._heading_index = None
        self._reset_escape()
        return self._set_output("STOP", 0, 0)

    def decide(self, lidar: SectorClearance, arduino: ArduinoStatus, _person: bool, now: float,
               camera_ready: bool = True) -> str:
        if now - self.started_at < self.standby_s:
            return self._set_stop(f"STANDBY {max(0, int(self.standby_s - (now - self.started_at)))}s")
        if now - arduino.received_at > 1.5:
            return self._set_stop("STOP:UNO_STATUS_STALE")
        if not lidar.fresh:
            return self._set_stop("STOP:LD19_STALE")
        if not camera_ready:
            return self._set_stop("STOP:CAMERA_STALE")
        if lidar.front_m is None and lidar.profile is None:
            return self._set_stop("STOP:LD19_FRONT_UNSEEN")

        # Close obstacles enter a bounded recovery state machine.  Reverse
        # clearance comes from the LD19, turn completion comes from the IMU when
        # available, and every phase has a finite endpoint.
        straight_clearance = lidar.front_m or 0.0
        best_clearance = straight_clearance
        if lidar.profile is not None:
            straight_index = int(np.argmin(np.abs(STEER_HEADINGS)))
            straight_clearance = float(lidar.profile[straight_index])
            best_clearance = float(np.max(lidar.profile))
        close_ultrasonic = (
            arduino.front_cm is not None and arduino.front_cm < CLOSE_ULTRASONIC_CM
        )
        # A close return straight ahead is not a reason to reverse when a
        # body-width forward arc is open.  Recovery is reserved for the case
        # where every candidate corridor is genuinely too short.
        close_lidar = best_clearance < CLOSE_LIDAR_M
        if self._escape_phase != "IDLE":
            recovery_command = self._continue_escape(
                lidar, arduino, straight_clearance, now
            )
            if recovery_command is not None:
                return recovery_command
        if close_ultrasonic or close_lidar:
            source = "ULTRASONIC" if close_ultrasonic else "LD19"
            return self._start_escape(lidar, now, source)

        # With a corridor profile the planner can steer continuously: it knows
        # how far its own body can travel along every heading, so it curves
        # around an object and slows in proportion to what is actually ahead,
        # instead of switching between a straight mode and an arc mode.
        if lidar.profile is not None:
            choice = self._heading_from_profile(lidar.profile, now)
            if choice is not None:
                heading, limit = choice
                speed = self._cruise_speed(limit)
                # Corridor headings describe a straight swept pose, while the
                # chassis reaches that pose along an arc.  Lead the requested
                # yaw so the arc itself stays outside the inflated obstacle.
                applied_heading = float(np.clip(
                    heading * CORRIDOR_STEERING_GAIN,
                    -MAX_GENTLE_HEADING_DEG,
                    MAX_GENTLE_HEADING_DEG,
                ))
                self._arc_active = abs(applied_heading) > 6.0
                self.drive_confidence = float(np.clip(limit / 2.0, 0.0, 1.0))
                if self.exploration.active:
                    self.reason = (
                        f"EXPLORE_{self.exploration.mode}:"
                        f"{applied_heading:+.0f}deg "
                        f"target{self.exploration.target_distance_m:.1f}m"
                    )
                else:
                    self.reason = (
                        f"DRIVE:{applied_heading:+.0f}deg "
                        f"{limit:.2f}m pwm{speed}"
                    )
                return self._differential(speed, applied_heading)
            return self._start_escape(lidar, now, "NO_CORRIDOR")

        # Use different enter/exit distances so a range return hovering near
        # one threshold cannot make the chassis alternate between arc/straight.
        if lidar.front_m < 1.00:
            self._arc_active = True
        elif lidar.front_m > 1.05:
            self._arc_active = False
        if self._arc_active:
            choice = self._choose_turn(lidar, now)
            if choice is not None and choice[1] >= 0.42:
                progress = float(np.clip((lidar.front_m - 0.42) / 0.63, 0.0, 1.0))
                base = self._cruise_speed(lidar.front_m)
                self.reason = f"ARC_AVOID:{choice[0]}"
                self.drive_confidence = 0.55 + 0.25 * progress
                heading = -MAX_GENTLE_HEADING_DEG if choice[0] == "L" else MAX_GENTLE_HEADING_DEG
                return self._differential(min(self.speed, base), heading)
            return self._start_escape(lidar, now, "ARC_BLOCKED")

        # Centre gently toward the clearer front quarter.  It is deliberately
        # capped so a noisy far-wall measurement cannot cause a sharp turn.
        left = self._side_score(lidar.front_left_m, lidar.left_m)
        right = self._side_score(lidar.front_right_m, lidar.right_m)
        bias = 0.0
        if left is not None and right is not None:
            target_bias = float(np.clip((left - right) / 2.0, -0.16, 0.16))
            self._guidance_bias = self._guidance_bias * 0.78 + target_bias * 0.22
            bias = self._guidance_bias
        else:
            self._guidance_bias *= 0.82
        front_margin = float(np.clip((lidar.front_m - 0.42) / 1.10, 0.0, 1.0))
        side_margin = 0.0 if left is None or right is None else float(np.clip(min(left, right) / 1.0, 0.0, 1.0))
        self.drive_confidence = front_margin * 0.65 + side_margin * 0.35
        left_pwm = int(self.speed * (1.0 - bias))
        right_pwm = int(self.speed * (1.0 + bias))
        self.reason = "CLEAR:GUIDED_FORWARD" if abs(bias) >= 0.04 else "CLEAR:FORWARD"
        return self._set_output("F", left_pwm, right_pwm)

    def send(self, link: ArduinoLink, command: str, now: float) -> None:
        # Renew a short control lease every decision.  ArduinoLink refreshes
        # the selected output independently, so bounded planner/display stalls
        # cannot become audible motor dropouts while a healthy main loop runs.
        output = (self.left_pwm, self.right_pwm)
        if link.differential_ready:
            link.publish_drive(output[0], output[1])
        else:
            # The legacy L/R fallback is a counter-rotating pivot and was
            # responsible for fast spins whenever CAPS DRIVE was missing or
            # delayed.  Smooth autonomous motion requires current firmware.
            link.send("STOP")
        if output != self._last_output or now - self.last_sent_at >= 0.12:
            self.last_command, self._last_output, self.last_sent_at = command, output, now


def draw_dashboard(local_map: np.ndarray, policy: AutonomousPolicy,
                   clearance: SectorClearance, status: ArduinoStatus, _person: bool,
                   camera_ready: bool, differential_ready: bool,
                   imu: IMUState, slam_lite: SlamLiteState,
                   camera_motion: CameraMotionState | None = None) -> np.ndarray:
    panel = local_map.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 182), (14, 22, 31), -1)
    front = "--" if clearance.front_m is None else f"{clearance.front_m:.2f}m"
    front_left = "--" if clearance.front_left_m is None else f"{clearance.front_left_m:.2f}m"
    front_right = "--" if clearance.front_right_m is None else f"{clearance.front_right_m:.2f}m"
    ultra = "--" if status.front_cm is None else f"{status.front_cm:.0f}cm"
    lidar_state = "LIVE" if clearance.fresh else "STALE"
    camera_state = "LIVE" if camera_ready else "STALE"
    if camera_motion is not None and camera_motion.fresh:
        camera_state += f" FLOW {camera_motion.confidence:.2f}/{camera_motion.tracked_features}"
    cv2.putText(panel, "VisionFSD Robot - LiDAR navigation", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 244, 250), 1, cv2.LINE_AA)
    cv2.putText(panel, f"POLICY {policy.reason}", (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 235, 130) if policy.last_command == "F" else (80, 190, 245), 1, cv2.LINE_AA)
    cv2.putText(panel,
                f"LD19 {lidar_state}  F {front}  FL {front_left}  FR {front_right}  "
                f"ULTRA {ultra}  CAM NAV {camera_state}",
                (12, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (195, 215, 230), 1, cv2.LINE_AA)
    drive_mode = "DIFFERENTIAL" if differential_ready else "WAITING FOR CAPS DRIVE - MOTORS HELD STOPPED"
    drive_color = (90, 235, 130) if differential_ready else (70, 95, 255)
    cv2.putText(panel,
                f"UNO {drive_mode}  CMD {policy.left_pwm:+d}/{policy.right_pwm:+d}  "
                f"ACTUAL {status.left_pwm:+d}/{status.right_pwm:+d}  BLOCKED {'YES' if status.blocked else 'NO'}",
                (12, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.35, drive_color, 1, cv2.LINE_AA)
    if not imu.connected:
        imu_state = "MISSING - LD19+COMMAND POSE ACTIVE"
        imu_color = (80, 190, 245)
    elif not imu.calibrated:
        imu_state = f"CALIBRATING {imu.calibration_progress * 100:.0f}%"
        imu_color = (80, 190, 245)
    elif not imu.fresh:
        imu_state = "STALE - LD19+COMMAND POSE ACTIVE"
        imu_color = (70, 95, 255)
    else:
        limiter = " RATE LIMIT" if policy.imu_limited else ""
        imu_state = f"LIVE  YAW {imu.yaw_deg:+.1f}deg  RATE {imu.gyro_z_dps:+.1f}dps{limiter}"
        imu_color = (90, 235, 130)
    cv2.putText(panel, f"IMU {imu.source} {imu_state}", (12, 121),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, imu_color, 1, cv2.LINE_AA)
    exploration = policy.exploration
    exploration_state = (
        f"{exploration.mode} target {exploration.target_distance_m:.1f}m "
        f"bearing {exploration.heading_error_deg:+.0f}deg "
        f"frontiers {exploration.frontier_count} "
        f"coverage {exploration.coverage_ratio * 100:.0f}% "
        f"plan {exploration.planning_ms:.0f}ms"
    )
    cv2.putText(
        panel,
        f"EXPLORATION {exploration_state}",
        (12, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (225, 190, 235) if exploration.active else (150, 170, 190),
        1,
        cv2.LINE_AA,
    )
    match = "MATCH" if slam_lite.matched else "PREDICT"
    cv2.putText(panel,
                f"NAV CONFIDENCE {policy.drive_confidence:.2f}   POSE {match} "
                f"{slam_lite.yaw_source} yaw {slam_lite.yaw_confidence:.2f} "
                f"xy {slam_lite.translation_confidence:.2f}",
                (12, 169), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (185, 205, 225), 1, cv2.LINE_AA)
    return panel


def open_dashboard_window() -> None:
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative VisionFSD Pi robot runtime")
    parser.add_argument("--arduino-port", default="auto", help="Normally /dev/ttyACM0")
    parser.add_argument("--lidar-port", default="auto", help="Normally /dev/ttyUSB0")
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--standby-seconds", type=float, default=25.0)
    parser.add_argument("--speed", type=int, default=DEFAULT_CRUISE_PWM, choices=range(MIN_MOVE_PWM, MAX_PWM + 1),
                        metavar=f"{MIN_MOVE_PWM}..{MAX_PWM}",
                        help="Cruise PWM in clear space. The planner slows below this "
                             "in proportion to measured clearance.")
    parser.add_argument("--min-move-pwm", type=int, default=MIN_MOVE_PWM,
                        help="Lowest PWM that turns a loaded wheel. Raise if the robot "
                             "buzzes without moving; lower if it is still too quick.")
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--lidar-front-offset-deg", type=float, default=0.0,
                        help="Physical LD19 zero-angle correction; positive rotates readings right")
    parser.add_argument("--imu-bus", type=int, default=1)
    parser.add_argument("--imu-address", type=lambda value: int(value, 0), default=0x68)
    parser.add_argument("--imu-mount-yaw-deg", type=float, default=180.0,
                        help="IMU board yaw relative to robot frame; this chassis uses 180")
    parser.add_argument("--no-imu", action="store_true",
                        help="Disable USB/GPIO IMUs and use camera+LD19 pose prediction")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.standby_seconds < 0:
        raise ValueError("--standby-seconds must be non-negative")
    arduino_port = discover_arduino_port() if args.arduino_port == "auto" else args.arduino_port
    if not arduino_port:
        print("No unique Arduino Uno port found. Use --arduino-port /dev/ttyACM0.", file=sys.stderr)
        return 2
    lidar_port = discover_ld19_port(arduino_port) if args.lidar_port == "auto" else args.lidar_port
    if not lidar_port:
        print("No unique LD19 port found. Use --lidar-port /dev/ttyUSB0.", file=sys.stderr)
        return 2
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
    arduino = ArduinoLink(arduino_port)
    lidar = LD19Link(lidar_port, args.lidar_front_offset_deg)
    camera = CameraSafety(args.camera, args.fov)
    imu = None if args.no_imu else AutoIMULink(
        args.imu_bus, args.imu_address, args.imu_mount_yaw_deg
    )
    print(
        f"VisionFSD Robot: Uno={arduino_port}, LD19={lidar_port}, "
        f"camera request={args.camera}, IMU={'disabled' if imu is None else 'auto USB/GPIO'}"
    )
    policy = AutonomousPolicy(args.standby_seconds, args.speed, args.min_move_pwm)
    # The map supplies a long-horizon exploration heading.  Current LD19
    # geometry remains the authority that decides whether motion is safe.
    local_map = LidarSlamLite()
    explorer = FrontierExplorer()
    slam_lite = local_map.state()
    exploration = ExplorationState()
    imu_state = IMUState(error="disabled") if imu is None else imu.state()
    next_display_at = 0.0
    next_telemetry_at = 0.0
    last_policy_state: tuple[str, str, str] | None = None
    last_imu_error: str | None = None
    calibrated_source: str | None = None
    keep_running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not args.no_display:
        open_dashboard_window()
    try:
        while keep_running:
            now = time.monotonic()
            arduino.poll_capabilities(now)
            if imu is not None:
                imu_state = imu.tick(
                    now, stationary=policy.left_pwm == 0 and policy.right_pwm == 0
                )
                if imu_state.error != last_imu_error:
                    if imu_state.error:
                        print(f"IMU unavailable; camera+LD19 pose active: {imu_state.error}")
                    elif last_imu_error:
                        print(f"{imu_state.source} connected; calibrating while stationary")
                    last_imu_error = imu_state.error
                if imu_state.calibrated and imu_state.source != calibrated_source:
                    print(f"{imu_state.source} calibrated; measured yaw enabled")
                    calibrated_source = imu_state.source
            policy.observe_imu(imu_state)
            camera.tick(policy.left_pwm, policy.right_pwm)
            clearance = lidar.clearance()
            status = arduino.status()
            camera_ready = camera.ready(now)
            points, _fresh = lidar.snapshot()
            imu_yaw_rate = (
                imu_state.gyro_z_dps
                if imu_state.connected and imu_state.calibrated and imu_state.fresh
                else None
            )
            camera_motion = camera.motion
            camera_flow_fresh = (
                camera_motion.fresh
                and now - camera_motion.captured_at <= 0.45
                and camera_motion.confidence >= 0.25
            )
            slam_lite = local_map.update(
                points,
                policy.left_pwm,
                policy.right_pwm,
                now,
                imu_yaw_rate,
                camera_motion.yaw_rate_dps if camera_flow_fresh else None,
                camera_motion.translation_scale if camera_flow_fresh else 1.0,
            )
            policy.observe_pose(slam_lite)
            # Make the safety decision and refresh the Uno watchdog before the
            # advisory global planner runs.  A bounded but non-trivial A* search
            # must never turn route computation into periodic motor dropouts.
            control_now = time.monotonic()
            command = policy.decide(
                clearance,
                status,
                False,
                control_now,
                camera_ready,
            )
            policy.send(arduino, command, control_now)
            policy_state = (
                command,
                policy._escape_phase,
                policy.reason.split(":", 1)[0],
            )
            if policy_state != last_policy_state:
                event_front = clearance.limit_at(0.0)
                if event_front is None:
                    event_front = clearance.front_m or 0.0
                print(
                    f"NAV_EVENT reason={policy.reason} "
                    f"cmd={policy.left_pwm}/{policy.right_pwm} "
                    f"front_path={event_front:.2f}"
                )
                last_policy_state = policy_state
            exploration = explorer.update(
                local_map.grid,
                local_map.observed,
                local_map.visits,
                local_map.x,
                local_map.y,
                local_map.heading,
                local_map.metres,
                slam_lite.map_updates,
                control_now,
            )
            policy.observe_exploration(exploration)
            if now >= next_telemetry_at:
                next_telemetry_at = now + TELEMETRY_PERIOD_S
                front = "--" if clearance.front_m is None else f"{clearance.front_m:.2f}"
                ultra = "--" if status.front_cm is None else f"{status.front_cm:.0f}"
                print(
                    f"NAV reason={policy.reason} front_m={front} ultra_cm={ultra} "
                    f"cmd={policy.left_pwm}/{policy.right_pwm} "
                    f"actual={status.left_pwm}/{status.right_pwm} blocked={int(status.blocked)} "
                    f"lidar={int(clearance.fresh)} camera={int(camera_ready)} "
                    f"drive_caps={int(arduino.differential_ready)} "
                    f"imu={int(imu_state.fresh)} imu_cal={int(imu_state.calibrated)} "
                    f"gyro_z={imu_state.gyro_z_dps:+.1f} "
                    f"pose_yaw={slam_lite.yaw_source} "
                    f"cam_flow={camera_motion.confidence:.2f} "
                    f"explore={exploration.mode} "
                    f"target_m={exploration.target_distance_m:.2f} "
                    f"bearing={exploration.heading_error_deg:+.0f} "
                    f"plan_ms={exploration.planning_ms:.1f}"
                )
            if not args.no_display and now >= next_display_at:
                next_display_at = now + DISPLAY_PERIOD_S
                target_xy = (
                    None
                    if exploration.target_x_m is None or exploration.target_y_m is None
                    else (exploration.target_x_m, exploration.target_y_m)
                )
                waypoint_xy = (
                    None
                    if exploration.waypoint_x_m is None or exploration.waypoint_y_m is None
                    else (exploration.waypoint_x_m, exploration.waypoint_y_m)
                )
                panel = draw_dashboard(
                    local_map.render(
                        size=640,
                        target_xy=target_xy,
                        waypoint_xy=waypoint_xy,
                    ),
                    policy,
                    clearance,
                    status,
                    False,
                    camera_ready,
                    arduino.differential_ready,
                    imu_state,
                    slam_lite,
                    camera_motion,
                )
                cv2.imshow(WINDOW_TITLE, panel)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            time.sleep(0.03)
    finally:
        arduino.close()
        lidar.close()
        camera.close()
        if imu is not None:
            imu.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
