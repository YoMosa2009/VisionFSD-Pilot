#!/usr/bin/env python3
"""Conservative Pi 3B robot runtime for an LD19, USB camera and Arduino Uno.

The Pi is the high-level planner.  The Uno is the real-time motor and
front-ultrasonic safety controller.  A missing serial link, stale LiDAR data,
or a close obstacle therefore always results in STOP rather than a guessed
movement command.

This is deliberately a low-speed indoor demonstrator.  The supplied OSOYOO
chassis has neither wheel encoders nor an IMU, so its on-screen LiDAR map uses
commanded-motion dead reckoning and is labelled approximate; it is not a
claim of metric SLAM.
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
from robot_slam_lite import LidarSlamLite, SlamLiteState
from visionfsd_pi import (
    AsyncDetector,
    LatestCamera,
    SceneObject,
    SceneObjectTracker,
    TFLiteVehicleDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_TITLE = "VisionFSD Pi Robot - standby"
UNO_BAUD = 115200
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
# At the low 118-PWM clear-space ceiling, the usable band above the loaded-wheel
# deadband is only 13 PWM counts.  A larger split would stall the inside wheel
# and turn an avoidance arc into a one-wheel spin.  Keep every ordinary forward
# command inside that band; a close obstacle is handled by a bounded reverse
# curve, never an in-place pivot.
MAX_GENTLE_HEADING_DEG = 24.0
MIN_TURN_MARGIN_PWM = 8
ESCAPE_TURN_MARGIN_PWM = 6
ESCAPE_REVERSE_SECONDS = 0.90
ESCAPE_REAR_CLEARANCE_M = 0.38
ESCAPE_FRONT_RELEASE_M = 0.50
ESCAPE_COMMIT_SECONDS = 0.85
FORWARD_PREFERENCE_CLEARANCE_M = 1.00
DISPLAY_PERIOD_S = 0.10
CAPS_RETRY_S = 0.50

# Measured chassis, in metres.  The planner needs its own width because a
# rectangle fits through a gap that a point always would: this is what lets it
# steer around an object rather than treat one sector as blocked.
ROBOT_WIDTH_M = 0.14
ROBOT_LENGTH_M = 0.15
SAFETY_MARGIN_M = 0.04
CORRIDOR_HALF_WIDTH_M = ROBOT_WIDTH_M / 2.0 + SAFETY_MARGIN_M
FRONT_OVERHANG_M = ROBOT_LENGTH_M / 2.0
PLANNING_HORIZON_M = 3.0
# Candidate headings for the corridor sweep, symmetric so straight ahead is
# itself an option rather than falling between two near-tied neighbours.
STEER_HEADINGS = np.arange(-72.0, 72.1, 4.0, dtype=np.float32)
_STEER_RADIANS = np.radians(STEER_HEADINGS)
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


def _signed_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _sector_clearance(points: list[tuple[int, object]], centre_deg: float, half_width_deg: float) -> float | None:
    """Return a conservative range only when adjacent LD19 returns agree.

    A raw nearest-point rule reacts to one bad LiDAR return.  Requiring a
    small angular cluster preserves small real obstacles while suppressing a
    lone speckle, which otherwise makes a lightweight robot twitch or spin.
    """
    samples = sorted(
        (float(point.angle_deg), point.distance_mm / 1000.0)
        for _index, point in points
        if abs(_signed_angle(point.angle_deg - centre_deg)) <= half_width_deg
        # Keep the useful indoor portion of the LD19's range.  The live map
        # itself rejects anything beyond 6 m; the old 4.5 m cut-off made an
        # open room look like an unseen path and caused unnecessary stops.
        and 0.08 <= point.distance_mm / 1000.0 <= 5.8
    )
    clustered: list[float] = []
    for angle, distance in samples:
        neighbours = [
            other_distance
            for other_angle, other_distance in samples
            if abs(_signed_angle(other_angle - angle)) <= 3.0
            and abs(other_distance - distance) <= max(0.14, distance * 0.22)
        ]
        if len(neighbours) >= 2:
            clustered.append(float(np.median(neighbours)))
    return min(clustered) if clustered else None


def corridor_profile(points: list[tuple[int, object]]) -> np.ndarray:
    """Clear travel ahead of the bumper for every candidate heading.

    For each heading the robot is swept forward as a rectangle of its own
    width, and any return entering that corridor limits how far it can go.
    A five-sector summary cannot express "there is a 40 cm gap 12 degrees to
    the left", which is exactly the information needed to drive around an
    object instead of stopping in front of it or swerving past the whole side.
    """
    if not points:
        return np.zeros(STEER_HEADINGS.size, dtype=np.float32)
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
    usable_indices = np.flatnonzero(supported)
    if usable_indices.size == 0:
        return np.full(STEER_HEADINGS.size, PLANNING_HORIZON_M, dtype=np.float32)
    angles = usable_indices.astype(np.float32)
    ranges = binned[usable_indices]

    delta = np.radians(((angles[None, :] - STEER_HEADINGS[:, None]) + 180.0) % 360.0 - 180.0)
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
        self._status = ArduinoStatus(None, "S", 0.0)
        self._supports_differential = False
        self._last_caps_sent_at = float("-inf")
        self._reader = threading.Thread(target=self._read_loop, name="uno-status", daemon=True)
        self._reader.start()
        self.send("STOP")
        self.poll_capabilities(time.monotonic())

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
            except serial.SerialException:
                break
            if not line:
                continue
            if line.startswith("STATUS "):
                self._status = self._parse_status(line, time.monotonic())
            elif line == "CAPS DRIVE":
                self._supports_differential = True
            self._lines.put(line)

    def send(self, command: str) -> None:
        if self._running and self._serial.is_open:
            self._serial.write((command + "\n").encode("ascii"))

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
        try:
            if self._serial.is_open:
                self._serial.write(b"STOP\n")
            self._serial.close()
        except serial.SerialException:
            pass
        self._running = False


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
            except serial.SerialException:
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
            points = self._map.fresh(now, 0.35)
            fresh = now - self._last_packet_at <= 0.45
        return points, fresh

    def clearance(self) -> SectorClearance:
        points, fresh = self.snapshot()
        if not fresh:
            return SectorClearance(None, None, None, False)
        scan_at = max((float(point.captured_at) for _index, point in points), default=0.0)
        if self._clearance_cache is not None and scan_at <= self._clearance_stamp:
            return self._clearance_cache
        clearance = SectorClearance(
            _sector_clearance(points, 0.0, 20.0),
            _sector_clearance(points, -75.0, 35.0),
            _sector_clearance(points, 75.0, 35.0),
            True,
            _sector_clearance(points, -35.0, 20.0),
            _sector_clearance(points, 35.0, 20.0),
            corridor_profile(points),
            _sector_clearance(points, 180.0, 25.0),
            scan_at,
        )
        self._clearance_stamp = scan_at
        self._clearance_cache = clearance
        return clearance

    def close(self) -> None:
        self._running = False
        try:
            self._serial.close()
        except serial.SerialException:
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
    """Camera health/person semantics are a veto; range sensing steers."""

    def __init__(self, model: Path, fallback: Path, camera: str, threads: int, fov: float) -> None:
        try:
            detector = TFLiteVehicleDetector(model, 0.50, threads)
        except Exception:
            detector = TFLiteVehicleDetector(fallback, 0.50, threads)
        self.camera = LatestCamera(camera, 640, 480, 25)
        self.worker = AsyncDetector(detector)
        self.tracker = SceneObjectTracker(fov)
        self._last_camera_sequence = 0
        self._last_result_sequence = 0
        self._last_frame_at = 0.0
        self.frame: np.ndarray | None = None
        self.people: list[SceneObject] = []

    def tick(self) -> None:
        sequence, frame, captured = self.camera.latest()
        if frame is not None:
            self.frame = frame
            if sequence > self._last_camera_sequence:
                self.worker.submit(sequence, frame, captured)
                self._last_camera_sequence = sequence
                self._last_frame_at = captured
        result = self.worker.latest_after(self._last_result_sequence)
        if result is not None and self.frame is not None:
            self.people = [item for item in self.tracker.update(
                result.detections, self.frame.shape[1], result.completed_time, self.frame.shape[0]
            ) if item.detection.label == "person" and abs(item.bearing_deg) <= 28.0]
            self._last_result_sequence = result.sequence

    def person_in_path(self) -> bool:
        # A confirmed person approximately inside the forward path is a stop
        # condition.  A one-frame model guess is intentionally ignored.
        return any(item.observed and item.distance_m < 1.7 for item in self.people)

    def ready(self, now: float) -> bool:
        """Do not drive blind if the webcam has stopped delivering frames."""
        return self.frame is not None and not self.camera.error and now - self._last_frame_at <= 1.0

    def annotated_frame(self) -> np.ndarray | None:
        if self.frame is None:
            return None
        panel = self.frame.copy()
        for item in self.people:
            x1, y1, x2, y2 = (int(value) for value in item.detection.box)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (50, 70, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, f"PERSON {item.distance_m:.1f}m", (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 70, 255), 1, cv2.LINE_AA)
        return panel

    def close(self) -> None:
        self.worker.close()
        self.camera.close()


class AutonomousPolicy:
    """Sensor-fused, low-speed differential-drive planner.

    The LD19 supplies geometry, the webcam supplies a live/person safety
    veto, and the Uno's ultrasonic sensor remains the final near-field guard.
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
        self._escape_started_at: float | None = None
        self._escape_until = 0.0
        self._escape_commit_until = 0.0
        self._arc_active = False
        self._guidance_bias = 0.0
        self.drive_confidence = 0.0
        self.last_command = "STOP"
        self.left_pwm = 0
        self.right_pwm = 0
        self._last_output = (0, 0)
        self.last_sent_at = 0.0
        self.reason = "BOOT_STANDBY"

    @staticmethod
    def _side_score(primary: float | None, outer: float | None) -> float | None:
        values = [value for value in (primary, outer) if value is not None]
        return min(values) if values else None

    @staticmethod
    def _ramp(current: int, target: int) -> int:
        """Use a direct, non-buzzing motor band with a controlled rise.

        A direct brushed-motor command below MIN_MOVE_PWM only buzzes under the
        chassis.  Starting at that floor, then rising in small steps, is smooth
        without the 50 Hz on/off gating that made the prior experiment pulse.
        Slowing and stopping remain immediate for safety.
        """
        if target == 0:
            return 0
        direction = 1 if target > 0 else -1
        target = direction * max(MIN_MOVE_PWM, abs(target))
        if current == 0:
            return direction * MIN_MOVE_PWM
        if (current > 0) != (target > 0):
            # Brake to zero before reversing; the following decision begins
            # the other direction at the non-stalling floor.
            return 0
        if abs(target) <= abs(current):
            return target
        return direction * min(abs(target), abs(current) + MAX_PWM_STEP)

    def _set_output(self, label: str, left_pwm: int, right_pwm: int) -> str:
        self.left_pwm = self._ramp(self.left_pwm, int(np.clip(left_pwm, -MAX_PWM, MAX_PWM)))
        self.right_pwm = self._ramp(self.right_pwm, int(np.clip(right_pwm, -MAX_PWM, MAX_PWM)))
        return label

    def _choose_turn(self, lidar: SectorClearance, now: float) -> tuple[str, float] | None:
        left = self._side_score(lidar.front_left_m, lidar.left_m)
        right = self._side_score(lidar.front_right_m, lidar.right_m)
        if left is None and right is None:
            return None
        scores = {"L": left, "R": right}
        locked = scores.get(self.turn_command)
        candidate = "L" if right is None or (left is not None and left >= right) else "R"
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
        outer = min(MAX_PWM, self.min_move_pwm + ESCAPE_TURN_MARGIN_PWM)
        inner = self.min_move_pwm
        if direction == "L":
            return self._set_output("L", -outer, -inner)
        return self._set_output("R", -inner, -outer)

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
        """Turn only by the usable PWM margin; both wheels stay driven."""
        # A turn needs actual headroom above the loaded-wheel floor.  At the
        # floor both PWM values are identical after normalisation, so there is
        # no steering authority at all.  Preserve a small, direct-PWM margin
        # instead of dropping the inside track to zero.
        speed = min(self.speed, max(self.min_move_pwm + MIN_TURN_MARGIN_PWM, speed))
        heading = float(np.clip(heading_deg, -MAX_GENTLE_HEADING_DEG, MAX_GENTLE_HEADING_DEG))
        turn_fraction = abs(heading) / MAX_GENTLE_HEADING_DEG
        outer = speed
        available_split = min(MIN_TURN_MARGIN_PWM, speed - self.min_move_pwm)
        inner = max(
            self.min_move_pwm,
            int(round(speed - available_split * turn_fraction)),
        )
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
        score = np.minimum(profile, 1.5) - np.abs(STEER_HEADINGS) / 90.0 * 1.10
        usable = profile >= 0.38
        if not np.any(usable):
            return None
        straight = int(np.argmin(np.abs(STEER_HEADINGS)))
        # If the body already has a full metre straight ahead, keep making
        # progress.  Previously a slightly longer side corridor won every scan,
        # causing the robot to orbit local objects instead of crossing open floor.
        prefer_straight = profile[straight] >= FORWARD_PREFERENCE_CLEARANCE_M
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
        self.drive_confidence = 0.0
        self.cruise_pwm = 0
        return self._set_output("STOP", 0, 0)

    def _set_stop(self, reason: str) -> str:
        self.reason = reason
        self._arc_active = False
        self.drive_confidence = 0.0
        self.cruise_pwm = 0
        self._heading_index = None
        self._escape_started_at = None
        self._escape_until = 0.0
        self._escape_commit_until = 0.0
        return self._set_output("STOP", 0, 0)

    def decide(self, lidar: SectorClearance, arduino: ArduinoStatus, person: bool, now: float,
               camera_ready: bool = True) -> str:
        if now - self.started_at < self.standby_s:
            return self._set_stop(f"STANDBY {max(0, int(self.standby_s - (now - self.started_at)))}s")
        if now - arduino.received_at > 1.5:
            return self._set_stop("STOP:UNO_STATUS_STALE")
        if not lidar.fresh:
            return self._set_stop("STOP:LD19_STALE")
        if not camera_ready:
            return self._set_stop("STOP:CAMERA_STALE")
        if person:
            return self._set_stop("STOP:CONFIRMED_PERSON")
        if lidar.front_m is None:
            return self._set_stop("STOP:LD19_FRONT_UNSEEN")

        # The Uno has no rear sensor.  A close front obstacle therefore gets one
        # bounded LD19-cleared reverse curve, not a pivot.  The old one-wheel
        # pivot made the chassis spin in place, and repeated recovery attempts
        # could eventually back it into an unseen rear object.
        close_ultrasonic = arduino.front_cm is not None and arduino.front_cm < 22.0
        close_lidar = lidar.front_m < 0.42
        recovery_not_released = (
            self._escape_started_at is not None
            and lidar.front_m < ESCAPE_FRONT_RELEASE_M
        )
        if close_ultrasonic or close_lidar or recovery_not_released:
            if self._escape_started_at is None:
                choice = self._choose_turn(lidar, now)
                if choice is None or choice[1] < 0.34:
                    return self._hold_stop("STOP:ESCAPE_SIDE_BLOCKED")
                self.turn_command = choice[0]
                self._escape_started_at = now
                self._escape_until = now + ESCAPE_REVERSE_SECONDS
            if lidar.rear_m is None or lidar.rear_m < ESCAPE_REAR_CLEARANCE_M:
                return self._hold_stop("STOP:ESCAPE_REAR_BLOCKED")
            if now >= self._escape_until:
                # Preserve the latch.  Resetting it here made the next 30 ms
                # cycle start another reverse, producing an endless hiccup loop.
                return self._hold_stop("STOP:ESCAPE_NO_FRONT_CLEARANCE")
            source = "ULTRASONIC" if close_ultrasonic else "LD19"
            self.reason = f"ESCAPE_{source}:{self.turn_command}"
            self.drive_confidence = 0.35
            return self._reverse_arc(self.turn_command)
        if self._escape_started_at is not None:
            self._escape_started_at = None
            self._escape_until = 0.0
            self._escape_commit_until = now + ESCAPE_COMMIT_SECONDS
            self._direction_lock_until = self._escape_commit_until
        if now < self._escape_commit_until:
            heading = -MAX_GENTLE_HEADING_DEG if self.turn_command == "L" else MAX_GENTLE_HEADING_DEG
            speed = self._cruise_speed(lidar.front_m)
            self.reason = f"ESCAPE_COMMIT:{self.turn_command}"
            self.drive_confidence = 0.50
            return self._differential(speed, heading)

        # With a corridor profile the planner can steer continuously: it knows
        # how far its own body can travel along every heading, so it curves
        # around an object and slows in proportion to what is actually ahead,
        # instead of switching between a straight mode and an arc mode.
        if lidar.profile is not None:
            choice = self._heading_from_profile(lidar.profile, now)
            if choice is not None:
                heading, limit = choice
                speed = self._cruise_speed(limit)
                applied_heading = float(np.clip(heading, -MAX_GENTLE_HEADING_DEG, MAX_GENTLE_HEADING_DEG))
                self._arc_active = abs(applied_heading) > 6.0
                self.drive_confidence = float(np.clip(limit / 2.0, 0.0, 1.0))
                self.reason = f"DRIVE:{applied_heading:+.0f}deg {limit:.2f}m pwm{speed}"
                return self._differential(speed, applied_heading)
            return self._set_stop("STOP:NO_CLEAR_CORRIDOR")

        # Use different enter/exit distances so a range return hovering near
        # one threshold cannot make the chassis alternate between arc/straight.
        if lidar.front_m < 0.86:
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
            return self._set_stop("STOP:ARC_SIDE_BLOCKED")

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
        # Refresh inside the Uno's 350 ms dead-man timeout, but do not spam the
        # USB serial link or waste Pi/Uno CPU on redundant writes.
        output = (self.left_pwm, self.right_pwm)
        if output != self._last_output or now - self.last_sent_at >= 0.12:
            if link.differential_ready:
                link.send(f"DRIVE {output[0]} {output[1]}")
            else:
                # The legacy L/R fallback is a counter-rotating pivot and was
                # responsible for fast spins whenever CAPS DRIVE was missing or
                # delayed.  Smooth autonomous motion requires current firmware.
                link.send("STOP")
            self.last_command, self._last_output, self.last_sent_at = command, output, now


def draw_dashboard(local_map: np.ndarray, policy: AutonomousPolicy,
                   clearance: SectorClearance, status: ArduinoStatus, person: bool,
                   camera_ready: bool, differential_ready: bool,
                   slam_lite: SlamLiteState) -> np.ndarray:
    panel = local_map.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 134), (14, 22, 31), -1)
    front = "--" if clearance.front_m is None else f"{clearance.front_m:.2f}m"
    front_left = "--" if clearance.front_left_m is None else f"{clearance.front_left_m:.2f}m"
    front_right = "--" if clearance.front_right_m is None else f"{clearance.front_right_m:.2f}m"
    ultra = "--" if status.front_cm is None else f"{status.front_cm:.0f}cm"
    lidar_state = "LIVE" if clearance.fresh else "STALE"
    camera_state = "LIVE" if camera_ready else "STALE"
    cv2.putText(panel, "VisionFSD Robot - LiDAR navigation", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 244, 250), 1, cv2.LINE_AA)
    cv2.putText(panel, f"POLICY {policy.reason}", (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 235, 130) if policy.last_command == "F" else (80, 190, 245), 1, cv2.LINE_AA)
    cv2.putText(panel,
                f"LD19 {lidar_state}  F {front}  FL {front_left}  FR {front_right}  "
                f"ULTRA {ultra}  CAM {camera_state}  PERSON {'YES' if person else 'NO'}",
                (12, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (195, 215, 230), 1, cv2.LINE_AA)
    drive_mode = "DIFFERENTIAL" if differential_ready else "WAITING FOR CAPS DRIVE - MOTORS HELD STOPPED"
    drive_color = (90, 235, 130) if differential_ready else (70, 95, 255)
    cv2.putText(panel,
                f"UNO {drive_mode}  CMD {policy.left_pwm:+d}/{policy.right_pwm:+d}  "
                f"ACTUAL {status.left_pwm:+d}/{status.right_pwm:+d}  BLOCKED {'YES' if status.blocked else 'NO'}",
                (12, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.35, drive_color, 1, cv2.LINE_AA)
    match = "MATCH" if slam_lite.matched else "PREDICT"
    cv2.putText(panel,
                f"NAV CONFIDENCE {policy.drive_confidence:.2f}   SLAM-LITE {match} "
                f"yaw {slam_lite.yaw_confidence:.2f} correction {slam_lite.yaw_correction_deg:+.1f}deg",
                (12, 121), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (185, 205, 225), 1, cv2.LINE_AA)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative VisionFSD Pi robot runtime")
    parser.add_argument("--arduino-port", default="auto", help="Normally /dev/ttyACM0")
    parser.add_argument("--lidar-port", default="auto", help="Normally /dev/ttyUSB0")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/vehicle_efficientdet_lite0_int8.tflite")
    parser.add_argument("--fallback-model", type=Path, default=PROJECT_ROOT / "models/vehicle_ssd_mobilenet_v1.tflite")
    parser.add_argument("--standby-seconds", type=float, default=25.0)
    parser.add_argument("--speed", type=int, default=DEFAULT_CRUISE_PWM, choices=range(MIN_MOVE_PWM, MAX_PWM + 1),
                        metavar=f"{MIN_MOVE_PWM}..{MAX_PWM}",
                        help="Cruise PWM in clear space. The planner slows below this "
                             "in proportion to measured clearance.")
    parser.add_argument("--min-move-pwm", type=int, default=MIN_MOVE_PWM,
                        help="Lowest PWM that turns a loaded wheel. Raise if the robot "
                             "buzzes without moving; lower if it is still too quick.")
    parser.add_argument("--threads", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--lidar-front-offset-deg", type=float, default=0.0,
                        help="Physical LD19 zero-angle correction; positive rotates readings right")
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
    camera = CameraSafety(args.model, args.fallback_model, args.camera, args.threads, args.fov)
    print(f"VisionFSD Robot: Uno={arduino_port}, LD19={lidar_port}, camera={args.camera}")
    policy = AutonomousPolicy(args.standby_seconds, args.speed, args.min_move_pwm)
    # The mapper is advisory: obstacle avoidance always uses the current LD19
    # sectors above, never a past map cell or a guessed pose.
    local_map = LidarSlamLite()
    slam_lite = local_map.state()
    next_display_at = 0.0
    keep_running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not args.no_display:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    try:
        while keep_running:
            now = time.monotonic()
            arduino.poll_capabilities(now)
            camera.tick()
            clearance = lidar.clearance()
            status = arduino.status()
            camera_ready = camera.ready(now)
            command = policy.decide(clearance, status, camera.person_in_path(), now, camera_ready)
            policy.send(arduino, command, now)
            points, _fresh = lidar.snapshot()
            slam_lite = local_map.update(points, policy.left_pwm, policy.right_pwm, now)
            if not args.no_display and now >= next_display_at:
                next_display_at = now + DISPLAY_PERIOD_S
                panel = draw_dashboard(local_map.render(size=640), policy, clearance, status,
                                       camera.person_in_path(), camera_ready,
                                       arduino.differential_ready, slam_lite)
                cv2.imshow(WINDOW_TITLE, panel)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            time.sleep(0.03)
    finally:
        arduino.close()
        lidar.close()
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
