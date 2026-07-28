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
from visionfsd_pi import (
    AsyncDetector,
    Detection,
    LatestCamera,
    SceneObject,
    SceneObjectTracker,
    TFLiteVehicleDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_TITLE = "VisionFSD Pi Robot - standby"
UNO_BAUD = 115200
LD19_BAUD = 230400


@dataclass(frozen=True)
class ArduinoStatus:
    front_cm: float | None
    motion: str
    received_at: float


@dataclass(frozen=True)
class SectorClearance:
    front_m: float | None
    left_m: float | None
    right_m: float | None
    fresh: bool


def _signed_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _sector_minimum(points: list[tuple[int, object]], centre_deg: float, half_width_deg: float) -> float | None:
    distances = [
        point.distance_mm / 1000.0
        for _index, point in points
        if abs(_signed_angle(point.angle_deg - centre_deg)) <= half_width_deg
    ]
    return min(distances) if distances else None


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
        self._reader = threading.Thread(target=self._read_loop, name="uno-status", daemon=True)
        self._reader.start()
        self.send("STOP")

    def _read_loop(self) -> None:
        while self._running:
            try:
                line = self._serial.readline().decode("ascii", "replace").strip()
            except serial.SerialException:
                break
            if not line:
                continue
            if line.startswith("STATUS "):
                fields = dict(part.split("=", 1) for part in line[7:].split() if "=" in part)
                try:
                    front = float(fields["front_cm"]) if fields.get("front_cm") not in (None, "NO_ECHO") else None
                except ValueError:
                    front = None
                self._status = ArduinoStatus(front, fields.get("motion", "S"), time.monotonic())
            self._lines.put(line)

    def send(self, command: str) -> None:
        if self._running and self._serial.is_open:
            self._serial.write((command + "\n").encode("ascii"))

    def status(self) -> ArduinoStatus:
        return self._status

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
        return SectorClearance(
            _sector_minimum(points, 0.0, 22.0),
            _sector_minimum(points, -75.0, 35.0),
            _sector_minimum(points, 75.0, 35.0),
            True,
        )

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

    def integrate_motion(self, motion: str, now: float) -> None:
        elapsed = min(0.20, max(0.0, now - self._last_motion_at))
        self._last_motion_at = now
        # Conservative nominal chassis speeds.  These are display-only and
        # intentionally do not feed movement decisions.
        if motion == "F":
            self.x += math.sin(math.radians(self.heading)) * 0.11 * elapsed
            self.y -= math.cos(math.radians(self.heading)) * 0.11 * elapsed
        elif motion == "B":
            self.x -= math.sin(math.radians(self.heading)) * 0.07 * elapsed
            self.y += math.cos(math.radians(self.heading)) * 0.07 * elapsed
        elif motion == "L":
            self.heading = (self.heading - 72.0 * elapsed) % 360.0
        elif motion == "R":
            self.heading = (self.heading + 72.0 * elapsed) % 360.0

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
    """Camera is a semantic veto only; it never overrides range sensing."""

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
        self.frame: np.ndarray | None = None
        self.people: list[SceneObject] = []

    def tick(self) -> None:
        sequence, frame, captured = self.camera.latest()
        if frame is not None:
            self.frame = frame
            if sequence > self._last_camera_sequence:
                self.worker.submit(sequence, frame, captured)
                self._last_camera_sequence = sequence
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

    def close(self) -> None:
        self.worker.close()
        self.camera.close()


class AutonomousPolicy:
    """One authority, ordered by safety: ultrasonic, LiDAR, camera, planner."""

    def __init__(self, standby_s: float, speed: int) -> None:
        self.started_at = time.monotonic()
        self.standby_s = standby_s
        self.speed = speed
        self.turn_until = 0.0
        self.turn_command = "L"
        self.last_command = "STOP"
        self.last_sent_at = 0.0
        self.reason = "BOOT_STANDBY"

    def decide(self, lidar: SectorClearance, arduino: ArduinoStatus, person: bool, now: float) -> str:
        if now - self.started_at < self.standby_s:
            self.reason = f"STANDBY {max(0, int(self.standby_s - (now - self.started_at)))}s"
            return "STOP"
        if now - arduino.received_at > 1.5:
            self.reason = "STOP:UNO_STATUS_STALE"
            return "STOP"
        if not lidar.fresh:
            self.reason = "STOP:LD19_STALE"
            return "STOP"
        if arduino.front_cm is not None and arduino.front_cm < 22.0:
            self.reason = "STOP:ULTRASONIC"
            return "STOP"
        if person:
            self.reason = "STOP:CONFIRMED_PERSON"
            return "STOP"
        if lidar.front_m is not None and lidar.front_m < 0.42:
            if now >= self.turn_until:
                self.turn_command = "L" if (lidar.left_m or 0.0) >= (lidar.right_m or 0.0) else "R"
                self.turn_until = now + 0.65
            self.reason = f"AVOID:{self.turn_command}"
            return self.turn_command
        if now < self.turn_until:
            self.reason = f"TURN:{self.turn_command}"
            return self.turn_command
        self.reason = "CLEAR:FORWARD"
        return "F"

    def send(self, link: ArduinoLink, command: str, now: float) -> None:
        # Refresh inside the Uno's 350 ms dead-man timeout, but do not spam the
        # USB serial link or waste Pi/Uno CPU on redundant writes.
        if command != self.last_command or now - self.last_sent_at >= 0.12:
            if command == "F":
                link.send(f"SPEED {self.speed}")
            elif command in ("L", "R"):
                link.send(f"SPEED {max(48, self.speed - 15)}")
            link.send(command)
            self.last_command, self.last_sent_at = command, now


def draw_dashboard(frame: np.ndarray, local_map: np.ndarray, policy: AutonomousPolicy,
                   clearance: SectorClearance, status: ArduinoStatus, person: bool) -> np.ndarray:
    height = max(frame.shape[0], local_map.shape[0])
    left = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    right = cv2.resize(local_map, (height, height))
    panel = np.hstack((left, right))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 72), (14, 22, 31), -1)
    front = "--" if clearance.front_m is None else f"{clearance.front_m:.2f}m"
    ultra = "--" if status.front_cm is None else f"{status.front_cm:.0f}cm"
    lidar_state = "LIVE" if clearance.fresh else "STALE"
    message = f"{policy.reason}   LD19 {lidar_state} FRONT {front}   ULTRASONIC {ultra}   PERSON {'YES' if person else 'NO'}"
    cv2.putText(panel, "VisionFSD Robot - sensor fused low-speed mode", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 244, 250), 1, cv2.LINE_AA)
    cv2.putText(panel, message, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 235, 130) if policy.last_command == "F" else (80, 190, 245), 1, cv2.LINE_AA)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative VisionFSD Pi robot runtime")
    parser.add_argument("--arduino-port", default="auto", help="Normally /dev/ttyACM0")
    parser.add_argument("--lidar-port", default="auto", help="Normally /dev/ttyUSB0")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/vehicle_efficientdet_lite0_int8.tflite")
    parser.add_argument("--fallback-model", type=Path, default=PROJECT_ROOT / "models/vehicle_ssd_mobilenet_v1.tflite")
    parser.add_argument("--standby-seconds", type=float, default=25.0)
    parser.add_argument("--speed", type=int, default=70, choices=range(45, 91))
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
    policy = AutonomousPolicy(args.standby_seconds, args.speed)
    local_map = LocalLidarMap()
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
            camera.tick()
            clearance = lidar.clearance()
            status = arduino.status()
            command = policy.decide(clearance, status, camera.person_in_path(), now)
            policy.send(arduino, command, now)
            points, _fresh = lidar.snapshot()
            local_map.integrate_motion(command, now)
            local_map.integrate_points(points)
            if not args.no_display and camera.frame is not None:
                panel = draw_dashboard(camera.frame, local_map.render(), policy, clearance, status,
                                       camera.person_in_path())
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
