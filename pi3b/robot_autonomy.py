#!/usr/bin/env python3
"""Pi 3B robot runtime for an LD19, a USB camera and an Arduino Uno.

The Pi is the high-level planner.  The Uno is the real-time motor driver and
the final front-ultrasonic guard.  A missing serial link, a stale LiDAR scan,
or a dead camera always results in STOP rather than a guessed movement.

Navigation itself lives in ``robot_navigation``: the robot's own width and
length are used to work out how far its body can actually travel along each
candidate heading, which is what lets it pick real gaps, arc smoothly, reverse
out of dead ends, and notice when it is orbiting or has stalled.

The chassis has no wheel encoders and no IMU.  Heading change is estimated by
matching successive LiDAR scans, so the on-screen map is a useful local sketch
and explicitly not metric SLAM.
"""

from __future__ import annotations

import argparse
import math
import os
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
from robot_navigation import (
    BIN_COUNT,
    PLANNING_HORIZON_M,
    DriveCommand,
    NavigationPlanner,
    PlannerTuning,
    RobotGeometry,
    ScanFrame,
    scan_from_points,
)
from visionfsd_pi import (
    AsyncDetector,
    LatestCamera,
    SceneObject,
    SceneObjectTracker,
    TFLiteVehicleDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_TITLE = "VisionFSD Pi Robot"
UNO_BAUD = 115200
LD19_BAUD = 230400
PLAN_PERIOD_S = 0.08
RENDER_PERIOD_S = 0.16


@dataclass(frozen=True)
class ArduinoStatus:
    front_cm: float | None
    motion: str
    received_at: float
    blocked: bool = False


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
        self._lines: queue.Queue[str] = queue.Queue(maxsize=200)
        self._running = True
        self._status = ArduinoStatus(None, "S", 0.0)
        self._supports_differential = False
        self.firmware = 1
        self._reader = threading.Thread(target=self._read_loop, name="uno-status", daemon=True)
        self._reader.start()
        self.send("CAPS")
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
                raw_front = fields.get("front_cm")
                try:
                    front = None if raw_front in (None, "NO_ECHO") else float(raw_front)
                except ValueError:
                    front = None
                self._status = ArduinoStatus(front, fields.get("motion", "S"), time.monotonic(),
                                             fields.get("blocked") == "1")
                try:
                    self.firmware = int(fields.get("fw", self.firmware))
                except ValueError:
                    pass
            elif line.startswith("CAPS") and "DRIVE" in line:
                self._supports_differential = True
            try:
                self._lines.put_nowait(line)
            except queue.Full:
                pass

    def send(self, command: str) -> None:
        if self._running and self._serial.is_open:
            try:
                self._serial.write((command + "\n").encode("ascii"))
            except serial.SerialException:
                self._running = False

    def status(self) -> ArduinoStatus:
        return self._status

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

    @property
    def scan_hz(self) -> float:
        return self._parser.speed_dps / 360.0

    @property
    def crc_errors(self) -> int:
        return self._parser.crc_errors

    def scan(self) -> ScanFrame:
        now = time.monotonic()
        with self._lock:
            points = self._map.fresh(now, 0.35)
            fresh = now - self._last_packet_at <= 0.45
        return scan_from_points(points, fresh, now)

    def close(self) -> None:
        self._running = False
        try:
            self._serial.close()
        except serial.SerialException:
            pass


class LocalLidarMap:
    """Display-only local occupancy sketch with an explicitly approximate pose.

    Heading comes from the planner's LiDAR scan matching rather than from the
    commanded PWM, which keeps a pivot from smearing the map into rings.
    Translation is still commanded-motion only, so this remains a sketch.
    """

    def __init__(self, cells: int = 110, metres: float = 5.5) -> None:
        self.cells = cells
        self.metres = metres
        self.grid = np.zeros((cells, cells), dtype=np.uint8)
        self.x = metres / 2.0
        self.y = metres / 2.0
        self.heading = 0.0
        self.trail: list[tuple[float, float]] = []
        self._last_motion_at = time.monotonic()

    def integrate_motion(self, left_pwm: int, right_pwm: int, measured_heading: float, now: float) -> None:
        elapsed = min(0.40, max(0.0, now - self._last_motion_at))
        self._last_motion_at = now
        self.heading = measured_heading % 360.0
        forward = (left_pwm + right_pwm) * 0.5
        # Only near-symmetric wheel commands are treated as translation; a
        # pivot has a mean near zero and correctly moves the pose very little.
        linear = (forward / 105.0) * 0.32
        self.x += math.sin(math.radians(self.heading)) * linear * elapsed
        self.y -= math.cos(math.radians(self.heading)) * linear * elapsed
        self.x = float(np.clip(self.x, 0.3, self.metres - 0.3))
        self.y = float(np.clip(self.y, 0.3, self.metres - 0.3))
        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 0.08:
            self.trail.append((self.x, self.y))
            del self.trail[:-260]

    def integrate_scan(self, scan: ScanFrame) -> None:
        finite = np.isfinite(scan.ranges)
        if not np.any(finite):
            return
        distances = scan.ranges[finite]
        angles = np.radians(np.arange(BIN_COUNT, dtype=np.float32)[finite] + self.heading)
        keep = (distances >= 0.10) & (distances <= self.metres / 1.6)
        if not np.any(keep):
            return
        scale = self.cells / self.metres
        columns = ((self.x + np.sin(angles[keep]) * distances[keep]) * scale).astype(np.int32)
        rows = ((self.y - np.cos(angles[keep]) * distances[keep]) * scale).astype(np.int32)
        inside = (rows >= 0) & (rows < self.cells) & (columns >= 0) & (columns < self.cells)
        self.grid = (self.grid.astype(np.float32) * 0.985).astype(np.uint8)
        np.add.at(self.grid, (rows[inside], columns[inside]), 40)

    def render(self, size: int = 460) -> np.ndarray:
        image = cv2.resize(self.grid, (size, size), interpolation=cv2.INTER_NEAREST)
        panel = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        scale = size / self.metres
        for index in range(1, len(self.trail)):
            start = (int(self.trail[index - 1][0] * scale), int(self.trail[index - 1][1] * scale))
            end = (int(self.trail[index][0] * scale), int(self.trail[index][1] * scale))
            cv2.line(panel, start, end, (120, 150, 90), 1, cv2.LINE_AA)
        px = int(np.clip(self.x * scale, 0, size - 1))
        py = int(np.clip(self.y * scale, 0, size - 1))
        radians = math.radians(self.heading)
        tip = (int(px + math.sin(radians) * 22), int(py - math.cos(radians) * 22))
        cv2.circle(panel, (px, py), 7, (80, 240, 100), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, (px, py), tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(panel, "LOCAL LIDAR MAP - POSE APPROXIMATE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (235, 245, 250), 1, cv2.LINE_AA)
        return panel


class CameraSafety:
    """Camera health and semantics are a veto; range sensing does the steering."""

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
        self.objects: list[SceneObject] = []
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
            self.objects = self.tracker.update(
                result.detections, self.frame.shape[1], result.completed_time, self.frame.shape[0]
            )
            self.people = [item for item in self.objects if item.detection.label == "person"]
            self._last_result_sequence = result.sequence

    def person_stop(self) -> bool:
        # A confirmed person approximately inside the forward path is a stop
        # condition.  A one-frame model guess is intentionally ignored.
        return any(item.observed and item.distance_m < 1.7 and abs(item.bearing_deg) <= 28.0
                   for item in self.people)

    def person_bearings(self) -> list[float]:
        """Bearings of confirmed people to steer away from before stopping is needed."""
        return [float(item.bearing_deg) for item in self.people
                if item.observed and item.distance_m < 3.5]

    def clutter_scale(self) -> float:
        """Slow down when the camera sees anything solid close ahead."""
        close = [item for item in self.objects
                 if item.observed and item.distance_m < 1.3 and abs(item.bearing_deg) <= 24.0]
        return 0.75 if close else 1.0

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


class CommandSender:
    """Refresh the Uno inside its 350 ms dead-man timeout, without spamming it."""

    def __init__(self) -> None:
        self._last_output = (0, 0)
        self._last_sent_at = 0.0

    def send(self, link: ArduinoLink, command: DriveCommand, now: float) -> None:
        output = (command.left_pwm, command.right_pwm)
        if output == self._last_output and now - self._last_sent_at < 0.12:
            return
        if link.differential_ready:
            link.send(f"DRIVE {output[0]} {output[1]}")
        elif output == (0, 0):
            link.send("STOP")
        elif output[0] >= 0 and output[1] >= 0:
            # Old firmware cannot arc.  It still avoids obstacles, so a missed
            # re-flash degrades the ride rather than the safety case.
            link.send("L" if output[0] < output[1] else ("R" if output[0] > output[1] else "F"))
        elif output[0] <= 0 and output[1] <= 0:
            link.send("B")
        else:
            link.send("L" if output[0] < output[1] else "R")
        self._last_output, self._last_sent_at = output, now


def draw_clearance_fan(panel: np.ndarray, planner: NavigationPlanner, limits: np.ndarray,
                       origin: tuple[int, int], radius: int) -> None:
    """Draw the body-inflated travel limit for every candidate heading."""
    cx, cy = origin
    headings = planner.corridor.headings
    for heading, limit in zip(headings, limits):
        radians = math.radians(float(heading))
        length = int(radius * float(limit) / PLANNING_HORIZON_M)
        end = (int(cx + math.sin(radians) * length), int(cy - math.cos(radians) * length))
        usable = limit >= planner.tuning.creep_limit_m
        colour = (90, 200, 110) if usable else (60, 70, 190)
        cv2.line(panel, (cx, cy), end, colour, 1, cv2.LINE_AA)
    chosen = math.radians(planner.chosen_heading)
    tip = (int(cx + math.sin(chosen) * radius), int(cy - math.cos(chosen) * radius))
    cv2.arrowedLine(panel, (cx, cy), tip, (255, 235, 120), 2, cv2.LINE_AA, tipLength=0.18)


def draw_dashboard(frame: np.ndarray, local_map: np.ndarray, planner: NavigationPlanner,
                   limits: np.ndarray, status: ArduinoStatus, scan: ScanFrame,
                   camera_ready: bool, differential_ready: bool, person: bool,
                   plan_hz: float) -> np.ndarray:
    height = max(frame.shape[0], local_map.shape[0])
    left = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    right = cv2.resize(local_map, (height, height))
    panel = np.hstack((left, right))
    header = 104
    cv2.rectangle(panel, (0, 0), (panel.shape[1], header), (14, 22, 31), -1)

    ultra = "--" if status.front_cm is None else f"{status.front_cm:.0f}cm"
    if not planner.ultrasonic.trusted:
        ultra += " UNTRUSTED"
    lidar_state = "LIVE" if scan.fresh else "STALE"
    camera_state = "LIVE" if camera_ready else "STALE"
    drive_mode = "DIFFERENTIAL" if differential_ready else "COMPATIBILITY"
    moving = planner.left_pwm or planner.right_pwm

    cv2.putText(panel, f"VisionFSD Robot - {planner.state}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 244, 250), 1, cv2.LINE_AA)
    cv2.putText(panel, planner.reason, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (90, 235, 130) if moving else (80, 190, 245), 1, cv2.LINE_AA)
    line = (f"LD19 {lidar_state} {scan.valid_count}pts  FWD {planner.forward_limit_m:.2f}m  "
            f"REAR {planner.rear_limit_m:.2f}m  ULTRASONIC {ultra}  CAMERA {camera_state}  "
            f"PERSON {'YES' if person else 'NO'}")
    cv2.putText(panel, line, (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 225, 235), 1, cv2.LINE_AA)
    stall = "  STALL" if planner.motion.stalled else ""
    line2 = (f"UNO {drive_mode}: left {planner.left_pwm:+d} right {planner.right_pwm:+d}  "
             f"YAW {planner.yaw.yaw_rate_dps:+.0f}deg/s  TURNSUM {planner.motion.turn_integral_deg:+.0f}  "
             f"RECOVER {planner.recovery_count}  PLAN {plan_hz:.0f}Hz{stall}")
    cv2.putText(panel, line2, (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 170, 120) if stall else (185, 200, 215), 1, cv2.LINE_AA)

    fan_origin = (left.shape[1] + height // 2, header + (height - header) // 2 + 40)
    draw_clearance_fan(panel, planner, limits, fan_origin, min(height // 3, 150))
    return panel


def parse_args() -> argparse.Namespace:
    def env_float(name: str, fallback: float) -> float:
        try:
            return float(os.environ[name])
        except (KeyError, ValueError):
            return fallback

    parser = argparse.ArgumentParser(description="VisionFSD Pi robot runtime")
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
    parser.add_argument("--robot-width-m", type=float, default=env_float("VISIONFSD_ROBOT_WIDTH_M", 0.17),
                        help="Widest part of the chassis, wheels included")
    parser.add_argument("--robot-length-m", type=float, default=env_float("VISIONFSD_ROBOT_LENGTH_M", 0.22),
                        help="Front bumper to rear bumper")
    parser.add_argument("--lidar-offset-m", type=float, default=env_float("VISIONFSD_LIDAR_OFFSET_M", 0.02),
                        help="How far the LD19 sits ahead of the middle of the robot")
    parser.add_argument("--safety-margin-m", type=float, default=env_float("VISIONFSD_SAFETY_MARGIN_M", 0.055),
                        help="Extra clearance added to each side of the body")
    parser.add_argument("--min-move-pwm", type=int, default=50,
                        help="Lowest PWM that actually turns these motors")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.standby_seconds < 0:
        raise ValueError("--standby-seconds must be non-negative")
    if not 0.05 <= args.robot_width_m <= 1.0 or not 0.05 <= args.robot_length_m <= 1.5:
        raise ValueError("--robot-width-m/--robot-length-m must be realistic metre values")
    if args.safety_margin_m < 0.0:
        raise ValueError("--safety-margin-m must be non-negative")
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
    geometry = RobotGeometry(args.robot_width_m, args.robot_length_m,
                             args.lidar_offset_m, args.safety_margin_m)
    tuning = PlannerTuning(speed=args.speed, min_move_pwm=args.min_move_pwm)
    arduino = ArduinoLink(arduino_port)
    lidar = LD19Link(lidar_port, args.lidar_front_offset_deg)
    camera = CameraSafety(args.model, args.fallback_model, args.camera, args.threads, args.fov)
    print(f"VisionFSD Robot: Uno={arduino_port}, LD19={lidar_port}, camera={args.camera}")
    print(f"Body {geometry.width_m:.2f}x{geometry.length_m:.2f} m, corridor half-width "
          f"{geometry.corridor_half_width_m:.3f} m, pivot radius {geometry.pivot_radius_m:.3f} m")

    planner = NavigationPlanner(geometry, tuning, args.standby_seconds)
    sender = CommandSender()
    local_map = LocalLidarMap()
    keep_running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not args.no_display:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    next_render = 0.0
    plan_hz = 0.0
    last_plan_at = time.monotonic()
    try:
        while keep_running:
            now = time.monotonic()
            camera.tick()
            scan = lidar.scan()
            status = arduino.status()
            camera_ready = camera.ready(now)
            person = camera.person_stop()
            command = planner.decide(
                scan,
                status.front_cm,
                now - status.received_at <= 1.5,
                camera_ready,
                person,
                camera.person_bearings(),
                now,
            )
            clutter = camera.clutter_scale()
            if clutter < 1.0 and command.left_pwm > 0 and command.right_pwm > 0:
                command = DriveCommand(int(command.left_pwm * clutter), int(command.right_pwm * clutter),
                                       command.state, command.reason + " CAM_SLOW",
                                       command.heading_deg, command.target_speed)
            sender.send(arduino, command, now)
            plan_hz = 0.85 * plan_hz + 0.15 / max(1e-3, now - last_plan_at)
            last_plan_at = now

            if not args.no_display and now >= next_render:
                next_render = now + RENDER_PERIOD_S
                local_map.integrate_motion(planner.left_pwm, planner.right_pwm,
                                           planner.yaw.heading_deg, now)
                local_map.integrate_scan(scan)
                display_frame = camera.annotated_frame()
                if display_frame is not None:
                    limits = planner.corridor.travel_limits(scan)
                    panel = draw_dashboard(display_frame, local_map.render(), planner, limits,
                                           status, scan, camera_ready, arduino.differential_ready,
                                           person, plan_hz)
                    cv2.imshow(WINDOW_TITLE, panel)
                    cv2.setWindowTitle(WINDOW_TITLE, f"VisionFSD Pi Robot - {planner.state.lower()}")
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        break
            time.sleep(max(0.0, PLAN_PERIOD_S - (time.monotonic() - now)))
    finally:
        arduino.close()
        lidar.close()
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
