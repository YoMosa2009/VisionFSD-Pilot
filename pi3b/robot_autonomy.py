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
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import serial
from serial.tools import list_ports

from lidar_visualizer import LD19Parser, LivePolarMap
from robot_navigation import (
    PLANNING_HORIZON_M,
    DriveCommand,
    NavigationPlanner,
    PlannerTuning,
    RobotGeometry,
    ScanFrame,
    scan_from_points,
)
from robot_slam import SlamConfig, SlamTracker, SlamWorker
from robot_vision import LowObstacleGuard
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
# Roughly two LD19 revolutions per scan match.  Matching faster buys little,
# because the sensor cannot show the world changing faster than it spins.
SLAM_PERIOD_S = 0.22


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
        # Raw returns keep their sub-degree angle and their own timestamp, which
        # the 1-degree planning grid throws away.  Scan matching wants both: the
        # angle for accuracy and the timestamp to undo the robot's own motion
        # across a revolution.
        self._raw: deque[tuple[float, float, float]] = deque(maxlen=1400)
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
                self._raw.extend(
                    (point.angle_deg, point.distance_mm / 1000.0, point.captured_at)
                    for point in rotated
                    if point.confidence >= 8 and 80 <= point.distance_mm <= 6000
                )
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

    def revolution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw returns from the last LD19 revolution, with their ages.

        The window is exactly one revolution, taken from the sensor's reported
        spin rate.  A longer window measures some directions twice from two
        different robot positions, which hands scan matching a smeared scan and
        makes it look unreliable when it is being fed badly.
        """
        spin = self._parser.speed_dps
        window_s = float(np.clip(360.0 / spin, 0.08, 0.16)) if spin > 0 else 0.10
        now = time.monotonic()
        with self._lock:
            recent = [item for item in self._raw if now - item[2] <= window_s]
        if not recent:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty
        data = np.asarray(recent, dtype=np.float32)
        return data[:, 0], data[:, 1], (now - data[:, 2]).astype(np.float32)

    def close(self) -> None:
        self._running = False
        try:
            self._serial.close()
        except serial.SerialException:
            pass


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


def draw_dashboard(frame: np.ndarray, map_panel: np.ndarray, planner: NavigationPlanner,
                   limits: np.ndarray, status: ArduinoStatus, scan: ScanFrame,
                   camera_ready: bool, differential_ready: bool, person: bool,
                   plan_hz: float, slam, lidar: "LD19Link", guard: LowObstacleGuard,
                   slam_ms: float) -> np.ndarray:
    height = max(frame.shape[0], map_panel.shape[0])
    left = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
    right = cv2.resize(map_panel, (height, height))
    panel = np.hstack((left, right))
    header = 126
    cv2.rectangle(panel, (0, 0), (panel.shape[1], header), (14, 22, 31), -1)

    ultra = "--" if status.front_cm is None else f"{status.front_cm:.0f}cm"
    if not planner.ultrasonic.trusted:
        ultra += " UNTRUSTED"
    lidar_state = "LIVE" if scan.fresh else "STALE"
    camera_state = "LIVE" if camera_ready else "STALE"
    drive_mode = "DIFFERENTIAL" if differential_ready else "COMPATIBILITY"
    moving = planner.left_pwm or planner.right_pwm
    slam_state = "TRACKING" if slam.trusted else ("SEARCHING" if slam.updates else "INIT")
    bearing, weight = planner.frontier

    cv2.putText(panel, f"VisionFSD Robot - {planner.state}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 244, 250), 1, cv2.LINE_AA)
    cv2.putText(panel, planner.reason, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (90, 235, 130) if moving else (80, 190, 245), 1, cv2.LINE_AA)
    line = (f"LD19 {lidar_state} {scan.valid_count}pts {lidar.scan_hz:.1f}Hz CRC {lidar.crc_errors}  "
            f"FWD {planner.forward_limit_m:.2f}m  REAR {planner.rear_limit_m:.2f}m  "
            f"ULTRASONIC {ultra}  CAMERA {camera_state}  PERSON {'YES' if person else 'NO'}")
    cv2.putText(panel, line, (12, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 225, 235), 1, cv2.LINE_AA)
    stall = "  STALL" if planner.motion.stalled else ""
    line2 = (f"UNO {drive_mode}: left {planner.left_pwm:+d} right {planner.right_pwm:+d}  "
             f"YAW {planner.yaw.yaw_rate_dps:+.0f}deg/s  TURNSUM {planner.motion.turn_integral_deg:+.0f}  "
             f"RECOVER {planner.recovery_count}  PLAN {plan_hz:.0f}Hz{stall}")
    cv2.putText(panel, line2, (12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 170, 120) if stall else (185, 200, 215), 1, cv2.LINE_AA)
    low = "  LOW-OBSTACLE" if guard.blocked else ""
    line3 = (f"SLAM {slam_state} res {slam.residual_m*100:.0f}cm {slam.matched_points}pts "
             f"{slam_ms:.0f}ms  FRONTIER {bearing:+.0f}deg w{weight:.2f}  "
             f"FLOOR {guard.coverage*100:.0f}%{low}")
    cv2.putText(panel, line3, (12, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 170, 120) if (low or not slam.trusted) else (170, 205, 190), 1, cv2.LINE_AA)

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
    parser.add_argument("--robot-width-m", type=float, default=env_float("VISIONFSD_ROBOT_WIDTH_M", 0.14),
                        help="Widest part of the chassis, wheels included")
    parser.add_argument("--robot-length-m", type=float, default=env_float("VISIONFSD_ROBOT_LENGTH_M", 0.15),
                        help="Front bumper to rear bumper")
    parser.add_argument("--lidar-offset-m", type=float, default=env_float("VISIONFSD_LIDAR_OFFSET_M", 0.0),
                        help="How far the LD19 sits ahead of the middle of the robot")
    parser.add_argument("--safety-margin-m", type=float, default=env_float("VISIONFSD_SAFETY_MARGIN_M", 0.040),
                        help="Extra clearance added to each side of the body")
    parser.add_argument("--min-move-pwm", type=int, default=50,
                        help="Lowest PWM that actually turns these motors")
    parser.add_argument("--no-slam", action="store_true",
                        help="Disable scan-matching SLAM and its exploration bias")
    parser.add_argument("--no-low-obstacle-guard", action="store_true",
                        help="Disable the camera floor-clutter slow-down")
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
    slam = SlamTracker(SlamConfig())
    slam_worker = None if args.no_slam else SlamWorker(slam)
    guard = LowObstacleGuard(enabled=not args.no_low_obstacle_guard)
    keep_running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if not args.no_display:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    next_render = 0.0
    next_slam = 0.0
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
            # The camera can only slow the robot down.  Whichever of the two
            # visual cues is more cautious wins; neither may steer.
            speed_scale = min(camera.clutter_scale(), guard.update(camera.frame))
            command = planner.decide(
                scan,
                status.front_cm,
                now - status.received_at <= 1.5,
                camera_ready,
                person,
                camera.person_bearings(),
                now,
                speed_scale,
                slam.frontier(),
            )
            sender.send(arduino, command, now)
            plan_hz = 0.85 * plan_hz + 0.15 / max(1e-3, now - last_plan_at)
            last_plan_at = now

            if slam_worker is not None and now >= next_slam:
                next_slam = now + SLAM_PERIOD_S
                bearings, ranges, ages = lidar.revolution()
                if bearings.size:
                    # Commanded speed is only a search-window centre; scan
                    # matching is what actually decides the pose.  The worker
                    # times its own interval, so a dropped sweep cannot make the
                    # prediction under-count how far the robot moved.
                    forward = (planner.left_pwm + planner.right_pwm) / 2.0 / 105.0 * 0.32
                    slam_worker.submit(bearings, ranges, ages, planner.yaw.yaw_rate_dps, forward)

            if not args.no_display and now >= next_render:
                next_render = now + RENDER_PERIOD_S
                display_frame = camera.annotated_frame()
                if display_frame is not None:
                    guard.annotate(display_frame)
                    limits = planner.corridor.travel_limits(scan)
                    state = slam.state()
                    map_panel = slam.map.render(state.x, state.y, state.heading_deg,
                                                trail=slam.trail)
                    panel = draw_dashboard(display_frame, map_panel, planner, limits,
                                           status, scan, camera_ready, arduino.differential_ready,
                                           person, plan_hz, state, lidar, guard,
                                           slam_worker.last_duration_ms if slam_worker else 0.0)
                    cv2.imshow(WINDOW_TITLE, panel)
                    cv2.setWindowTitle(WINDOW_TITLE, f"VisionFSD Pi Robot - {planner.state.lower()}")
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        break
            time.sleep(max(0.0, PLAN_PERIOD_S - (time.monotonic() - now)))
    finally:
        arduino.close()
        lidar.close()
        camera.close()
        if slam_worker is not None:
            slam_worker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
