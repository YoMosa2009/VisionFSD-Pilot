"""Robot controls for an AI pilot running in the Claude desktop app.

An optional experiment, separate from the robot's autonomous runtime. The pilot
(Claude Haiku 5.5 in the Claude desktop app) runs these commands from this
computer; each one connects to the robot's dashboard, does one thing, prints
what the robot now senses, and exits.

    python haiku_pilot/robot.py status
    python haiku_pilot/robot.py observe
    python haiku_pilot/robot.py manual on
    python haiku_pilot/robot.py drive forward 1.0 0.5 --say "heading for the doorway"
    python haiku_pilot/robot.py turn left 0.5 0.5
    python haiku_pilot/robot.py stop
    python haiku_pilot/robot.py manual off

Moves go through the dashboard's Manual Control, exactly like the phone's
arrow buttons. At the operator's request there is no proximity limit in
Manual Control (robot v1.9.31 onwards): the robot may drive right up to, and
into, things, and this program does not refuse close moves either. What still
stops it is what stops a robot nobody controls: STOP or turning Manual
Control off on the dashboard ends any move at once, a held command expires
within 0.35 s if this program stops sending it, and every move is capped.
The Arduino's own 18 cm ultrasonic stop straight ahead also remains, because
only reflashing its firmware can remove it; a move it holds back is reported.

Each report also draws a top-down picture of the LiDAR scan. A model reads the
shape of a room - openings, doorways, a gap 30 degrees to the left - far more
reliably from a picture than from hundreds of coordinates. Drawing it uses
numpy and OpenCV from the project's .venv; everything else is the standard
library, and without them the report simply has no picture.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import urllib.request

DEFAULT_ROBOT = os.environ.get("VISIONFSD_ROBOT", "192.168.0.17:8080")
VIEW_DIR = os.path.join(tempfile.gettempdir(), "visionfsd_pilot")
VIEW_PATH = os.path.join(VIEW_DIR, "latest_view.jpg")
LIDAR_PATH = os.path.join(VIEW_DIR, "lidar_topdown.png")
LIDAR_VIEW_RADIUS_M = 3.0
LIDAR_VIEW_PX = 600
# A corridor counts as an opening when the robot's own width (plus margin)
# is clear this far along it.
OPENING_MIN_M = 0.8
CORRIDOR_HALF_WIDTH_M = 0.16

DRIVE_MAX_S = 2.0
TURN_MAX_S = 1.5
HOLD_PERIOD_S = 0.05          # the dashboard refreshes a held button this often
SETTLE_S = 0.5                # let the chassis stop and a fresh scan arrive
LANE_HALF_WIDTH_M = 0.15      # the chassis is 0.23 m wide
BODY_OVERHANG_M = 0.13        # the chassis extends this far ahead of and behind the LiDAR
TELEMETRY_WAIT_S = 4.0
# When the robot is busy it sends LiDAR points only every couple of seconds
# (full telemetry is shed to protect driving), while the light messages -
# including the range readings manual driving is gated on - keep arriving
# several times a second. Safety decisions use the light readings; the map
# and directions use the newest scan, whose age the report states.
SCAN_WAIT_S = 10.0
# Telemetry arrives every ~0.36 s from an idle robot and gaps of 0.9 s are
# normal (measured 2026-10-08); 1.5 s ended moves on ordinary Wi-Fi or CPU
# hiccups, which the pilot reported as "LiDAR dropouts". The robot itself
# stops within 0.35 s of this program going quiet, so waiting longer here
# costs no safety.
TELEMETRY_STALE_S = 3.0
# The Arduino ramps motor power up by 4 PWM every 20 ms, so from a standstill
# the wheels need about half a second to reach driving power, plus one robot
# control tick to pick the command up. A move's seconds are counted from when
# the wheels are actually driving - seen in telemetry, or after this long.
# Before, a 0.6 s forward command moved ~1 cm and short turns did nothing.
SPINUP_MAX_S = 0.7
DRIVING_PWM = 100

SECTORS = (
    ("ahead", 0.0), ("ahead-right", 45.0), ("right", 90.0), ("behind-right", 135.0),
    ("behind", 180.0), ("behind-left", -135.0), ("left", -90.0), ("ahead-left", -45.0),
)
COMMANDS = {"forward": "F", "backward": "B", "left": "L", "right": "R"}


class Override(Exception):
    """A person took over, or the robot could not be reached."""


# --------------------------------------------------------------------------- robot link


def _masked(data: bytes) -> bytes:
    key = os.urandom(4)
    return key + bytes(byte ^ key[index % 4] for index, byte in enumerate(data))


class RobotLink:
    """WebSocket client for the robot dashboard: telemetry in, control out.

    Closing the connection makes the robot stop any held command.
    """

    def __init__(self, address: str) -> None:
        host, _, port = address.partition(":")
        self.host, self.port = host, int(port or 8080)
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=5.0)
        except OSError as error:
            raise Override(f"cannot reach the robot at {address}: {error}") from error
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self.telemetry: dict | None = None
        self.full: dict | None = None
        self.telemetry_at = 0.0
        self.control: dict = {}
        self.full_count = 0
        self.light_count = 0
        self.full_at = 0.0
        self._direct_control = False
        self.closed = False
        self._buffer = bytearray(self._handshake())
        self._sock.settimeout(None)
        threading.Thread(target=self._read_loop, name="pilot-ws", daemon=True).start()
        self.send({"type": "subscribe", "level": "full"})

    def _handshake(self) -> bytes:
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall(
            f"GET /ws HTTP/1.1\r\nHost: {self.host}:{self.port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise Override("the robot closed the connection during the handshake")
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        if b" 101" not in head.split(b"\r\n")[0]:
            raise Override("the robot dashboard did not accept the connection")
        return rest

    def send(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        header = bytearray([0x81])
        if len(data) < 126:
            header.append(0x80 | len(data))
        elif len(data) < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", len(data))
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", len(data))
        with self._send_lock:
            self._sock.sendall(bytes(header) + _masked(data))

    def _need(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise OSError("connection closed")
            self._buffer += chunk
        out = bytes(self._buffer[:count])
        del self._buffer[:count]
        return out

    def _read_loop(self) -> None:
        try:
            while True:
                first, second = self._need(2)
                opcode, length = first & 0x0F, second & 0x7F
                if length == 126:
                    length = struct.unpack(">H", self._need(2))[0]
                elif length == 127:
                    length = struct.unpack(">Q", self._need(8))[0]
                mask = self._need(4) if second & 0x80 else b""
                payload = self._need(length)
                if mask:
                    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
                if opcode == 0x8:
                    break
                if opcode != 0x1:
                    continue
                try:
                    message = json.loads(payload)
                except ValueError:
                    continue
                with self._lock:
                    if message.get("type") == "hello" and isinstance(message.get("control"), dict):
                        self.control = message["control"]
                        self._direct_control = True
                    elif message.get("type") == "control":
                        # Sent the moment the control state changes - STOP
                        # pressed, Manual toggled - so it is authoritative.
                        self.control = {k: v for k, v in message.items() if k != "type"}
                        self._direct_control = True
                    elif "health" in message:
                        self.telemetry = message
                        self.light_count += 1
                        if "scan" in message:
                            self.full = message
                            self.full_count += 1
                            self.full_at = time.monotonic()
                        self.telemetry_at = time.monotonic()
                        # Telemetry also carries the control state, but it can
                        # be a moment older than a direct control message; use
                        # it only until the first one arrives.
                        if not self._direct_control and isinstance(message.get("control"), dict):
                            self.control = message["control"]
        except OSError:
            pass
        finally:
            self.closed = True

    def snapshot(self) -> tuple[dict | None, dict | None, float, dict]:
        with self._lock:
            return self.telemetry, self.full, self.telemetry_at, dict(self.control)

    def marks(self) -> tuple[int, int]:
        with self._lock:
            return self.light_count, self.full_count

    def wait_for_fresh(self, after: tuple[int, int] = (0, 0), want_new_scan: bool = False,
                       timeout: float = TELEMETRY_WAIT_S, scan_timeout: float | None = None) -> bool:
        """Wait for telemetry the robot published after the ``after`` marks.

        Fresh light telemetry is required: it carries the range readings
        every safety decision uses, and the robot sends it several times a
        second. The robot answers a new connection with its last cached
        message, so two past the mark guarantee one fresh one.

        A LiDAR scan is only wanted, never required: a busy robot can go
        several seconds without sending one, and the scan only feeds the map
        and directions. Returns whether a scan newer than the mark arrived.
        """
        light_mark, full_mark = after
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self.light_count >= light_mark + 2:
                    break
            if self.closed or time.monotonic() >= deadline:
                raise Override("no fresh telemetry from the robot; it may be busy or starting up")
            time.sleep(0.05)
        if not want_new_scan:
            return False
        deadline = time.monotonic() + (SCAN_WAIT_S if scan_timeout is None else scan_timeout)
        while time.monotonic() < deadline and not self.closed:
            with self._lock:
                if self.full_count >= full_mark + 1:
                    return True
            time.sleep(0.05)
        return False

    def scan_age(self) -> float | None:
        with self._lock:
            return None if not self.full_at else time.monotonic() - self.full_at

    def wait_for_telemetry(self, need_scan: bool = True, timeout: float = TELEMETRY_WAIT_S) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            telemetry, full, _at, _control = self.snapshot()
            if (full if need_scan else telemetry) is not None:
                return
            if self.closed:
                break
            time.sleep(0.05)
        telemetry, _full, _at, _control = self.snapshot()
        if telemetry is None:
            raise Override("no telemetry from the robot; it may still be starting up")

    def wait_for_control(self, predicate, timeout: float = 2.0) -> bool:
        """Wait until the robot's control state satisfies predicate."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self.snapshot()[3]):
                return True
            if self.closed:
                return False
            time.sleep(0.05)
        return False

    def close(self) -> None:
        """Close without resetting the connection.

        Closing a socket that still has unread telemetry waiting makes some
        systems reset the connection, and the robot can then lose the last
        command sent. Half-close first and let the robot finish.
        """
        try:
            self._sock.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + 0.5
            while not self.closed and time.monotonic() < deadline:
                time.sleep(0.02)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


def fetch_camera_frame(address: str, timeout: float = 4.0) -> bytes | None:
    """One current JPEG from the robot's webcam stream, or None.

    The stream's first frame is the robot's cached last frame - often from
    the previous command, before the robot moved. The robot only encodes
    frames while someone watches, so the cache is not refreshed between
    commands. Skip it and return the next frame, published after this
    connection opened.
    """
    deadline = time.monotonic() + timeout
    frames = 0
    try:
        with urllib.request.urlopen(f"http://{address}/camera.mjpg", timeout=timeout) as stream:
            buffer = b""
            while time.monotonic() < deadline:
                chunk = stream.read(8192)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start < 0 or end < 0:
                        break
                    frame = buffer[start:end + 2]
                    buffer = buffer[end + 2:]
                    frames += 1
                    if frames >= 2:
                        return frame
                if len(buffer) > 2_000_000:
                    buffer = b""
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- perception


def summarize(telemetry: dict | None, fresh: dict | None = None) -> dict:
    """What the robot senses, in metres, in the robot's own frame.

    ``telemetry`` supplies the LiDAR points (a full message); ``fresh`` is
    the newest message of any kind, whose range readings are combined with
    the scan so the lane distances are never older than the freshest data.
    """
    summary: dict = {"sectors": {}, "lane_ahead_m": None, "lane_behind_m": None, "range": {}}
    if not telemetry:
        return summary
    scan = telemetry.get("scan") or {}
    points = [
        (x / 100.0, y / 100.0)
        for x, y in zip(scan.get("x", []), scan.get("y", []))
        if math.hypot(x, y) >= 5
    ]
    for name, centre in SECTORS:
        nearest = None
        for x, y in points:
            bearing = math.degrees(math.atan2(x, y))
            if abs((bearing - centre + 180.0) % 360.0 - 180.0) <= 22.5:
                distance = math.hypot(x, y)
                nearest = distance if nearest is None else min(nearest, distance)
        summary["sectors"][name] = None if nearest is None else round(nearest, 2)
    ahead = [(y, x) for x, y in points if y > 0.0 and abs(x) <= LANE_HALF_WIDTH_M]
    behind = [(-y, x) for x, y in points if y < 0.0 and abs(x) <= LANE_HALF_WIDTH_M]
    for name, lane in (("ahead", ahead), ("behind", behind)):
        if not lane:
            continue
        nearest, side = min(lane)
        summary[f"lane_{name}_m"] = round(max(0.0, nearest - BODY_OVERHANG_M), 2)
        # Something nearly as close on both sides of centre is a wall or
        # edge across the lane, not one thing at a corner.
        close = [x for distance, x in lane if distance <= nearest + 0.03]
        spans = min(close) <= -0.04 and max(close) >= 0.04
        summary[f"lane_{name}_side_cm"] = "across" if spans else round(side * 100)
    summary["lanes_from_scan"] = bool(points)
    latest = fresh or telemetry
    summary["range"] = (latest.get("health") or {}).get("range") or {}
    if not points:
        # No scan to measure the lane from: fall back to the robot's own
        # corridor readings, which include a 5.5 cm margin each side.
        for lane_key, range_key in (("lane_ahead_m", "front_m"), ("lane_behind_m", "rear_m")):
            reading = summary["range"].get(range_key)
            if reading is not None:
                summary[lane_key] = round(reading, 2)
    summary["reason"] = latest.get("reason")
    pose = latest.get("pose") or {}
    summary["pose"] = (pose.get("x"), pose.get("y"), pose.get("h"))
    return summary


def _where(side_cm) -> str:
    """Where across the robot's width the nearest thing in the lane is."""
    if side_cm is None:
        return ""
    if side_cm == "across":
        return " (across your whole width)"
    if abs(side_cm) <= 4:
        return " (dead centre)"
    corner = abs(side_cm) >= 9
    side = "right" if side_cm > 0 else "left"
    return f" ({'at your ' + side + ' corner' if corner else 'slightly ' + side}, {abs(side_cm)} cm off centre)"


def describe(summary: dict, control: dict) -> str:
    def metres(value) -> str:
        return "no return" if value is None else f"{value:.2f} m"

    rng = summary.get("range") or {}
    ultra = rng.get("ultra_cm")
    mode = (
        "STOPPED by a person on the dashboard" if control.get("halted")
        else "Manual Control (you may drive)" if control.get("manual")
        else "autonomous (run `manual on` before driving)"
    )
    return "\n".join([
        f"Mode: {mode}",
        f"Clear in your own lane: ahead {metres(summary.get('lane_ahead_m'))}"
        + _where(summary.get("lane_ahead_side_cm"))
        + f", behind {metres(summary.get('lane_behind_m'))}"
        + _where(summary.get("lane_behind_side_cm")),
        "Nearest LiDAR return by direction: " + ", ".join(
            f"{name} {metres(value)}" for name, value in summary.get("sectors", {}).items()
        ),
        "Ultrasonic straight ahead: " + ("no echo" if ultra is None else f"{ultra} cm")
        + " (a narrow beam at its own height; it can disagree with the LiDAR lane"
        + " when something is off-centre or above/below the LiDAR's scan)",
    ] + _low_object_warning(summary.get("lane_ahead_m"), ultra))


def _low_object_warning(lane_ahead_m, ultra_cm) -> list[str]:
    """Flag something the ultrasonic sees ahead that the LiDAR does not.

    The LiDAR scans one flat slice of the room; a short object (a bottle, a
    toy, a shoe) sits entirely below it. On 2026-10-08 the LiDAR lane read
    1.68 m clear with a 10 cm bottle 35 cm ahead; only the ultrasonic and the
    camera saw it.
    """
    if ultra_cm is None:
        return []
    if lane_ahead_m is not None and ultra_cm / 100.0 >= lane_ahead_m - 0.15:
        return []
    return [
        f"WARNING: the ultrasonic sees something {ultra_cm} cm ahead that the LiDAR does not"
        " - probably a low object below the LiDAR's scan. Check the camera before driving forward."
    ]


def _points(telemetry: dict | None, key: str = "scan") -> list[tuple[float, float]]:
    block = (telemetry or {}).get(key) or {}
    return [
        (x / 100.0, y / 100.0)
        for x, y in zip(block.get("x", []), block.get("y", []))
        if math.hypot(x, y) >= 5
    ]


def corridor_clearance(points: list[tuple[float, float]], bearing_deg: float) -> float:
    """Free distance from the bumper along a robot-wide corridor at a bearing.

    Bearing 0 is straight ahead, positive to the right, negative to the left.
    """
    radians = math.radians(bearing_deg)
    ux, uy = math.sin(radians), math.cos(radians)
    nearest = math.inf
    for x, y in points:
        along = x * ux + y * uy
        if along <= 0.0:
            continue
        if abs(x * uy - y * ux) <= CORRIDOR_HALF_WIDTH_M:
            nearest = min(nearest, along)
    return max(0.0, nearest - BODY_OVERHANG_M)


def find_openings(points: list[tuple[float, float]], step_deg: int = 5) -> list[dict]:
    """The standout directions to drive: peaks in robot-wide clear distance.

    A threshold alone is useless in an open room - every direction clears
    it, and the answer is one 360-degree "opening". What matters is where the
    space runs on furthest: a doorway, a hallway, the long side of a room. So
    this finds local peaks of corridor clearance around the robot, keeps those
    at least OPENING_MIN_M long, drops weaker peaks within 30 degrees of a
    stronger one, and reports each with the span over which the clearance
    stays within 70% of its peak without rising into a clearer direction.
    """
    if not points:
        return []
    bearings = list(range(-180, 180, step_deg))
    count = len(bearings)
    clear = [min(corridor_clearance(points, bearing), 9.9) for bearing in bearings]
    if max(clear) - min(clear) < 1e-9:
        # Everywhere the same: no direction stands out.
        peaks = [(count // 2, clear[0])] if clear[0] >= OPENING_MIN_M else []
    else:
        # Walk the circle starting at the least clear direction, so a peak or
        # plateau can never straddle the starting point.
        origin = clear.index(min(clear))
        order = [(origin + offset) % count for offset in range(count)]
        peaks = []
        position = 1
        while position < count:
            run_end = position
            while run_end + 1 < count and clear[order[run_end + 1]] == clear[order[position]]:
                run_end += 1
            value = clear[order[position]]
            before = clear[order[position - 1]]
            after = clear[order[(run_end + 1) % count]]
            if value >= OPENING_MIN_M and value > before and value > after:
                peaks.append((order[(position + run_end) // 2], value))
            position = run_end + 1
    peaks.sort(key=lambda item: -item[1])
    chosen: list[tuple[int, float]] = []
    for peak_index, value in peaks:
        separation = min(
            (abs((bearings[peak_index] - bearings[other] + 180) % 360 - 180) for other, _v in chosen),
            default=360,
        )
        if separation >= 30:
            chosen.append((peak_index, value))
    openings = []
    for peak_index, value in chosen:
        span = 1
        for direction in (-1, 1):
            step = 1
            while step < count and 0.7 * value <= clear[(peak_index + direction * step) % count] <= value:
                span += 1
                step += 1
        openings.append({
            "bearing": bearings[peak_index],
            "span": min(360, span * step_deg),
            "clear_m": round(value, 2),
        })
    return openings


def describe_openings(openings: list[dict]) -> str:
    if not openings:
        return "Open corridors wide enough for the robot: none within sensing range"
    parts = []
    for item in openings[:4]:
        bearing = item["bearing"]
        if bearing == 0:
            side = "straight ahead"
        elif bearing > 0:
            side = f"{bearing} deg right"
        else:
            side = f"{-bearing} deg left"
        if abs(bearing) >= 135:
            side += " (behind you)"
        clear = "over 9.9 m" if item["clear_m"] >= 9.9 else f"{item['clear_m']:.1f} m"
        parts.append(f"{side}: {clear} clear, {item['span']} deg wide")
    return (
        "Open corridors wide enough for the robot (0 = ahead, right/left = which way to turn): "
        + "; ".join(parts)
    )


def draw_lidar(telemetry: dict | None, summary: dict, openings: list[dict],
               path: str = LIDAR_PATH) -> str | None:
    """Top-down picture of the scan, robot in the middle facing up.

    Returns the saved path, or None when numpy/OpenCV are unavailable.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    size = LIDAR_VIEW_PX
    centre = size // 2
    scale = (size / 2 - 20) / LIDAR_VIEW_RADIUS_M
    image = np.full((size, size, 3), 24, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def pixel(x: float, y: float) -> tuple[int, int]:
        return int(round(centre + x * scale)), int(round(centre - y * scale))

    # Distance rings every 0.5 m, labelled every metre.
    for tenths in range(5, int(LIDAR_VIEW_RADIUS_M * 10) + 1, 5):
        radius = tenths / 10.0
        cv2.circle(image, (centre, centre), int(radius * scale), (70, 70, 70), 1, cv2.LINE_AA)
        if tenths % 10 == 0:
            cv2.putText(image, f"{radius:.0f} m", (centre + 4, centre - int(radius * scale) + 14),
                        font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(image, (centre, 0), (centre, size), (50, 50, 50), 1)
    cv2.line(image, (0, centre), (size, centre), (50, 50, 50), 1)
    # The lane the robot would drive straight through, green up to the first
    # thing in it.
    lane = summary.get("lane_ahead_m")
    lane_end = min((lane if lane is not None else LIDAR_VIEW_RADIUS_M) + BODY_OVERHANG_M, LIDAR_VIEW_RADIUS_M)
    overlay = image.copy()
    cv2.rectangle(overlay, pixel(-LANE_HALF_WIDTH_M, lane_end), pixel(LANE_HALF_WIDTH_M, BODY_OVERHANG_M),
                  (60, 150, 60), -1)
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    # Open corridors as arrows, labelled with the turn that faces them.
    for item in openings[:4]:
        radians = math.radians(item["bearing"])
        reach = min(item["clear_m"] + BODY_OVERHANG_M, LIDAR_VIEW_RADIUS_M - 0.25)
        tip = pixel(math.sin(radians) * reach, math.cos(radians) * reach)
        cv2.arrowedLine(image, (centre, centre), tip, (40, 210, 250), 2, cv2.LINE_AA, tipLength=0.06)
        bearing = item["bearing"]
        label = "ahead" if bearing == 0 else (f"R{bearing}" if bearing > 0 else f"L{-bearing}")
        cv2.putText(image, label, (tip[0] + 4, tip[1] - 4), font, 0.45, (40, 210, 250), 1, cv2.LINE_AA)
    # Remembered obstacles from the last few seconds, dim grey.
    for x, y in _points(telemetry, "mem"):
        if math.hypot(x, y) <= LIDAR_VIEW_RADIUS_M:
            cv2.circle(image, pixel(x, y), 1, (110, 110, 110), -1)
    # Live LiDAR returns, coloured by distance from the robot.
    for x, y in _points(telemetry, "scan"):
        distance = math.hypot(x, y)
        if distance > LIDAR_VIEW_RADIUS_M:
            continue
        if distance < 0.5:
            colour = (60, 60, 255)
        elif distance < 1.0:
            colour = (60, 170, 255)
        else:
            colour = (235, 235, 235)
        cv2.circle(image, pixel(x, y), 2, colour, -1)
    # The robot to scale - 23 cm wide, 27 cm long, LiDAR at its centre - with
    # an arrow showing which way it faces.
    cv2.rectangle(image, pixel(-0.115, 0.135), pixel(0.115, -0.135), (255, 160, 60), 2)
    cv2.arrowedLine(image, (centre, centre), pixel(0.0, 0.30), (255, 160, 60), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(image, "AHEAD", (centre - 26, 16), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "LEFT", (6, centre - 6), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "RIGHT", (size - 56, centre - 6), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "BEHIND", (centre - 30, size - 8), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    legend = (
        ("blue box: the robot, arrow = facing", (255, 160, 60)),
        ("green: clear lane straight ahead", (90, 190, 90)),
        ("yellow arrows: open corridors (R/L deg)", (40, 210, 250)),
        ("dots: LiDAR - red <0.5 m, orange <1 m", (60, 170, 255)),
        ("grey dots: recently seen, out of view", (130, 130, 130)),
    )
    for row, (text, colour) in enumerate(legend):
        cv2.putText(image, text, (8, 18 + row * 16), font, 0.4, colour, 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path if cv2.imwrite(path, image) else None


def pose_change(before, after) -> str:
    try:
        moved = math.hypot(after[0] - before[0], after[1] - before[1])
        turned = (after[2] - before[2] + 180.0) % 360.0 - 180.0
    except (TypeError, IndexError):
        return "Movement estimate unavailable."
    return f"Estimated change: moved {moved:.2f} m, turned {turned:+.0f} degrees (approximate)."


# --------------------------------------------------------------------------- actions


def check_override(link: RobotLink, clock=time.monotonic) -> None:
    _telemetry, _full, at, control = link.snapshot()
    if control.get("halted"):
        raise Override("STOP was pressed on the dashboard - a person has taken over")
    if control and not control.get("manual"):
        raise Override("Manual Control is off - the robot is not yours to drive right now")
    if link.closed:
        raise Override("the connection to the robot was lost")
    if at and clock() - at > TELEMETRY_STALE_S:
        raise Override("telemetry from the robot stopped arriving")


def _wheels_driving(telemetry: dict | None, command: str) -> bool:
    """Whether the Uno reports the wheels at driving power for this command."""
    actual = ((telemetry or {}).get("drive") or {}).get("act") or [0, 0]
    try:
        left, right = int(actual[0]), int(actual[1])
    except (TypeError, ValueError, IndexError):
        return False
    wanted = {"F": (1, 1), "B": (-1, -1), "L": (-1, 1), "R": (1, -1)}.get(command)
    if wanted is None:
        return False
    return (left * wanted[0] >= DRIVING_PWM) and (right * wanted[1] >= DRIVING_PWM)


def hold(link: RobotLink, command: str, power: float, seconds: float,
         clock=time.monotonic, sleep=time.sleep) -> tuple[str, float]:
    """Hold a drive command like a pressed button. Always ends stopped.

    ``seconds`` is time with the wheels actually driving: the clock starts
    when telemetry shows the wheels at driving power, or SPINUP_MAX_S after
    the first command, whichever comes first. Returns (note, spin-up seconds).
    """
    started = clock()
    driving_since = None
    note = ""
    try:
        while True:
            now = clock()
            if driving_since is not None and now - driving_since >= seconds:
                break
            check_override(link, clock)
            link.send({"type": "control", "drive": command, "mag": power})
            telemetry, _full, _at, _control = link.snapshot()
            if driving_since is None and (
                _wheels_driving(telemetry, command) or now - started >= SPINUP_MAX_S
            ):
                driving_since = now
            reason = (telemetry or {}).get("reason") or ""
            rng = ((telemetry or {}).get("health") or {}).get("range") or {}
            if reason.startswith("STOP:MANUAL_FORWARD_BLOCKED"):
                # Robots before v1.9.31 still refuse close forward moves.
                note = "The robot cut the move short: something is close ahead."
                break
            if command == "F" and rng.get("uno_blocked"):
                note = ("The Arduino's 18 cm ultrasonic stop is holding the robot: "
                        "something is under 18 cm straight ahead.")
                break
            sleep(HOLD_PERIOD_S)
    finally:
        link.send({"type": "control", "drive": "STOP"})
    spinup = (driving_since - started) if driving_since is not None else clock() - started
    return note, spinup


def report(link: RobotLink, address: str, heading: str = "",
           after: tuple[int, int] = (0, 0), new_scan: bool = True) -> str:
    """Current senses as text, and the camera frame saved for viewing."""
    link.wait_for_fresh(after, want_new_scan=new_scan)
    telemetry, full, _at, control = link.snapshot()
    summary = summarize(full or telemetry, fresh=telemetry)
    openings = find_openings(_points(full or telemetry))
    lines = [heading] if heading else []
    lines.append(describe(summary, control))
    lines.append(describe_openings(openings))
    picture = draw_lidar(full, summary, openings) if full is not None else None
    age = link.scan_age()
    if full is None:
        lines.append(
            "LiDAR map: no scan received yet (the robot sends one every few seconds "
            "when busy). The lane and range numbers above are current."
        )
    elif picture:
        lines.append(
            f"LiDAR map saved: {picture}  (top-down, robot in the middle facing up; open it)"
            + ("" if age is None or age < 1.0 else f" - scan is {age:.1f} s old")
        )
    else:
        lines.append("LiDAR map: not drawn (numpy/OpenCV missing - use the project's .venv Python)")
    frame = fetch_camera_frame(address)
    if frame:
        os.makedirs(VIEW_DIR, exist_ok=True)
        with open(VIEW_PATH, "wb") as handle:
            handle.write(frame)
        lines.append(f"Camera image saved: {VIEW_PATH}  (open it to see what the robot sees)")
    else:
        lines.append("Camera image: not available right now.")
    return "\n".join(lines)


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="robot.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--robot", default=DEFAULT_ROBOT, help="dashboard host:port")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="mode and version, no camera")
    sub.add_parser("observe", help="what the robot senses now, plus a camera image")
    manual = sub.add_parser("manual", help="take (on) or hand back (off) Manual Control")
    manual.add_argument("state", choices=("on", "off"))
    for name, choices, limit in (("drive", ("forward", "backward"), DRIVE_MAX_S),
                                 ("turn", ("left", "right"), TURN_MAX_S)):
        move = sub.add_parser(name, help=f"{name} for up to {limit} s, then stop and observe")
        move.add_argument("direction", choices=choices)
        move.add_argument("seconds", type=float)
        move.add_argument("power", type=float, nargs="?", default=0.5)
        move.add_argument("--say", default="", help="short note shown on the dashboard")
    sub.add_parser("stop", help="stop immediately")
    args = parser.parse_args(argv)

    try:
        link = RobotLink(args.robot)
    except Override as error:
        print(f"ERROR: {error}")
        return 2
    try:
        if args.command == "stop":
            link.send({"type": "control", "drive": "STOP"})
            link.wait_for_telemetry(need_scan=False)
            print("Stopped.")
            return 0
        if args.command == "status":
            link.wait_for_telemetry(need_scan=False)
            telemetry, _full, _at, control = link.snapshot()
            print(f"Robot v{telemetry.get('v')} reachable at {args.robot}.")
            print(describe(summarize(None), control).splitlines()[0])
            print(f"Robot reports: {telemetry.get('reason')}")
            return 0
        if args.command == "observe":
            print(report(link, args.robot))
            return 0
        if args.command == "manual":
            link.wait_for_telemetry(need_scan=False)
            _telemetry, _full, _at, control = link.snapshot()
            if args.state == "on":
                if control.get("halted"):
                    print("REFUSED: the robot is STOPPED from the dashboard. Ask the person to press Resume.")
                    return 3
                link.send({"type": "control", "manual": True})
                link.send({"type": "pilot", "note": "taking Manual Control", "model": "claude-haiku-5-5"})
                if not link.wait_for_control(lambda state: state.get("manual") is True):
                    print("ERROR: the robot did not confirm Manual Control. Try `manual on` again.")
                    return 2
                print(report(link, args.robot, "Manual Control is on. The robot will only move when you tell it to."))
            else:
                link.send({"type": "control", "manual": False})
                link.send({"type": "pilot", "note": ""})
                if not link.wait_for_control(lambda state: state.get("manual") is False):
                    print("ERROR: the robot did not confirm. Try `manual off` again.")
                    return 2
                print("Manual Control is off. The robot is driving itself again.")
            return 0

        # drive / turn
        link.wait_for_fresh()
        check_override(link)
        telemetry, full, _at, _control = link.snapshot()
        before = summarize(full or telemetry, fresh=telemetry)
        limit = DRIVE_MAX_S if args.command == "drive" else TURN_MAX_S
        seconds = min(limit, max(0.1, args.seconds))
        power = min(1.0, max(0.0, args.power))
        note = args.say or f"{args.command} {args.direction} {seconds:.1f}s"
        link.send({"type": "pilot", "note": note[:240], "model": "claude-haiku-5-5"})
        cut_short, spinup = hold(link, COMMANDS[args.direction], power, seconds)
        time.sleep(SETTLE_S)
        stopped_at = link.marks()
        link.wait_for_fresh(stopped_at, want_new_scan=True)
        telemetry, full, _at, _control = link.snapshot()
        after = summarize(full or telemetry, fresh=telemetry)
        heading = (
            f"Did: {args.command} {args.direction} for {seconds:.1f} s at power {power:.1f} "
            f"(wheels took {spinup:.1f} s to reach driving power first). "
            + (cut_short + " " if cut_short else "")
            + pose_change(before.get("pose"), after.get("pose"))
        )
        # The wait for a post-move scan already happened above.
        print(report(link, args.robot, heading, after=stopped_at, new_scan=False))
        return 4 if cut_short else 0
    except Override as error:
        try:
            link.send({"type": "control", "drive": "STOP"})
        except OSError:
            pass
        print(f"STOPPED: {error}. Do not move again until the person says so.")
        return 5
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
