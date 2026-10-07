"""Robot controls for an AI pilot running in the Claude desktop app.

An optional experiment, separate from the robot's autonomous runtime. The pilot
(Claude Haiku 5.5 in the Claude desktop app) runs these commands from this
computer; each one connects to the robot's dashboard, does one thing, prints
what the robot now senses, and exits. Standard library only.

    python haiku_pilot/robot.py status
    python haiku_pilot/robot.py observe
    python haiku_pilot/robot.py manual on
    python haiku_pilot/robot.py drive forward 1.0 0.5 --say "heading for the doorway"
    python haiku_pilot/robot.py turn left 0.5 0.5
    python haiku_pilot/robot.py stop
    python haiku_pilot/robot.py manual off

Moves go through the dashboard's Manual Control, exactly like the phone's
arrow buttons, so every manual-mode check on the robot still applies: forward
is refused when the ultrasonic or LiDAR sees something close ahead, the Uno's
18 cm stop and command timeout sit beneath that, and a held command expires
within 0.35 s if this program stops sending it. This program adds what manual
mode does not check - it will not reverse into something behind - and caps
every move. A person pressing STOP or turning Manual Control off on the
dashboard ends any move at once, and no command here will take control back.
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

DRIVE_MAX_S = 2.0
TURN_MAX_S = 1.5
HOLD_PERIOD_S = 0.05          # the dashboard refreshes a held button this often
SETTLE_S = 0.5                # let the chassis stop and a fresh scan arrive
FORWARD_MIN_M = 0.40          # this tool's forward gate (the robot's own is 0.32 m)
ULTRASONIC_MIN_CM = 30.0
REVERSE_MIN_M = 0.30          # manual mode does not check behind; this tool does
LANE_HALF_WIDTH_M = 0.15      # the chassis is 0.23 m wide
BODY_OVERHANG_M = 0.13        # the chassis extends this far ahead of and behind the LiDAR
TELEMETRY_WAIT_S = 4.0
TELEMETRY_STALE_S = 1.5

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
                        if "scan" in message:
                            self.full = message
                            self.full_count += 1
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

    def wait_for_fresh_scan(self, after_count: int = 0, timeout: float = TELEMETRY_WAIT_S) -> None:
        """Wait for a scan published after ``after_count`` scans were seen.

        The robot answers a new connection with its last cached telemetry,
        which can be a couple of seconds old when it is busy. Deciding whether
        a move is safe, or reporting what a move did, needs telemetry the
        robot published afterwards. Two scans past the mark guarantees one.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.full_count >= after_count + 2:
                    return
            if self.closed:
                break
            time.sleep(0.05)
        raise Override("no fresh LiDAR data from the robot; it may be busy or starting up")

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
    """One JPEG from the robot's webcam stream, or None.

    The robot only encodes camera frames while someone is watching; opening
    the stream counts as watching, so the first frame can take a moment.
    """
    deadline = time.monotonic() + timeout
    try:
        with urllib.request.urlopen(f"http://{address}/camera.mjpg", timeout=timeout) as stream:
            buffer = b""
            while time.monotonic() < deadline:
                chunk = stream.read(8192)
                if not chunk:
                    break
                buffer += chunk
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    return buffer[start:end + 2]
                if len(buffer) > 2_000_000:
                    buffer = b""
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- perception


def summarize(telemetry: dict | None) -> dict:
    """What the robot senses, in metres, in the robot's own frame."""
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
    ahead = [y for x, y in points if y > 0.0 and abs(x) <= LANE_HALF_WIDTH_M]
    behind = [-y for x, y in points if y < 0.0 and abs(x) <= LANE_HALF_WIDTH_M]
    if ahead:
        summary["lane_ahead_m"] = round(max(0.0, min(ahead) - BODY_OVERHANG_M), 2)
    if behind:
        summary["lane_behind_m"] = round(max(0.0, min(behind) - BODY_OVERHANG_M), 2)
    summary["range"] = (telemetry.get("health") or {}).get("range") or {}
    summary["reason"] = telemetry.get("reason")
    pose = telemetry.get("pose") or {}
    summary["pose"] = (pose.get("x"), pose.get("y"), pose.get("h"))
    return summary


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
        f"Clear in your own lane: ahead {metres(summary.get('lane_ahead_m'))}, "
        f"behind {metres(summary.get('lane_behind_m'))}",
        "Nearest LiDAR return by direction: " + ", ".join(
            f"{name} {metres(value)}" for name, value in summary.get("sectors", {}).items()
        ),
        "Ultrasonic straight ahead: " + ("no echo" if ultra is None else f"{ultra} cm"),
    ])


def check_move(kind: str, direction: str, summary: dict) -> str | None:
    """Why this tool refuses the move before asking the robot, or None."""
    rng = summary.get("range") or {}
    if kind == "drive" and direction == "forward":
        lane = summary.get("lane_ahead_m")
        if lane is not None and lane < FORWARD_MIN_M:
            return f"only {lane:.2f} m clear in your lane ahead (need {FORWARD_MIN_M:.2f} m)"
        ultra = rng.get("ultra_cm")
        if ultra is not None and ultra < ULTRASONIC_MIN_CM:
            return f"the ultrasonic sees something {ultra} cm ahead"
    if kind == "drive" and direction == "backward":
        lane = summary.get("lane_behind_m")
        if lane is None:
            return "no LiDAR returns behind, so reversing would be blind"
        if lane < REVERSE_MIN_M:
            return f"only {lane:.2f} m clear in your lane behind (need {REVERSE_MIN_M:.2f} m)"
    return None


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


def hold(link: RobotLink, command: str, power: float, seconds: float,
         clock=time.monotonic, sleep=time.sleep) -> str:
    """Hold a drive command like a pressed button. Always ends stopped."""
    started = clock()
    note = ""
    try:
        while clock() - started < seconds:
            check_override(link, clock)
            link.send({"type": "control", "drive": command, "mag": power})
            telemetry, _full, _at, _control = link.snapshot()
            if ((telemetry or {}).get("reason") or "").startswith("STOP:MANUAL_FORWARD_BLOCKED"):
                note = "The robot cut the move short: something is close ahead."
                break
            sleep(HOLD_PERIOD_S)
    finally:
        link.send({"type": "control", "drive": "STOP"})
    return note


def report(link: RobotLink, address: str, heading: str = "", after_count: int = 0) -> str:
    """Current senses as text, and the camera frame saved for viewing."""
    link.wait_for_fresh_scan(after_count)
    telemetry, full, _at, control = link.snapshot()
    summary = summarize(full or telemetry)
    lines = [heading] if heading else []
    lines.append(describe(summary, control))
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
        link.wait_for_fresh_scan()
        check_override(link)
        telemetry, full, _at, _control = link.snapshot()
        before = summarize(full or telemetry)
        limit = DRIVE_MAX_S if args.command == "drive" else TURN_MAX_S
        seconds = min(limit, max(0.1, args.seconds))
        power = min(1.0, max(0.0, args.power))
        refusal = check_move(args.command, args.direction, before)
        if refusal:
            print(report(link, args.robot, f"REFUSED, nothing moved: {refusal}. Choose a different move."))
            return 3
        note = args.say or f"{args.command} {args.direction} {seconds:.1f}s"
        link.send({"type": "pilot", "note": note[:240], "model": "claude-haiku-5-5"})
        cut_short = hold(link, COMMANDS[args.direction], power, seconds)
        time.sleep(SETTLE_S)
        stopped_at = link.full_count
        link.wait_for_fresh_scan(stopped_at)
        telemetry, full, _at, _control = link.snapshot()
        after = summarize(full or telemetry)
        heading = (
            f"Did: {args.command} {args.direction} for {seconds:.1f} s at power {power:.1f}. "
            + (cut_short + " " if cut_short else "")
            + pose_change(before.get("pose"), after.get("pose"))
        )
        print(report(link, args.robot, heading, after_count=stopped_at))
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
