"""Phone and laptop dashboard for the Pi robot runtime.

Three views are served from one page on the robot's LAN address:

* **LiDAR** - a live visualiser drawn by the browser from raw telemetry: every
  LD19 return, the obstacle memory, the planned arc and route, moving objects
  with their predicted paths, and the chassis at true scale.
* **Camera** - the onboard USB webcam.
* **Dashboard** - the same rendered panel the HDMI monitor shows.

The LiDAR view is drawn client-side on purpose. Rendering and JPEG-encoding a
detailed picture is exactly the work a Pi 3B cannot spare, while the phone
showing it has a GPU sitting idle. The robot sends compact numbers; the phone
does the drawing. That is how the view can carry far more detail than the old
mirrored image and still cost the robot less.

Design constraints, in priority order:

1. **Never interfere with motor control.** The control loop only hands over
   references and small dicts under short locks. Encoding, socket writes and
   client handling all run on other threads, and every failure is swallowed:
   a broken browser tab must never stop the robot loop or crash the runtime.
2. **Do no work nobody is watching.** Every stream counts its viewers, and the
   runtime asks before rendering, encoding or building telemetry.
3. **No accumulating backlog.** Exactly one frame or telemetry message is
   retained per stream. Slow Wi-Fi drops intermediate updates; nothing queues.
4. **Honest staleness.** The viewer is always told how old what it sees is.

Control - halt, resume and manual driving - travels over a WebSocket so a held
button reaches the robot in tens of milliseconds, with an HTTP fallback. It
fails safe: manual commands expire on their own. It cannot authenticate, so the
port belongs on a trusted network only.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import math
from pathlib import Path
import select
import socket
import socketserver
import struct
import threading
import time

import cv2
import numpy as np


class RobotControl:
    """Operator halt and manual driving, shared with the control loop.

    Fails safe in every direction:

    * a manual command expires on its own after a fraction of a second, so a
      dropped phone, a closed tab or a walk out of Wi-Fi range stops the
      robot rather than leaving it driving;
    * leaving manual mode clears any held command, and so does a halt;
    * the halt is sticky and has to be released explicitly; and
    * nothing here can weaken the Uno's ultrasonic stop or the LD19 forward
      check in the policy, which both still apply to manual driving.
    """

    #: A held button is refreshed every 50 ms over the WebSocket. Several
    #: consecutive refreshes have to go missing before the robot stops on its
    #: own - long enough to ride out Wi-Fi jitter, short enough that a lost
    #: connection is not felt as the robot carrying on.
    COMMAND_TTL_S = 0.35

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._halted = False
        self._manual = False
        self._command = "STOP"
        self._magnitude = 0.0
        self._command_at = 0.0
        self.revision = 0

    @property
    def halted(self) -> bool:
        with self._lock:
            return self._halted

    @property
    def manual(self) -> bool:
        with self._lock:
            return self._manual

    def halt(self) -> None:
        with self._lock:
            self._halted = True
            self._command = "STOP"
            self._magnitude = 0.0
            self.revision += 1

    def resume(self) -> None:
        with self._lock:
            self._halted = False
            self.revision += 1

    def set_manual(self, enabled: bool) -> None:
        with self._lock:
            self._manual = bool(enabled)
            self._command = "STOP"
            self._magnitude = 0.0
            self._command_at = 0.0
            self.revision += 1

    def drive(self, command: str, magnitude: float = 1.0) -> bool:
        """Set the held manual command. ``magnitude`` is 0..1 of the range.

        The page raises magnitude the longer a button is held, so a tap is a
        small, precise nudge and a long press builds to full rate - which is
        what makes a turn proportional to the press instead of a fixed jump.
        """
        command = str(command).upper()
        if command not in ("F", "B", "L", "R", "STOP"):
            return False
        try:
            magnitude = float(magnitude)
        except (TypeError, ValueError):
            magnitude = 0.0
        if not math.isfinite(magnitude):
            magnitude = 0.0
        with self._lock:
            if not self._manual:
                return False
            self._command = command
            self._magnitude = 0.0 if command == "STOP" else min(1.0, max(0.0, magnitude))
            self._command_at = time.monotonic()
        return True

    def manual_command(self, now: float | None = None) -> str:
        """The command to apply now, or STOP once it has gone stale."""
        return self.manual_input(now)[0]

    def manual_input(self, now: float | None = None) -> tuple[str, float]:
        """(command, magnitude) to apply now; STOP once stale."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self._manual or self._command == "STOP":
                return "STOP", 0.0
            if now - self._command_at > self.COMMAND_TTL_S:
                return "STOP", 0.0
            return self._command, self._magnitude

    def state(self) -> dict:
        with self._lock:
            return {
                "halted": self._halted,
                "manual": self._manual,
                "command": self._command,
                "magnitude": round(self._magnitude, 2),
            }


DEFAULT_PORT = 8080
DEFAULT_FPS = 5.0
DEFAULT_QUALITY = 70
# The viewer marks the picture stale once the runtime has not published a new
# frame for this long. Well above the normal publish period, low enough that a
# hung render is obvious while walking beside the robot.
STALE_AFTER_S = 2.0
_BOUNDARY = "visionfsdframe"
_PAGE_PATH = Path(__file__).resolve().parent / "web" / "index.html"
_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_FALLBACK_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionFSD robot</title></head><body style="background:#0a0f14;color:#e6eef5;
font-family:system-ui,sans-serif"><h1>VisionFSD robot</h1>
<p>The dashboard page is missing from this install (pi3b/web/index.html).</p>
<p><button id="halt">STOP</button> <button id="manual">MANUAL CONTROL</button>
<button data-drive="F">F</button></p><img src="stream.mjpg" alt="dashboard">
</body></html>"""


def load_page() -> str:
    try:
        page = _PAGE_PATH.read_text(encoding="utf-8")
    except OSError:
        page = _FALLBACK_PAGE
    return page.replace("STALE_AFTER_PLACEHOLDER", repr(STALE_AFTER_S))


class DashboardStream:
    """Hold the newest image frame and encode it on a private thread."""

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
        version: str = "",
        control: RobotControl | None = None,
        name: str = "dashboard-encoder",
    ) -> None:
        self.control = control
        self.fps = max(0.5, min(15.0, fps))
        self.quality = int(np.clip(quality, 30, 95))
        self.version = version
        self._period_s = 1.0 / self.fps
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._pending_at = 0.0
        self._jpeg: bytes | None = None
        self._jpeg_at = 0.0
        self._published_at = 0.0
        self._encoded_count = 0
        self._encode_started_at = time.monotonic()
        self._frame_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._viewers = 0
        self._thread = threading.Thread(
            target=self._encode_loop, name=name, daemon=True
        )
        self._thread.start()

    @property
    def publish_period_s(self) -> float:
        """How often the control loop needs to hand over a rendered frame."""
        return self._period_s

    @property
    def viewers(self) -> int:
        """Clients currently streaming. Zero means no work is needed."""
        with self._lock:
            return self._viewers

    def _add_viewer(self, delta: int) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers + delta)

    def publish(self, frame: np.ndarray) -> None:
        """Accept the newest frame. Called from the control loop.

        Only stores a reference and returns, so the cost to the control loop
        is a lock acquisition; encoding happens on the encoder thread.
        Replacing an unencoded frame is intentional - the viewer always wants
        the newest picture, never a backlog of old ones.
        """
        now = time.monotonic()
        with self._lock:
            self._pending = frame
            self._pending_at = now
            self._published_at = now
            self._frame_ready.notify_all()

    def _encode_loop(self) -> None:
        encode_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        while not self._stop.is_set():
            with self._lock:
                while self._pending is None and not self._stop.is_set():
                    self._frame_ready.wait(0.5)
                if self._stop.is_set():
                    return
                frame = self._pending
                frame_at = self._pending_at
                self._pending = None
            try:
                encoded, buffer = cv2.imencode(".jpg", frame, encode_parameters)
            except Exception:
                encoded, buffer = False, None
            if encoded:
                with self._lock:
                    self._jpeg = buffer.tobytes()
                    self._jpeg_at = frame_at
                    self._encoded_count += 1
                    self._frame_ready.notify_all()
            # Pace encoding rather than the caller: the control loop stays
            # free to publish whenever it likes without paying for JPEG work.
            self._stop.wait(self._period_s)

    def latest(self) -> tuple[bytes | None, float]:
        with self._lock:
            return self._jpeg, self._jpeg_at

    def wait_for_frame(
        self, after: float, timeout: float = 1.0
    ) -> tuple[bytes | None, float]:
        """Block until an encoded frame newer than ``after`` exists."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._jpeg is None or self._jpeg_at <= after:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or self._stop.is_set():
                    return self._jpeg, self._jpeg_at
                self._frame_ready.wait(remaining)
            return self._jpeg, self._jpeg_at

    def status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            has_frame = self._jpeg is not None
            frame_age = now - self._jpeg_at if has_frame else 0.0
            elapsed = max(1e-3, now - self._encode_started_at)
            published_fps = self._encoded_count / elapsed
        return {
            "version": self.version,
            "has_frame": has_frame,
            "frame_age_s": round(frame_age, 2),
            "published_fps": round(published_fps, 2),
            "stale_after_s": STALE_AFTER_S,
            "stale": (not has_frame) or frame_age > STALE_AFTER_S,
            **(self.control.state() if self.control is not None else {}),
        }

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame_ready.notify_all()


class TelemetryHub:
    """The latest telemetry message, encoded once however many clients watch.

    Clients subscribe at one of two levels. ``full`` (the LiDAR tab) carries
    every scan point, the memory, the arc, the route and the tracks. ``light``
    (the camera and dashboard tabs) carries only what the on-screen status
    needs. The runtime asks which levels are wanted before building anything.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._seq = 0
        self._stamp = 0.0
        self._full: dict | None = None
        self._light: dict | None = None
        self._encoded: dict[tuple[int, str], bytes] = {}
        self._subscribers = {"full": 0, "light": 0}
        self._map_png: bytes | None = None
        self._map_seq = 0

    def wants(self, level: str) -> bool:
        with self._lock:
            if level == "light":
                return self._subscribers["light"] + self._subscribers["full"] > 0
            return self._subscribers.get(level, 0) > 0

    def _subscribe(self, old: str | None, new: str | None) -> None:
        with self._lock:
            if old in self._subscribers:
                self._subscribers[old] = max(0, self._subscribers[old] - 1)
            if new in self._subscribers:
                self._subscribers[new] += 1

    def publish(self, light: dict, full: dict | None = None) -> None:
        """Replace the latest message. ``full`` extends ``light``."""
        with self._lock:
            self._seq += 1
            self._stamp = time.monotonic()
            self._light = light
            self._full = full
            self._encoded.clear()
            self._ready.notify_all()

    def publish_map(self, png: bytes) -> None:
        with self._lock:
            self._map_png = png
            self._map_seq += 1

    def map_png(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._map_png, self._map_seq

    def wait(self, after_seq: int, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._ready.wait(remaining)
            return self._seq

    def encoded(self, level: str) -> tuple[int, bytes | None]:
        with self._lock:
            seq = self._seq
            key = (seq, level)
            if key in self._encoded:
                return seq, self._encoded[key]
            message = self._full if level == "full" and self._full is not None else self._light
            if message is None:
                return seq, None
            payload = dict(message)
            payload["type"] = "telemetry"
            payload["level"] = "full" if level == "full" and self._full is not None else "light"
            payload["age_ms"] = round((time.monotonic() - self._stamp) * 1000.0, 1)
            payload["map_seq"] = self._map_seq
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._encoded[key] = data
            return seq, data


# ---------------------------------------------------------------------------
# Minimal RFC 6455 WebSocket framing, enough for small JSON messages.


def websocket_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key.strip() + _WEBSOCKET_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """A single unmasked server-to-client frame."""
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", 0x80 | opcode, length)
    elif length < 65536:
        header = struct.pack("!BBH", 0x80 | opcode, 126, length)
    else:
        header = struct.pack("!BBQ", 0x80 | opcode, 127, length)
    return header + payload


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionResetError("websocket closed")
        data.extend(chunk)
    return bytes(data)


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    """Read one client frame. Returns (opcode, unmasked payload)."""
    first, second = _recv_exact(sock, 2)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", _recv_exact(sock, 2))
    elif length == 127:
        (length,) = struct.unpack("!Q", _recv_exact(sock, 8))
    if length > 65536:
        raise ConnectionResetError("websocket frame too large")
    mask = _recv_exact(sock, 4) if masked else b"\x00\x00\x00\x00"
    payload = bytearray(_recv_exact(sock, length))
    for index in range(length):
        payload[index] ^= mask[index % 4]
    return opcode, bytes(payload)


class _DashboardHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    stream: DashboardStream
    camera: DashboardStream | None = None
    telemetry: TelemetryHub | None = None
    page: str = _FALLBACK_PAGE

    def log_message(self, _format: str, *_args) -> None:
        # Per-request logging would interleave with the robot's own NAV
        # telemetry on stdout for every browser poll.
        return

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _discard_body(self) -> None:
        """Read an unwanted request body before answering.

        Replying without reading it leaves unread bytes in the socket, and
        some platforms then reset the connection before the client sees the
        response.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if 0 < length <= 65536:
            try:
                self.rfile.read(length)
            except OSError:
                pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._send_bytes(self.page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/ws":
                self._websocket()
            elif path == "/status.json":
                status = self.stream.status()
                if self.camera is not None:
                    status["camera_has_frame"] = self.camera.status()["has_frame"]
                payload = json.dumps(status).encode("utf-8")
                self._send_bytes(payload, "application/json")
            elif path == "/frame.jpg":
                self._single(self.stream, "no dashboard frame yet")
            elif path == "/stream.mjpg":
                self._stream(self.stream)
            elif path == "/camera.jpg":
                self._single(self.camera, "no camera frame yet")
            elif path == "/camera.mjpg":
                if self.camera is None:
                    self.send_error(503, "camera stream is not enabled")
                    return
                self._stream(self.camera)
            elif path == "/map.png":
                png = None if self.telemetry is None else self.telemetry.map_png()[0]
                if png is None:
                    self.send_error(503, "no map yet")
                    return
                self._send_bytes(png, "image/png")
            else:
                self.send_error(404, "not found")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            return
        except Exception:
            # A viewer must never be able to take the runtime down.
            try:
                self.send_error(500, "dashboard error")
            except Exception:
                return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?", 1)[0] != "/control":
            self._discard_body()
            self.send_error(404, "not found")
            return
        try:
            self._control()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            return
        except Exception:
            try:
                self.send_error(500, "control error")
            except Exception:
                return

    def _single(self, stream: DashboardStream | None, missing: str) -> None:
        jpeg = None if stream is None else stream.latest()[0]
        if jpeg is None:
            self.send_error(503, missing)
            return
        self._send_bytes(jpeg, "image/jpeg")

    @staticmethod
    def _apply_control(control: RobotControl, payload: dict) -> None:
        # A halt is honoured before anything else in the same message, and
        # leaving manual mode always clears whatever was held.
        if payload.get("halt"):
            control.halt()
        if payload.get("resume"):
            control.resume()
        if "manual" in payload:
            control.set_manual(bool(payload["manual"]))
        drive = payload.get("drive")
        if isinstance(drive, str):
            control.drive(drive, payload.get("mag", 1.0))

    def _control(self) -> None:
        control = getattr(self.stream, "control", None)
        length = int(self.headers.get("Content-Length") or 0)
        body = b""
        if 0 < length <= 4096:
            body = self.rfile.read(length)
        elif length > 4096:
            self._discard_body()
        if control is None:
            self.send_error(503, "control is not enabled")
            return
        payload: dict = {}
        try:
            payload = json.loads(body or b"{}")
        except (ValueError, UnicodeDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        self._apply_control(control, payload)
        self._send_bytes(
            json.dumps(control.state()).encode("utf-8"), "application/json"
        )

    def _stream(self, stream: DashboardStream) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )
        self.end_headers()
        stream._add_viewer(1)
        try:
            last_at = 0.0
            while True:
                jpeg, frame_at = stream.wait_for_frame(last_at, timeout=1.0)
                if jpeg is None:
                    # Nothing rendered yet. Keep the connection open so the
                    # page does not flap between reconnect attempts; the
                    # viewer count is already telling the runtime to render.
                    continue
                last_at = frame_at
                header = (
                    f"--{_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        finally:
            stream._add_viewer(-1)

    def _websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or "websocket" not in (self.headers.get("Upgrade") or "").lower():
            self.send_error(400, "expected a websocket upgrade")
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", websocket_accept_key(key))
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True
        sock = self.connection
        sock.settimeout(2.0)
        control: RobotControl | None = getattr(self.stream, "control", None)
        hub = self.telemetry
        level: str | None = None
        last_seq = 0
        last_control_revision = -1
        light_period = 0.25
        full_period = 0.10
        next_send_at = 0.0

        def send(message: dict) -> None:
            sock.sendall(encode_frame(json.dumps(message, separators=(",", ":")).encode("utf-8")))

        try:
            send({"type": "hello", "version": self.stream.version,
                  "control": None if control is None else control.state()})
            while True:
                readable, _w, _x = select.select([sock], [], [], 0.02)
                if readable:
                    opcode, payload = read_frame(sock)
                    if opcode == 0x8:
                        try:
                            sock.sendall(encode_frame(payload[:2], 0x8))
                        except OSError:
                            pass
                        return
                    if opcode == 0x9:
                        sock.sendall(encode_frame(payload, 0xA))
                        continue
                    if opcode != 0x1:
                        continue
                    try:
                        message = json.loads(payload.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if not isinstance(message, dict):
                        continue
                    kind = message.get("type")
                    if kind == "subscribe" and hub is not None:
                        wanted = message.get("level")
                        wanted = wanted if wanted in ("full", "light") else None
                        hub._subscribe(level, wanted)
                        level = wanted
                        next_send_at = 0.0
                    elif kind == "control" and control is not None:
                        self._apply_control(control, message)
                    elif kind == "ping":
                        send({"type": "pong", "c": message.get("c")})
                if control is not None and control.revision != last_control_revision:
                    last_control_revision = control.revision
                    send({"type": "control", **control.state()})
                now = time.monotonic()
                if hub is not None and level is not None and now >= next_send_at:
                    seq, data = hub.encoded(level)
                    if data is not None and seq != last_seq:
                        last_seq = seq
                        sock.sendall(encode_frame(data))
                        next_send_at = now + (full_period if level == "full" else light_period)
        except (OSError, ConnectionResetError, ValueError, struct.error):
            return
        finally:
            if hub is not None:
                hub._subscribe(level, None)
            if control is not None:
                # A closed control connection must not leave a held command
                # running until it happens to expire.
                control.drive("STOP")


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # A slow or vanished client must not hold a worker thread forever.
    timeout = 5.0


class DashboardWebServer:
    """Serve the dashboard, camera, telemetry and control over HTTP."""

    def __init__(
        self,
        stream: DashboardStream,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        camera: DashboardStream | None = None,
        telemetry: TelemetryHub | None = None,
    ) -> None:
        self.stream = stream
        self.camera = camera
        self.telemetry = telemetry
        self.host = host
        self.port = port
        handler = type(
            "BoundDashboardHandler",
            (_DashboardHandler,),
            {
                "stream": stream,
                "camera": camera,
                "telemetry": telemetry,
                "page": load_page(),
            },
        )
        self._server = _ThreadedHTTPServer((host, port), handler)
        self._server.socket.settimeout(5.0)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dashboard-web",
            kwargs={"poll_interval": 0.5},
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://{local_ip_address()}:{self.port}/"

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
        self.stream.close()
        if self.camera is not None:
            self.camera.close()


def local_ip_address() -> str:
    """Best-effort LAN address for printing a reachable URL in the log."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packet is sent; this only asks the routing table which local
        # interface would be used to reach off-link traffic.
        probe.connect(("192.0.2.1", 1))
        return str(probe.getsockname()[0])
    except Exception:
        return "127.0.0.1"
    finally:
        probe.close()


def start_dashboard_server(
    port: int = DEFAULT_PORT,
    fps: float = DEFAULT_FPS,
    quality: int = DEFAULT_QUALITY,
    version: str = "",
    host: str = "0.0.0.0",
    control: RobotControl | None = None,
    camera_fps: float | None = None,
    telemetry: TelemetryHub | None = None,
) -> tuple[DashboardStream, DashboardWebServer] | None:
    """Start serving, or return None if the port cannot be bound.

    A remote view is a convenience. Failing to bind (port in use, no network
    yet) must degrade to "no remote view", never to "no robot".
    """
    stream = DashboardStream(
        fps=fps, quality=quality, version=version, control=control
    )
    camera = (
        None
        if camera_fps is None
        else DashboardStream(
            fps=camera_fps, quality=quality, version=version,
            control=control, name="camera-encoder",
        )
    )
    try:
        server = DashboardWebServer(
            stream, host=host, port=port, camera=camera, telemetry=telemetry
        )
    except OSError:
        stream.close()
        if camera is not None:
            camera.close()
        return None
    return stream, server
