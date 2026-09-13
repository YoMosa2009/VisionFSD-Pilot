"""View-only dashboard streaming for the Pi robot runtime.

The Pi normally renders its dashboard into a fullscreen OpenCV window, which
means watching the robot requires an HDMI monitor physically tethered to it.
This module mirrors that same rendered dashboard over HTTP so a phone or
laptop on the same network can watch it instead.

Design constraints, in priority order:

1. **Never interfere with motor control.**  The control loop only ever hands
   over a reference to an already-rendered frame under a short lock.  JPEG
   encoding, socket writes and client handling all happen on other threads,
   and every failure path is swallowed - a broken browser tab must never be
   able to stop the robot loop or crash the runtime.
2. **No accumulating backlog.**  Exactly one encoded frame is retained.  A
   client on slow Wi-Fi simply misses intermediate frames; frames are never
   queued per client, so memory and latency stay bounded.
3. **Honest staleness.**  The viewer is told the age of what it is looking
   at, so a frozen picture can never be mistaken for a live robot.

The stream is read-only: there are no control endpoints, and nothing here can
command the chassis.
"""

from __future__ import annotations

import http.server
import json
import socket
import socketserver
import threading
import time

import cv2
import numpy as np

DEFAULT_PORT = 8080
DEFAULT_FPS = 5.0
DEFAULT_QUALITY = 70
# The viewer marks the picture stale once the runtime has not published a new
# dashboard frame for this long. Well above the normal publish period, low
# enough that a hung render is obvious while walking beside the robot.
STALE_AFTER_S = 2.0
_BOUNDARY = "visionfsdframe"

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionFSD robot dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0a0f14; color: #e6eef5;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
           padding: 8px 12px; background: #0e1620; font-size: 14px; }
  #badge { padding: 2px 10px; border-radius: 999px; font-weight: 600;
           font-size: 12px; letter-spacing: 0.04em; }
  .live { background: #12482a; color: #7cf0a6; }
  .stale { background: #4d1d12; color: #ffb59b; }
  .lost { background: #4a1020; color: #ff9bb5; }
  #wrap { display: flex; justify-content: center; padding: 8px; }
  img { max-width: 100%; height: auto; image-rendering: auto;
        border-radius: 6px; background: #05080b; }
  .dim { opacity: 0.35; transition: opacity 0.3s; }
  .muted { color: #8fa3b5; }
</style>
</head>
<body>
<header>
  <strong>VisionFSD robot</strong>
  <span id="badge" class="stale">CONNECTING</span>
  <span id="detail" class="muted">waiting for first frame</span>
  <span id="version" class="muted"></span>
</header>
<div id="wrap"><img id="view" class="dim" alt="robot dashboard"></div>
<script>
(function () {
  var view = document.getElementById('view');
  var badge = document.getElementById('badge');
  var detail = document.getElementById('detail');
  var version = document.getElementById('version');
  var staleAfter = STALE_AFTER_PLACEHOLDER;

  function attach() {
    view.src = 'stream.mjpg?t=' + Date.now();
  }
  view.addEventListener('error', function () {
    badge.className = 'lost';
    badge.textContent = 'DISCONNECTED';
    detail.textContent = 'stream dropped, retrying';
    view.classList.add('dim');
    setTimeout(attach, 1500);
  });
  view.addEventListener('load', function () {
    view.classList.remove('dim');
  });

  function poll() {
    fetch('status.json', { cache: 'no-store' })
      .then(function (response) { return response.json(); })
      .then(function (status) {
        version.textContent = 'v' + status.version;
        if (!status.has_frame) {
          badge.className = 'stale';
          badge.textContent = 'NO FRAME';
          detail.textContent = 'runtime has not rendered a dashboard yet';
          return;
        }
        var age = status.frame_age_s;
        if (age > staleAfter) {
          badge.className = 'stale';
          badge.textContent = 'STALE';
          detail.textContent = 'last frame ' + age.toFixed(1) + 's ago';
          view.classList.add('dim');
        } else {
          badge.className = 'live';
          badge.textContent = 'LIVE';
          detail.textContent = age.toFixed(1) + 's behind, '
            + status.published_fps.toFixed(1) + ' fps';
          view.classList.remove('dim');
        }
      })
      .catch(function () {
        badge.className = 'lost';
        badge.textContent = 'OFFLINE';
        detail.textContent = 'cannot reach the robot';
        view.classList.add('dim');
      });
  }
  attach();
  poll();
  setInterval(poll, 1000);
})();
</script>
</body>
</html>
"""


class DashboardStream:
    """Hold the newest dashboard frame and encode it on a private thread."""

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        quality: int = DEFAULT_QUALITY,
        version: str = "",
    ) -> None:
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
        self._thread = threading.Thread(
            target=self._encode_loop, name="dashboard-encoder", daemon=True
        )
        self._thread.start()

    @property
    def publish_period_s(self) -> float:
        """How often the control loop needs to hand over a rendered frame."""
        return self._period_s

    def publish(self, frame: np.ndarray) -> None:
        """Accept the newest dashboard frame. Called from the control loop.

        This only stores a reference and returns, so the cost to the control
        loop is a lock acquisition; encoding happens on the encoder thread.
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
            # free to render whenever it likes without paying for JPEG work.
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
        }

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame_ready.notify_all()


class _DashboardHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    stream: DashboardStream

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

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                page = _PAGE.replace(
                    "STALE_AFTER_PLACEHOLDER", repr(STALE_AFTER_S)
                )
                self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/status.json":
                payload = json.dumps(self.stream.status()).encode("utf-8")
                self._send_bytes(payload, "application/json")
            elif path == "/frame.jpg":
                jpeg, _at = self.stream.latest()
                if jpeg is None:
                    self.send_error(503, "no dashboard frame yet")
                    return
                self._send_bytes(jpeg, "image/jpeg")
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self.send_error(404, "not found")
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return
        except Exception:
            # A viewer must never be able to take the runtime down.
            try:
                self.send_error(500, "dashboard stream error")
            except Exception:
                return

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )
        self.end_headers()
        last_at = 0.0
        while True:
            jpeg, frame_at = self.stream.wait_for_frame(last_at, timeout=1.0)
            if jpeg is None:
                # Nothing rendered yet. Keep the connection open so the page
                # does not flap between reconnect attempts.
                continue
            if frame_at <= last_at:
                # Timed out waiting for something new. Resend the last frame
                # so an idle TCP connection cannot look identical to a dead
                # one; the page's own status poll reports the real age.
                pass
            last_at = frame_at
            header = (
                f"--{_BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode("ascii")
            self.wfile.write(header)
            self.wfile.write(jpeg)
            self.wfile.write(b"\r\n")


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # A slow or vanished client must not hold a worker thread forever.
    timeout = 5.0


class DashboardWebServer:
    """Serve the rendered dashboard read-only over HTTP."""

    def __init__(
        self,
        stream: DashboardStream,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
    ) -> None:
        self.stream = stream
        self.host = host
        self.port = port
        handler = type(
            "BoundDashboardHandler", (_DashboardHandler,), {"stream": stream}
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
) -> tuple[DashboardStream, DashboardWebServer] | None:
    """Start streaming, or return None if the port cannot be bound.

    A dashboard viewer is a convenience. Failing to bind (port in use, no
    network yet) must degrade to "no remote view", never to "no robot".
    """
    stream = DashboardStream(fps=fps, quality=quality, version=version)
    try:
        server = DashboardWebServer(stream, host=host, port=port)
    except OSError:
        stream.close()
        return None
    return stream, server
