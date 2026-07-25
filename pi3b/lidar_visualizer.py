#!/usr/bin/env python3
"""Live, read-only LD19 360 degree point-cloud visualizer.

The LD19 continuously emits UART packets at 230400 baud.  This program only
reads those packets; it sends no motion, configuration, or motor commands.
It runs on Raspberry Pi OS and Windows, which makes it useful for first
testing the sensor on a laptop before mounting it on the Pi.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


PACKET_HEADER = 0x54
PACKET_VER_LEN = 0x2C
POINT_COUNT = 12
PACKET_SIZE = 47
DEFAULT_BAUD = 230400
WINDOW_TITLE = "VisionFSD LD19 - read only"


@dataclass(frozen=True)
class LidarPoint:
    """One polar LD19 measurement, with angle zero at the sensor front."""

    angle_deg: float
    distance_mm: int
    confidence: int
    captured_at: float


class LD19Parser:
    """Incremental parser for the documented 47-byte LD19 UART packet."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.packets = 0
        self.crc_errors = 0

    @staticmethod
    def crc8(payload: bytes) -> int:
        """LD19 CRC-8: poly 0x4D, init 0, no reflection/final xor."""
        crc = 0
        for byte in payload:
            crc ^= byte
            for _ in range(8):
                crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        return crc

    def feed(self, raw: bytes, captured_at: float | None = None) -> list[LidarPoint]:
        if raw:
            self._buffer.extend(raw)
        now = time.monotonic() if captured_at is None else captured_at
        result: list[LidarPoint] = []
        while True:
            start = self._buffer.find(bytes((PACKET_HEADER, PACKET_VER_LEN)))
            if start < 0:
                # Preserve a possible header byte at the end for the next read.
                self._buffer[:] = self._buffer[-1:] if self._buffer[-1:] == bytes((PACKET_HEADER,)) else b""
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < PACKET_SIZE:
                break
            packet = bytes(self._buffer[:PACKET_SIZE])
            del self._buffer[:PACKET_SIZE]
            if self.crc8(packet[:-1]) != packet[-1]:
                self.crc_errors += 1
                # The data may have been shifted by a dropped byte. Re-scan all
                # but the first byte for the next real packet header.
                self._buffer[:0] = packet[1:]
                continue
            self.packets += 1
            speed_dps = int.from_bytes(packet[2:4], "little")
            start_cd = int.from_bytes(packet[4:6], "little")
            end_cd = int.from_bytes(packet[42:44], "little")
            span_cd = (end_cd - start_cd) % 36000
            for index in range(POINT_COUNT):
                offset = 6 + index * 3
                distance = int.from_bytes(packet[offset:offset + 2], "little")
                confidence = packet[offset + 2]
                angle_cd = (start_cd + span_cd * index / (POINT_COUNT - 1)) % 36000
                result.append(LidarPoint(angle_cd / 100.0, distance, confidence, now))
            # Keep the latest speed available without changing the public point type.
            self.speed_dps = speed_dps
        return result


def discover_port() -> str | None:
    """Pick a common USB-UART adapter, without guessing an arbitrary COM port."""
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is missing. Install pi3b/requirements.txt first.") from exc
    preferred = ("CP210", "SILICON LABS", "CH340", "CH341", "FTDI", "USB SERIAL", "UART")
    ports = list(list_ports.comports())
    for item in ports:
        description = f"{item.description} {item.manufacturer or ''}".upper()
        if any(name in description for name in preferred):
            return item.device
    return ports[0].device if len(ports) == 1 else None


def open_serial(port_name: str, baud: int):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is missing. Install pi3b/requirements.txt first.") from exc
    return serial.Serial(port_name, baudrate=baud, timeout=0.05)


def _draw_grid(panel: np.ndarray, center: tuple[int, int], radius: int, max_range_m: float) -> None:
    cx, cy = center
    for fraction in (0.25, 0.5, 0.75, 1.0):
        ring = int(radius * fraction)
        colour = (38, 57, 72) if fraction < 1.0 else (65, 90, 110)
        cv2.circle(panel, center, ring, colour, 1, cv2.LINE_AA)
        label = f"{max_range_m * fraction:.0f}m"
        cv2.putText(panel, label, (cx + 5, cy - ring - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    for angle in range(0, 360, 30):
        radians = math.radians(angle)
        x = int(cx + math.sin(radians) * radius)
        y = int(cy - math.cos(radians) * radius)
        cv2.line(panel, center, (x, y), (34, 49, 62), 1, cv2.LINE_AA)
    cv2.putText(panel, "FRONT", (cx - 20, max(18, cy - radius - 11)), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (210, 225, 235), 1, cv2.LINE_AA)
    cv2.arrowedLine(panel, (cx, cy), (cx, cy - min(radius, 42)), (235, 245, 250), 2, cv2.LINE_AA, tipLength=0.28)


def render(points: list[LidarPoint], parser: LD19Parser, max_range_m: float, size: int) -> np.ndarray:
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    panel[:] = (12, 21, 29)
    margin = 54
    center = (size // 2, size // 2 + 18)
    radius = max(80, min(center[0] - margin, center[1] - margin, size - center[1] - margin))
    _draw_grid(panel, center, radius, max_range_m)
    now = time.monotonic()
    visible = 0
    for point in points:
        if now - point.captured_at > 1.2 or point.distance_mm <= 0:
            continue
        distance_m = point.distance_mm / 1000.0
        if distance_m > max_range_m:
            continue
        radians = math.radians(point.angle_deg)
        scale = distance_m / max_range_m
        x = int(center[0] + math.sin(radians) * radius * scale)
        y = int(center[1] - math.cos(radians) * radius * scale)
        # Near objects are warmer/brighter. Confidence controls the blue channel.
        proximity = int(255 * (1.0 - scale))
        colour = (min(255, 60 + point.confidence), 90 + proximity // 2, proximity)
        cv2.circle(panel, (x, y), 2 if point.confidence > 80 else 1, colour, -1, cv2.LINE_AA)
        visible += 1
    cv2.circle(panel, center, 10, (210, 230, 242), -1, cv2.LINE_AA)
    cv2.putText(panel, "VisionFSD LD19 360-degree point cloud", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (225, 235, 242), 1, cv2.LINE_AA)
    status = f"POINTS {visible}  PACKETS {parser.packets}  CRC ERRORS {parser.crc_errors}"
    cv2.putText(panel, status, (14, size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (155, 184, 201), 1, cv2.LINE_AA)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only LD19 360-degree visualizer")
    parser.add_argument("--port", default="auto", help="Serial port, e.g. COM4 or /dev/ttyUSB0 (default: auto)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--max-range", type=float, default=12.0, help="Radar display radius in metres")
    parser.add_argument("--size", type=int, default=720, help="Square window size in pixels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_range <= 0 or args.size < 240:
        raise ValueError("--max-range must be positive and --size must be at least 240")
    port_name = discover_port() if args.port.lower() == "auto" else args.port
    if not port_name:
        print("No usable USB serial port found. Check the LD19 USB-UART driver and run with --port COMx.", file=sys.stderr)
        return 2
    try:
        device = open_serial(port_name, args.baud)
    except Exception as exc:
        print(f"Could not open LD19 on {port_name}: {exc}", file=sys.stderr)
        return 2
    print(f"Reading LD19 from {port_name} at {args.baud} baud. Press Q or Esc to quit.")
    parser = LD19Parser()
    points: deque[LidarPoint] = deque(maxlen=5400)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            chunk = device.read(max(1, device.in_waiting))
            points.extend(parser.feed(chunk))
            image = render(list(points), parser, args.max_range, args.size)
            cv2.imshow(WINDOW_TITLE, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        device.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
