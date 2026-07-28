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


@dataclass(frozen=True)
class ObstacleCluster:
    """A contiguous group of fresh range returns, not an object class."""

    angle_deg: float
    distance_mm: int
    point_count: int


class LivePolarMap:
    """Newest measurement per one-degree direction, with no slow history trail."""

    def __init__(self, bin_count: int = 360) -> None:
        self._bin_count = bin_count
        self._points: list[LidarPoint | None] = [None] * bin_count

    def update(self, points: list[LidarPoint], min_confidence: int,
               min_range_mm: int, max_range_mm: int) -> int:
        accepted = 0
        for point in points:
            if not (min_range_mm <= point.distance_mm <= max_range_mm) or point.confidence < min_confidence:
                continue
            index = int(round(point.angle_deg * self._bin_count / 360.0)) % self._bin_count
            previous = self._points[index]
            # Newest wins, and returns from the same sweep are broken toward the
            # nearer one.  The old rule also let a higher-confidence *older*
            # return replace a fresh one, which can hide an object that has just
            # moved into that direction.
            if (previous is None
                    or point.captured_at > previous.captured_at
                    or (point.captured_at == previous.captured_at
                        and point.distance_mm < previous.distance_mm)):
                self._points[index] = point
            accepted += 1
        return accepted

    def fresh(self, now: float, persistence_s: float) -> list[tuple[int, LidarPoint]]:
        return [
            (index, point)
            for index, point in enumerate(self._points)
            if point is not None and now - point.captured_at <= persistence_s
        ]


def obstacle_clusters(fresh: list[tuple[int, LidarPoint]], bin_count: int,
                      minimum_points: int = 3) -> list[ObstacleCluster]:
    """Group adjacent fresh returns and reject isolated speckle for display."""
    if not fresh:
        return []
    by_index = dict(fresh)
    groups: list[list[tuple[int, LidarPoint]]] = []
    current: list[tuple[int, LidarPoint]] = []
    previous_index: int | None = None
    previous_distance: int | None = None
    for index in sorted(by_index):
        point = by_index[index]
        adjacent = previous_index is not None and index - previous_index <= 2
        similar_range = previous_distance is not None and abs(point.distance_mm - previous_distance) <= max(
            250, int(min(point.distance_mm, previous_distance) * 0.28)
        )
        if current and not (adjacent and similar_range):
            groups.append(current)
            current = []
        current.append((index, point))
        previous_index, previous_distance = index, point.distance_mm
    if current:
        groups.append(current)
    if len(groups) > 1 and groups[0][0][0] <= 1 and groups[-1][-1][0] >= bin_count - 2:
        edge_a, edge_b = groups[-1][-1][1], groups[0][0][1]
        if abs(edge_a.distance_mm - edge_b.distance_mm) <= max(250, int(min(edge_a.distance_mm, edge_b.distance_mm) * 0.28)):
            groups[0] = groups[-1] + groups[0]
            groups.pop()
    result: list[ObstacleCluster] = []
    for group in groups:
        if len(group) < minimum_points:
            continue
        distances = sorted(point.distance_mm for _, point in group)
        middle = group[len(group) // 2][1]
        result.append(ObstacleCluster(middle.angle_deg, distances[len(distances) // 2], len(group)))
    return sorted(result, key=lambda item: item.distance_mm)


class LD19Parser:
    """Incremental parser for the documented 47-byte LD19 UART packet."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.packets = 0
        self.crc_errors = 0
        self.speed_dps = 0

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
    return serial.Serial(port_name, baudrate=baud, timeout=0.01)


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


def render(fresh: list[tuple[int, LidarPoint]], clusters: list[ObstacleCluster],
           parser: LD19Parser, max_range_m: float, persistence_s: float, size: int) -> np.ndarray:
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    panel[:] = (12, 21, 29)
    margin = 54
    center = (size // 2, size // 2 + 18)
    radius = max(80, min(center[0] - margin, center[1] - margin, size - center[1] - margin))
    _draw_grid(panel, center, radius, max_range_m)
    visible = 0
    for _index, point in fresh:
        distance_m = point.distance_mm / 1000.0
        radians = math.radians(point.angle_deg)
        scale = distance_m / max_range_m
        x = int(center[0] + math.sin(radians) * radius * scale)
        y = int(center[1] - math.cos(radians) * radius * scale)
        # Near objects are warmer/brighter. Confidence controls the blue channel.
        proximity = int(255 * (1.0 - scale))
        colour = (min(255, 60 + point.confidence), 90 + proximity // 2, proximity)
        cv2.circle(panel, (x, y), 2 if point.confidence > 80 else 1, colour, -1, cv2.LINE_AA)
        visible += 1
    for cluster in clusters[:6]:
        distance_m = cluster.distance_mm / 1000.0
        radians = math.radians(cluster.angle_deg)
        x = int(center[0] + math.sin(radians) * radius * (distance_m / max_range_m))
        y = int(center[1] - math.cos(radians) * radius * (distance_m / max_range_m))
        cv2.circle(panel, (x, y), 8, (235, 235, 90), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{distance_m:.1f}m", (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (225, 235, 145), 1, cv2.LINE_AA)
    cv2.circle(panel, center, 10, (210, 230, 242), -1, cv2.LINE_AA)
    cv2.putText(panel, "VisionFSD LD19 360-degree point cloud", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (225, 235, 242), 1, cv2.LINE_AA)
    status = (f"LIVE {visible}  CLUSTERS {len(clusters)}  SCAN {parser.speed_dps / 360.0:.1f}Hz  "
              f"HOLD {persistence_s * 1000:.0f}ms  CRC {parser.crc_errors}")
    cv2.putText(panel, status, (14, size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (155, 184, 201), 1, cv2.LINE_AA)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only LD19 360-degree visualizer")
    parser.add_argument("--port", default="auto", help="Serial port, e.g. COM4 or /dev/ttyUSB0 (default: auto)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--max-range", type=float, default=12.0, help="Radar display radius in metres")
    parser.add_argument("--size", type=int, default=720, help="Square window size in pixels")
    parser.add_argument("--persistence", type=float, default=0.30,
                        help="Seconds a direction remains visible after its latest return")
    parser.add_argument("--min-confidence", type=int, default=8,
                        help="Drop very weak intensity returns below this 0-255 value")
    parser.add_argument("--min-range-mm", type=int, default=80,
                        help="Drop near-sensor noise closer than this distance")
    parser.add_argument("--cluster-min-points", type=int, default=3,
                        help="Minimum nearby returns required for a displayed obstacle cluster")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_range <= 0 or args.size < 240 or args.persistence <= 0:
        raise ValueError("--max-range and --persistence must be positive and --size must be at least 240")
    if not (0 <= args.min_confidence <= 255) or args.min_range_mm < 0 or args.cluster_min_points < 1:
        raise ValueError("Invalid LiDAR quality-filter setting")
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
    polar_map = LivePolarMap()
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            chunk = device.read(max(1, device.in_waiting))
            polar_map.update(parser.feed(chunk), args.min_confidence, args.min_range_mm, int(args.max_range * 1000.0))
            now = time.monotonic()
            fresh = polar_map.fresh(now, args.persistence)
            clusters = obstacle_clusters(fresh, 360, args.cluster_min_points)
            image = render(fresh, clusters, parser, args.max_range, args.persistence, args.size)
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
