from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_visualizer import LD19Parser, PACKET_SIZE, POINT_COUNT


def packet(start_cd: int, end_cd: int) -> bytes:
    raw = bytearray((0x54, 0x2C))
    raw.extend((0x10, 0x27))  # 10000 degrees/second
    raw.extend(start_cd.to_bytes(2, "little"))
    for index in range(POINT_COUNT):
        raw.extend((1000 + index * 100).to_bytes(2, "little"))
        raw.append(90 + index)
    raw.extend(end_cd.to_bytes(2, "little"))
    raw.extend((0, 0))  # Timestamp is not required by this visualizer.
    raw.append(LD19Parser.crc8(raw))
    assert len(raw) == PACKET_SIZE
    return bytes(raw)


class LD19ProtocolTests(unittest.TestCase):
    def test_decodes_twelve_points_and_interpolates_angles(self) -> None:
        parser = LD19Parser()
        points = parser.feed(packet(35000, 1000), captured_at=12.0)
        self.assertEqual(len(points), POINT_COUNT)
        self.assertEqual(parser.packets, 1)
        self.assertAlmostEqual(points[0].angle_deg, 350.0)
        self.assertAlmostEqual(points[-1].angle_deg, 10.0)
        self.assertEqual(points[0].distance_mm, 1000)
        self.assertEqual(points[-1].confidence, 101)

    def test_rejects_bad_crc_then_resynchronizes(self) -> None:
        parser = LD19Parser()
        broken = bytearray(packet(0, 1100))
        broken[-1] ^= 0xFF
        points = parser.feed(bytes(broken) + packet(1200, 2300), captured_at=12.0)
        self.assertEqual(parser.crc_errors, 1)
        self.assertEqual(parser.packets, 1)
        self.assertEqual(len(points), POINT_COUNT)
        self.assertAlmostEqual(points[0].angle_deg, 12.0)


if __name__ == "__main__":
    unittest.main()
