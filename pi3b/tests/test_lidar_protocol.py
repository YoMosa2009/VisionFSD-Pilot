from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_visualizer import LD19Parser, LidarPoint, LivePolarMap, PACKET_SIZE, POINT_COUNT, obstacle_clusters


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

    def test_live_map_replaces_old_direction_instead_of_leaving_a_trail(self) -> None:
        live_map = LivePolarMap()
        live_map.update([LidarPoint(10.0, 3000, 100, 1.0)], 8, 80, 12000)
        live_map.update([LidarPoint(10.2, 900, 120, 1.2)], 8, 80, 12000)
        fresh = live_map.fresh(1.25, 0.30)
        self.assertEqual(len(fresh), 1)
        self.assertEqual(fresh[0][1].distance_mm, 900)
        self.assertEqual(live_map.fresh(1.51, 0.30), [])

    def test_clusters_need_neighboring_returns_and_reject_speckle(self) -> None:
        points = [
            (10, LidarPoint(10.0, 1000, 90, 1.0)),
            (11, LidarPoint(11.0, 1010, 90, 1.0)),
            (12, LidarPoint(12.0, 1020, 90, 1.0)),
            (90, LidarPoint(90.0, 800, 120, 1.0)),
        ]
        clusters = obstacle_clusters(points, 360, minimum_points=3)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].point_count, 3)


if __name__ == "__main__":
    unittest.main()
