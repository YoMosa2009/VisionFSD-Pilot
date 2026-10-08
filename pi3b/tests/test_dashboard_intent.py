"""Regression tests for the dashboard intent overlay and the map reset.

The dashboard previously showed only the raw policy reason and a single
target marker, so watching the robot gave no way to tell where it believed it
was going. These cover the route polyline, the applied-steering indicator and
the intent line, plus the map reset that a displacement triggers.
"""

from __future__ import annotations

import math
import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar_visualizer import LidarPoint
from robot_autonomy import (
    _INTENT_COLOR,
    ArduinoStatus,
    AutonomousPolicy,
    SectorClearance,
    draw_dashboard,
)
from robot_explorer import FrontierExplorer
from robot_imu import IMUState
from robot_slam_lite import LidarSlamLite


def _room_scan(radius_m: float = 2.4, count: int = 360, captured_at: float = 0.0):
    return [
        (
            index,
            LidarPoint(
                angle_deg=index * (360.0 / count),
                distance_mm=int(radius_m * 1000.0),
                confidence=200,
                captured_at=captured_at,
            ),
        )
        for index in range(count)
    ]


class RenderIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = LidarSlamLite(cells=120, metres=6.0)
        self.mapper.update(_room_scan(captured_at=1.0), 0, 0, 1.0, None, None, 1.0, None)

    def test_render_without_a_plan_still_works(self) -> None:
        panel = self.mapper.render(size=240)
        self.assertEqual(panel.shape, (240, 240, 3))

    def test_route_polyline_is_drawn(self) -> None:
        plain = self.mapper.render(size=240)
        path = tuple(
            (self.mapper.x + 0.1 * step, self.mapper.y - 0.1 * step)
            for step in range(12)
        )
        with_path = self.mapper.render(size=240, path_xy=path)

        self.assertFalse(np.array_equal(plain, with_path))

    def test_a_single_point_route_is_not_drawn(self) -> None:
        plain = self.mapper.render(size=240)
        single = self.mapper.render(
            size=240, path_xy=((self.mapper.x, self.mapper.y),)
        )
        self.assertTrue(np.array_equal(plain, single))

    def test_applied_steering_indicator_is_drawn(self) -> None:
        plain = self.mapper.render(size=240)
        steered = self.mapper.render(size=240, steering_deg=35.0)
        self.assertFalse(np.array_equal(plain, steered))

    def test_route_far_outside_the_viewport_does_not_raise(self) -> None:
        far = tuple((900.0 + step, -900.0 - step) for step in range(6))
        panel = self.mapper.render(size=240, path_xy=far)
        self.assertEqual(panel.shape, (240, 240, 3))


class MapResetTests(unittest.TestCase):
    def test_reset_clears_the_map_and_recentres_the_pose(self) -> None:
        """A displaced chassis cannot relate its old map frame to the new
        one, so the map is rebuilt from where it now stands."""
        mapper = LidarSlamLite(cells=120, metres=6.0)
        mapper.update(_room_scan(captured_at=1.0), 0, 0, 1.0, None, None, 1.0, None)
        mapper.x += 0.8
        mapper.heading = 55.0
        self.assertGreater(int(np.count_nonzero(mapper.observed)), 0)

        mapper.reset()

        self.assertEqual(int(np.count_nonzero(mapper.observed)), 0)
        self.assertEqual(int(np.count_nonzero(mapper.grid)), 0)
        self.assertEqual(int(np.count_nonzero(mapper.visits)), 0)
        self.assertAlmostEqual(mapper.x, mapper.metres / 2.0)
        self.assertAlmostEqual(mapper.y, mapper.metres / 2.0)
        self.assertEqual(mapper.heading, 0.0)
        self.assertEqual(mapper.state().map_updates, 0)

    def test_the_mapper_keeps_working_after_a_reset(self) -> None:
        mapper = LidarSlamLite(cells=120, metres=6.0)
        mapper.update(_room_scan(captured_at=1.0), 0, 0, 1.0, None, None, 1.0, None)
        mapper.reset()
        mapper.update(_room_scan(captured_at=2.0), 0, 0, 2.0, None, None, 1.0, None)

        self.assertGreater(int(np.count_nonzero(mapper.observed)), 0)


class ExplorationPathTests(unittest.TestCase):
    def test_planned_route_is_exposed_in_map_metres(self) -> None:
        mapper = LidarSlamLite(cells=160, metres=6.0)
        explorer = FrontierExplorer()
        now = time.monotonic()
        state = None
        for index in range(30):
            tick = now + index * 0.1
            mapper.update(
                _room_scan(captured_at=tick), 0, 0, tick, None, None, 1.0, None
            )
            state = explorer.update(
                mapper.grid,
                mapper.observed,
                mapper.visits,
                mapper.x,
                mapper.y,
                mapper.heading,
                mapper.metres,
                mapper.state().map_updates,
                now + index * 0.1,
            )
        self.assertIsNotNone(state)
        if not state.active:
            self.skipTest("no reachable frontier in this synthetic room")
        self.assertGreaterEqual(len(state.path_xy_m), 2)
        for x_m, y_m in state.path_xy_m:
            self.assertTrue(0.0 <= x_m <= mapper.metres)
            self.assertTrue(0.0 <= y_m <= mapper.metres)

    def test_route_ends_at_the_selected_target(self) -> None:
        mapper = LidarSlamLite(cells=160, metres=6.0)
        explorer = FrontierExplorer()
        now = time.monotonic()
        state = None
        for index in range(30):
            tick = now + index * 0.1
            mapper.update(
                _room_scan(captured_at=tick), 0, 0, tick, None, None, 1.0, None
            )
            state = explorer.update(
                mapper.grid, mapper.observed, mapper.visits,
                mapper.x, mapper.y, mapper.heading, mapper.metres,
                mapper.state().map_updates, now + index * 0.1,
            )
        if state is None or not state.active or not state.path_xy_m:
            self.skipTest("no reachable frontier in this synthetic room")
        end_x, end_y = state.path_xy_m[-1]
        self.assertLess(
            math.hypot(end_x - state.target_x_m, end_y - state.target_y_m),
            0.20,
        )

    def test_no_plan_means_an_empty_route(self) -> None:
        explorer = FrontierExplorer()
        empty = np.zeros((120, 120), dtype=np.uint8)
        state = explorer.update(
            empty, empty, empty.astype(np.uint16),
            3.0, 3.0, 0.0, 6.0, 0, time.monotonic(),
        )
        self.assertEqual(state.path_xy_m, ())


class DashboardIntentTests(unittest.TestCase):
    def _panel(self, policy: AutonomousPolicy) -> np.ndarray:
        mapper = LidarSlamLite(cells=120, metres=6.0)
        mapper.update(_room_scan(captured_at=1.0), 0, 0, 1.0, None, None, 1.0, None)
        return draw_dashboard(
            mapper.render(size=480),
            policy,
            SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0),
            ArduinoStatus(front_cm=90.0, motion="S", received_at=time.monotonic()),
            False,
            True,
            True,
            IMUState(connected=True, calibrated=True, fresh=True),
            mapper.state(),
        )

    def test_dashboard_renders_with_the_intent_row(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        policy.decide(
            SectorClearance(2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=1.0),
            ArduinoStatus(front_cm=None, motion="S", received_at=time.monotonic()),
            False,
            time.monotonic(),
        )
        panel = self._panel(policy)
        self.assertEqual(panel.shape[2], 3)
        # The header band grew to fit the intent row.
        self.assertTrue(np.any(panel[185:205, :20] != 0))

    def test_every_intent_label_has_a_colour(self) -> None:
        for label in AutonomousPolicy.INTENT_LABELS.values():
            self.assertIn(label, _INTENT_COLOR)
        self.assertIn("REORIENTING", _INTENT_COLOR)

    def test_dashboard_survives_an_unmapped_intent(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        policy.intent = "SOMETHING NEW"
        policy.intent_detail = "unmapped"
        panel = self._panel(policy)
        self.assertEqual(panel.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
