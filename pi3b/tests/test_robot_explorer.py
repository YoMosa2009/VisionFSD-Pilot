from __future__ import annotations

import math
import pathlib
import sys
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar_visualizer import LidarPoint
from robot_autonomy import (
    STEER_HEADINGS,
    ArduinoStatus,
    AutonomousPolicy,
    SectorClearance,
    _sector_clearance,
    corridor_profile,
)
from robot_explorer import ExplorationState, FrontierExplorer
from robot_imu import IMUState
from robot_slam_lite import LidarSlamLite


class FrontierExplorerTests(unittest.TestCase):
    def test_astar_abandons_work_after_control_loop_deadline(self) -> None:
        free = np.ones((220, 220), dtype=bool)
        free[:, 110] = False
        path = FrontierExplorer._astar(
            free,
            (110, 30),
            (110, 190),
            deadline=time.perf_counter() - 0.001,
        )
        self.assertIsNone(path)

    def test_cached_route_waypoint_advances_between_full_replans(self) -> None:
        free = np.ones((80, 80), dtype=bool)
        path = [(40, col) for col in range(10, 61)]
        first = FrontierExplorer._select_waypoint(path, (40, 10), 10.0, free)
        advanced = FrontierExplorer._select_waypoint(path, (40, 16), 10.0, free)
        self.assertIsNotNone(first)
        self.assertIsNotNone(advanced)
        self.assertGreater(advanced[1], first[1])

    def test_timed_out_replan_keeps_last_safe_route(self) -> None:
        cells = 120
        metres = 6.0
        grid = np.zeros((cells, cells), dtype=np.uint8)
        observed = np.zeros_like(grid)
        visits = np.zeros((cells, cells), dtype=np.uint16)
        cv2.circle(observed, (60, 60), 32, 255, -1)
        explorer = FrontierExplorer()
        first = explorer.update(
            grid, observed, visits, 3.0, 3.0, 0.0, metres, 10, 1.0
        )
        self.assertTrue(first.active)

        explorer.PLAN_TIME_BUDGET_S = 0.0
        retained = explorer.update(
            grid, observed, visits, 3.0, 3.0, 0.0, metres, 11, 2.0
        )
        self.assertTrue(retained.active)
        self.assertEqual(retained.target_x_m, first.target_x_m)
        self.assertEqual(retained.target_y_m, first.target_y_m)

    def test_pose_inside_inflation_reconnects_to_nearby_known_free_space(self) -> None:
        cells = 120
        metres = 6.0
        grid = np.zeros((cells, cells), dtype=np.uint8)
        observed = np.full_like(grid, 255)
        visits = np.zeros((cells, cells), dtype=np.uint16)
        grid[60, 63] = 255
        explorer = FrontierExplorer()
        state = explorer.update(
            grid, observed, visits, 3.0, 3.0, 0.0, metres, 10, 1.0
        )
        self.assertTrue(state.active)
        self.assertEqual(state.mode, "PATROL")

    def test_reachable_unknown_boundary_becomes_frontier_target(self) -> None:
        cells = 120
        metres = 6.0
        grid = np.zeros((cells, cells), dtype=np.uint8)
        observed = np.zeros_like(grid)
        visits = np.zeros((cells, cells), dtype=np.uint16)
        cv2.circle(observed, (60, 60), 32, 255, -1)

        explorer = FrontierExplorer()
        state = explorer.update(
            grid,
            observed,
            visits,
            x_m=3.0,
            y_m=3.0,
            heading_deg=0.0,
            metres=metres,
            map_updates=10,
            now=1.0,
        )

        self.assertTrue(state.active)
        self.assertEqual(state.mode, "FRONTIER")
        self.assertGreater(state.frontier_count, 0)
        self.assertGreater(state.target_distance_m, 1.0)
        self.assertIsNotNone(state.waypoint_x_m)
        self.assertIsNotNone(state.waypoint_y_m)

    def test_astar_routes_through_gap_in_inflated_free_space(self) -> None:
        free = np.ones((40, 40), dtype=bool)
        free[:, 20] = False
        free[28:33, 20] = True
        path = FrontierExplorer._astar(free, (10, 8), (10, 32))

        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(any(row >= 28 for row, _col in path))
        self.assertTrue(all(free[row, col] for row, col in path))

    def test_exploration_heading_biases_an_open_corridor(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        policy.observe_exploration(ExplorationState(
            active=True,
            mode="FRONTIER",
            heading_error_deg=55.0,
            target_distance_m=2.0,
        ))
        profile = np.full(STEER_HEADINGS.shape, 1.5, dtype=np.float32)
        choice = policy._heading_from_profile(profile, 1.0)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertGreater(choice[0], 20.0)

    def test_live_corridor_safety_overrides_exploration_target(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        policy.observe_exploration(ExplorationState(
            active=True,
            mode="FRONTIER",
            heading_error_deg=60.0,
            target_distance_m=2.0,
        ))
        profile = np.full(STEER_HEADINGS.shape, 0.20, dtype=np.float32)
        profile[(STEER_HEADINGS >= -55.0) & (STEER_HEADINGS <= -25.0)] = 1.2
        choice = policy._heading_from_profile(profile, 1.0)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertLess(choice[0], 0.0)

    def test_closed_loop_no_imu_exploration_covers_multiple_room_axes(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        mapper = LidarSlamLite()
        explorer = FrontierExplorer()
        x = y = heading = 0.0
        positions: list[tuple[float, float]] = []
        collided = False

        for step in range(320):
            now = policy.started_at + step * 0.1
            points = []
            for index, angle_deg in enumerate(range(0, 360, 2)):
                ray = heading + math.radians(angle_deg)
                dx, dy = math.sin(ray), math.cos(ray)
                distances = []
                if abs(dx) > 1e-6:
                    for wall_x in (-1.45, 1.45):
                        hit = (wall_x - x) / dx
                        if hit > 0.0:
                            distances.append(hit)
                if abs(dy) > 1e-6:
                    for wall_y in (-1.45, 1.45):
                        hit = (wall_y - y) / dy
                        if hit > 0.0:
                            distances.append(hit)

                obstacle_x, obstacle_y, obstacle_radius = 0.0, 0.72, 0.16
                offset_x, offset_y = x - obstacle_x, y - obstacle_y
                b = 2.0 * (offset_x * dx + offset_y * dy)
                c = (
                    offset_x * offset_x
                    + offset_y * offset_y
                    - obstacle_radius * obstacle_radius
                )
                discriminant = b * b - 4.0 * c
                if discriminant >= 0.0:
                    hit = (-b - math.sqrt(discriminant)) / 2.0
                    if hit > 0.08:
                        distances.append(hit)
                distance = min(distances)
                points.append((
                    index,
                    LidarPoint(
                        float(angle_deg),
                        int(distance * 1000),
                        120,
                        now,
                    ),
                ))

            clearance = SectorClearance(
                _sector_clearance(points, 0.0, 20.0),
                _sector_clearance(points, -75.0, 35.0),
                _sector_clearance(points, 75.0, 35.0),
                True,
                _sector_clearance(points, -35.0, 20.0),
                _sector_clearance(points, 35.0, 20.0),
                corridor_profile(points),
                _sector_clearance(points, 180.0, 25.0),
            )
            slam = mapper.update(
                points,
                policy.left_pwm,
                policy.right_pwm,
                now,
                imu_yaw_rate_dps=None,
            )
            policy.observe_pose(slam)
            exploration = explorer.update(
                mapper.grid,
                mapper.observed,
                mapper.visits,
                mapper.x,
                mapper.y,
                mapper.heading,
                mapper.metres,
                slam.map_updates,
                now,
            )
            policy.observe_exploration(exploration)
            policy.observe_imu(IMUState(error="missing"))
            status = ArduinoStatus(
                front_cm=None if clearance.front_m is None else clearance.front_m * 100.0,
                motion="S",
                received_at=now,
            )
            policy.decide(clearance, status, False, now, True)

            left_speed = policy.left_pwm / 255.0 * 0.26
            right_speed = policy.right_pwm / 255.0 * 0.26
            linear_speed = (left_speed + right_speed) / 2.0
            turn_rate = (left_speed - right_speed) / 0.14
            heading += turn_rate * 0.1
            x += math.sin(heading) * linear_speed * 0.1
            y += math.cos(heading) * linear_speed * 0.1
            positions.append((x, y))

            wall_collision = abs(x) >= 1.35 or abs(y) >= 1.35
            obstacle_collision = math.hypot(x, y - 0.72) <= 0.26
            if wall_collision or obstacle_collision:
                collided = True
                break

        x_positions = [position[0] for position in positions]
        y_positions = [position[1] for position in positions]
        self.assertFalse(collided)
        self.assertGreater(max(x_positions) - min(x_positions), 0.65)
        self.assertGreater(max(y_positions) - min(y_positions), 0.85)
        self.assertGreater(int(np.count_nonzero(mapper.observed)), 4_000)
        self.assertGreater(exploration.replans, 10)


if __name__ == "__main__":
    unittest.main()
