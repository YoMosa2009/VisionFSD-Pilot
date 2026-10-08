"""Regressions for measured openings, map memory and bounded global guidance."""
import math
import pathlib
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from robot_autonomy import (AutonomousPolicy, SectorClearance, ArduinoLink,
                            UNO_CONTROL_LEASE_S, CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT)
from robot_explorer import AsyncExplorer, ExplorationState, FrontierExplorer
from robot_local_planner import PlannerLimits, find_gap
from robot_slam_lite import LidarSlamLite


def opening_scan(bearing):
    angles = np.arange(0., 360., 1., dtype=np.float32)
    error = (angles - bearing + 180.) % 360. - 180.
    ranges = np.where(np.abs(error) <= 55., 2.6, .50).astype(np.float32)
    radians = np.radians(angles)
    return angles, ranges, np.sin(radians) * ranges, np.cos(radians) * ranges


class GapGeometryTests(unittest.TestCase):
    def test_opening_bearings_are_rotation_equivariant(self):
        # Includes a doorway straddling zero, approached from either side.
        for bearing in range(-170, 180, 10):
            with self.subTest(bearing=bearing):
                _, _, x, y = opening_scan(bearing)
                gap = find_gap(x, y, PlannerLimits())
                self.assertTrue(gap.found)
                error = (gap.bearing_deg - bearing + 180.) % 360. - 180.
                self.assertLessEqual(abs(error), 5.)

    def test_gap_seek_converges_through_scan_wrap_without_imu(self):
        policy = AutonomousPolicy(0., 112)
        heading = 0.
        goal = 100.
        for tick in range(60):
            now = 10. + tick * .1
            bearing = (goal - heading + 180.) % 360. - 180.
            angles, ranges, _, _ = opening_scan(bearing)
            policy.observe_scan(angles, ranges, now, now)
            clear = SectorClearance(.5, .5, .5, True, .5, .5, rear_m=.5, scan_at=now)
            command = policy._seek_gap(clear, now)
            if policy._gap_aligned:
                self.assertLessEqual(abs(bearing), 20.)
                return
            self.assertIn(command, ('L', 'R'))
            # Simulated response only: five degrees per pivot, no inferred distance.
            heading += 5. if command == 'R' else -5.
        self.fail('gap pivot never aligned with the actual opening')


class RecoveryMemoryTests(unittest.TestCase):
    def test_non_imu_pivot_rotates_memory_and_gap_by_the_same_step(self):
        policy = AutonomousPolicy(0., 112)
        policy.local_planner.memory.add_scan(np.array([0.]), np.array([1.]), 10.)
        _, initial_y = policy.local_planner.memory.cartesian()
        policy.left_pwm, policy.right_pwm = 117, -117
        policy._last_motion_track_at = 10.
        policy._committed_bearing_deg = 80.
        policy._track_local_motion(10.1)
        yaw = 234. / 255. * CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT * .1
        x, y = policy.local_planner.memory.cartesian()
        self.assertAlmostEqual(float(x[0]), -float(initial_y[0]) * math.sin(math.radians(yaw)), places=5)
        self.assertAlmostEqual(float(y[0]), float(initial_y[0]) * math.cos(math.radians(yaw)), places=5)
        self.assertAlmostEqual(policy._committed_bearing_deg, 80. - yaw, places=5)

    def test_reverse_moves_remembered_front_obstacle_away(self):
        policy = AutonomousPolicy(0., 112)
        policy.local_planner.memory.add_scan(np.array([0.]), np.array([1.]), 10.)
        _, initial_y = policy.local_planner.memory.cartesian()
        policy.left_pwm = policy.right_pwm = -105
        policy._last_motion_track_at = 10.
        policy._track_local_motion(10.1)
        _, y = policy.local_planner.memory.cartesian()
        distance = 105. / 255. * policy.local_planner.limits.top_speed_mps * .1
        self.assertAlmostEqual(float(y[0]), float(initial_y[0]) + distance, places=5)

class OccupancyMemoryTests(unittest.TestCase):
    def test_actual_grid_half_life_is_independent_of_scan_rate(self):
        results = []
        for period in (.1, .5, 1.):
            mapper = LidarSlamLite(cells=60, metres=6.)
            mapper.grid[10, 10] = 200
            mapper._integrate_points([], 0.)
            for tick in range(1, round(12. / period) + 1):
                mapper._integrate_points([], tick * period)
            results.append(int(mapper.grid[10, 10]))
        for value in results:
            self.assertAlmostEqual(value, 100, delta=1)
        self.assertLessEqual(max(results) - min(results), 1)

    def test_fractional_evidence_moves_and_resets_with_map(self):
        mapper = LidarSlamLite(cells=60, metres=6.)
        mapper.grid[30, 30] = 200
        mapper._integrate_points([], 0.)
        mapper._integrate_points([], .1)
        mapper.x = .5
        mapper._recenter_if_needed()
        self.assertEqual(np.count_nonzero(mapper._decay_remainder), 1)
        mapper.reset()
        self.assertEqual(np.count_nonzero(mapper._decay_remainder), 0)
        mapper._integrate_points([], 1.)
        self.assertEqual(np.count_nonzero(mapper.grid), 0)


class GlobalBudgetTests(unittest.TestCase):
    def test_committed_goal_search_uses_the_same_deadline(self):
        explorer = FrontierExplorer()
        grid = np.zeros((120, 120), dtype=np.uint8)
        grid[[0, -1], :] = 255
        grid[:, [0, -1]] = 255
        observed = np.full_like(grid, 255)
        visits = np.zeros_like(grid, dtype=np.uint16)
        args = (grid, observed, visits, 3., 3., 0., 6., 20)
        first = explorer.update(*args, 10.)
        self.assertTrue(first.active)
        # Cover ordinary commitment and the no-progress alternate-goal path.
        for now in (11., 25.):
            # Since v1.9.28 a still-clear route is reused rather than searched
            # again. Discard it so this replan has to search, which is what
            # this test checks the deadline of.
            explorer._path_cells = []
            with mock.patch.object(explorer, '_astar', wraps=explorer._astar) as search:
                explorer.update(*args, now)
            self.assertTrue(search.call_args_list)
            deadlines = [call.args[3] for call in search.call_args_list]
            self.assertTrue(all(value is not None for value in deadlines), deadlines)
            self.assertEqual(len(set(deadlines)), 1)

    def test_timeout_does_not_reuse_route_after_new_obstacle(self):
        explorer = FrontierExplorer()
        grid = np.zeros((120, 120), dtype=np.uint8)
        grid[[0, -1], :] = 255
        grid[:, [0, -1]] = 255
        observed = np.full_like(grid, 255)
        visits = np.zeros_like(grid, dtype=np.uint16)
        args = (grid, observed, visits, 3., 3., 0., 6., 20)
        first = explorer.update(*args, 10.)
        self.assertTrue(first.active)
        row = int(round(first.target_y_m * 20))
        col = int(round(first.target_x_m * 20))
        grid[max(0, row-2):row+3, max(0, col-2):col+3] = 255
        explorer.PLAN_TIME_BUDGET_S = 0.
        self.assertFalse(explorer.update(*args, 11.).active)

    def test_stale_global_snapshot_cannot_keep_steering(self):
        worker = object.__new__(AsyncExplorer)
        worker._lock = threading.Lock()
        worker._state = ExplorationState(active=True, mode='FRONTIER')
        worker._state_snapshot_at = time.monotonic() - 10.
        self.assertFalse(worker.state().active)
        self.assertEqual(worker.state().mode, 'STALE_MAP')

    def test_main_loop_lease_expires_before_firmware_silence_timeout(self):
        self.assertLessEqual(UNO_CONTROL_LEASE_S, .30)
        link = object.__new__(ArduinoLink)
        link._drive_lock = threading.Lock()
        link._drive_command = 'DRIVE 105 105'
        link._drive_last_write = 10.
        link._drive_lease_until = 10. + UNO_CONTROL_LEASE_S
        link._drive_expired = False
        link.drive_lease_expirations = 0
        self.assertEqual(link._heartbeat_command(10.31), 'STOP')


if __name__ == '__main__':
    unittest.main()
