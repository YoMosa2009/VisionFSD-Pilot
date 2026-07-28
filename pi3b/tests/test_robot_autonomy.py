from __future__ import annotations

import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import ArduinoStatus, AutonomousPolicy, DirectionalClearance, SectorClearance


class AutonomousPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = ArduinoStatus(front_cm=80.0, motion="S", received_at=time.monotonic())

    def test_boot_standby_never_moves(self) -> None:
        policy = AutonomousPolicy(25.0, 70)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        self.assertEqual(policy.decide(clear, self.status, False, policy.started_at + 3.0), "STOP")

    def test_stale_lidar_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        stale = SectorClearance(None, None, None, False)
        self.assertEqual(policy.decide(stale, self.status, False, time.monotonic()), "STOP")

    def test_ultrasonic_wins_over_clear_lidar(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=time.monotonic())
        clear_sides = SectorClearance(2.0, 2.0, 1.0, True, 1.8, 0.8)
        self.assertEqual(policy.decide(clear_sides, blocked, False, time.monotonic()), "L")
        self.assertLess(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)

    def test_confirmed_person_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        self.assertEqual(policy.decide(clear, self.status, True, time.monotonic()), "STOP")

    def test_front_obstacle_turns_toward_clearer_side(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        obstacle = SectorClearance(0.25, 0.7, 1.6, True, 0.7, 1.4)
        self.assertEqual(policy.decide(obstacle, self.status, False, time.monotonic()), "R")

    def test_midrange_obstacle_uses_forward_arc(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        obstacle = SectorClearance(0.60, 1.8, 0.7, True, 1.5, 0.7)
        self.assertEqual(policy.decide(obstacle, self.status, False, time.monotonic()), "F")
        self.assertLess(policy.left_pwm, policy.right_pwm)
        self.assertGreater(policy.left_pwm, 0)

    def test_arc_hysteresis_avoids_threshold_flicker(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        first = SectorClearance(0.80, 1.5, 0.8, True, 1.4, 0.8)
        policy.decide(first, self.status, False, time.monotonic())
        held = SectorClearance(0.94, 1.5, 0.8, True, 1.4, 0.8)
        self.assertEqual(policy.decide(held, self.status, False, time.monotonic() + 0.10), "F")
        self.assertLess(policy.left_pwm, policy.right_pwm)

    def test_direction_lock_prevents_small_side_measurement_flip(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        now = time.monotonic()
        first = SectorClearance(0.70, 1.5, 1.0, True, 1.4, 1.0)
        policy.decide(first, self.status, False, now)
        # Right improves a little, but not enough to discard the selected left
        # arc in the next scan.
        second = SectorClearance(0.70, 1.2, 1.35, True, 1.2, 1.35)
        policy.decide(second, self.status, False, now + 0.10)
        self.assertLess(policy.left_pwm, policy.right_pwm)

    def test_arc_keeps_both_motors_out_of_low_pwm_stall_range(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        obstacle = SectorClearance(0.50, 1.6, 0.8, True, 1.5, 0.8)
        policy.decide(obstacle, self.status, False, time.monotonic())
        self.assertGreaterEqual(min(policy.left_pwm, policy.right_pwm), 48)

    def test_stale_camera_stops_after_standby(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        self.assertEqual(policy.decide(clear, self.status, False, time.monotonic(), False), "STOP")

    def test_unseen_front_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        unseen = SectorClearance(None, 2.0, 2.0, True, 2.0, 2.0)
        self.assertEqual(policy.decide(unseen, self.status, False, time.monotonic()), "STOP")

    def test_clear_path_moves_forward(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(1.2, 1.0, 1.0, True)
        self.assertEqual(policy.decide(clear, self.status, False, time.monotonic()), "F")

    def test_dense_lidar_profile_selects_a_measured_exploration_arc(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        now = time.monotonic()
        # The left-forward opening is both close enough to be safe and more
        # useful than the right.  This must become a measured path choice,
        # not an arbitrary left/right reaction.
        directions = DirectionalClearance((
            (-60.0, 1.1), (-40.0, 1.6), (-20.0, 2.0), (0.0, 0.70),
            (20.0, 1.0), (40.0, 0.85), (60.0, 0.80),
        ), True)
        obstacle = SectorClearance(0.70, 1.6, 0.8, True, 1.5, 0.8)
        self.assertEqual(policy.decide(obstacle, self.status, False, now, directions=directions), "F")
        self.assertLess(policy.left_pwm, policy.right_pwm)
        self.assertIn("EXPLORE_ARC", policy.reason)
        self.assertLess(policy.goal_bearing, 0.0)

    def test_exploration_goal_is_locked_against_small_scan_changes(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        now = time.monotonic()
        obstacle = SectorClearance(0.72, 1.5, 1.2, True, 1.4, 1.1)
        first = DirectionalClearance((
            (-50.0, 1.2), (-30.0, 1.9), (-10.0, 1.45), (10.0, 1.2),
            (30.0, 1.1), (50.0, 1.0),
        ), True)
        policy.decide(obstacle, self.status, False, now, directions=first)
        first_bearing = policy.goal_bearing
        # A distant right reading improves slightly.  The safe, active left
        # route should remain locked until its short commitment expires.
        second = DirectionalClearance((
            (-50.0, 1.2), (-30.0, 1.75), (-10.0, 1.45), (10.0, 1.2),
            (30.0, 2.0), (50.0, 1.0),
        ), True)
        policy.decide(obstacle, self.status, False, now + 0.15, directions=second)
        self.assertAlmostEqual(policy.goal_bearing, first_bearing)
        self.assertLess(policy.left_pwm, policy.right_pwm)


if __name__ == "__main__":
    unittest.main()
