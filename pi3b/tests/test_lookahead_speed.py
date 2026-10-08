"""Regression tests for v1.9.23: look-ahead past each arc, and calm speed.

From driving v1.9.22: too fast, contact with walls, and choices that only
looked a short way ahead instead of heading for openings further out.
"""

from __future__ import annotations

import math
import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import (
    ARC_STEER_OPTIONS,
    CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT,
    EFFECTIVE_TURN_SPLIT_PWM,
    MAX_GENTLE_HEADING_DEG,
    MAX_PWM,
    MIN_MOVE_PWM,
    ROBOT_FOOTPRINT,
    TURN_OUTER_CAP_PWM,
    AutonomousPolicy,
)
from robot_local_planner import ArcBank, PlannerLimits, evaluate_arcs

SPEEDS = tuple((pwm, 0.55 * pwm / 255.0) for pwm in (105, 112))


def _bank() -> ArcBank:
    limits = PlannerLimits(
        footprint=ROBOT_FOOTPRINT,
        yaw_rate_dps_at_full_steer=EFFECTIVE_TURN_SPLIT_PWM / MAX_PWM * CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT,
        full_steer_deg=MAX_GENTLE_HEADING_DEG,
    )
    return ArcBank(limits, ARC_STEER_OPTIONS, SPEEDS)


def _points(segments, spacing=0.03):
    xs, ys = [], []
    for (x1, y1), (x2, y2) in segments:
        count = max(2, int(math.hypot(x2 - x1, y2 - y1) / spacing))
        xs.extend(np.linspace(x1, x2, count))
        ys.extend(np.linspace(y1, y2, count))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class LookAheadTests(unittest.TestCase):
    def test_prefers_the_arc_that_leads_into_a_long_opening(self) -> None:
        """Ahead-left is a wall 2.4 m away; ahead-right opens into a long clear
        corridor. The arcs themselves are equally clear for their length; only
        what lies beyond separates them."""
        obstacle_x, obstacle_y = _points([
            ((-3.0, 2.4), (0.2, 2.4)),      # wall across the left half, 2.4 m ahead
            ((1.6, 0.8), (1.6, 6.0)),       # far right boundary of the opening
            ((-3.0, 0.0), (-3.0, 2.4)),
        ])
        choice = evaluate_arcs(obstacle_x, obstacle_y, _bank(), None, 0.0)
        self.assertTrue(choice.admissible)
        self.assertGreater(choice.steering_deg, 0.0)

    def test_open_floor_still_drives_straight(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        choice = evaluate_arcs(empty, empty, _bank(), None, 0.0)
        self.assertEqual(choice.steering_deg, 0.0)


class CalmSpeedTests(unittest.TestCase):
    def test_cruise_is_used_only_with_open_space_ahead(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        open_choice = evaluate_arcs(empty, empty, _bank(), None, 0.0)
        self.assertEqual(open_choice.pwm, 112)

        wall_x, wall_y = _points([((-2.0, 2.2), (2.0, 2.2))])
        near_choice = evaluate_arcs(wall_x, wall_y, _bank(), None, 0.0)
        self.assertTrue(near_choice.admissible)
        self.assertEqual(near_choice.pwm, MIN_MOVE_PWM)

    def test_no_wheel_exceeds_the_turn_cap_while_steering(self) -> None:
        policy = AutonomousPolicy(0.0, 112)
        for _ in range(80):
            policy._note_output_tick(time.monotonic())
            policy._differential(112, MAX_GENTLE_HEADING_DEG)
            time.sleep(0.001)
        self.assertLessEqual(max(policy.left_pwm, policy.right_pwm), TURN_OUTER_CAP_PWM)
        self.assertNotEqual(policy.left_pwm, policy.right_pwm)

    def test_a_real_turn_drops_to_the_movement_floor(self) -> None:
        policy = AutonomousPolicy(0.0, 112)
        for _ in range(80):
            policy._note_output_tick(time.monotonic())
            policy._differential(112, MAX_GENTLE_HEADING_DEG)
            time.sleep(0.001)
        self.assertEqual(min(policy.left_pwm, policy.right_pwm), MIN_MOVE_PWM)

    def test_planner_turn_model_uses_the_capped_split(self) -> None:
        """A planner that believes it can turn harder than the capped wheels
        allow commits to arcs it cannot follow."""
        policy = AutonomousPolicy(0.0, 112)
        expected = EFFECTIVE_TURN_SPLIT_PWM / MAX_PWM * CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT
        self.assertAlmostEqual(
            policy.local_planner.limits.yaw_rate_dps_at_full_steer, expected, places=6
        )
        self.assertEqual(EFFECTIVE_TURN_SPLIT_PWM, TURN_OUTER_CAP_PWM - MIN_MOVE_PWM)


if __name__ == "__main__":
    unittest.main()
