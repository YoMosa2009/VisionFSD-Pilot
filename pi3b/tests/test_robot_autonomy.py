from __future__ import annotations

import math
import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import (
    MAX_PWM,
    MIN_MOVE_PWM,
    STEER_HEADINGS,
    ArduinoStatus,
    AutonomousPolicy,
    SectorClearance,
    corridor_profile,
)


class _Return:
    """Minimal stand-in for an LD19 point."""

    def __init__(self, angle_deg: float, distance_mm: int) -> None:
        self.angle_deg = angle_deg
        self.distance_mm = distance_mm


def wall_scene(front_m: float, gap: tuple[int, int] | None = None, span: int = 75):
    """A flat wall ahead, optionally with an opening, plus open space behind."""
    points = []
    for angle in range(-span, span + 1):
        distance = front_m / math.cos(math.radians(angle))
        if gap is not None and gap[0] <= angle <= gap[1]:
            distance = 4.5
        points.append((angle % 360, _Return(angle % 360, int(min(distance, 5.5) * 1000))))
    for angle in range(span + 1, 360 - span):
        points.append((angle, _Return(angle, 4000)))
    return points


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
        policy = AutonomousPolicy(0.0, 150)
        obstacle = SectorClearance(0.50, 1.6, 0.8, True, 1.5, 0.8)
        policy.decide(obstacle, self.status, False, time.monotonic())
        self.assertGreaterEqual(min(policy.left_pwm, policy.right_pwm), MIN_MOVE_PWM)

    def test_every_motion_clears_the_loaded_stall_floor(self) -> None:
        """A wheel that turns freely in the air still stalls under the chassis.

        Any commanded motion must therefore be either zero or genuinely above
        the loaded deadband; in between, the motor only buzzes and sags the
        battery, which is what made the robot creep and pause on the floor.
        """
        scenarios = [
            SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0),      # open cruise
            SectorClearance(0.50, 1.6, 0.8, True, 1.5, 0.8),     # arc
            SectorClearance(0.25, 0.7, 1.6, True, 0.7, 1.4),     # pivot escape
            SectorClearance(1.2, 0.6, 1.9, True, 0.6, 1.8),      # guided forward
        ]
        for index, clearance in enumerate(scenarios):
            policy = AutonomousPolicy(0.0, 150)
            policy.decide(clearance, self.status, False, time.monotonic())
            for value in (policy.left_pwm, policy.right_pwm):
                self.assertTrue(value == 0 or abs(value) >= MIN_MOVE_PWM,
                                f"scenario {index} produced stalling PWM {value}")
                self.assertLessEqual(abs(value), MAX_PWM)

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


class CorridorProfileTests(unittest.TestCase):
    def test_straight_ahead_is_a_candidate_heading(self) -> None:
        # An even split leaves the two nearest options tied either side of
        # centre, and the chassis weaves while believing it drives straight.
        self.assertIn(0.0, {float(value) for value in STEER_HEADINGS})

    def test_wall_limits_travel_to_the_bumper(self) -> None:
        profile = corridor_profile(wall_scene(1.20))
        straight = profile[int(np.argmin(np.abs(STEER_HEADINGS)))]
        self.assertAlmostEqual(float(straight), 1.20 - 0.075, places=2)

    def test_a_gap_the_body_fits_is_seen_as_open(self) -> None:
        """The five-sector summary cannot express "there is a gap 20 deg right"."""
        profile = corridor_profile(wall_scene(1.20, gap=(12, 28)))
        through = profile[int(np.argmin(np.abs(STEER_HEADINGS - 20.0)))]
        straight = profile[int(np.argmin(np.abs(STEER_HEADINGS)))]
        self.assertGreater(float(through), 2.5)
        self.assertLess(float(straight), 1.3)

    def test_returns_behind_never_limit_forward_travel(self) -> None:
        behind = [(angle, _Return(angle, 150)) for angle in range(160, 201)]
        behind += [(angle, _Return(angle, 4000)) for angle in range(-60, 61)]
        profile = corridor_profile(behind)
        straight = profile[int(np.argmin(np.abs(STEER_HEADINGS)))]
        self.assertGreater(float(straight), 2.0)


class SpeedGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = ArduinoStatus(front_cm=80.0, motion="S", received_at=time.monotonic())

    def _drive(self, front_m: float, speed: int = 125) -> AutonomousPolicy:
        policy = AutonomousPolicy(0.0, speed)
        clearance = SectorClearance(front_m, 2.0, 2.0, True, 2.0, 2.0,
                                    corridor_profile(wall_scene(front_m)))
        policy.decide(clearance, self.status, False, time.monotonic())
        return policy

    def test_speed_rises_with_clearance(self) -> None:
        """The reported failure: one flat-out speed regardless of surroundings."""
        speeds = [max(self._drive(d).left_pwm, self._drive(d).right_pwm)
                  for d in (0.80, 1.20, 1.60, 2.40)]
        self.assertEqual(speeds, sorted(speeds), f"not monotonic: {speeds}")
        self.assertLess(speeds[0], speeds[-1])

    def test_cruise_never_exceeds_the_configured_ceiling(self) -> None:
        for front_m in (0.80, 1.20, 2.00, 3.00):
            policy = self._drive(front_m, speed=125)
            self.assertLessEqual(max(abs(policy.left_pwm), abs(policy.right_pwm)), 125)

    def test_turning_is_never_faster_than_driving_straight(self) -> None:
        """Scaling a turn up rather than down makes every corner alarming."""
        policy = AutonomousPolicy(0.0, 125)
        clearance = SectorClearance(1.20, 2.0, 2.0, True, 2.0, 2.0,
                                    corridor_profile(wall_scene(1.20, gap=(12, 28))))
        policy.decide(clearance, self.status, False, time.monotonic())
        self.assertLessEqual(max(policy.left_pwm, policy.right_pwm), 125)
        self.assertNotEqual(policy.left_pwm, policy.right_pwm)

    def test_governed_output_still_clears_the_stall_floor(self) -> None:
        for front_m in (0.60, 0.90, 1.40, 2.50):
            policy = self._drive(front_m)
            for value in (policy.left_pwm, policy.right_pwm):
                self.assertTrue(value == 0 or abs(value) >= MIN_MOVE_PWM,
                                f"{front_m} m produced stalling PWM {value}")


if __name__ == "__main__":
    unittest.main()
