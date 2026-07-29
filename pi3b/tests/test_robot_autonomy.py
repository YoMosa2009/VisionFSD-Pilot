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


def settle(policy, clearance, status, person=False, cycles=45, camera_ready=True):
    """Run the policy to a steady state.

    Output is slew-rate limited, so a single decision only moves the wheels a
    few PWM off their previous value.  Tests that inspect commanded speed have
    to let it reach the value it is actually asking for.
    """
    now = time.monotonic()
    label = "STOP"
    for index in range(cycles):
        label = policy.decide(clearance, status, person, now + index * 0.03, camera_ready)
    return label


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
        clear_sides = SectorClearance(2.0, 2.0, 1.0, True, 1.8, 0.8, None, 1.0)
        self.assertEqual(policy.decide(clear_sides, blocked, False, time.monotonic()), "L")
        self.assertLess(policy.left_pwm, 0)
        self.assertLess(policy.right_pwm, 0)

    def test_close_obstacle_never_commands_a_one_wheel_pivot(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=time.monotonic())
        clear_sides = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        settle(policy, clear_sides, blocked, cycles=10)
        self.assertLess(policy.left_pwm, 0)
        self.assertLess(policy.right_pwm, 0)
        self.assertNotEqual(policy.left_pwm, policy.right_pwm)

    def test_close_obstacle_recovery_does_not_restart_after_timeout(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        clearance = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        outputs = []
        for index in range(48):
            policy.decide(clearance, blocked, False, now + index * 0.03)
            outputs.append((policy.left_pwm, policy.right_pwm))
        first_stop = next(index for index, output in enumerate(outputs) if index > 0 and output == (0, 0))
        self.assertTrue(all(output == (0, 0) for output in outputs[first_stop:]))

    def test_recovery_commits_to_selected_side_after_front_clears(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        close = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        policy.decide(close, blocked, False, now)
        clear_status = ArduinoStatus(front_cm=80.0, motion="S", received_at=now)
        released = SectorClearance(0.65, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        self.assertEqual(policy.decide(released, clear_status, False, now + 0.20), "F")
        # Direction reversals intentionally pass through one zero-output cycle.
        policy.decide(released, clear_status, False, now + 0.23)
        self.assertEqual(policy.reason, "ESCAPE_COMMIT:L")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)

    def test_close_obstacle_with_blocked_rear_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=time.monotonic())
        clearance = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 0.20)
        self.assertEqual(policy.decide(clearance, blocked, False, time.monotonic()), "STOP")

    def test_motion_starts_at_the_loaded_wheel_floor_not_a_tiny_pwm(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        policy.decide(clear, self.status, False, time.monotonic())
        self.assertEqual(policy.left_pwm, MIN_MOVE_PWM)
        self.assertEqual(policy.right_pwm, MIN_MOVE_PWM)

    def test_acceleration_rises_in_small_steps_after_the_floor(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        policy.decide(clear, self.status, False, time.monotonic())
        first = policy.left_pwm
        policy.decide(clear, self.status, False, time.monotonic() + 0.04)
        self.assertGreaterEqual(policy.left_pwm, first)
        self.assertLessEqual(policy.left_pwm - first, 4)

    def test_confirmed_person_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        self.assertEqual(policy.decide(clear, self.status, True, time.monotonic()), "STOP")

    def test_front_obstacle_turns_toward_clearer_side(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        obstacle = SectorClearance(0.25, 0.7, 1.6, True, 0.7, 1.4, None, 1.0)
        self.assertEqual(policy.decide(obstacle, self.status, False, time.monotonic()), "R")

    def test_midrange_obstacle_uses_forward_arc(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        obstacle = SectorClearance(0.60, 1.8, 0.7, True, 1.5, 0.7)
        self.assertEqual(settle(policy, obstacle, self.status), "F")
        self.assertLess(policy.left_pwm, policy.right_pwm)
        self.assertGreater(policy.right_pwm, 0)
        self.assertIn(policy.left_pwm, (0, MIN_MOVE_PWM))

    def test_arc_hysteresis_avoids_threshold_flicker(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        first = SectorClearance(0.80, 1.5, 0.8, True, 1.4, 0.8)
        settle(policy, first, self.status)
        held = SectorClearance(0.94, 1.5, 0.8, True, 1.4, 0.8)
        self.assertEqual(settle(policy, held, self.status), "F")
        self.assertLess(policy.left_pwm, policy.right_pwm)

    def test_direction_lock_prevents_small_side_measurement_flip(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        first = SectorClearance(0.70, 1.5, 1.0, True, 1.4, 1.0)
        settle(policy, first, self.status)
        # Right improves a little, but not enough to discard the selected left
        # arc in the next scan.
        second = SectorClearance(0.70, 1.2, 1.35, True, 1.2, 1.35)
        settle(policy, second, self.status)
        self.assertLess(policy.left_pwm, policy.right_pwm)

    def test_arc_never_uses_a_low_pwm_stall_range(self) -> None:
        policy = AutonomousPolicy(0.0, 150)
        obstacle = SectorClearance(0.50, 1.6, 0.8, True, 1.5, 0.8)
        settle(policy, obstacle, self.status)
        for value in (policy.left_pwm, policy.right_pwm):
            self.assertTrue(value == 0 or abs(value) >= MIN_MOVE_PWM)

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
            policy = AutonomousPolicy(0.0, 85)
            settle(policy, clearance, self.status)
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

    def test_wide_open_heading_still_uses_a_forward_arc_not_a_pivot(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.zeros(STEER_HEADINGS.size, dtype=np.float32)
        profile[int(np.argmin(np.abs(STEER_HEADINGS - 72.0)))] = 3.0
        clearance = SectorClearance(1.0, 2.0, 2.0, True, 2.0, 2.0, profile)
        self.assertEqual(settle(policy, clearance, self.status), "F")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)
        self.assertGreaterEqual(min(policy.left_pwm, policy.right_pwm), MIN_MOVE_PWM)
        self.assertLessEqual(abs(policy.left_pwm - policy.right_pwm), 8)

    def test_multi_object_profile_cannot_immediately_flip_turn_direction(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        left_profile = np.full(STEER_HEADINGS.size, 0.20, dtype=np.float32)
        left_profile[int(np.argmin(np.abs(STEER_HEADINGS + 28.0)))] = 2.0
        left = SectorClearance(0.70, 1.8, 1.6, True, 1.7, 1.5, left_profile)
        for index in range(4):
            policy.decide(left, self.status, False, now + index * 0.03)
        self.assertLess(policy.left_pwm, policy.right_pwm)

        right_profile = np.full(STEER_HEADINGS.size, 0.20, dtype=np.float32)
        right_profile[int(np.argmin(np.abs(STEER_HEADINGS + 28.0)))] = 0.50
        right_profile[int(np.argmin(np.abs(STEER_HEADINGS - 28.0)))] = 2.5
        right = SectorClearance(0.70, 1.0, 2.0, True, 0.8, 1.9, right_profile)
        policy.decide(right, self.status, False, now + 0.15)
        self.assertLess(policy.left_pwm, policy.right_pwm)

    def test_open_straight_corridor_is_preferred_over_a_longer_side(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 1.10, dtype=np.float32)
        profile[int(np.argmin(np.abs(STEER_HEADINGS - 32.0)))] = 3.0
        clearance = SectorClearance(1.10, 2.0, 2.0, True, 2.0, 2.0, profile)
        settle(policy, clearance, self.status)
        self.assertEqual(policy.left_pwm, policy.right_pwm)

    def test_planner_returns_to_forward_progress_when_straight_opens(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        side_profile = np.full(STEER_HEADINGS.size, 0.20, dtype=np.float32)
        side_profile[int(np.argmin(np.abs(STEER_HEADINGS + 28.0)))] = 2.0
        side = SectorClearance(0.70, 2.0, 0.7, True, 2.0, 0.7, side_profile)
        settle(policy, side, self.status)
        self.assertLess(policy.left_pwm, policy.right_pwm)

        open_profile = np.full(STEER_HEADINGS.size, 1.20, dtype=np.float32)
        open_profile[int(np.argmin(np.abs(STEER_HEADINGS + 28.0)))] = 3.0
        open_path = SectorClearance(1.20, 2.0, 2.0, True, 2.0, 2.0, open_profile)
        settle(policy, open_path, self.status)
        self.assertEqual(policy.left_pwm, policy.right_pwm)

    def test_missing_differential_capability_can_only_send_stop(self) -> None:
        class Link:
            differential_ready = False

            def __init__(self) -> None:
                self.commands = []

            def send(self, command: str) -> None:
                self.commands.append(command)

        policy = AutonomousPolicy(0.0, 118)
        policy.left_pwm, policy.right_pwm = 118, 105
        link = Link()
        policy.send(link, "F", time.monotonic())
        self.assertEqual(link.commands, ["STOP"])


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

    def test_isolated_lidar_speckle_does_not_block_a_clear_corridor(self) -> None:
        speckle = [(0, _Return(0, 180))]
        profile = corridor_profile(speckle)
        straight = profile[int(np.argmin(np.abs(STEER_HEADINGS)))]
        self.assertGreater(float(straight), 2.0)


class SpeedGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = ArduinoStatus(front_cm=80.0, motion="S", received_at=time.monotonic())

    def _drive(self, front_m: float, speed: int = 118) -> AutonomousPolicy:
        policy = AutonomousPolicy(0.0, speed)
        clearance = SectorClearance(front_m, 2.0, 2.0, True, 2.0, 2.0,
                                    corridor_profile(wall_scene(front_m)))
        settle(policy, clearance, self.status)
        return policy

    def test_speed_rises_with_clearance(self) -> None:
        """The reported failure: one flat-out speed regardless of surroundings."""
        speeds = [max(self._drive(d).left_pwm, self._drive(d).right_pwm)
                  for d in (0.80, 1.20, 1.60, 2.40)]
        self.assertEqual(speeds, sorted(speeds), f"not monotonic: {speeds}")
        self.assertLess(speeds[0], speeds[-1])

    def test_cruise_never_exceeds_the_configured_ceiling(self) -> None:
        for front_m in (0.80, 1.20, 2.00, 3.00):
            policy = self._drive(front_m, speed=118)
            self.assertLessEqual(max(abs(policy.left_pwm), abs(policy.right_pwm)), 118)

    def test_turning_is_never_faster_than_driving_straight(self) -> None:
        """Scaling a turn up rather than down makes every corner alarming."""
        policy = AutonomousPolicy(0.0, 118)
        clearance = SectorClearance(0.80, 2.0, 2.0, True, 2.0, 2.0,
                                    corridor_profile(wall_scene(0.80, gap=(12, 28))))
        settle(policy, clearance, self.status)
        self.assertLessEqual(max(policy.left_pwm, policy.right_pwm), 118)
        self.assertNotEqual(policy.left_pwm, policy.right_pwm)

    def test_governed_output_still_clears_the_stall_floor(self) -> None:
        for front_m in (0.60, 0.90, 1.40, 2.50):
            policy = self._drive(front_m)
            for value in (policy.left_pwm, policy.right_pwm):
                self.assertTrue(value == 0 or abs(value) >= MIN_MOVE_PWM,
                                f"{front_m} m produced stalling PWM {value}")


if __name__ == "__main__":
    unittest.main()
