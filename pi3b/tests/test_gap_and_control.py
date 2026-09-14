"""Regression tests for v1.9.20.

Covers the measured chassis footprint, gap-seeking in clutter, per-second
drive ramps, and the dashboard's operator halt and manual driving.
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
    CORRIDOR_HALF_WIDTH_M,
    FRONT_OVERHANG_M,
    GAP_ALIGNED_DEG,
    MANUAL_DRIVE,
    MANUAL_FORWARD_MIN_M,
    MAX_PWM_RATE_PER_S,
    MAX_STEERING_RATE_DPS,
    MIN_MOVE_PWM,
    ROBOT_FOOTPRINT,
    ROBOT_LENGTH_M,
    ROBOT_WIDTH_M,
    ArduinoStatus,
    AutonomousPolicy,
    SectorClearance,
)
from robot_local_planner import PlannerLimits, find_gap
from robot_web import RobotControl


def _status(now: float, front_cm: float | None = None) -> ArduinoStatus:
    return ArduinoStatus(front_cm=front_cm, motion="S", received_at=now)


def _ring(clearance_m: float, openings: tuple[tuple[float, float, float], ...] = ()):
    """A 360-degree scan at a fixed range, with optional deeper sectors."""
    angles = np.arange(0.0, 360.0, 2.0, dtype=np.float32)
    ranges = np.full(angles.size, clearance_m, dtype=np.float32)
    for start, end, depth in openings:
        ranges[(angles >= start) & (angles <= end)] = depth
    radians = np.radians(angles)
    return (
        (np.sin(radians) * ranges).astype(np.float32),
        (np.cos(radians) * ranges).astype(np.float32),
    )


class FootprintTests(unittest.TestCase):
    """The chassis measures 9 x 10.5 inches. The previous 0.14 x 0.15 m
    model understated it badly, so every clearance check was computed for a
    smaller robot than the one driving."""

    def test_dimensions_match_the_measured_chassis(self) -> None:
        self.assertAlmostEqual(ROBOT_WIDTH_M, 0.2286, places=4)
        self.assertAlmostEqual(ROBOT_LENGTH_M, 0.2667, places=4)

    def test_the_corridor_covers_the_real_half_width(self) -> None:
        self.assertGreater(CORRIDOR_HALF_WIDTH_M, ROBOT_WIDTH_M / 2.0)

    def test_travel_is_measured_from_the_bumper(self) -> None:
        self.assertAlmostEqual(FRONT_OVERHANG_M, ROBOT_LENGTH_M / 2.0, places=6)

    def test_two_circles_cover_the_rectangle(self) -> None:
        """Every corner of the footprint must fall inside one of the two
        circles, or the collision check would miss it."""
        half_width = ROBOT_WIDTH_M / 2.0
        half_length = ROBOT_LENGTH_M / 2.0
        for corner_x in (-half_width, half_width):
            for corner_y in (-half_length, half_length):
                nearest = min(
                    math.hypot(corner_x, corner_y - ROBOT_FOOTPRINT.offset_m),
                    math.hypot(corner_x, corner_y + ROBOT_FOOTPRINT.offset_m),
                )
                self.assertLessEqual(nearest, ROBOT_FOOTPRINT.radius_m + 1e-9)

    def test_the_cover_is_tighter_than_a_circumscribed_circle(self) -> None:
        """Two circles are the point: one circumscribed circle would have a
        17.6 cm radius against a true 11.4 cm half-width, and the robot would
        refuse gaps it fits through comfortably."""
        circumscribed = math.hypot(ROBOT_WIDTH_M, ROBOT_LENGTH_M) / 2.0
        self.assertLess(ROBOT_FOOTPRINT.radius_m, circumscribed)


class GapFindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = PlannerLimits(footprint=ROBOT_FOOTPRINT)

    def test_an_opening_behind_the_robot_is_found(self) -> None:
        """The arc search only sees plus or minus 36 degrees. Surrounded by
        furniture, the only way out may be well outside that."""
        x, y = _ring(0.45, ((100.0, 150.0, 2.6),))
        gap = find_gap(x, y, self.limits)
        self.assertTrue(gap.found)
        self.assertGreater(gap.bearing_deg, 90.0)
        self.assertLess(gap.bearing_deg, 160.0)

    def test_a_fully_enclosed_robot_reports_no_gap(self) -> None:
        x, y = _ring(0.35)
        self.assertFalse(find_gap(x, y, self.limits).found)

    def test_a_shallow_alcove_is_not_a_route(self) -> None:
        x, y = _ring(0.35, ((80.0, 120.0, 0.55),))
        self.assertFalse(find_gap(x, y, self.limits).found)

    def test_a_slot_too_narrow_for_the_chassis_is_rejected(self) -> None:
        """Angular width alone is not enough: a narrow slot far away subtends
        the same angle as a doorway nearby."""
        x, y = _ring(0.40, ((88.0, 92.0, 3.0),))
        self.assertFalse(find_gap(x, y, self.limits).found)

    def test_the_wider_of_two_openings_wins(self) -> None:
        x, y = _ring(0.40, ((85.0, 95.0, 2.0), (200.0, 260.0, 2.0)))
        gap = find_gap(x, y, self.limits)
        self.assertTrue(gap.found)
        self.assertGreater(gap.width_deg, 30.0)

    def test_hysteresis_holds_the_opening_already_being_followed(self) -> None:
        """Two similar gaps swapping rank between scans is the classic
        Follow-The-Gap zigzag, and is what spinning in place looked like."""
        x, y = _ring(0.40, ((60.0, 100.0, 2.0), (260.0, 300.0, 2.0)))
        left = find_gap(x, y, self.limits, held_bearing_deg=-80.0)
        right = find_gap(x, y, self.limits, held_bearing_deg=80.0)
        self.assertTrue(left.found and right.found)
        self.assertLess(left.bearing_deg, 0.0)
        self.assertGreater(right.bearing_deg, 0.0)

    def test_the_goal_direction_influences_the_choice(self) -> None:
        x, y = _ring(0.40, ((60.0, 100.0, 2.0), (260.0, 300.0, 2.0)))
        toward_left = find_gap(x, y, self.limits, goal_heading_deg=-80.0)
        toward_right = find_gap(x, y, self.limits, goal_heading_deg=80.0)
        self.assertLess(toward_left.bearing_deg, 0.0)
        self.assertGreater(toward_right.bearing_deg, 0.0)

    def test_completely_open_space_needs_no_turn(self) -> None:
        gap = find_gap(
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            self.limits,
        )
        self.assertTrue(gap.found)
        self.assertEqual(gap.bearing_deg, 0.0)


class GapSeekingPolicyTests(unittest.TestCase):
    def _boxed_policy(self) -> AutonomousPolicy:
        policy = AutonomousPolicy(0.0, 112)
        angles = np.arange(0.0, 360.0, 2.0, dtype=np.float32)
        ranges = np.full(angles.size, 0.42, dtype=np.float32)
        ranges[(angles >= 100.0) & (angles <= 150.0)] = 2.6
        policy.observe_scan(angles, ranges, 1.0, 1.0)
        return policy

    def test_a_boxed_in_robot_turns_toward_the_measured_opening(self) -> None:
        policy = self._boxed_policy()
        now = time.monotonic()
        clearance = SectorClearance(
            0.42, 0.42, 0.42, True, 0.42, 0.42, rear_m=0.42, scan_at=now
        )

        policy.decide(clearance, _status(now), False, now)

        self.assertIn("GAP_SEEK", policy.reason)
        # The opening is on the robot's right, so it pivots right.
        self.assertEqual(policy.turn_command, "R")
        self.assertNotEqual((policy.left_pwm, policy.right_pwm), (0, 0))

    def test_the_chosen_opening_is_published_for_the_dashboard(self) -> None:
        policy = self._boxed_policy()
        now = time.monotonic()
        clearance = SectorClearance(
            0.42, 0.42, 0.42, True, 0.42, 0.42, rear_m=0.42, scan_at=now
        )
        policy.decide(clearance, _status(now), False, now)
        self.assertTrue(policy.gap.found)
        self.assertGreater(policy.gap.width_deg, 0.0)

    def test_gap_seeking_is_not_used_when_a_forward_arc_exists(self) -> None:
        policy = AutonomousPolicy(0.0, 112)
        angles = np.arange(0.0, 360.0, 2.0, dtype=np.float32)
        ranges = np.full(angles.size, 3.0, dtype=np.float32)
        now = time.monotonic()
        policy.observe_scan(angles, ranges, now, now)
        clearance = SectorClearance(
            3.0, 3.0, 3.0, True, 3.0, 3.0, rear_m=3.0, scan_at=now
        )

        command = policy.decide(clearance, _status(now), False, now)

        self.assertEqual(command, "F")
        self.assertNotIn("GAP_SEEK", policy.reason)

    def test_alignment_tolerance_is_a_real_angle(self) -> None:
        self.assertGreater(GAP_ALIGNED_DEG, 0.0)
        self.assertLess(GAP_ALIGNED_DEG, 45.0)


class RateRampTests(unittest.TestCase):
    """Ramps are per second. Per-tick ramps silently coupled responsiveness to
    loop rate, so a tick spent rendering advanced the drive no further than an
    idle one and the chassis felt laggy exactly when the scene was busiest."""

    def test_a_longer_gap_between_decisions_ramps_further(self) -> None:
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        outputs = []
        for gap_s in (0.02, 0.08):
            policy = AutonomousPolicy(0.0, 200)
            now = time.monotonic()
            policy.decide(clear, _status(now), False, now)
            first = policy.left_pwm
            policy.decide(clear, _status(now + gap_s), False, now + gap_s)
            outputs.append(policy.left_pwm - first)
        self.assertGreater(outputs[1], outputs[0])

    def test_the_ramp_never_exceeds_its_rate(self) -> None:
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        policy = AutonomousPolicy(0.0, 200)
        now = time.monotonic()
        policy.decide(clear, _status(now), False, now)
        first = policy.left_pwm
        policy.decide(clear, _status(now + 0.05), False, now + 0.05)
        self.assertLessEqual(
            policy.left_pwm - first, math.ceil(MAX_PWM_RATE_PER_S * 0.05)
        )

    def test_rates_are_positive_and_bounded(self) -> None:
        self.assertGreater(MAX_STEERING_RATE_DPS, 0.0)
        self.assertGreater(MAX_PWM_RATE_PER_S, 0.0)


class RobotControlTests(unittest.TestCase):
    def test_a_fresh_control_is_autonomous(self) -> None:
        control = RobotControl()
        self.assertFalse(control.halted)
        self.assertFalse(control.manual)

    def test_halt_is_sticky_until_resumed(self) -> None:
        control = RobotControl()
        control.halt()
        self.assertTrue(control.halted)
        control.resume()
        self.assertFalse(control.halted)

    def test_drive_is_refused_outside_manual_mode(self) -> None:
        control = RobotControl()
        self.assertFalse(control.drive("F"))
        self.assertEqual(control.manual_command(), "STOP")

    def test_a_manual_command_expires_on_its_own(self) -> None:
        """A dropped phone, a closed tab or a walk out of Wi-Fi range has to
        stop the robot rather than leave it driving on the last thing it
        heard."""
        control = RobotControl()
        control.set_manual(True)
        now = time.monotonic()
        control.drive("F")
        self.assertEqual(control.manual_command(now), "F")
        self.assertEqual(
            control.manual_command(now + RobotControl.COMMAND_TTL_S + 0.1),
            "STOP",
        )

    def test_leaving_manual_mode_clears_a_held_command(self) -> None:
        control = RobotControl()
        control.set_manual(True)
        control.drive("F")
        control.set_manual(False)
        self.assertEqual(control.manual_command(), "STOP")

    def test_an_unknown_direction_is_rejected(self) -> None:
        control = RobotControl()
        control.set_manual(True)
        self.assertFalse(control.drive("LAUNCH"))

    def test_halting_drops_a_held_manual_command(self) -> None:
        control = RobotControl()
        control.set_manual(True)
        control.drive("F")
        control.halt()
        self.assertEqual(control.manual_command(), "STOP")


class OperatorControlPolicyTests(unittest.TestCase):
    def _clear(self, now: float) -> SectorClearance:
        return SectorClearance(
            2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=2.0, scan_at=now
        )

    def _policy(self) -> tuple[AutonomousPolicy, RobotControl]:
        policy = AutonomousPolicy(0.0, 112)
        control = RobotControl()
        policy.control = control
        return policy, control

    def test_a_halt_stops_the_robot(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        control.halt()

        command = policy.decide(self._clear(now), _status(now), False, now)

        self.assertEqual(command, "STOP")
        self.assertEqual(policy.reason, "STOP:HALTED_BY_OPERATOR")
        self.assertEqual((policy.left_pwm, policy.right_pwm), (0, 0))

    def test_resuming_returns_to_autonomy(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        control.halt()
        policy.decide(self._clear(now), _status(now), False, now)
        control.resume()

        command = policy.decide(
            self._clear(now + 0.1), _status(now + 0.1), False, now + 0.1
        )

        self.assertEqual(command, "F")

    def test_a_halt_outranks_stuck_recovery(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        policy._stuck_phase = "RECOVER"
        policy.stuck_phase = "RECOVER"
        control.halt()

        policy.decide(self._clear(now), _status(now), False, now)

        self.assertEqual(policy.stuck_phase, "IDLE")
        self.assertEqual((policy.left_pwm, policy.right_pwm), (0, 0))

    def test_manual_reverse_is_applied(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        control.set_manual(True)
        control.drive("B")

        for index in range(20):
            tick = now + index * 0.03
            policy.decide(self._clear(tick), _status(tick), False, tick)

        self.assertEqual(policy.reason, "MANUAL:B")
        self.assertLess(policy.left_pwm, 0)
        self.assertLess(policy.right_pwm, 0)

    def test_manual_pivots_counter_rotate(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        control.set_manual(True)
        control.drive("R")

        for index in range(20):
            tick = now + index * 0.03
            policy.decide(self._clear(tick), _status(tick), False, tick)

        self.assertGreater(policy.left_pwm, 0)
        self.assertLess(policy.right_pwm, 0)

    def test_manual_forward_is_refused_against_a_close_obstacle(self) -> None:
        """Manual driving keeps the independent safety layers underneath it.
        Reverse and pivots stay free, because driving out of somewhere the
        planner could not is the whole point."""
        policy, control = self._policy()
        now = time.monotonic()
        control.set_manual(True)
        control.drive("F")
        blocked = SectorClearance(
            0.20, 2.0, 2.0, True, 2.0, 2.0, rear_m=2.0, scan_at=now
        )

        command = policy.decide(blocked, _status(now), False, now)

        self.assertEqual(command, "STOP")
        self.assertEqual(policy.reason, "STOP:MANUAL_FORWARD_BLOCKED")

    def test_manual_reverse_is_allowed_against_a_close_front_obstacle(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        control.set_manual(True)
        control.drive("B")
        blocked = SectorClearance(
            0.20, 2.0, 2.0, True, 2.0, 2.0, rear_m=2.0, scan_at=now
        )

        for index in range(20):
            tick = now + index * 0.03
            policy.decide(blocked, _status(tick), False, tick)

        self.assertEqual(policy.reason, "MANUAL:B")
        self.assertLess(policy.left_pwm, 0)

    def test_manual_forward_is_refused_on_the_ultrasonic_alone(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        control.set_manual(True)
        control.drive("F")

        command = policy.decide(
            self._clear(now), _status(now, front_cm=12.0), False, now
        )

        self.assertEqual(command, "STOP")

    def test_an_expired_manual_command_stops_the_robot(self) -> None:
        policy, control = self._policy()
        now = time.monotonic()
        control.set_manual(True)
        control.drive("B")
        for index in range(20):
            tick = now + index * 0.03
            policy.decide(self._clear(tick), _status(tick), False, tick)
        self.assertLess(policy.left_pwm, 0)

        stale = now + RobotControl.COMMAND_TTL_S + 1.0
        for index in range(40):
            tick = stale + index * 0.03
            policy.decide(self._clear(tick), _status(tick), False, tick)

        self.assertEqual((policy.left_pwm, policy.right_pwm), (0, 0))

    def test_manual_forward_uses_the_movement_floor(self) -> None:
        self.assertEqual(MANUAL_DRIVE["F"], (MIN_MOVE_PWM, MIN_MOVE_PWM))
        self.assertEqual(MANUAL_DRIVE["STOP"], (0, 0))
        self.assertGreater(MANUAL_FORWARD_MIN_M, 0.0)


if __name__ == "__main__":
    unittest.main()
