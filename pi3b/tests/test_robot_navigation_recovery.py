"""Regression tests for the v1.9.18 navigation and recovery changes.

Each test here corresponds to an observed physical failure:

* the chassis pulsing drive/stop/drive and pivoting in place while the LD19
  could plainly see a broad opening a few degrees away,
* a wheel stall in open floor that never registered as being stuck, and
* the runtime continuing a route computed for a position it had been lifted
  out of.
"""

from __future__ import annotations

import pathlib
import sys
import time
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import (
    CLOSE_FRONT_TURN_M,
    DISPLACED_HOLD_S,
    ESCAPE_PIVOT_BOOST_PWM,
    MIN_MOVE_PWM,
    STEER_HEADINGS,
    ArduinoStatus,
    AutonomousPolicy,
    CameraMotionState,
    SectorClearance,
)
from robot_imu import IMUState
from robot_motion import ScanMotionResult


def _status(
    now: float, front_cm: float | None = None, blocked: bool = False
) -> ArduinoStatus:
    return ArduinoStatus(
        front_cm=front_cm, motion="S", received_at=now, blocked=blocked
    )


def _profile(straight_m: float, opening_m: float, span: tuple[float, float]):
    profile = np.full(STEER_HEADINGS.size, straight_m, dtype=np.float32)
    inside = (STEER_HEADINGS >= span[0]) & (STEER_HEADINGS <= span[1])
    profile[inside] = opening_m
    return profile


def _live_imu(**overrides) -> IMUState:
    base = dict(
        connected=True,
        calibrated=True,
        fresh=True,
        still_energy_g=0.005,
        motion_energy_g=0.050,
    )
    base.update(overrides)
    return IMUState(**base)


class SteerAroundTests(unittest.TestCase):
    """A short straight corridor beside a broad opening must be driven
    around, not answered with a stop-and-pivot escape."""

    def test_broad_side_opening_keeps_rolling(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = _profile(0.35, 2.0, (20.0, 60.0))
        clearance = SectorClearance(
            0.35, 2.0, 2.0, True, 2.0, 2.0, profile, rear_m=1.0
        )

        command = policy.decide(clearance, _status(time.monotonic()), False,
                                time.monotonic())

        self.assertEqual(command, "F")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)
        self.assertTrue(policy.reason.startswith("STEER_AROUND"))
        self.assertEqual(policy._escape_phase, "IDLE")

    def test_steer_around_turns_toward_the_opening(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = _profile(0.35, 2.0, (20.0, 60.0))
        clearance = SectorClearance(
            0.35, 2.0, 2.0, True, 2.0, 2.0, profile, rear_m=1.0
        )
        now = time.monotonic()
        for index in range(30):
            policy.decide(clearance, _status(now), False, now + index * 0.03)

        # Right-hand opening means the outer (left) wheel leads.
        self.assertGreater(policy.steering_deg, 0.0)
        self.assertGreater(policy.left_pwm, policy.right_pwm)

    def test_close_ultrasonic_still_forces_a_real_escape(self) -> None:
        """The Uno's forward cone keeps its authority: the steer-around band
        never runs while the near-field sensor is triggered."""
        policy = AutonomousPolicy(0.0, 118)
        profile = _profile(0.35, 2.0, (20.0, 60.0))
        clearance = SectorClearance(
            0.35, 2.0, 2.0, True, 2.0, 2.0, profile, rear_m=1.0
        )

        policy.decide(
            clearance, _status(time.monotonic(), front_cm=20.0), False,
            time.monotonic(),
        )

        self.assertIn("ESCAPE", policy.reason)
        self.assertNotIn("STEER_AROUND", policy.reason)

    def test_blocked_in_every_direction_still_escapes(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 0.30, dtype=np.float32)
        clearance = SectorClearance(
            0.30, 0.30, 0.30, True, 0.30, 0.30, profile, rear_m=1.0
        )

        policy.decide(clearance, _status(time.monotonic()), False,
                      time.monotonic())

        self.assertNotIn("STEER_AROUND", policy.reason)

    def test_a_narrow_gap_is_not_treated_as_an_opening(self) -> None:
        """Only a genuinely broad alternative avoids the escape machine; a
        single lucky ray between two objects must not."""
        policy = AutonomousPolicy(0.0, 118)
        profile = _profile(0.35, 2.0, (29.0, 31.0))
        clearance = SectorClearance(
            0.35, 2.0, 2.0, True, 2.0, 2.0, profile, rear_m=1.0
        )

        policy.decide(clearance, _status(time.monotonic()), False,
                      time.monotonic())

        self.assertNotIn("STEER_AROUND", policy.reason)

    def test_opening_straight_ahead_is_left_to_the_normal_planner(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 2.0, dtype=np.float32)
        clearance = SectorClearance(
            2.0, 2.0, 2.0, True, 2.0, 2.0, profile, rear_m=1.0
        )

        command = policy.decide(clearance, _status(time.monotonic()), False,
                                time.monotonic())

        self.assertEqual(command, "F")
        self.assertNotIn("STEER_AROUND", policy.reason)

    def test_steer_around_band_sits_below_the_front_turn_threshold(self) -> None:
        profile = _profile(0.35, 2.0, (20.0, 60.0))
        straight = int(np.argmin(np.abs(STEER_HEADINGS)))
        self.assertLess(float(profile[straight]), CLOSE_FRONT_TURN_M)


class PivotAuthorityTests(unittest.TestCase):
    def test_pivots_drive_above_the_straight_movement_floor(self) -> None:
        """A pivot scrubs both tyres, so min_move_pwm - measured for straight
        rolling - under-drives it and the turn times out instead of turning.

        Output is slew-rate limited, so the boost is reached over a few
        control ticks rather than in one step.
        """
        policy = AutonomousPolicy(0.0, 118, MIN_MOVE_PWM)
        for _ in range(30):
            policy._pivot_crawl("L")
        self.assertEqual(
            policy.left_pwm, -(MIN_MOVE_PWM + ESCAPE_PIVOT_BOOST_PWM)
        )
        self.assertEqual(policy.right_pwm, 0)

        centre = AutonomousPolicy(0.0, 118, MIN_MOVE_PWM)
        for _ in range(30):
            centre._center_pivot_crawl("R")
        self.assertEqual(
            centre.left_pwm, MIN_MOVE_PWM + ESCAPE_PIVOT_BOOST_PWM
        )
        self.assertEqual(
            centre.right_pwm, -(MIN_MOVE_PWM + ESCAPE_PIVOT_BOOST_PWM)
        )

    def test_pivot_pwm_stays_within_the_hardware_range(self) -> None:
        policy = AutonomousPolicy(0.0, 255, 250)
        self.assertLessEqual(policy._pivot_pwm(), 255)


class OpenFloorStuckTests(unittest.TestCase):
    """Every range metres away: the case where no previous evidence source
    could see a wheel stall."""

    def _drive_stalled(
        self,
        policy: AutonomousPolicy,
        ticks: int = 60,
        scan_verdict: str = "NOT_MOVING",
        front_cm: float | None = None,
        imu: IMUState | None = None,
    ) -> str:
        stalled_camera = CameraMotionState(
            fresh=True, confidence=0.9, motion_observed=False
        )
        now = time.monotonic()
        command = "F"
        for index in range(ticks):
            tick_now = now + index * 0.05
            clearance = SectorClearance(
                2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=1.0, scan_at=tick_now
            )
            if imu is not None:
                policy.observe_imu(imu)
            policy.observe_scan_motion(
                ScanMotionResult(verdict=scan_verdict), tick_now
            )
            command = policy.decide(
                clearance,
                _status(tick_now, front_cm=front_cm),
                False,
                tick_now,
                True,
                True,
                replace(stalled_camera, captured_at=tick_now),
            )
        return command

    def test_scan_and_camera_agreement_triggers_recovery(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        self._drive_stalled(policy)
        self.assertIn(policy.stuck_phase, ("RECOVER", "LATCHED"))
        self.assertIn("STUCK", policy.reason)

    def test_a_moving_scan_verdict_keeps_the_robot_driving(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        command = self._drive_stalled(policy, scan_verdict="MOVING")
        self.assertEqual(policy.stuck_phase, "IDLE")
        self.assertEqual(command, "F")

    def test_unchanging_ultrasonic_corroborates_a_stall(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        self._drive_stalled(policy, scan_verdict="UNKNOWN", front_cm=90.0)
        self.assertIn(policy.stuck_phase, ("RECOVER", "LATCHED"))

    def test_advancing_ultrasonic_is_evidence_of_progress(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        stalled_camera = CameraMotionState(
            fresh=True, confidence=0.9, motion_observed=False
        )
        now = time.monotonic()
        front = 150.0
        for index in range(60):
            tick_now = now + index * 0.05
            clearance = SectorClearance(
                2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=1.0, scan_at=tick_now
            )
            policy.decide(
                clearance,
                _status(tick_now, front_cm=front),
                False,
                tick_now,
                True,
                True,
                replace(stalled_camera, captured_at=tick_now),
            )
            front -= 1.0
        self.assertEqual(policy.stuck_phase, "IDLE")

    def test_imu_energy_at_the_stationary_floor_corroborates_a_stall(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        still = _live_imu(still_energy_g=0.005, motion_energy_g=0.006)
        self._drive_stalled(policy, scan_verdict="UNKNOWN", imu=still)
        self.assertIn(policy.stuck_phase, ("RECOVER", "LATCHED"))

    def test_imu_energy_never_votes_moving_on_its_own(self) -> None:
        """A stalled motor buzzing against a rug produces plenty of energy,
        so high energy must not be read as evidence of travel."""
        policy = AutonomousPolicy(0.0, 118)
        policy.observe_imu(_live_imu(motion_energy_g=0.50))
        self.assertEqual(policy._imu_energy_evidence(True), "UNKNOWN")

    def test_energy_evidence_is_silent_without_a_commanded_drive(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        policy.observe_imu(_live_imu(motion_energy_g=0.001))
        self.assertEqual(policy._imu_energy_evidence(False), "UNKNOWN")


class DisplacementTests(unittest.TestCase):
    """Being picked up invalidates every stored phase and the whole map."""

    def _clear(self, now: float) -> SectorClearance:
        return SectorClearance(
            2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=1.0, scan_at=now
        )

    def test_imu_handling_stops_and_counts_a_displacement(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy.observe_imu(_live_imu(handled=True))

        command = policy.decide(self._clear(now), _status(now), False, now)

        self.assertEqual(command, "STOP")
        self.assertEqual(policy.reason, "STOP:DISPLACED_REORIENT")
        self.assertEqual(policy.displacement_count, 1)
        self.assertEqual(policy.left_pwm, 0)
        self.assertEqual(policy.right_pwm, 0)

    def test_scan_displacement_stops_the_chassis(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy.observe_scan_motion(
            ScanMotionResult(displaced=True, displacement_m=0.9), now
        )

        policy.decide(self._clear(now), _status(now), False, now)

        self.assertEqual(policy.reason, "STOP:DISPLACED_REORIENT")
        self.assertEqual(policy.displacement_count, 1)

    def test_displacement_clears_a_latched_stuck_state(self) -> None:
        """The reported workflow: the robot latches STUCK, a person lifts it
        clear, and it must not resume the maneuver it latched on."""
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy._stuck_phase = "LATCHED"
        policy.stuck_phase = "LATCHED"
        policy._stuck_latched_at = now
        policy._escape_phase = "TURN"
        policy.observe_imu(_live_imu(handled=True))

        policy.decide(self._clear(now), _status(now), False, now)

        self.assertEqual(policy.stuck_phase, "IDLE")
        self.assertEqual(policy._escape_phase, "IDLE")

    def test_displacement_holds_still_briefly_then_resumes(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy.observe_imu(_live_imu(handled=True))
        policy.decide(self._clear(now), _status(now), False, now)

        policy.observe_imu(_live_imu(handled=False))
        settling = policy.decide(
            self._clear(now + 0.3), _status(now + 0.3), False, now + 0.3
        )
        self.assertEqual(settling, "STOP")
        self.assertEqual(policy.reason, "STOP:DISPLACED_SETTLING")

        later = now + DISPLACED_HOLD_S + 0.1
        resumed = policy.decide(
            self._clear(later), _status(later), False, later
        )
        self.assertEqual(resumed, "F")

    def test_one_displacement_episode_counts_once(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy.observe_imu(_live_imu(handled=True))
        for index in range(10):
            tick = now + index * 0.05
            policy.decide(self._clear(tick), _status(tick), False, tick)
        self.assertEqual(policy.displacement_count, 1)

    def test_steering_commitment_is_dropped_on_displacement(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        # Straight clearance below FORWARD_PREFERENCE_CLEARANCE_M so the
        # planner actually commits to the side opening rather than holding
        # the centre heading.
        profile = _profile(0.90, 2.5, (20.0, 60.0))
        clearance = SectorClearance(
            0.90, 2.0, 2.0, True, 2.0, 2.0, profile, rear_m=1.0
        )
        for index in range(20):
            policy.decide(clearance, _status(now), False, now + index * 0.03)
        self.assertNotEqual(policy.steering_deg, 0.0)

        later = now + 1.0
        policy.observe_imu(_live_imu(handled=True))
        policy.decide(clearance, _status(later), False, later)

        self.assertEqual(policy.steering_deg, 0.0)
        self.assertIsNone(policy._heading_index)


class IntentReportingTests(unittest.TestCase):
    def test_intent_describes_driving(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clearance = SectorClearance(2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=1.0)
        now = time.monotonic()

        policy.decide(clearance, _status(now), False, now)

        self.assertIn(policy.intent, ("DRIVING", "CRUISING", "EXPLORING"))
        self.assertTrue(policy.intent_detail)

    def test_intent_reports_being_displaced(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy.observe_imu(_live_imu(handled=True))
        clearance = SectorClearance(2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=1.0)

        policy.decide(clearance, _status(now), False, now)

        self.assertEqual(policy.intent, "REORIENTING")
        self.assertIn("picked up", policy.intent_detail)

    def test_intent_reports_a_latched_stuck_state(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        policy._stuck_phase = "LATCHED"
        policy._update_intent()

        self.assertEqual(policy.intent, "STUCK")
        self.assertIn("recovery exhausted", policy.intent_detail)

    def test_standby_intent_before_the_timer_expires(self) -> None:
        policy = AutonomousPolicy(5.0, 118)
        clearance = SectorClearance(2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=1.0)
        now = time.monotonic()

        policy.decide(clearance, _status(now), False, now)

        self.assertEqual(policy.intent, "STANDBY")


if __name__ == "__main__":
    unittest.main()
