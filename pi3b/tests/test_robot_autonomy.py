from __future__ import annotations

import math
import pathlib
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import (
    MAX_PWM,
    MIN_MOVE_PWM,
    RUNTIME_VERSION,
    STEER_HEADINGS,
    STUCK_RELATCH_RETRY_S,
    UNO_CONTROL_LEASE_S,
    WINDOW_TITLE,
    ArduinoLink,
    ArduinoStatus,
    AutonomousPolicy,
    CameraMotionState,
    CameraSafety,
    LD19Link,
    SectorClearance,
    _sector_clearance,
    corridor_profile,
    discover_arduino_port,
    discover_ld19_port,
    maximize_dashboard_window,
    open_dashboard_window,
)
from robot_imu import IMUState
from robot_slam_lite import SlamLiteState


class _Return:
    """Minimal stand-in for an LD19 point."""

    def __init__(self, angle_deg: float, distance_mm: int, confidence: int = 90) -> None:
        self.angle_deg = angle_deg
        self.distance_mm = distance_mm
        self.confidence = confidence


class SerialDiscoveryTests(unittest.TestCase):
    @staticmethod
    def _port(device: str, vid: int | None, pid: int | None, description: str = ""):
        return SimpleNamespace(
            device=device,
            vid=vid,
            pid=pid,
            description=description,
            manufacturer=None,
        )

    @mock.patch("robot_autonomy.list_ports.comports")
    def test_exact_robot_usb_identities_win_over_ambiguous_descriptions(self, comports) -> None:
        comports.return_value = [
            self._port("/dev/ttyUSB0", 0x10C4, 0xEA60, "USB UART"),
            self._port("/dev/ttyACM0", 0x2341, 0x0043, "USB Serial"),
            self._port("/dev/ttyUSB1", 0x1A86, 0x7523, "USB Serial"),
        ]
        self.assertEqual(discover_arduino_port(), "/dev/ttyACM0")
        self.assertEqual(discover_ld19_port("/dev/ttyACM0"), "/dev/ttyUSB0")

    @mock.patch("robot_autonomy.list_ports.comports")
    def test_unique_device_node_fallback_handles_missing_metadata(self, comports) -> None:
        comports.return_value = [
            self._port("/dev/ttyACM0", None, None),
            self._port("/dev/ttyUSB0", None, None),
        ]
        self.assertEqual(discover_arduino_port(), "/dev/ttyACM0")
        self.assertEqual(discover_ld19_port("/dev/ttyACM0"), "/dev/ttyUSB0")


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
        self.assertEqual(policy.right_pwm, 0)

    def test_ultrasonic_obstacle_prefers_bounded_turn_over_backing_out(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=time.monotonic())
        clear_sides = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        settle(policy, clear_sides, blocked, cycles=10)
        self.assertLess(policy.left_pwm, 0)
        self.assertEqual(policy.right_pwm, 0)
        self.assertEqual(policy._escape_phase, "TURN")

    def test_close_obstacle_enters_bounded_turn_without_reverse_cycle(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        clearance = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        outputs = []
        for index in range(48):
            policy.decide(clearance, blocked, False, now + index * 0.03)
            outputs.append((policy.left_pwm, policy.right_pwm))
        self.assertFalse(any(left < 0 and right < 0 for left, right in outputs))
        self.assertTrue(any((left < 0) != (right < 0) for left, right in outputs))
        self.assertNotEqual(outputs[-1], (0, 0))
        self.assertTrue(policy.reason.startswith("ESCAPE_TURN:L"))

    def test_imu_turn_angle_releases_recovery_into_forward_commit(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        close = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        policy.observe_imu(IMUState(
            connected=True, calibrated=True, fresh=True, yaw_deg=0.0
        ))
        policy.decide(close, blocked, False, now)
        policy.decide(close, blocked, False, now + 0.71)

        released = SectorClearance(0.75, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        clear_status = ArduinoStatus(front_cm=80.0, motion="S", received_at=now)
        policy.observe_imu(IMUState(
            connected=True, calibrated=True, fresh=True, yaw_deg=35.0
        ))
        policy.decide(released, clear_status, False, now + 0.80)
        policy.decide(released, clear_status, False, now + 0.83)
        self.assertEqual(policy.reason, "ESCAPE_COMMIT:L")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)

    def test_ld19_pose_bounds_recovery_when_imu_is_missing(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        close = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        policy.observe_imu(IMUState(error="missing"))
        policy.observe_pose(SlamLiteState(
            heading_deg=0.0,
            yaw_confidence=0.5,
            yaw_correction_deg=0.0,
            matched=True,
            map_updates=10,
            yaw_source="COMMAND+LD19",
        ))
        policy.decide(close, blocked, False, now)
        policy.decide(close, blocked, False, now + 0.71)

        released = SectorClearance(0.75, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        clear_status = ArduinoStatus(front_cm=80.0, motion="S", received_at=now)
        policy.observe_pose(SlamLiteState(
            heading_deg=35.0,
            yaw_confidence=0.5,
            yaw_correction_deg=0.0,
            matched=True,
            map_updates=11,
            yaw_source="COMMAND+LD19",
        ))
        policy.decide(released, clear_status, False, now + 0.80)
        policy.decide(released, clear_status, False, now + 0.83)

        self.assertEqual(policy.reason, "ESCAPE_COMMIT:L")
        self.assertEqual(policy._escape_turn_yaw_source, "LD19+COMMAND")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)

    def test_escape_turn_stops_at_hard_angle_when_no_exit_exists(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        no_right_exit = SectorClearance(
            0.30, 1.8, 0.20, True, 1.7, 0.20, None, 1.0
        )
        policy.observe_imu(IMUState(
            connected=True, calibrated=True, fresh=True, yaw_deg=0.0
        ))
        policy.decide(no_right_exit, blocked, False, now)
        policy.decide(no_right_exit, blocked, False, now + 0.71)
        policy.observe_imu(IMUState(
            connected=True, calibrated=True, fresh=True, yaw_deg=90.0
        ))
        self.assertEqual(
            policy.decide(no_right_exit, blocked, False, now + 0.80), "STOP"
        )
        self.assertEqual(policy.reason, "STOP:BOXED_IN")
        changed_geometry = SectorClearance(
            0.30, 0.20, 1.6, True, 0.20, 1.5, None, 1.0
        )
        self.assertEqual(
            policy.decide(changed_geometry, blocked, False, now + 1.30), "R"
        )
        self.assertEqual(policy.reason, "ESCAPE_OPENING_TURN:R")

    def test_ambiguous_sides_search_rear_then_use_new_opening(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        boxed = SectorClearance(0.30, 0.20, 0.20, True, 0.20, 0.20, None, 1.0)
        self.assertEqual(policy.decide(boxed, blocked, False, now), "B")
        self.assertEqual(policy.reason, "ESCAPE_REVERSE_SEARCH_ULTRASONIC")
        opened = SectorClearance(0.30, 1.5, 0.20, True, 1.4, 0.20, None, 1.0)
        self.assertEqual(policy.decide(opened, blocked, False, now + 0.53), "L")
        self.assertLess(policy.left_pwm, 0)
        self.assertEqual(policy.right_pwm, 0)

    def test_body_corridor_overrides_noncritical_broad_side_minimum(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        misleading_profile = np.full(
            STEER_HEADINGS.shape, 3.0, dtype=np.float32
        )
        clearance = SectorClearance(
            0.30, 0.20, 1.4, True, 0.20, 1.3, misleading_profile, 1.0
        )
        self.assertEqual(policy.decide(clearance, blocked, False, now), "L")
        self.assertEqual(policy.turn_command, "L")

    def test_no_forward_corridor_searches_clear_rear_before_boxed_stop(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        status = ArduinoStatus(front_cm=80.0, motion="S", received_at=now)
        no_corridor = SectorClearance(
            0.60, 1.5, 0.70, True, 1.4, 0.70,
            np.full(STEER_HEADINGS.shape, 0.25, dtype=np.float32), 1.0,
        )
        self.assertEqual(policy.decide(no_corridor, status, False, now), "B")
        self.assertEqual(policy.reason, "ESCAPE_REVERSE_SEARCH_LD19")

    def test_recovery_commits_to_selected_side_after_front_clears(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        close = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        policy.decide(close, blocked, False, now)
        clear_status = ArduinoStatus(front_cm=80.0, motion="S", received_at=now)
        released = SectorClearance(0.75, 1.8, 0.8, True, 1.7, 0.8, None, 1.0)
        self.assertEqual(policy.decide(released, clear_status, False, now + 0.20), "F")
        # Direction reversals intentionally pass through one zero-output cycle.
        policy.decide(released, clear_status, False, now + 0.23)
        self.assertEqual(policy.reason, "ESCAPE_COMMIT:L")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)

    def test_close_obstacle_with_blocked_rear_uses_clear_side_opening(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=time.monotonic())
        clearance = SectorClearance(0.30, 1.8, 0.8, True, 1.7, 0.8, None, 0.20)
        self.assertEqual(policy.decide(clearance, blocked, False, time.monotonic()), "L")
        self.assertLess(policy.left_pwm, 0)
        self.assertEqual(policy.right_pwm, 0)

    def test_close_obstacle_stops_when_rear_and_both_turn_sweeps_are_blocked(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=time.monotonic())
        clearance = SectorClearance(0.30, 0.24, 0.23, True, 0.22, 0.21, None, 0.20)
        self.assertEqual(policy.decide(clearance, blocked, False, time.monotonic()), "STOP")

    def test_rear_becoming_blocked_uses_lidar_cleared_centre_pivot(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        reverse_first = SectorClearance(
            0.30, 0.32, 0.20, True, 0.31, 0.20, None, 1.0
        )
        self.assertEqual(policy.decide(reverse_first, blocked, False, now), "L")
        self.assertEqual(policy._escape_phase, "REVERSE")
        self.assertLess(policy.left_pwm, 0)
        self.assertLess(policy.right_pwm, 0)

        left_open_rear_blocked = SectorClearance(
            0.30, 1.6, 0.20, True, 1.5, 0.20, None, 0.18
        )
        self.assertEqual(
            policy.decide(
                left_open_rear_blocked, blocked, False, now + 0.10
            ),
            "L",
        )
        policy.decide(left_open_rear_blocked, blocked, False, now + 0.14)
        self.assertEqual(policy._escape_phase, "TURN")
        self.assertEqual(policy.reason.split()[0], "ESCAPE_TURN:L")
        self.assertLess(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)

    def test_no_imu_side_opening_turn_flows_directly_into_forward_commit(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy.observe_pose(SlamLiteState(0.0, 0.8, 0.0, True, 10))
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=now)
        opening = SectorClearance(0.30, 1.8, 0.24, True, 1.7, 0.23, None, 0.20)
        self.assertEqual(policy.decide(opening, blocked, False, now), "L")

        policy.observe_pose(SlamLiteState(-32.0, 0.8, 0.0, True, 11))
        clear = SectorClearance(0.90, 1.6, 0.35, True, 1.5, 0.34, None, 0.22)
        released = ArduinoStatus(front_cm=90.0, motion="S", received_at=now + 0.4)
        command = policy.decide(clear, released, False, now + 0.4)
        self.assertEqual(command, "F")
        released = ArduinoStatus(front_cm=90.0, motion="S", received_at=now + 0.45)
        command = policy.decide(clear, released, False, now + 0.45)
        self.assertEqual(command, "F")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)

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

    def test_camera_person_flag_is_ignored_in_navigation_only_mode(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        self.assertEqual(policy.decide(clear, self.status, True, time.monotonic()), "F")

    def test_close_straight_return_pivots_into_open_body_corridor_before_contact(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 0.20, dtype=np.float32)
        profile[(STEER_HEADINGS >= 22.0) & (STEER_HEADINGS <= 38.0)] = 1.20
        clearance = SectorClearance(
            0.24, 0.35, 1.4, True, 0.30, 1.3, profile, 1.0
        )

        command = policy.decide(clearance, self.status, False, time.monotonic())

        self.assertEqual(command, "R")
        self.assertEqual(policy.left_pwm, 0)
        self.assertLessEqual(policy.right_pwm, -MIN_MOVE_PWM)
        self.assertEqual(policy._escape_phase, "TURN")

    def test_multi_obstacle_corridors_remain_continuous_forward_motion(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        outputs = []
        for index in range(80):
            profile = np.full(STEER_HEADINGS.size, 0.55, dtype=np.float32)
            centre = 32.0 if (index // 20) % 2 == 0 else -32.0
            profile[np.abs(STEER_HEADINGS - centre) <= 10.0] = 1.10
            clearance = SectorClearance(
                0.55, 0.9, 0.9, True, 0.8, 0.8, profile, 1.0
            )
            decision_at = now + index * 0.04
            status = ArduinoStatus(
                front_cm=80.0, motion="S", received_at=decision_at
            )
            command = policy.decide(
                clearance, status, False, decision_at
            )
            outputs.append((command, policy.left_pwm, policy.right_pwm))

        self.assertTrue(all(command == "F" for command, _left, _right in outputs))
        self.assertTrue(all(left > 0 and right > 0 for _command, left, right in outputs))
        self.assertEqual(policy._escape_phase, "IDLE")

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

    def test_detected_uncalibrated_imu_blocks_motion(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)

        command = policy.decide(
            clear,
            self.status,
            False,
            time.monotonic(),
            camera_ready=True,
            imu_ready=False,
        )

        self.assertEqual(command, "STOP")
        self.assertEqual(policy.reason, "STOP:IMU_CALIBRATING")

    def test_unseen_front_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        unseen = SectorClearance(None, 2.0, 2.0, True, 2.0, 2.0)
        self.assertEqual(policy.decide(unseen, self.status, False, time.monotonic()), "STOP")

    def test_unseen_fixed_front_sector_uses_live_open_corridor_profile(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 1.5, dtype=np.float32)
        clearance = SectorClearance(None, None, None, True, profile=profile, rear_m=1.0)

        self.assertEqual(
            policy.decide(clearance, self.status, False, time.monotonic()), "F"
        )

    def test_clear_path_moves_forward(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(1.2, 1.0, 1.0, True)
        self.assertEqual(policy.decide(clear, self.status, False, time.monotonic()), "F")

    def test_wide_open_heading_still_uses_a_forward_arc_not_a_pivot(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 0.70, dtype=np.float32)
        profile[(STEER_HEADINGS >= 58.0) & (STEER_HEADINGS <= 72.0)] = 3.0
        clearance = SectorClearance(1.0, 2.0, 2.0, True, 2.0, 2.0, profile)
        self.assertEqual(settle(policy, clearance, self.status), "F")
        self.assertGreater(policy.left_pwm, 0)
        self.assertGreater(policy.right_pwm, 0)
        self.assertGreaterEqual(min(policy.left_pwm, policy.right_pwm), MIN_MOVE_PWM)
        self.assertGreaterEqual(abs(policy.left_pwm - policy.right_pwm), 24)

    def test_multi_object_profile_cannot_immediately_flip_turn_direction(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        left_profile = np.full(STEER_HEADINGS.size, 0.60, dtype=np.float32)
        left_profile[np.abs(STEER_HEADINGS + 28.0) <= 7.0] = 2.0
        left = SectorClearance(0.70, 1.8, 1.6, True, 1.7, 1.5, left_profile)
        for index in range(4):
            policy.decide(left, self.status, False, now + index * 0.03)
        self.assertLess(policy.left_pwm, policy.right_pwm)

        right_profile = np.full(STEER_HEADINGS.size, 0.60, dtype=np.float32)
        right_profile[np.abs(STEER_HEADINGS + 28.0) <= 7.0] = 0.50
        right_profile[np.abs(STEER_HEADINGS - 28.0) <= 7.0] = 2.5
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
        side_profile = np.full(STEER_HEADINGS.size, 0.60, dtype=np.float32)
        side_profile[np.abs(STEER_HEADINGS + 28.0) <= 7.0] = 2.0
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

    def test_fast_measured_yaw_removes_additional_turn_split(self) -> None:
        baseline = AutonomousPolicy(0.0, 118)
        limited = AutonomousPolicy(0.0, 118)
        limited.observe_imu(IMUState(
            connected=True,
            calibrated=True,
            fresh=True,
            gyro_z_dps=60.0,
        ))
        for _index in range(20):
            baseline._differential(118, 40.0)
            limited._differential(118, 40.0)
        self.assertGreater(abs(baseline.left_pwm - baseline.right_pwm), 0)
        self.assertEqual(limited.left_pwm, limited.right_pwm)
        self.assertTrue(limited.imu_limited)


class ArduinoLinkTests(unittest.TestCase):
    def test_published_drive_survives_bounded_half_second_control_stall(self) -> None:
        link = object.__new__(ArduinoLink)
        link._drive_lock = threading.Lock()
        link._drive_command = "STOP"
        link._drive_lease_until = 0.0
        link._drive_last_write = 0.0
        link._drive_expired = True
        link._write = mock.Mock(return_value=True)

        with mock.patch("robot_autonomy.time.monotonic", return_value=10.0):
            link.publish_drive(110, 112)

        self.assertEqual(link._drive_lease_until, 10.0 + UNO_CONTROL_LEASE_S)
        self.assertEqual(link._heartbeat_command(10.45), "DRIVE 110 112")
        self.assertEqual(link._heartbeat_command(10.51), "STOP")

    def test_drive_heartbeat_refreshes_lease_then_expires_to_stop(self) -> None:
        link = object.__new__(ArduinoLink)
        link._drive_lock = __import__("threading").Lock()
        link._drive_command = "DRIVE 110 112"
        link._drive_lease_until = 10.4
        link._drive_last_write = 10.0
        link._drive_expired = False

        self.assertEqual(link._heartbeat_command(10.1), "DRIVE 110 112")
        self.assertEqual(link._heartbeat_command(10.5), "STOP")
        self.assertIsNone(link._heartbeat_command(10.6))

    def test_capability_handshake_retries_until_drive_is_confirmed(self) -> None:
        link = object.__new__(ArduinoLink)
        link._serial = mock.Mock()
        link._supports_differential = False
        link._last_caps_sent_at = float("-inf")
        commands = []
        link.send = commands.append

        link.poll_capabilities(10.0)
        link.poll_capabilities(10.1)
        link.poll_capabilities(10.6)
        self.assertEqual(commands, ["CAPS", "CAPS"])

        link._supports_differential = True
        link.poll_capabilities(11.2)
        self.assertEqual(commands, ["CAPS", "CAPS"])

    def test_write_io_error_drops_port_without_crashing_runtime(self) -> None:
        connection = mock.Mock()
        connection.is_open = True
        connection.write.side_effect = OSError(5, "Input/output error")
        link = object.__new__(ArduinoLink)
        link._write_lock = threading.Lock()
        link._running = True
        link._serial = connection
        link._supports_differential = True
        link._status = ArduinoStatus(80.0, "F", 10.0, False, 110, 112)
        link._last_io_error = None
        link._next_reconnect_at = 0.0

        self.assertFalse(link._write("STOP"))
        self.assertIsNone(link._serial)
        self.assertFalse(link.differential_ready)
        self.assertEqual(link.status().motion, "S")
        self.assertEqual(link.status().received_at, 0.0)
        self.assertIn("Input/output error", link._last_io_error)
        connection.close.assert_called_once()

    @mock.patch("robot_autonomy.time.sleep")
    @mock.patch("robot_autonomy.serial.Serial")
    @mock.patch("robot_autonomy.discover_arduino_port", return_value="/dev/ttyACM2")
    def test_reconnect_rediscovers_current_uno_node_and_restarts_handshake(
        self, discover, serial_open, sleep
    ) -> None:
        connection = mock.Mock()
        connection.is_open = True
        serial_open.return_value = connection
        link = object.__new__(ArduinoLink)
        link._port = "/dev/ttyACM0"
        link._serial = None
        link._running = True
        link._write_lock = threading.Lock()
        link._supports_differential = False
        link._last_caps_sent_at = float("-inf")
        link._last_io_error = "Input/output error"
        link._next_reconnect_at = 9.0

        link.poll_capabilities(10.0)

        discover.assert_called_once_with()
        serial_open.assert_called_once_with(
            "/dev/ttyACM2", 115200, timeout=0.05, write_timeout=0.2
        )
        sleep.assert_called_once_with(2.1)
        self.assertEqual(link._port, "/dev/ttyACM2")
        self.assertEqual(
            connection.write.call_args_list,
            [mock.call(b"STOP\n"), mock.call(b"CAPS\n")],
        )

    def test_status_parser_exposes_actual_motor_outputs_and_hard_stop(self) -> None:
        status = ArduinoLink._parse_status(
            "STATUS motion=F front_cm=14.8 left_pwm=0 right_pwm=0 blocked=1",
            123.0,
        )
        self.assertEqual(status.front_cm, 14.8)
        self.assertEqual(status.motion, "F")
        self.assertEqual(status.left_pwm, 0)
        self.assertEqual(status.right_pwm, 0)
        self.assertTrue(status.blocked)
        self.assertEqual(status.received_at, 123.0)

    def test_ld19_reader_treats_closed_descriptor_as_shutdown(self) -> None:
        link = object.__new__(LD19Link)
        link._running = True
        link._serial = mock.Mock()
        link._serial.in_waiting = None
        link._read_loop()

    def test_ld19_close_joins_reader_before_closing_serial(self) -> None:
        events = []
        link = object.__new__(LD19Link)
        link._running = True
        link._thread = mock.Mock()
        link._thread.join.side_effect = lambda timeout: events.append(("join", timeout))
        link._serial = mock.Mock()
        link._serial.close.side_effect = lambda: events.append(("close", None))
        link.close()
        self.assertEqual(events, [("join", 0.5), ("close", None)])


class CameraSafetyTests(unittest.TestCase):
    def test_camera_capture_can_wait_for_imu_calibration(self) -> None:
        camera = mock.Mock()
        with (
            mock.patch.object(CameraSafety, "_candidate_sources", return_value=["0"]),
            mock.patch("robot_autonomy.LatestCamera", return_value=camera) as latest,
        ):
            safety = CameraSafety("auto", 62.0, auto_start=False)
            latest.assert_not_called()
            safety.start(10.0)
            latest.assert_called_once()
            self.assertIs(safety.camera, camera)
            safety.start(11.0)
            latest.assert_called_once()
            safety.close()

    def test_reopened_camera_gets_its_own_startup_timeout(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        first = mock.Mock()
        first.error = ""
        first.latest.return_value = (1, frame, 1.0)
        second = mock.Mock()
        second.error = ""
        second.latest.return_value = (0, None, 0.0)

        with (
            mock.patch.object(CameraSafety, "_candidate_sources", return_value=["0"]),
            mock.patch("robot_autonomy.LatestCamera", side_effect=[first, second]),
        ):
            with mock.patch("robot_autonomy.time.monotonic", return_value=1.0):
                safety = CameraSafety("auto", 62.0)
            with mock.patch("robot_autonomy.time.monotonic", return_value=1.1):
                safety.tick()
            first.error = "unplugged"
            with mock.patch("robot_autonomy.time.monotonic", return_value=3.1):
                safety.tick()
            with mock.patch("robot_autonomy.time.monotonic", return_value=4.2):
                safety.tick()
            with mock.patch("robot_autonomy.time.monotonic", return_value=4.3):
                safety.tick()

            self.assertIs(safety.camera, second)
            second.close.assert_not_called()
            second.latest.return_value = (1, frame, 4.4)
            with mock.patch("robot_autonomy.time.monotonic", return_value=4.4):
                safety.tick()
            self.assertTrue(safety.ready(4.4))
            safety.close()

    def test_stationary_camera_frame_skips_optical_flow_work(self) -> None:
        safety = object.__new__(CameraSafety)
        safety._flow_gray = np.ones((120, 160), dtype=np.uint8)
        safety._flow_at = 1.0
        safety.motion = CameraMotionState(fresh=True)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        with mock.patch("robot_autonomy.cv2.resize") as resize:
            safety._update_motion(frame, 2.0, 0, 0)

        resize.assert_not_called()
        self.assertIsNone(safety._flow_gray)
        self.assertFalse(safety.motion.fresh)

    def test_recent_live_frame_bridges_a_short_camera_usb_reset(self) -> None:
        safety = object.__new__(CameraSafety)
        safety._has_live_frame = True
        safety._last_frame_at = 10.0
        safety.camera = None
        safety.frame = None

        self.assertTrue(safety.ready(10.80))
        self.assertFalse(safety.ready(11.01))

    def test_optical_flow_supplies_non_imu_turn_measurement(self) -> None:
        safety = object.__new__(CameraSafety)
        safety._fov = 70.0
        safety._flow_gray = None
        safety._flow_at = 0.0
        safety.motion = None
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        shifted = np.roll(frame, -4, axis=1)

        safety._update_motion(frame, 1.0, 118, 90)
        safety._update_motion(shifted, 1.1, 118, 90)

        self.assertTrue(safety.motion.fresh)
        self.assertGreater(safety.motion.confidence, 0.25)
        self.assertIsNotNone(safety.motion.yaw_rate_dps)
        self.assertGreater(safety.motion.yaw_rate_dps, 0.0)

    def test_dashboard_window_opens_full_screen(self) -> None:
        with (
            mock.patch("robot_autonomy.cv2.namedWindow") as named,
            mock.patch("robot_autonomy.cv2.moveWindow") as move,
            mock.patch("robot_autonomy.cv2.setWindowProperty") as fullscreen,
        ):
            open_dashboard_window()
        named.assert_called_once()
        move.assert_called_once_with(WINDOW_TITLE, 0, 0)
        fullscreen.assert_called_once()

    def test_dashboard_fullscreen_can_be_reapplied_after_first_frame(self) -> None:
        with (
            mock.patch("robot_autonomy.cv2.moveWindow") as move,
            mock.patch("robot_autonomy.cv2.setWindowProperty") as fullscreen,
        ):
            maximize_dashboard_window()
        move.assert_called_once_with(WINDOW_TITLE, 0, 0)
        fullscreen.assert_called_once_with(
            WINDOW_TITLE,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    def test_missing_camera_keeps_runtime_in_safe_stale_state(self) -> None:
        with (
            mock.patch.object(CameraSafety, "_candidate_sources", return_value=["0", "1"]),
            mock.patch("robot_autonomy.LatestCamera", side_effect=RuntimeError("not available")),
        ):
            safety = CameraSafety("auto", 62.0)
            self.assertIsNone(safety.camera)
            self.assertFalse(safety.ready(time.monotonic()))
            self.assertFalse(hasattr(safety, "worker"))
            safety.tick()
            safety.close()

    def test_auto_camera_moves_to_next_usable_video_node(self) -> None:
        working_camera = mock.Mock()
        working_camera.error = ""
        with (
            mock.patch.object(CameraSafety, "_candidate_sources", return_value=["0", "1"]),
            mock.patch(
                "robot_autonomy.LatestCamera",
                side_effect=[RuntimeError("index zero failed"), working_camera],
            ),
        ):
            safety = CameraSafety("auto", 62.0)
            self.assertIs(safety.camera, working_camera)
            self.assertEqual(safety.camera_source, "1")
            safety.close()
            working_camera.close.assert_called_once()


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

    def test_rear_corridor_ignores_obstacle_outside_robot_width(self) -> None:
        points = [
            (0, _Return(180, 1200)),
            (1, _Return(177, 1200)),
            (2, _Return(135, 220)),
            (3, _Return(138, 225)),
        ]
        rear = corridor_profile(points, np.array([180.0], dtype=np.float32))[0]
        self.assertGreater(float(rear), 1.0)

    def test_isolated_lidar_speckle_does_not_block_a_clear_corridor(self) -> None:
        speckle = [(0, _Return(0, 180, confidence=20))]
        profile = corridor_profile(speckle)
        straight = profile[int(np.argmin(np.abs(STEER_HEADINGS)))]
        self.assertGreater(float(straight), 2.0)

    def test_unconfirmed_single_return_does_not_create_stop_step(self) -> None:
        thin_leg = [(0, _Return(0, 500, confidence=120))]
        profile = corridor_profile(thin_leg)
        straight = profile[int(np.argmin(np.abs(STEER_HEADINGS)))]
        self.assertGreater(float(straight), 2.0)

    def test_very_strong_close_single_return_is_retained(self) -> None:
        thin_leg = [(0, _Return(0, 500, confidence=220))]
        profile = corridor_profile(thin_leg)
        straight = profile[int(np.argmin(np.abs(STEER_HEADINGS)))]
        self.assertLess(float(straight), 0.50)


class SpeedGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = ArduinoStatus(front_cm=80.0, motion="S", received_at=time.monotonic())

    def _drive(self, front_m: float, speed: int = 118) -> AutonomousPolicy:
        policy = AutonomousPolicy(0.0, speed)
        clearance = SectorClearance(front_m, 2.0, 2.0, True, 2.0, 2.0,
                                    np.full(STEER_HEADINGS.size, front_m, dtype=np.float32))
        status = ArduinoStatus(
            front_cm=80.0,
            motion="S",
            received_at=time.monotonic(),
        )
        settle(policy, clearance, status)
        return policy

    def test_speed_rises_with_clearance(self) -> None:
        """The reported failure: one flat-out speed regardless of surroundings."""
        speeds = [(self._drive(d).left_pwm + self._drive(d).right_pwm) / 2.0
                  for d in (0.80, 1.20, 1.60, 2.40)]
        self.assertEqual(speeds, sorted(speeds), f"not monotonic: {speeds}")
        self.assertLess(speeds[0], speeds[-1])

    def test_straight_cruise_never_exceeds_the_configured_ceiling(self) -> None:
        for front_m in (0.80, 1.20, 2.00, 3.00):
            policy = self._drive(front_m, speed=118)
            if policy.left_pwm == policy.right_pwm:
                self.assertLessEqual(policy.left_pwm, 118)

    def test_turning_adds_yaw_without_increasing_average_speed(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clearance = SectorClearance(0.80, 2.0, 2.0, True, 2.0, 2.0,
                                    corridor_profile(wall_scene(0.80, gap=(12, 28))))
        settle(policy, clearance, self.status)
        self.assertLessEqual((policy.left_pwm + policy.right_pwm) / 2.0, 119.0)
        self.assertNotEqual(policy.left_pwm, policy.right_pwm)

    def test_governed_output_still_clears_the_stall_floor(self) -> None:
        for front_m in (0.60, 0.90, 1.40, 2.50):
            policy = self._drive(front_m)
            for value in (policy.left_pwm, policy.right_pwm):
                self.assertTrue(value == 0 or abs(value) >= MIN_MOVE_PWM,
                                f"{front_m} m produced stalling PWM {value}")


class NavigationSimulationTests(unittest.TestCase):
    def test_close_obstacle_recovery_turns_and_resumes_exploration(self) -> None:
        """A close obstacle must not leave the policy permanently stopped."""
        policy = AutonomousPolicy(0.0, 118)
        x = y = heading = previous_heading = 0.0
        obstacle_x, obstacle_y, obstacle_radius = 0.0, 0.45, 0.16
        moved_after_recovery = False
        entered_turn = False
        collided = False

        for step in range(160):
            points = []
            for angle_deg in range(0, 360, 2):
                distance = 3.5
                ray = heading + math.radians(angle_deg)
                dx, dy = math.sin(ray), math.cos(ray)
                ox, oy = x - obstacle_x, y - obstacle_y
                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - obstacle_radius * obstacle_radius
                discriminant = b * b - 4.0 * c
                if discriminant >= 0.0:
                    hit = (-b - math.sqrt(discriminant)) / 2.0
                    if 0.08 <= hit < distance:
                        distance = hit
                points.append((
                    angle_deg,
                    _Return(angle_deg, int(distance * 1000), confidence=120),
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
            now = policy.started_at + step * 0.1
            # A live gyro reports the rotation the chassis actually just made,
            # not just the derived heading used elsewhere in this simulation;
            # keep it consistent so the stuck detector's IMU-progress source
            # sees a real turn rate during the escape turn below instead of a
            # permanently zero rate that would misread as not moving.
            gyro_z_dps = -math.degrees(heading - previous_heading) / 0.1
            previous_heading = heading
            policy.observe_imu(IMUState(
                connected=True,
                calibrated=True,
                fresh=True,
                yaw_deg=-math.degrees(heading),
                gyro_z_dps=gyro_z_dps,
            ))
            front_cm = None if clearance.front_m is None else clearance.front_m * 100.0
            status = ArduinoStatus(front_cm=front_cm, motion="S", received_at=now)
            policy.decide(clearance, status, False, now, True)
            entered_turn = entered_turn or policy.reason.startswith("ESCAPE_TURN")
            if entered_turn and policy.left_pwm > 0 and policy.right_pwm > 0:
                moved_after_recovery = True

            left_speed = policy.left_pwm / 255.0 * 0.26
            right_speed = policy.right_pwm / 255.0 * 0.26
            linear_speed = (left_speed + right_speed) / 2.0
            heading += ((left_speed - right_speed) / 0.14) * 0.1
            x += math.sin(heading) * linear_speed * 0.1
            y += math.cos(heading) * linear_speed * 0.1
            if math.hypot(x - obstacle_x, y - obstacle_y) <= obstacle_radius + 0.10:
                collided = True
                break

        self.assertFalse(collided)
        self.assertTrue(entered_turn)
        self.assertTrue(moved_after_recovery)
        self.assertGreater(math.hypot(x, y), 0.65)

    def test_closed_loop_arc_clears_a_central_obstacle(self) -> None:
        """Approximate kinematics catch a planner that only commands wide arcs."""
        policy = AutonomousPolicy(0.0, 118)
        x = y = heading = 0.0
        obstacle_x, obstacle_y, obstacle_radius = 0.0, 1.2, 0.20
        collided = False

        for step in range(120):
            points = []
            for angle_deg in range(0, 360, 2):
                distance = 3.5
                ray = heading + math.radians(angle_deg)
                dx, dy = math.sin(ray), math.cos(ray)
                ox, oy = x - obstacle_x, y - obstacle_y
                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - obstacle_radius * obstacle_radius
                discriminant = b * b - 4.0 * c
                if discriminant >= 0.0:
                    hit = (-b - math.sqrt(discriminant)) / 2.0
                    if 0.08 <= hit < distance:
                        distance = hit
                points.append((
                    angle_deg,
                    _Return(angle_deg, int(distance * 1000), confidence=120),
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
            now = policy.started_at + step * 0.1
            status = ArduinoStatus(front_cm=None, motion="S", received_at=now)
            policy.decide(clearance, status, False, now, True)

            left_speed = policy.left_pwm / 255.0 * 0.26
            right_speed = policy.right_pwm / 255.0 * 0.26
            linear_speed = (left_speed + right_speed) / 2.0
            heading += ((left_speed - right_speed) / 0.14) * 0.1
            x += math.sin(heading) * linear_speed * 0.1
            y += math.cos(heading) * linear_speed * 0.1
            if math.hypot(x - obstacle_x, y - obstacle_y) <= obstacle_radius + 0.10:
                collided = True
                break

        self.assertFalse(collided)
        self.assertGreater(y, 1.25)


class DashboardVersionTests(unittest.TestCase):
    def test_runtime_version_matches_the_version_file_and_reaches_the_dashboard(self) -> None:
        version_file = pathlib.Path(__file__).resolve().parents[1] / "VERSION"
        self.assertEqual(RUNTIME_VERSION, version_file.read_text(encoding="utf-8").strip())
        self.assertIn(RUNTIME_VERSION, WINDOW_TITLE)


class TurnSideScoreTests(unittest.TestCase):
    """_turn_side_score must not be fooled by a single lucky LiDAR ray."""

    def test_ignores_a_single_ray_spike_in_an_otherwise_narrow_sector(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 0.20, dtype=np.float32)
        profile[int(np.argmin(np.abs(STEER_HEADINGS - 40.0)))] = 3.0
        clearance = SectorClearance(0.30, 1.8, 0.20, True, 1.7, 0.19, profile, 1.0)

        score = policy._turn_side_score(clearance, "R")

        self.assertLess(score, 0.25)

    def test_recognizes_a_genuinely_broad_opening(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        profile = np.full(STEER_HEADINGS.size, 0.20, dtype=np.float32)
        profile[STEER_HEADINGS >= 30.0] = 3.0
        clearance = SectorClearance(0.30, 1.8, 0.20, True, 1.7, 0.19, profile, 1.0)

        score = policy._turn_side_score(clearance, "R")

        self.assertGreater(score, 1.0)


class StuckDetectionTests(unittest.TestCase):
    """Independent motion evidence, and recovery when it disagrees with a
    commanded drive for long enough. See robot_autonomy.py's "Stuck
    detection" section for the MOVING/NOT_MOVING/UNKNOWN contract."""

    def _status(self, now: float, blocked: bool = False) -> ArduinoStatus:
        return ArduinoStatus(front_cm=80.0, motion="S", received_at=now, blocked=blocked)

    def test_all_unknown_evidence_never_latches_stuck(self) -> None:
        """A static test fixture that never reports real progress must not
        misread as being stuck: every source has to opt in with a real
        signal (fresh camera flow, a live IMU, a real scan timestamp, or a
        Uno block) before silence counts as evidence."""
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(1.0, 2.0, 2.0, True, 2.0, 2.0)
        now = time.monotonic()
        for index in range(80):
            tick_now = now + index * 0.05
            policy.decide(clear, self._status(tick_now), False, tick_now)
        self.assertEqual(policy.stuck_phase, "IDLE")

    def test_camera_confirms_no_motion_triggers_recovery(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        stalled_camera = CameraMotionState(fresh=True, confidence=0.9, motion_observed=False)
        now = time.monotonic()
        command = "F"
        for index in range(60):
            tick_now = now + index * 0.05
            command = policy.decide(
                clear, self._status(tick_now), False, tick_now, True, True, stalled_camera
            )
        self.assertEqual(policy.stuck_phase, "RECOVER")
        self.assertTrue(policy.reason.startswith("STUCK_RECOVER_"))
        self.assertNotEqual(command, "F")

    def test_moving_vote_resets_accumulating_suspicion(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        stalled = CameraMotionState(fresh=True, confidence=0.9, motion_observed=False)
        moving = CameraMotionState(fresh=True, confidence=0.9, motion_observed=True)
        now = time.monotonic()
        for index in range(20):
            tick_now = now + index * 0.05
            policy.decide(clear, self._status(tick_now), False, tick_now, True, True, stalled)
        self.assertIsNotNone(policy._stuck_window_start)

        tick_now = now + 20 * 0.05
        policy.decide(clear, self._status(tick_now), False, tick_now, True, True, moving)

        self.assertIsNone(policy._stuck_window_start)
        self.assertEqual(policy.stuck_phase, "IDLE")

    def test_a_stale_sensor_stop_is_not_overridden_while_recovering(self) -> None:
        """decide()'s own safety stops (stale camera/LiDAR/Uno status,
        standby, IMU calibrating, an existing boxed-in stop) must win over an
        in-progress stuck-recovery maneuver: driving through one would be
        blind driving, and a single stale LD19 scan would otherwise make
        every candidate maneuver look unsafe and latch STUCK permanently."""
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        stalled_camera = CameraMotionState(fresh=True, confidence=0.9, motion_observed=False)
        now = time.monotonic()
        for index in range(40):
            tick_now = now + index * 0.05
            policy.decide(
                clear, self._status(tick_now), False, tick_now, True, True, stalled_camera
            )
        self.assertEqual(policy.stuck_phase, "RECOVER")

        tick_now = now + 40 * 0.05
        command = policy.decide(
            clear,
            self._status(tick_now),
            False,
            tick_now,
            camera_ready=False,
            imu_ready=True,
            camera_motion=stalled_camera,
        )

        self.assertEqual(command, "STOP")
        self.assertEqual(policy.reason, "STOP:CAMERA_STALE")
        self.assertEqual(policy.left_pwm, 0)
        self.assertEqual(policy.right_pwm, 0)

    def test_recovery_exhausts_attempts_and_latches_with_a_clear_reason(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        stalled_camera = CameraMotionState(fresh=True, confidence=0.9, motion_observed=False)
        now = time.monotonic()
        command = "F"
        for index in range(140):
            tick_now = now + index * 0.05
            command = policy.decide(
                clear, self._status(tick_now), False, tick_now, True, True, stalled_camera
            )
        self.assertEqual(policy.stuck_phase, "LATCHED")
        self.assertEqual(command, "STOP")
        self.assertIn("NEEDS_RESET", policy.reason)
        self.assertEqual(policy.left_pwm, 0)
        self.assertEqual(policy.right_pwm, 0)

    def test_no_safe_recovery_candidates_latches_immediately(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        boxed = SectorClearance(0.30, 0.24, 0.23, True, 0.22, 0.21, None, 0.20)

        command = policy._advance_stuck_recovery(boxed, time.monotonic())

        self.assertEqual(command, "STOP")
        self.assertEqual(policy.stuck_phase, "LATCHED")
        self.assertIn("NEEDS_RESET", policy.reason)

    def test_latched_stuck_clears_on_external_motion_and_resumes(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy._stuck_phase = "LATCHED"
        policy._stuck_latched_reason = "STOP:STUCK_STALLED_NEEDS_RESET"
        policy._stuck_latched_at = now
        policy.imu_yaw_rate_dps = 20.0  # well above STUCK_EXTERNAL_YAW_RATE_DPS
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)

        command = policy.decide(clear, self._status(now), False, now)

        self.assertEqual(policy.stuck_phase, "IDLE")
        self.assertEqual(command, "F")

    def test_latched_stuck_holds_without_external_motion(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy._stuck_phase = "LATCHED"
        policy._stuck_latched_reason = "STOP:STUCK_STALLED_NEEDS_RESET"
        policy._stuck_latched_at = now
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)

        command = policy.decide(clear, self._status(now), False, now)

        self.assertEqual(command, "STOP")
        self.assertEqual(policy.stuck_phase, "LATCHED")

    def test_latched_stuck_re_arms_after_a_timeout_without_an_imu(self) -> None:
        """Without a live IMU, _external_motion_detected() has nothing to
        observe, so a periodic re-arm is the only way a latch clears on its
        own; confirm it does, and that a still-genuinely-stuck chassis
        re-latches after another bounded attempt burst rather than driving
        forever."""
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy._stuck_phase = "LATCHED"
        policy._stuck_latched_reason = "STOP:STUCK_STALLED_NEEDS_RESET"
        policy._stuck_latched_at = now
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)

        re_armed_at = now + STUCK_RELATCH_RETRY_S + 0.01
        command = policy.decide(clear, self._status(re_armed_at), False, re_armed_at)

        self.assertNotEqual(policy.stuck_phase, "LATCHED")
        self.assertEqual(command, "F")

    def test_still_stuck_after_re_arm_latches_again(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        policy._stuck_phase = "LATCHED"
        policy._stuck_latched_reason = "STOP:STUCK_STALLED_NEEDS_RESET"
        policy._stuck_latched_at = now
        clear = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        stalled_camera = CameraMotionState(fresh=True, confidence=0.9, motion_observed=False)

        re_armed_at = now + STUCK_RELATCH_RETRY_S + 0.01
        command = "F"
        for index in range(140):
            tick_now = re_armed_at + index * 0.05
            command = policy.decide(
                clear, self._status(tick_now), False, tick_now, True, True, stalled_camera
            )

        self.assertEqual(policy.stuck_phase, "LATCHED")
        self.assertEqual(command, "STOP")

    def test_camera_evidence_requires_fresh_confident_translation(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        policy.left_pwm, policy.right_pwm = 118, 118
        lidar = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        arduino = self._status(time.monotonic())
        now = time.monotonic()

        stale = CameraMotionState(fresh=False, confidence=0.9, motion_observed=False)
        self.assertEqual(policy._motion_evidence(lidar, arduino, now, stale)["camera"], "UNKNOWN")

        low_confidence = CameraMotionState(fresh=True, confidence=0.1, motion_observed=False)
        self.assertEqual(
            policy._motion_evidence(lidar, arduino, now, low_confidence)["camera"], "UNKNOWN"
        )

        confident_still = CameraMotionState(fresh=True, confidence=0.9, motion_observed=False)
        self.assertEqual(
            policy._motion_evidence(lidar, arduino, now, confident_still)["camera"], "NOT_MOVING"
        )

        confident_moving = CameraMotionState(fresh=True, confidence=0.9, motion_observed=True)
        self.assertEqual(
            policy._motion_evidence(lidar, arduino, now, confident_moving)["camera"], "MOVING"
        )

        policy.left_pwm, policy.right_pwm = -118, 0  # pivot, not a translation
        self.assertEqual(
            policy._motion_evidence(lidar, arduino, now, confident_moving)["camera"], "UNKNOWN"
        )

    def test_imu_evidence_requires_a_meaningful_commanded_turn(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        lidar = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        arduino = self._status(time.monotonic())
        now = time.monotonic()

        policy.left_pwm, policy.right_pwm = 118, 118  # straight, not a turn
        policy.imu_yaw_rate_dps = 0.0
        self.assertEqual(policy._motion_evidence(lidar, arduino, now, None)["imu"], "UNKNOWN")

        policy.left_pwm, policy.right_pwm = -105, 105  # pivot
        policy.imu_yaw_rate_dps = None  # IMU not ready
        self.assertEqual(policy._motion_evidence(lidar, arduino, now, None)["imu"], "UNKNOWN")

        policy.imu_yaw_rate_dps = 0.5
        self.assertEqual(policy._motion_evidence(lidar, arduino, now, None)["imu"], "NOT_MOVING")

        policy.imu_yaw_rate_dps = 30.0
        self.assertEqual(policy._motion_evidence(lidar, arduino, now, None)["imu"], "MOVING")

    def test_uno_blocked_flag_only_counts_during_a_forward_component(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        lidar = SectorClearance(2.0, 2.0, 2.0, True, 2.0, 2.0)
        blocked_status = self._status(time.monotonic(), blocked=True)
        now = time.monotonic()

        policy.left_pwm, policy.right_pwm = 118, 118
        self.assertEqual(
            policy._motion_evidence(lidar, blocked_status, now, None)["uno"], "NOT_MOVING"
        )

        policy.left_pwm, policy.right_pwm = -118, -118  # reverse: the firmware
        # guard only ever gates a forward component.
        self.assertEqual(
            policy._motion_evidence(lidar, blocked_status, now, None)["uno"], "UNKNOWN"
        )

    def test_lidar_progress_evidence_requires_a_real_scan_timestamp(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        stale_fixture = SectorClearance(0.50, 2.0, 2.0, True, 2.0, 2.0)
        now = time.monotonic()
        vote = "UNKNOWN"
        for index in range(30):
            tick_now = now + index * 0.05
            vote = policy._lidar_progress_evidence(stale_fixture, 118, 118, tick_now)
        self.assertEqual(vote, "UNKNOWN")

    def test_lidar_progress_evidence_detects_no_advance_with_a_real_scan(self) -> None:
        # Every window-close tick resets the reference for the next window,
        # so only that closing tick reports a verdict; the following ticks
        # go back to UNKNOWN until the next window elapses. Collect votes
        # across several windows instead of asserting on an arbitrary tick.
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        votes = []
        for index in range(60):
            tick_now = now + index * 0.05
            clearance = SectorClearance(0.50, 2.0, 2.0, True, 2.0, 2.0, scan_at=tick_now)
            votes.append(policy._lidar_progress_evidence(clearance, 118, 118, tick_now))
        self.assertIn("NOT_MOVING", votes)
        self.assertNotIn("MOVING", votes)

    def test_lidar_progress_evidence_detects_real_advance(self) -> None:
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        votes = []
        front = 1.00
        for index in range(60):
            tick_now = now + index * 0.05
            clearance = SectorClearance(front, 2.0, 2.0, True, 2.0, 2.0, scan_at=tick_now)
            votes.append(policy._lidar_progress_evidence(clearance, 118, 118, tick_now))
            front -= 0.01
        self.assertIn("MOVING", votes)
        self.assertNotIn("NOT_MOVING", votes)


if __name__ == "__main__":
    unittest.main()
