from __future__ import annotations

import pathlib
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from robot_autonomy import (
    ArduinoLink, ArduinoStatus, AutonomousPolicy, CameraMotionState,
    SectorClearance, STEER_HEADINGS,
)
from robot_explorer import ExplorationState, FrontierExplorer
from robot_imu import AsyncIMULink, IMUState, LSM6DS3MCP2221Link
from test_robot_imu import _FakeLSMBus, _lsm_sample


class RuntimeRegressionTests(unittest.TestCase):
    def test_expired_heartbeat_cannot_overtake_new_drive(self):
        link = object.__new__(ArduinoLink)
        link._drive_lock = threading.Lock()
        link._drive_command = 'DRIVE 110 112'
        link._drive_lease_until = 10.4
        link._drive_last_write = 10.0
        link._drive_expired = False
        link._running = True
        selected_stop = threading.Event()
        release_stop = threading.Event()
        publisher_started = threading.Event()
        writes = []

        def write(command):
            if command == 'STOP':
                selected_stop.set()
                release_stop.wait(1.0)
            writes.append(command)
            return True

        def publish():
            publisher_started.set()
            link.publish_drive(118, 118)

        link._write = write
        def finish_loop(_delay):
            link._running = False
        with mock.patch('robot_autonomy.time.monotonic', return_value=10.6), mock.patch(
            'robot_autonomy.time.sleep', side_effect=finish_loop
        ):
            heartbeat = threading.Thread(target=link._heartbeat_loop)
            heartbeat.start()
            try:
                self.assertTrue(selected_stop.wait(1.0))
                publisher = threading.Thread(target=publish)
                publisher.start()
                self.assertTrue(publisher_started.wait(1.0))
                publisher.join(0.05)
            finally:
                release_stop.set()
                heartbeat.join(1.0)
            publisher.join(1.0)
        self.assertFalse(heartbeat.is_alive())
        self.assertFalse(publisher.is_alive())
        self.assertEqual(writes, ['STOP', 'DRIVE 118 118'])

    def test_watchdog_diagnostics_count_distinct_expirations(self):
        link = object.__new__(ArduinoLink)
        link._drive_lock = threading.Lock()
        link._drive_command = 'DRIVE 118 118'
        link._drive_lease_until = 10.4
        link._drive_last_write = 10.0
        link._drive_expired = False
        link.drive_lease_expirations = 0
        self.assertEqual(link._heartbeat_command(10.5), 'STOP')
        self.assertIsNone(link._heartbeat_command(10.6))
        self.assertEqual(link.drive_lease_expirations, 1)
        link.uno_watchdog_stops = 0
        link._running = True
        lines = iter([b'STOP:COMMAND_TIMEOUT', b'STATUS motion=S front_cm=50', b''])
        def read():
            line = next(lines)
            if not line:
                link._running = False
            return line
        link._serial = mock.Mock()
        link._serial.readline.side_effect = read
        link._read_loop()
        self.assertEqual(link.uno_watchdog_stops, 1)
        self.assertEqual(link.status().motion, 'S')

    def test_usb_stall_expires_cached_imu_state(self):
        imu = object.__new__(AsyncIMULink)
        imu._lock = threading.Lock()
        imu._state = IMUState(connected=True, calibrated=True, fresh=True,
                              updated_at=10.0, gyro_z_dps=30.0)
        self.assertTrue(imu.state(10.1).fresh)
        self.assertFalse(imu.state(10.3).fresh)
        self.assertFalse(imu.tick(10.3, stationary=False).fresh)
        self.assertTrue(imu._state.fresh)  # immutable cached measurement

    def test_usb_gap_does_not_invent_integrated_yaw(self):
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(calibration_samples=20, bus_factory=lambda _: bus)
        for i in range(20):
            imu.tick(1.0 + i * 0.03)
        bus.sample = _lsm_sample(gz=3000)
        before = imu.tick(1.60, stationary=False)
        after = imu.tick(2.60, stationary=False)
        self.assertEqual(after.yaw_deg, before.yaw_deg)
        self.assertEqual(after.sample_gaps, 1)
        resumed = imu.tick(2.63, stationary=False)
        self.assertGreater(resumed.yaw_deg, after.yaw_deg)

    def test_recovery_evaluates_preceding_pivot_not_new_forward_plan(self):
        policy = AutonomousPolicy(0.0, 118)
        policy.left_pwm, policy.right_pwm = -105, 105
        policy.imu_yaw_rate_dps = 0.0
        now = time.monotonic()
        clear = SectorClearance(2., 2., 2., True, 2., 2., rear_m=2.)
        def next_forward(*args, **kwargs):
            policy.left_pwm = policy.right_pwm = 118
            return 'F'
        with mock.patch.object(policy, '_plan', side_effect=next_forward):
            policy.decide(clear, ArduinoStatus(100., 'L', now), False, now)
        self.assertEqual(policy.stuck_votes['imu'], 'NOT_MOVING')

    def test_stalled_pivot_has_two_independent_sources(self):
        policy = AutonomousPolicy(0.0, 118)
        policy.left_pwm, policy.right_pwm = -105, 105
        policy.imu_yaw_rate_dps = 0.0
        now = time.monotonic()
        clear = SectorClearance(2., 2., 2., True, 2., 2., rear_m=2.)
        for i in range(30):
            tick = now + i * .05
            camera = CameraMotionState(fresh=True, confidence=.9, captured_at=tick)
            # A continuing commanded pivot that neither sensor sees moving.
            with mock.patch.object(policy, '_plan', return_value='L'):
                policy.decide(clear, ArduinoStatus(100., 'L', tick), False, tick,
                              camera_motion=camera)
            if policy.stuck_phase == 'RECOVER':
                break
        self.assertEqual(policy.stuck_phase, 'RECOVER')

    def test_cached_camera_flow_is_not_current_motion_evidence(self):
        policy = AutonomousPolicy(0.0, 118)
        policy.left_pwm = policy.right_pwm = 118
        camera = CameraMotionState(fresh=True, confidence=.9, captured_at=10.)
        votes = policy._motion_evidence(
            SectorClearance(2., 2., 2., True), ArduinoStatus(100., 'F', 11.), 11., camera)
        self.assertEqual(votes['camera'], 'UNKNOWN')

    def test_failed_reverse_switches_to_another_safe_maneuver(self):
        policy = AutonomousPolicy(0.0, 118)
        policy.left_pwm = policy.right_pwm = -118
        clear = SectorClearance(2., 2., 2., True, 2., 2., rear_m=2.)
        policy._advance_stuck_recovery(clear, time.monotonic())
        self.assertNotEqual(policy._stuck_maneuver, 'REVERSE')

    def test_first_motion_does_not_cancel_bounded_recovery(self):
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        clear = SectorClearance(2., 2., 2., True, 2., 2., rear_m=2.)
        policy._advance_stuck_recovery(clear, now)
        camera = CameraMotionState(fresh=True, confidence=.9,
                                   motion_observed=True, captured_at=now + .1)
        def next_forward(*args, **kwargs):
            policy.left_pwm = policy.right_pwm = 118
            return 'F'
        with mock.patch.object(policy, '_plan', side_effect=next_forward):
            policy.decide(clear, ArduinoStatus(100., 'B', now + .1), False, now + .1,
                          camera_motion=camera)
        self.assertEqual(policy.stuck_phase, 'RECOVER')
        self.assertLess(policy.left_pwm, 0)
        self.assertLess(policy.right_pwm, 0)

    def test_successful_recovery_finishes_and_respects_camera_stop(self):
        policy = AutonomousPolicy(0.0, 118)
        now = time.monotonic()
        clear = SectorClearance(2., 2., 2., True, 2., 2., rear_m=2.)
        policy._advance_stuck_recovery(clear, now)
        moving = CameraMotionState(fresh=True, confidence=.9,
                                   motion_observed=True, captured_at=now + .1)
        policy.decide(clear, ArduinoStatus(100., 'B', now + .1), False, now + .1,
                      camera_motion=moving)
        command = policy.decide(clear, ArduinoStatus(100., 'B', now + .2), False,
                                now + .2, camera_ready=False, camera_motion=moving)
        self.assertEqual(command, 'STOP')
        self.assertEqual((policy.left_pwm, policy.right_pwm), (0, 0))
        command = policy.decide(clear, ArduinoStatus(100., 'S', now + 1.), False, now + 1.)
        self.assertEqual(policy.stuck_phase, 'IDLE')
        self.assertEqual(command, 'F')

    def test_reverse_arc_aborts_when_side_becomes_blocked(self):
        policy = AutonomousPolicy(0.0, 118)
        clear_rear_blocked_side = SectorClearance(.5, .10, .10, True, .10, .10, rear_m=2.)
        self.assertIsNone(policy._drive_stuck_maneuver('REVERSE_ARC_L', clear_rear_blocked_side))

    def test_waypoint_shortcut_does_not_cut_obstacle_corner(self):
        free = np.ones((5, 5), dtype=bool)
        free[1, 2] = False
        self.assertFalse(FrontierExplorer._line_is_clear(free, (1, 1), (2, 2)))
        self.assertFalse(FrontierExplorer._line_is_clear(free, (2, 2), (1, 1)))
        self.assertTrue(FrontierExplorer._line_is_clear(free, (1, 1), (3, 1)))

    def test_waypoint_fallback_cannot_point_through_wall(self):
        free = np.ones((6, 6), dtype=bool)
        free[:, 2] = False
        self.assertIsNone(FrontierExplorer._select_waypoint(
            [(2, 3), (2, 4)], (2, 1), 10., free))

    def test_live_broad_opening_outweighs_short_map_corridor(self):
        policy = AutonomousPolicy(0.0, 118)
        policy.observe_exploration(ExplorationState(active=True, heading_error_deg=45.))
        profile = np.full(STEER_HEADINGS.size, .90, dtype=np.float32)
        profile[np.abs(STEER_HEADINGS + 25.) < 15.] = 1.15
        heading, clearance = policy._heading_from_profile(profile, time.monotonic())
        self.assertLess(heading, 0.)
        self.assertGreater(clearance, 1.)


if __name__ == '__main__':
    unittest.main()
