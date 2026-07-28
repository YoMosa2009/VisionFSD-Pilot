"""Hardware-free tests for the robot's local navigation.

These deliberately encode the field reports that motivated the planner: the
robot pivoting on the spot instead of exploring, refusing gaps it physically
fits through, driving into thin obstacles, and stopping until it was picked up.
"""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_navigation import (
    BIN_COUNT,
    CorridorModel,
    ExplorationMemory,
    MotionMonitor,
    NavigationPlanner,
    PlannerTuning,
    RobotGeometry,
    ScanFrame,
    UltrasonicTrust,
    YawTracker,
    despeckle,
    pivot_clearance,
    reverse_limit,
    scan_from_ranges,
)


# The measured OSOYOO chassis: 140 mm wide, 150 mm long, LD19 taken as centred.
GEOMETRY = RobotGeometry(width_m=0.14, length_m=0.15, lidar_forward_offset_m=0.0,
                         safety_margin_m=0.040)


def open_room(distance_m: float = 2.6) -> ScanFrame:
    return scan_from_ranges({}, default=distance_m)


def wall_with_gap(wall_m: float, gap_half_width_deg: float, beyond_m: float = 4.0,
                  span_deg: int = 70) -> ScanFrame:
    """A flat wall across the front with a centred opening."""
    ranges: dict[int, float] = {}
    for angle in range(-span_deg, span_deg + 1):
        if abs(angle) <= gap_half_width_deg:
            ranges[angle] = beyond_m
        else:
            ranges[angle] = wall_m / math.cos(math.radians(angle))
    for angle in range(span_deg + 1, 360 - span_deg):
        ranges[angle] = 3.2
    return scan_from_ranges(ranges)


def drive_cycles(planner: NavigationPlanner, scan: ScanFrame, start: float, cycles: int,
                 front_cm: float | None = None, step: float = 0.1, allow_stall: bool = False):
    """Replay identical scans through the planner.

    A byte-identical synthetic scan looks exactly like a robot whose wheels are
    not turning, so stall detection is disabled unless a test is exercising it.
    """
    if not allow_stall:
        planner.motion.stall_after_s = float("inf")
    command = None
    for index in range(cycles):
        command = planner.decide(scan, front_cm, True, True, False, [], start + index * step)
    return command


class GeometryTests(unittest.TestCase):
    def test_derived_dimensions(self) -> None:
        self.assertAlmostEqual(GEOMETRY.corridor_half_width_m, 0.070 + 0.040)
        self.assertAlmostEqual(GEOMETRY.front_overhang_m, 0.075)
        self.assertAlmostEqual(GEOMETRY.rear_overhang_m, 0.075)
        self.assertGreater(GEOMETRY.pivot_radius_m, GEOMETRY.corridor_half_width_m)


class CorridorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corridor = CorridorModel(GEOMETRY)

    def test_wall_ahead_limits_travel_to_the_bumper(self) -> None:
        scan = scan_from_ranges({angle: 1.0 / math.cos(math.radians(angle))
                                 for angle in range(-40, 41)}, default=3.0)
        limit = self.corridor.limit_at(scan, 0.0)
        self.assertAlmostEqual(limit, 1.0 - GEOMETRY.front_overhang_m, places=2)

    def test_returns_behind_the_heading_never_limit_forward_travel(self) -> None:
        scan = scan_from_ranges({170: 0.12, 180: 0.12, 190: 0.12}, default=2.8)
        self.assertGreater(self.corridor.limit_at(scan, 0.0), 2.0)

    def test_gap_wider_than_the_robot_is_passable(self) -> None:
        # 1.5 m away, +/-12 deg is a 0.63 m opening for a 0.28 m corridor.
        wide = wall_with_gap(1.5, 12.0)
        self.assertGreater(self.corridor.limit_at(wide, 0.0), 2.0)

    def test_gap_narrower_than_the_robot_is_not_passable(self) -> None:
        # +/-3 deg at 1.5 m is a 0.16 m opening: narrower than the 0.28 m body
        # corridor, so travel must still be limited by the wall.
        narrow = wall_with_gap(1.5, 3.0)
        limit = self.corridor.limit_at(narrow, 0.0)
        self.assertLess(limit, 1.6)
        self.assertAlmostEqual(limit, 1.5 - GEOMETRY.front_overhang_m, places=1)

    def test_wider_robot_rejects_a_gap_a_narrow_robot_accepts(self) -> None:
        gap = wall_with_gap(1.5, 8.0)
        narrow_robot = CorridorModel(RobotGeometry(0.14, 0.22, 0.02, 0.02))
        wide_robot = CorridorModel(RobotGeometry(0.36, 0.22, 0.02, 0.06))
        self.assertGreater(narrow_robot.limit_at(gap, 0.0), 2.0)
        self.assertLess(wide_robot.limit_at(gap, 0.0), 1.6)


class ScanFilterTests(unittest.TestCase):
    def test_confident_thin_obstacle_survives_and_blocks(self) -> None:
        scan = scan_from_ranges({0: 0.80}, default=2.6, confidence=200.0)
        cleaned = despeckle(scan)
        self.assertTrue(np.isfinite(cleaned.ranges[0]))
        self.assertAlmostEqual(CorridorModel(GEOMETRY).limit_at(cleaned, 0.0),
                               0.80 - GEOMETRY.front_overhang_m, places=2)

    def test_weak_isolated_spike_is_removed(self) -> None:
        ranges = np.full(BIN_COUNT, 2.6, dtype=np.float32)
        confidence = np.full(BIN_COUNT, 200.0, dtype=np.float32)
        ranges[0] = 0.60
        confidence[0] = 10.0
        cleaned = despeckle(ScanFrame(ranges, confidence, True, 0.0))
        self.assertFalse(np.isfinite(cleaned.ranges[0]))


class RearAndPivotTests(unittest.TestCase):
    def test_reverse_limit_uses_the_rear_overhang(self) -> None:
        scan = scan_from_ranges({angle % 360: 1.0 / abs(math.cos(math.radians(angle)))
                                 for angle in range(160, 201)}, default=3.0)
        self.assertAlmostEqual(reverse_limit(scan, GEOMETRY), 1.0 - GEOMETRY.rear_overhang_m, places=2)

    def test_pivot_clearance_reports_the_closest_return(self) -> None:
        scan = scan_from_ranges({90: 0.30}, default=3.0)
        self.assertAlmostEqual(pivot_clearance(scan, GEOMETRY), 0.30, places=1)


class YawTrackerTests(unittest.TestCase):
    @staticmethod
    def _textured_room() -> np.ndarray:
        angles = np.arange(BIN_COUNT, dtype=np.float32)
        return (2.0 + 0.8 * np.sin(np.radians(angles) * 3.0)).astype(np.float32)

    def test_right_rotation_is_measured_as_positive_yaw(self) -> None:
        profile = self._textured_room()
        tracker = YawTracker()
        tracker.update(ScanFrame(profile, np.full(BIN_COUNT, 200.0), True, 0.0), 0.0)
        # Turning right by 8 degrees moves every feature 8 bins anticlockwise.
        rotated = np.roll(profile, -8)
        tracker.update(ScanFrame(rotated, np.full(BIN_COUNT, 200.0), True, 0.5), 0.5)
        self.assertGreater(tracker.yaw_rate_dps, 0.0)
        self.assertAlmostEqual(tracker.heading_deg, 8.0, places=0)

    def test_featureless_room_does_not_invent_a_turn_rate(self) -> None:
        """Every shift scores alike in a symmetric room; that must read as no rotation."""
        flat = np.full(BIN_COUNT, 2.4, dtype=np.float32)
        tracker = YawTracker()
        tracker.update(ScanFrame(flat, np.full(BIN_COUNT, 200.0), True, 0.0), 0.0)
        tracker.update(ScanFrame(flat.copy(), np.full(BIN_COUNT, 200.0), True, 0.5), 0.5)
        self.assertEqual(tracker.yaw_rate_dps, 0.0)
        self.assertEqual(tracker.heading_deg, 0.0)

    def test_static_scene_reports_no_rotation(self) -> None:
        profile = self._textured_room()
        tracker = YawTracker()
        tracker.update(ScanFrame(profile, np.full(BIN_COUNT, 200.0), True, 0.0), 0.0)
        tracker.update(ScanFrame(profile.copy(), np.full(BIN_COUNT, 200.0), True, 0.5), 0.5)
        self.assertAlmostEqual(tracker.yaw_rate_dps, 0.0, places=3)
        self.assertLess(tracker.scene_change, 0.01)


class MotionMonitorTests(unittest.TestCase):
    def test_commanded_motion_without_scene_change_is_a_stall(self) -> None:
        yaw = YawTracker()
        yaw.scene_change, yaw.yaw_rate_dps = 0.001, 0.0
        monitor = MotionMonitor(stall_after_s=1.0)
        for index in range(30):
            monitor.update(70, 70, yaw, index * 0.1)
        self.assertTrue(monitor.stalled)

    def test_moving_scene_is_not_a_stall(self) -> None:
        yaw = YawTracker()
        yaw.scene_change, yaw.yaw_rate_dps = 0.20, 0.0
        monitor = MotionMonitor(stall_after_s=1.0)
        for index in range(30):
            monitor.update(70, 70, yaw, index * 0.1)
        self.assertFalse(monitor.stalled)

    def test_sustained_one_way_turning_accumulates(self) -> None:
        yaw = YawTracker()
        yaw.scene_change, yaw.yaw_rate_dps = 0.20, 60.0
        monitor = MotionMonitor()
        for index in range(60):
            monitor.update(60, -60, yaw, index * 0.1)
        self.assertGreater(monitor.turn_integral_deg, 240.0)


class UltrasonicTrustTests(unittest.TestCase):
    def test_sustained_disagreement_with_lidar_revokes_trust(self) -> None:
        trust = UltrasonicTrust(disagreement_s=1.0)
        for index in range(30):
            trust.update(18.0, 2.5, index * 0.1)
        self.assertFalse(trust.trusted)

    def test_agreement_keeps_trust(self) -> None:
        trust = UltrasonicTrust(disagreement_s=1.0)
        for index in range(30):
            trust.update(40.0, 0.45, index * 0.1)
        self.assertTrue(trust.trusted)


class ExplorationMemoryTests(unittest.TestCase):
    def test_repeated_heading_is_penalised_relative_to_a_fresh_one(self) -> None:
        memory = ExplorationMemory(half_life_s=0.0)
        for index in range(40):
            memory.update(0.0, True, index * 0.5)
        penalties = memory.penalty(np.array([0.0, 180.0], dtype=np.float32))
        self.assertGreater(penalties[0], 0.9)
        self.assertEqual(penalties[1], 0.0)


class PlannerTests(unittest.TestCase):
    def make(self, standby: float = 0.0, **tuning) -> NavigationPlanner:
        return NavigationPlanner(GEOMETRY, PlannerTuning(**tuning), standby, started_at=0.0)

    # ---- safety gates ----
    def test_boot_standby_never_moves(self) -> None:
        planner = self.make(standby=25.0)
        command = planner.decide(open_room(), 80.0, True, True, False, [], 3.0)
        self.assertEqual((command.left_pwm, command.right_pwm), (0, 0))
        self.assertEqual(command.state, "STANDBY")

    def test_stale_lidar_stops(self) -> None:
        planner = self.make()
        stale = ScanFrame(np.full(BIN_COUNT, 2.0, dtype=np.float32),
                          np.full(BIN_COUNT, 200.0, dtype=np.float32), False, 0.0)
        command = planner.decide(stale, 80.0, True, True, False, [], 1.0)
        self.assertEqual(command.state, "SAFETY_STOP")
        self.assertEqual((command.left_pwm, command.right_pwm), (0, 0))

    def test_stale_uno_stops(self) -> None:
        command = self.make().decide(open_room(), 80.0, False, True, False, [], 1.0)
        self.assertEqual(command.state, "SAFETY_STOP")

    def test_stale_camera_stops(self) -> None:
        command = self.make().decide(open_room(), 80.0, True, False, False, [], 1.0)
        self.assertEqual(command.state, "SAFETY_STOP")

    def test_confirmed_person_stops(self) -> None:
        command = self.make().decide(open_room(), 80.0, True, True, True, [], 1.0)
        self.assertEqual(command.state, "SAFETY_STOP")
        self.assertEqual((command.left_pwm, command.right_pwm), (0, 0))

    # ---- the field reports ----
    def test_close_ultrasonic_with_clear_lidar_still_makes_forward_progress(self) -> None:
        """The reported failure: a near ultrasonic return pinned it into a pivot loop."""
        planner = self.make()
        command = drive_cycles(planner, open_room(2.8), 1.0, 12, front_cm=18.0)
        self.assertGreater(command.left_pwm, 0)
        self.assertGreater(command.right_pwm, 0)
        self.assertNotEqual(command.state, "PIVOT")
        self.assertNotEqual(command.state, "HOLD")

    def test_persistently_disagreeing_ultrasonic_is_dropped_and_it_drives_straight(self) -> None:
        planner = self.make()
        command = drive_cycles(planner, open_room(2.8), 1.0, 60, front_cm=18.0)
        self.assertFalse(planner.ultrasonic.trusted)
        self.assertGreater(min(command.left_pwm, command.right_pwm), 0)
        self.assertLess(abs(command.heading_deg), 10.0)

    def test_open_room_drives_forward_at_speed(self) -> None:
        planner = self.make(speed=70)
        command = drive_cycles(planner, open_room(2.8), 1.0, 12)
        self.assertIn(command.state, ("CRUISE", "ARC"))
        self.assertGreater(min(command.left_pwm, command.right_pwm), 0)
        self.assertGreaterEqual(max(command.left_pwm, command.right_pwm), 60)

    def test_straight_ahead_is_an_actual_candidate(self) -> None:
        """An even candidate split leaves -1 and +1 tying, and the robot weaves."""
        headings = NavigationPlanner(GEOMETRY, PlannerTuning(), 0.0, started_at=0.0).corridor.headings
        self.assertIn(0.0, set(float(value) for value in headings))

    def test_open_room_does_not_weave(self) -> None:
        """The reported failure: it wandered left and right while driving straight."""
        planner = self.make()
        planner.motion.stall_after_s = float("inf")
        headings = []
        for index in range(60):
            command = planner.decide(open_room(2.8), None, True, True, False, [], 1.0 + index * 0.1)
            headings.append(command.heading_deg)
        settled = headings[10:]
        self.assertEqual(set(settled), {0.0}, f"heading wandered: {sorted(set(settled))}")
        self.assertEqual(command.left_pwm, command.right_pwm)

    def test_exploration_memory_does_not_curl_a_straight_run(self) -> None:
        """Penalising the heading currently being driven turns a straight line into an arc."""
        planner = self.make()
        planner.motion.stall_after_s = float("inf")
        for index in range(120):
            command = planner.decide(open_room(2.8), None, True, True, False, [], 1.0 + index * 0.1)
        self.assertAlmostEqual(command.heading_deg, 0.0, places=6)

    def test_obstacle_ahead_produces_a_forward_arc_not_a_stop(self) -> None:
        scan = scan_from_ranges({angle: 0.75 for angle in range(-14, 15)}, default=2.8)
        planner = self.make()
        command = drive_cycles(planner, scan, 1.0, 12)
        self.assertIn(command.state, ("ARC", "CRUISE"))
        self.assertGreater(min(command.left_pwm, command.right_pwm), -1)
        self.assertNotEqual(command.left_pwm, command.right_pwm)

    def test_dead_end_reverses_instead_of_giving_up(self) -> None:
        """The reported failure: it stopped and had to be picked up."""
        blocked = {angle % 360: 0.22 for angle in range(-100, 101)}
        planner = self.make()
        command = drive_cycles(planner, scan_from_ranges(blocked, default=3.0), 1.0, 3)
        self.assertEqual(command.state, "REVERSE")
        self.assertLessEqual(max(command.left_pwm, command.right_pwm), 0)
        self.assertLess(min(command.left_pwm, command.right_pwm), 0)

    def test_too_tight_to_turn_still_nudges_backwards(self) -> None:
        """Between "can reverse freely" and "hopeless" there must still be a move."""
        ranges = {angle % 360: 0.30 for angle in range(-180, 181)}
        ranges.update({angle % 360: 0.12 for angle in range(-100, 101)})
        planner = self.make()
        command = drive_cycles(planner, scan_from_ranges(ranges), 1.0, 3)
        self.assertEqual(command.state, "REVERSE")
        self.assertIn("NUDGE BACK", planner.reason)

    def test_fully_enclosed_robot_holds_rather_than_driving_blind(self) -> None:
        planner = self.make()
        command = drive_cycles(planner, scan_from_ranges({}, default=0.11), 1.0, 3)
        self.assertEqual(command.state, "HOLD")
        self.assertEqual((command.left_pwm, command.right_pwm), (0, 0))

    def test_clear_side_opening_is_entered_by_turning_toward_it(self) -> None:
        ranges = {angle % 360: 0.30 for angle in range(-180, 181)}
        for angle in range(60, 105):
            ranges[angle] = 3.0
        planner = self.make()
        command = drive_cycles(planner, scan_from_ranges(ranges), 1.0, 3)
        self.assertIn(command.state, ("PIVOT", "REVERSE"))
        if command.state == "PIVOT":
            self.assertGreater(command.left_pwm, command.right_pwm)  # turning right

    def test_driving_with_an_unchanging_world_is_treated_as_a_stall(self) -> None:
        """A sagging motor battery leaves the Pi commanding motion that never happens."""
        planner = self.make()
        planner.motion.stall_after_s = 1.0
        states = set()
        for index in range(40):
            command = planner.decide(open_room(2.8), None, True, True, False, [], 1.0 + index * 0.1)
            states.add(command.state)
        self.assertTrue(planner.recovery_count > 0)
        self.assertTrue(states & {"REVERSE", "PIVOT"}, states)

    def test_people_bias_steering_away_before_a_stop_is_needed(self) -> None:
        planner = self.make()
        command = drive_cycles(planner, open_room(2.8), 1.0, 6)
        straight_heading = command.heading_deg
        planner_with_person = self.make()
        biased = None
        for index in range(6):
            biased = planner_with_person.decide(open_room(2.8), None, True, True, False, [-30.0],
                                                1.0 + index * 0.1)
        self.assertGreaterEqual(biased.heading_deg, straight_heading)

    # ---- output shaping ----
    def test_no_wheel_is_ever_left_inside_the_motor_deadband(self) -> None:
        planner = self.make(min_move_pwm=50)
        for cycle in range(40):
            command = planner.decide(open_room(2.8), None, True, True, False, [], 1.0 + cycle * 0.1)
            for value in (command.left_pwm, command.right_pwm):
                self.assertTrue(value == 0 or abs(value) >= 50, f"deadband PWM {value}")
                self.assertLessEqual(abs(value), 105)

    def test_tight_arc_idles_the_inner_wheel_but_stays_a_forward_command(self) -> None:
        """A tight arc emits values like (105, 0), which the Uno must still gate."""
        planner = self.make()
        command = None
        for _ in range(6):
            command = planner._arc(35.0, 62)
        self.assertEqual(command.right_pwm, 0)
        self.assertGreater(command.left_pwm, 0)
        # Both wheels non-negative with one positive: a forward component, and
        # visionfsd_pi_autonomy.ino gates exactly this shape.
        self.assertGreaterEqual(min(command.left_pwm, command.right_pwm), 0)

    def test_arcs_across_the_steering_range_respect_the_deadband(self) -> None:
        for heading in (-40.0, -25.0, -10.0, 0.0, 10.0, 25.0, 40.0):
            planner = self.make(min_move_pwm=50)
            for _ in range(6):
                command = planner._arc(heading, 62)
            for value in (command.left_pwm, command.right_pwm):
                self.assertTrue(value == 0 or abs(value) >= 50, f"{heading}deg gave {value}")
                self.assertLessEqual(abs(value), 105)

    def test_camera_slow_down_never_drops_a_wheel_into_the_deadband(self) -> None:
        planner = self.make(min_move_pwm=50)
        planner.motion.stall_after_s = float("inf")
        for index in range(20):
            command = planner.decide(open_room(2.8), None, True, True, False, [],
                                     1.0 + index * 0.1, speed_scale=0.75)
        for value in (command.left_pwm, command.right_pwm):
            self.assertTrue(value == 0 or abs(value) >= 50, f"deadband PWM {value}")

    def test_pivot_direction_signs_match_the_angle_convention(self) -> None:
        planner = self.make()
        right = planner._pivot(1)
        self.assertGreater(right.left_pwm, 0)
        self.assertLess(right.right_pwm, 0)
        planner.left_pwm = planner.right_pwm = 0
        left = planner._pivot(-1)
        self.assertLess(left.left_pwm, 0)
        self.assertGreater(left.right_pwm, 0)

    def test_anti_orbit_penalty_discourages_turning_the_same_way(self) -> None:
        planner = self.make()
        limits = np.full(len(planner.corridor.headings), 2.5, dtype=np.float32)
        neutral = planner._score(open_room(), limits, [])
        planner.motion.turn_integral_deg = 600.0
        penalised = planner._score(open_room(), limits, [])
        right = int(np.argmin(np.abs(planner.corridor.headings - 60.0)))
        left = int(np.argmin(np.abs(planner.corridor.headings + 60.0)))
        self.assertLess(penalised[right], neutral[right])
        self.assertAlmostEqual(float(penalised[left]), float(neutral[left]), places=4)


if __name__ == "__main__":
    unittest.main()
