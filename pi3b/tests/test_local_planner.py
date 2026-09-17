"""Regression tests for the arc planner and its obstacle memory.

Each group corresponds to something observed on the robot running v1.9.18:
contact with walls that the ultrasonic did not stop, a chassis that could not
plan past a thin obstacle, and steering decisions made from a single scan
with no recollection of what had just passed out of view.
"""

from __future__ import annotations

import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import (
    ARC_STEER_OPTIONS,
    CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT,
    MAX_GENTLE_HEADING_DEG,
    MAX_PWM,
    MAX_TURN_SPLIT_PWM,
    ROBOT_FOOTPRINT,
)
from robot_local_planner import (
    ArcBank,
    LocalPlanner,
    ObstacleMemory,
    PlannerLimits,
    evaluate_arcs,
    reduce_obstacles,
    stopping_distance_m,
)

SPEEDS = tuple((pwm, 0.55 * pwm / 255.0) for pwm in (105, 111, 118))


def _limits(**overrides) -> PlannerLimits:
    """Planner limits with the runtime's own chassis turn model."""
    base = dict(
        footprint=ROBOT_FOOTPRINT,
        yaw_rate_dps_at_full_steer=(
            MAX_TURN_SPLIT_PWM / MAX_PWM * CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT
        ),
        full_steer_deg=MAX_GENTLE_HEADING_DEG,
    )
    base.update(overrides)
    return PlannerLimits(**base)


def _bank(limits: PlannerLimits | None = None) -> ArcBank:
    return ArcBank(limits or _limits(), ARC_STEER_OPTIONS, SPEEDS)


def _wall(distance_m: float, half_width_m: float = 2.0, count: int = 400):
    x = np.linspace(-half_width_m, half_width_m, count).astype(np.float32)
    return x, np.full(count, distance_m, dtype=np.float32)


def _choose(obstacle_x, obstacle_y, goal=None, steering=0.0, limits=None):
    return evaluate_arcs(
        obstacle_x, obstacle_y, _bank(limits), goal, steering
    )


class StoppingDistanceTests(unittest.TestCase):
    def test_stopping_distance_grows_with_speed(self) -> None:
        limits = PlannerLimits()
        slow = stopping_distance_m(0.15, limits)
        fast = stopping_distance_m(0.50, limits)
        self.assertGreater(fast, slow)

    def test_reaction_latency_is_counted_at_full_speed(self) -> None:
        """The robot is committed to the reaction distance before any new
        measurement can change the command, so it is not part of braking."""
        limits = PlannerLimits(braking_mps2=1e6)
        self.assertAlmostEqual(
            stopping_distance_m(0.4, limits), 0.4 * limits.reaction_latency_s,
            places=4,
        )

    def test_a_stopped_chassis_needs_no_distance(self) -> None:
        self.assertEqual(stopping_distance_m(0.0, PlannerLimits()), 0.0)


class ArcSelectionTests(unittest.TestCase):
    def test_open_floor_drives_straight_at_the_top_level(self) -> None:
        choice = _choose(
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        )
        self.assertTrue(choice.admissible)
        self.assertEqual(choice.steering_deg, 0.0)
        self.assertEqual(choice.pwm, 118)

    def test_a_close_wall_has_no_admissible_arc(self) -> None:
        """The collision case. Every arc into a wall this close is consumed
        by stopping distance, so the planner must refuse rather than commit
        and rely on the ultrasonic, whose ~240 ms detection latency and
        narrow cone cannot catch it."""
        choice = _choose(*_wall(0.40))
        self.assertFalse(choice.admissible)
        self.assertEqual(choice.reason, "NO_ADMISSIBLE_ARC")

    def test_a_distant_wall_is_still_drivable(self) -> None:
        choice = _choose(*_wall(2.5))
        self.assertTrue(choice.admissible)

    def test_an_obstacle_with_open_floor_beside_it_is_steered_around(self) -> None:
        """A wall spanning the whole field has nowhere better to go inside the
        lookahead, so it is met head-on and then refused. A partial
        obstruction is the case that should produce a turn."""
        x = np.linspace(-2.0, 0.25, 260).astype(np.float32)
        choice = _choose(x, np.full(x.size, 0.95, dtype=np.float32))
        self.assertTrue(choice.admissible)
        self.assertGreater(choice.steering_deg, 0.0)

    def test_a_full_width_wall_is_refused_rather_than_driven_into(self) -> None:
        near = _choose(*_wall(0.40))
        far = _choose(*_wall(1.20))
        self.assertFalse(near.admissible)
        self.assertTrue(far.admissible)

    def test_every_admissible_arc_can_stop_inside_its_own_clearance(self) -> None:
        for distance in (0.45, 0.6, 0.9, 1.4, 2.2):
            choice = _choose(*_wall(distance))
            if not choice.admissible:
                continue
            with self.subTest(distance=distance):
                self.assertGreaterEqual(
                    choice.reachable_m,
                    choice.stopping_m + _limits().stop_buffer_m,
                )

    def test_a_thin_obstacle_is_routed_around_not_stopped_for(self) -> None:
        """A pole is a few centimetres wide and the floor beside it is open;
        the planner should curve past rather than treat it as a blockage."""
        pole_x = np.array([0.0, 0.02, -0.02], dtype=np.float32)
        pole_y = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        choice = _choose(pole_x, pole_y)
        self.assertTrue(choice.admissible)
        self.assertNotEqual(choice.steering_deg, 0.0)
        self.assertGreater(choice.reachable_m, 1.0)

    def test_a_gap_in_a_wall_is_aimed_at(self) -> None:
        """The gap is offset to the robot's right and far enough ahead that
        this chassis's turn radius can actually reach it. A tighter or nearer
        gap is not a scoring question but a physical one: with roughly a one
        metre turn radius the robot cannot always line up on a doorway in a
        single arc, and the escape machine owns that case."""
        x = np.linspace(-2.0, 2.0, 400).astype(np.float32)
        solid = ~((x > 0.1) & (x < 1.0))
        choice = _choose(
            x[solid], np.full(int(solid.sum()), 1.8, dtype=np.float32)
        )
        self.assertTrue(choice.admissible)
        # Positive steering is to the robot's right, toward the gap.
        self.assertGreater(choice.steering_deg, 0.0)

    def test_a_curling_arc_does_not_beat_real_forward_progress(self) -> None:
        """Scoring arc length rather than distance made good rewards the arc
        that curls tightly away from everything: it stays clear for its whole
        length while going nowhere. That is what made the robot orbit local
        objects instead of crossing open floor."""
        x = np.linspace(-2.0, 2.0, 400).astype(np.float32)
        solid = ~((x > 0.1) & (x < 1.0))
        choice = _choose(
            x[solid], np.full(int(solid.sum()), 1.8, dtype=np.float32)
        )
        forward = choice.path_xy[-1][1]
        self.assertGreater(forward, 1.0)

    def test_a_straight_corridor_is_driven_straight(self) -> None:
        count = 200
        walls_x = np.concatenate(
            (
                np.full(count, -0.45, dtype=np.float32),
                np.full(count, 0.45, dtype=np.float32),
            )
        )
        walls_y = np.concatenate(
            [np.linspace(0.0, 3.0, count, dtype=np.float32)] * 2
        )
        choice = _choose(walls_x, walls_y)
        self.assertTrue(choice.admissible)
        self.assertEqual(choice.steering_deg, 0.0)

    def test_the_goal_heading_breaks_ties_in_open_space(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        left = _choose(empty, empty, goal=-30.0)
        right = _choose(empty, empty, goal=30.0)
        self.assertLess(left.steering_deg, 0.0)
        self.assertGreater(right.steering_deg, 0.0)

    def test_current_steering_is_preferred_among_near_ties(self) -> None:
        """Without this the search re-answers from scratch every tick and two
        near-tied arcs alternate, which is what the weaving looked like."""
        pole_x = np.array([0.0, 0.02, -0.02], dtype=np.float32)
        pole_y = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        held_left = _choose(pole_x, pole_y, steering=-18.0)
        held_right = _choose(pole_x, pole_y, steering=18.0)
        self.assertLess(held_left.steering_deg, 0.0)
        self.assertGreater(held_right.steering_deg, 0.0)

    def test_reported_clearance_covers_only_the_usable_arc(self) -> None:
        choice = _choose(*_wall(0.9))
        if choice.admissible:
            self.assertGreaterEqual(choice.clearance_m, 0.0)

    def test_the_chosen_arc_is_published_for_the_overlay(self) -> None:
        choice = _choose(
            np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        )
        self.assertGreaterEqual(len(choice.path_xy), 2)
        # Robot frame: the arc leads forward (+y).
        self.assertGreater(choice.path_xy[-1][1], 0.0)

    def test_an_overstated_turn_rate_is_not_used(self) -> None:
        """The planner must not believe it can dodge harder than the wheel
        split allows, or it commits to an arc it cannot follow."""
        modelled = (
            MAX_TURN_SPLIT_PWM / MAX_PWM * CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT
        )
        self.assertAlmostEqual(
            _limits().yaw_rate_dps_at_full_steer, modelled, places=6
        )
        self.assertLessEqual(modelled, 30.0)


class SpeedGovernorTests(unittest.TestCase):
    def test_a_higher_assumed_top_speed_is_more_cautious(self) -> None:
        """top_speed_mps is an estimate, so overestimating must only ever
        cost caution, never safety."""
        cautious = _limits(top_speed_mps=1.2)
        optimistic = _limits(top_speed_mps=0.3)
        fast_speeds = tuple((pwm, 1.2 * pwm / 255.0) for pwm in (105, 118))
        slow_speeds = tuple((pwm, 0.3 * pwm / 255.0) for pwm in (105, 118))
        wall_x, wall_y = _wall(0.75)
        fast = evaluate_arcs(
            wall_x, wall_y,
            ArcBank(cautious, ARC_STEER_OPTIONS, fast_speeds),
            None, 0.0,
        )
        slow = evaluate_arcs(
            wall_x, wall_y,
            ArcBank(optimistic, ARC_STEER_OPTIONS, slow_speeds),
            None, 0.0,
        )
        self.assertGreaterEqual(slow.stopping_m * 4.0, 0.0)
        if fast.admissible and slow.admissible:
            self.assertGreaterEqual(fast.stopping_m, slow.stopping_m)

    def test_lookahead_is_a_distance_not_a_time(self) -> None:
        """A time horizon collapses to centimetres at this chassis's speed."""
        bank = _bank()
        self.assertGreaterEqual(bank.horizon_m, 1.5)
        self.assertLessEqual(bank.horizon_m, 2.4)


class ObstacleMemoryTests(unittest.TestCase):
    def _scan(self, angle_deg: float, range_m: float):
        return (
            np.array([angle_deg], dtype=np.float32),
            np.array([range_m], dtype=np.float32),
        )

    def test_a_scan_is_stored_in_the_robot_frame(self) -> None:
        memory = ObstacleMemory()
        memory.add_scan(*self._scan(0.0, 1.0), 1.0)
        x, y = memory.cartesian()
        self.assertEqual(x.size, 1)
        self.assertAlmostEqual(float(x[0]), 0.0, places=3)
        self.assertGreater(float(y[0]), 0.9)

    def test_driving_forward_brings_remembered_points_closer(self) -> None:
        memory = ObstacleMemory()
        memory.add_scan(*self._scan(0.0, 2.0), 1.0)
        memory.integrate_motion(0.5, 0.0)
        _x, y = memory.cartesian()
        self.assertAlmostEqual(float(y[0]), 1.54, places=2)

    def test_turning_rotates_remembered_points(self) -> None:
        """This is what keeps an obstacle in the set while the chassis turns
        away from it and the live scan can no longer see it."""
        memory = ObstacleMemory()
        memory.add_scan(*self._scan(0.0, 1.0), 1.0)
        memory.integrate_motion(0.0, 90.0)
        x, y = memory.cartesian()
        # A point dead ahead ends up on the robot's left after turning right.
        self.assertLess(float(x[0]), -0.9)
        self.assertAlmostEqual(float(y[0]), 0.0, places=2)

    def test_points_expire_after_the_horizon(self) -> None:
        memory = ObstacleMemory(horizon_s=0.5)
        memory.add_scan(*self._scan(0.0, 1.0), 1.0)
        self.assertEqual(memory.size, 1)
        memory.add_scan(*self._scan(90.0, 1.0), 2.0)
        self.assertEqual(memory.size, 1)

    def test_a_replayed_scan_timestamp_is_ignored(self) -> None:
        memory = ObstacleMemory()
        memory.add_scan(*self._scan(0.0, 1.0), 1.0)
        memory.add_scan(*self._scan(0.0, 1.0), 1.0)
        self.assertEqual(memory.size, 1)

    def test_memory_is_bounded(self) -> None:
        memory = ObstacleMemory(max_points=50)
        angles = np.linspace(0.0, 359.0, 400).astype(np.float32)
        ranges = np.full(400, 1.5, dtype=np.float32)
        memory.add_scan(angles, ranges, 1.0)
        self.assertLessEqual(memory.size, 50)

    def test_remembered_returns_are_held_slightly_further_away(self) -> None:
        """Dead-reckoned points are less trustworthy than the live scan, so
        memory alone must not be able to manufacture a hard block."""
        memory = ObstacleMemory()
        memory.add_scan(*self._scan(0.0, 1.0), 1.0)
        _x, y = memory.cartesian()
        self.assertGreater(float(y[0]), 1.0)

    def test_reset_empties_the_memory(self) -> None:
        memory = ObstacleMemory()
        memory.add_scan(*self._scan(0.0, 1.0), 1.0)
        memory.reset()
        self.assertEqual(memory.size, 0)

    def test_memory_lets_the_planner_see_what_the_scan_no_longer_does(self) -> None:
        planner = LocalPlanner()
        angles = np.linspace(-20.0, 20.0, 60).astype(np.float32) % 360.0
        ranges = np.full(60, 0.55, dtype=np.float32)
        planner.memory.add_scan(angles, ranges, 1.0)
        remembered_x, remembered_y = planner.memory.cartesian()
        blind = evaluate_arcs(
            remembered_x, remembered_y, _bank(), None, 0.0
        )
        empty = np.zeros(0, dtype=np.float32)
        forgetful = evaluate_arcs(
            empty, empty, _bank(), None, 0.0
        )
        self.assertTrue(forgetful.admissible)
        self.assertLess(blind.reachable_m, forgetful.reachable_m)


class ObstacleReductionTests(unittest.TestCase):
    def test_reduction_keeps_the_nearest_return_per_direction(self) -> None:
        x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        y = np.array([2.0, 1.0, 3.0], dtype=np.float32)
        reduced_x, reduced_y = reduce_obstacles(x, y)
        self.assertEqual(reduced_x.size, 1)
        self.assertAlmostEqual(float(np.hypot(reduced_x[0], reduced_y[0])), 1.0,
                               places=2)

    def test_reduction_is_bounded_by_the_bin_count(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.uniform(-3.0, 3.0, 4000).astype(np.float32)
        y = rng.uniform(-3.0, 3.0, 4000).astype(np.float32)
        reduced_x, _reduced_y = reduce_obstacles(x, y, bins=180)
        self.assertLessEqual(reduced_x.size, 180)

    def test_reduction_matches_a_direct_per_bin_minimum(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.uniform(-3.0, 3.0, 900).astype(np.float32)
        y = rng.uniform(-3.0, 3.0, 900).astype(np.float32)
        bins = 180
        ranges = np.hypot(x, y)
        keep = (ranges >= 0.05) & (ranges <= 3.2)
        angles = np.arctan2(x[keep], y[keep])
        index = np.clip(
            np.floor((angles + np.pi) / (2.0 * np.pi) * bins).astype(int),
            0, bins - 1,
        )
        expected: dict[int, float] = {}
        for bin_index, reach in zip(index, ranges[keep]):
            expected[int(bin_index)] = min(
                expected.get(int(bin_index), np.inf), float(reach)
            )
        reduced_x, reduced_y = reduce_obstacles(x, y, bins=bins)
        self.assertEqual(reduced_x.size, len(expected))
        for px, py in zip(reduced_x, reduced_y):
            angle = np.arctan2(px, py)
            bin_index = int(np.clip(
                np.floor((angle + np.pi) / (2.0 * np.pi) * bins), 0, bins - 1
            ))
            self.assertAlmostEqual(
                float(np.hypot(px, py)), expected[bin_index], places=4
            )

    def test_out_of_range_points_are_dropped(self) -> None:
        x = np.array([0.0, 0.0], dtype=np.float32)
        y = np.array([9.0, 0.01], dtype=np.float32)
        reduced_x, _reduced_y = reduce_obstacles(x, y)
        self.assertEqual(reduced_x.size, 0)


class PlannerBudgetTests(unittest.TestCase):
    """The planner shares a Pi 3B with LiDAR parsing, optical flow and the
    dashboard, and must never be able to starve the Uno command lease."""

    def test_the_arc_bank_is_cached_between_ticks(self) -> None:
        planner = LocalPlanner()
        first = planner.bank(ARC_STEER_OPTIONS, SPEEDS)
        second = planner.bank(ARC_STEER_OPTIONS, SPEEDS)
        self.assertIs(first, second)

    def test_changing_the_speed_options_rebuilds_the_bank(self) -> None:
        planner = LocalPlanner()
        first = planner.bank(ARC_STEER_OPTIONS, SPEEDS)
        second = planner.bank(
            ARC_STEER_OPTIONS, tuple((pwm, 0.4 * pwm / 255.0) for pwm in (105,))
        )
        self.assertIsNot(first, second)

    def test_a_search_stays_far_inside_the_uno_control_lease(self) -> None:
        planner = LocalPlanner()
        bank = planner.bank(ARC_STEER_OPTIONS, SPEEDS)
        rng = np.random.default_rng(0)
        x = rng.uniform(-3.0, 3.0, 1500).astype(np.float32)
        y = rng.uniform(-3.0, 3.0, 1500).astype(np.float32)
        evaluate_arcs(x, y, bank, 10.0, 0.0)
        started = time.perf_counter()
        for _ in range(20):
            evaluate_arcs(x, y, bank, 10.0, 0.0)
        elapsed = (time.perf_counter() - started) / 20.0
        # Generous versus this desktop so the bound still means something on a
        # Pi 3B, but far below the 0.5 s Uno lease it must not threaten.
        self.assertLess(elapsed, 0.05)

    def test_the_candidate_set_stays_small(self) -> None:
        bank = _bank()
        self.assertLessEqual(len(bank.candidates) * bank.steps, 300)


if __name__ == "__main__":
    unittest.main()
