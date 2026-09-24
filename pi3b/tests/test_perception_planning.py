"""Regression tests for v1.9.22 perception and planning.

Covers the LD19 ingestion changes, the manufacturer's mixed-pixel filter,
moving-object tracking, route-guided arc selection, obstacle-memory clearing,
exploration goal commitment and roaming, and the planner thread.
"""

from __future__ import annotations

import math
import pathlib
import random
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lidar_visualizer import LD19Parser, LidarPoint
from robot_autonomy import (
    ARC_STEER_OPTIONS,
    CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT,
    MAX_GENTLE_HEADING_DEG,
    MAX_PWM,
    MAX_TURN_SPLIT_PWM,
    ROBOT_FOOTPRINT,
    ArduinoStatus,
    AutonomousPolicy,
    LD19Link,
    SectorClearance,
)
from robot_explorer import AsyncExplorer, ExplorationState, FrontierExplorer
from robot_local_planner import (
    ArcBank,
    ObstacleMemory,
    PlannerLimits,
    evaluate_arcs,
    route_progress,
)
from robot_scan import frame_from_points, mixed_pixel_keep_mask
from robot_slam_lite import LidarSlamLite
from robot_tracking import (
    MovingObjectTracker,
    TrackedObject,
    closest_approach,
    predicted_obstacles,
    segment_scan,
)


def _limits() -> PlannerLimits:
    return PlannerLimits(
        footprint=ROBOT_FOOTPRINT,
        yaw_rate_dps_at_full_steer=(
            MAX_TURN_SPLIT_PWM / MAX_PWM * CHASSIS_TURN_RATE_DPS_AT_FULL_SPLIT
        ),
        full_steer_deg=MAX_GENTLE_HEADING_DEG,
    )


# --------------------------------------------------------------------------- LD19


class LD19IngestionTests(unittest.TestCase):
    def test_table_crc_matches_the_bitwise_definition(self) -> None:
        """The table is a speed-up only; it must agree on every payload."""

        def bitwise(payload: bytes) -> int:
            crc = 0
            for byte in payload:
                crc ^= byte
                for _ in range(8):
                    crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
            return crc

        generator = random.Random(4)
        for _ in range(300):
            payload = bytes(generator.randrange(256) for _ in range(46))
            self.assertEqual(LD19Parser.crc8(payload), bitwise(payload))

    def test_snapshot_is_not_rebuilt_without_new_packets(self) -> None:
        """Consumers call snapshot several times per control tick; walking all
        720 angular bins in Python each time was pure waste."""
        import threading

        calls = []

        class _Map:
            def fresh(self, now, history):
                calls.append(now)
                return [(0, LidarPoint(0.0, 1000, 200, now))]

        link = object.__new__(LD19Link)
        link._lock = threading.Lock()
        link._map = _Map()
        link._seq = 5
        link._snapshot_seq = -1
        link._snapshot_points = []
        link._scan_history_s = 0.2
        link._last_packet_at = time.monotonic()
        link.snapshot()
        link.snapshot()
        link.snapshot()
        self.assertEqual(len(calls), 1)
        link._seq += 1
        link.snapshot()
        self.assertEqual(len(calls), 2)

    def test_reader_accepts_the_ld19_rated_range(self) -> None:
        from lidar_visualizer import LD19_MAX_RANGE_MM

        self.assertEqual(LD19_MAX_RANGE_MM, 12000)


class MixedPixelFilterTests(unittest.TestCase):
    """The artefact flag is display-only and must never hide a real wall."""

    def _glancing_wall(self, offset_m: float = 0.5):
        angles = np.arange(0.0, 360.0, 0.8, dtype=np.float32)
        ranges = []
        power = []
        for angle in angles:
            dx = math.sin(math.radians(float(angle)))
            if dx < -1e-3:
                distance = offset_m / -dx
                incidence = abs(dx)
            else:
                distance = 12.5
                incidence = 1.0
            ranges.append(distance * 1000.0)
            power.append(220.0 * incidence ** 0.5 - 6.0 * min(distance, 12.0))
        return (
            angles,
            np.array(ranges, dtype=np.float32),
            np.clip(np.array(power, dtype=np.float32), 0, 255),
        )

    def test_a_wall_seen_at_a_glancing_angle_is_kept(self) -> None:
        """The v1.9.22 regression. A range-jump filter flagged 68-95% of this
        wall at 1-2 m and all of it at 2-3 m, so the planner could not see
        walls the robot was driving alongside until they were 0.6 m away."""
        angles, ranges, power = self._glancing_wall()
        keep = mixed_pixel_keep_mask(angles, ranges, power, 3600.0)
        wall = (ranges > 1000.0) & (ranges < 3000.0)
        self.assertGreater(int(wall.sum()), 10)
        self.assertTrue(np.all(keep[wall]))

    def test_a_continuous_surface_is_kept(self) -> None:
        angles = np.linspace(0.0, 360.0, 450, endpoint=False).astype(np.float32)
        ranges = np.full(450, 2000.0, dtype=np.float32)
        power = np.full(450, 150.0, dtype=np.float32)
        self.assertTrue(np.all(mixed_pixel_keep_mask(angles, ranges, power, 3600.0)))

    def test_a_ghost_between_two_surfaces_is_flagged(self) -> None:
        angles = np.arange(0.0, 360.0, 0.8, dtype=np.float32)
        ranges = np.full(angles.size, 1000.0, dtype=np.float32)
        power = np.full(angles.size, 210.0, dtype=np.float32)
        door = (angles > 20) & (angles < 45)
        ranges[door] = 3500.0
        ghost = int(np.argmax(angles > 20))
        ranges[ghost] = 2100.0
        power[ghost] = 80.0
        keep = mixed_pixel_keep_mask(angles, ranges, power, 3600.0)
        self.assertFalse(keep[ghost])

    def test_a_strong_isolated_return_is_a_real_thin_object(self) -> None:
        angles = np.arange(0.0, 360.0, 0.8, dtype=np.float32)
        ranges = np.full(angles.size, 4000.0, dtype=np.float32)
        power = np.full(angles.size, 200.0, dtype=np.float32)
        ranges[10] = 1500.0
        power[10] = 235.0
        self.assertTrue(mixed_pixel_keep_mask(angles, ranges, power, 3600.0)[10])

    def test_planning_uses_every_return_not_the_flag(self) -> None:
        """The flag can mark a real dim chair leg. It must never reach the
        planner."""
        source = pathlib.Path(__file__).resolve().parents[1] / "robot_autonomy.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("scan_angles, scan_ranges = scan.kept()", text)
        self.assertIn("scan_angles, scan_ranges = scan.angles_deg, scan.ranges_m", text)

    def test_frame_arrays_are_consistent(self) -> None:
        now = 2.0
        points = [(i, LidarPoint(float(a), 1500, 200, now)) for i, a in enumerate(np.arange(0, 360, 0.8))]
        frame = frame_from_points(points, 7, 3600.0)
        self.assertEqual(frame.seq, 7)
        self.assertEqual(frame.stamp, now)
        self.assertEqual(frame.angles_deg.size, frame.ranges_m.size)
        x, y = frame.cartesian(kept_only=False)
        self.assertAlmostEqual(float(np.max(np.hypot(x, y))), 1.5, places=3)

    def test_empty_input_gives_an_empty_frame(self) -> None:
        frame = frame_from_points([], 3)
        self.assertEqual(frame.size, 0)


# --------------------------------------------------------------------------- tracking


def _scene(t: float, person_speed: float = 0.8):
    angles = np.arange(0.0, 360.0, 0.8, dtype=np.float32)
    radians = np.radians(angles)
    ranges = np.minimum(
        3.0 / np.maximum(np.abs(np.cos(radians)), 1e-6),
        3.0 / np.maximum(np.abs(np.sin(radians)), 1e-6),
    ).astype(np.float32)
    px, py = -1.2 + person_speed * t, 1.5
    for index, angle in enumerate(radians):
        dx, dy = math.sin(angle), math.cos(angle)
        along = px * dx + py * dy
        if along <= 0:
            continue
        miss_sq = px * px + py * py - along * along
        if miss_sq <= 0.12 ** 2:
            ranges[index] = min(ranges[index], along - math.sqrt(0.12 ** 2 - miss_sq))
    return angles, ranges


class TrackingTests(unittest.TestCase):
    def test_walls_are_not_tracking_candidates(self) -> None:
        angles = np.arange(0.0, 360.0, 0.8, dtype=np.float32)
        ranges = np.full(angles.size, 2.0, dtype=np.float32)
        self.assertEqual(segment_scan(angles, ranges), [])

    def test_a_compact_object_is_segmented(self) -> None:
        angles, ranges = _scene(0.0)
        clusters = segment_scan(angles, ranges)
        self.assertEqual(len(clusters), 1)
        cx, cy, _radius = clusters[0]
        self.assertAlmostEqual(cx, -1.2, delta=0.12)
        self.assertAlmostEqual(cy, 1.5, delta=0.15)

    def test_a_crossing_person_is_tracked_with_the_right_velocity(self) -> None:
        tracker = MovingObjectTracker()
        objects = ()
        for step in range(30):
            angles, ranges = _scene(step * 0.1)
            objects = tracker.update(angles, ranges, step * 0.1, 6.0, 6.0, 0.0, 0.0)
        movers = [item for item in objects if item.moving]
        self.assertEqual(len(movers), 1)
        self.assertAlmostEqual(movers[0].robot_vx_mps, 0.8, delta=0.15)
        self.assertAlmostEqual(movers[0].robot_vy_mps, 0.0, delta=0.15)

    def test_a_still_object_is_never_reported_as_moving(self) -> None:
        tracker = MovingObjectTracker()
        objects = ()
        for step in range(40):
            angles, ranges = _scene(0.0, person_speed=0.0)
            objects = tracker.update(angles, ranges, step * 0.1, 6.0, 6.0, 0.0, 0.0)
        self.assertTrue(objects)
        self.assertFalse(any(item.moving for item in objects))

    def test_fast_rotation_suppresses_new_motion_claims(self) -> None:
        """Pose error while the chassis spins makes still objects appear to
        move; motion must not be declared from that."""
        tracker = MovingObjectTracker()
        objects = ()
        for step in range(30):
            angles, ranges = _scene(step * 0.1)
            objects = tracker.update(angles, ranges, step * 0.1, 6.0, 6.0, 0.0, 90.0)
        self.assertFalse(any(item.moving for item in objects))

    def test_only_moving_tracks_produce_predictions(self) -> None:
        still = TrackedObject(1, 0, 0, 0, 0, 0.1, False, 9, 0.0, 1.0, 0.0, 0.0)
        moving = TrackedObject(2, 0, 0, 1, 0, 0.1, True, 9, -1.0, 1.0, 0.8, 0.0)
        px, _py = predicted_obstacles((still,))
        self.assertEqual(px.size, 0)
        px, py = predicted_obstacles((moving,))
        self.assertGreater(px.size, 0)
        # The furthest prediction lies ahead along the direction of travel.
        self.assertGreater(float(px.max()), 0.0)

    def test_closest_approach_for_a_head_on_crossing(self) -> None:
        crossing = TrackedObject(1, 0, 0, 0, 0, 0.1, True, 9, -1.0, 1.0, 1.0, 0.0)
        seconds, distance = closest_approach(crossing, 0.0)
        self.assertAlmostEqual(seconds, 1.0, places=3)
        self.assertAlmostEqual(distance, 1.0, places=3)

    def test_closest_approach_for_a_separating_object(self) -> None:
        leaving = TrackedObject(1, 0, 0, 0, 0, 0.1, True, 9, 1.0, 1.0, 1.0, 0.0)
        seconds, _distance = closest_approach(leaving, 0.0)
        self.assertEqual(seconds, 0.0)


# --------------------------------------------------------------------------- route guidance


class RouteGuidanceTests(unittest.TestCase):
    def test_progress_along_a_straight_route(self) -> None:
        route_x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        route_y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        along, cross = route_progress(
            route_x, route_y,
            np.array([0.0, 0.3], dtype=np.float32),
            np.array([1.5, 1.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(along[0]), 1.5, places=4)
        self.assertAlmostEqual(float(cross[0]), 0.0, places=4)
        self.assertAlmostEqual(float(cross[1]), 0.3, places=4)

    def test_a_turn_ahead_is_started_early_at_a_junction(self) -> None:
        """At a T-junction with the plan turning left, heading-based guidance
        drives straight past. Scoring along the route starts the turn."""
        points = []
        for y in np.arange(0.0, 3.0, 0.03):
            points.append((0.6, y))
            if not 1.0 <= y <= 1.9:
                points.append((-0.6, y))
        for x in np.arange(-3.0, -0.6, 0.03):
            points.append((x, 1.0))
            points.append((x, 1.9))
        obstacle_x = np.array([p[0] for p in points], dtype=np.float32)
        obstacle_y = np.array([p[1] for p in points], dtype=np.float32)
        route = np.array(
            [[0, 0], [0, 0.7], [-0.2, 1.3], [-0.8, 1.45], [-1.6, 1.45], [-2.6, 1.45]],
            dtype=np.float32,
        )
        bank = ArcBank(_limits(), ARC_STEER_OPTIONS, tuple((p, 0.55 * p / 255) for p in (105, 112)))
        plain = evaluate_arcs(obstacle_x, obstacle_y, bank, None, 0.0)
        guided = evaluate_arcs(obstacle_x, obstacle_y, bank, None, 0.0, route_xy=route)
        self.assertTrue(guided.admissible)
        self.assertLess(guided.steering_deg, plain.steering_deg)

        def off_route(choice):
            ex, ey = choice.path_xy[-1]
            _along, cross = route_progress(
                route[:, 0], route[:, 1],
                np.array([ex], dtype=np.float32), np.array([ey], dtype=np.float32),
            )
            return float(cross[0])

        self.assertLess(off_route(guided), off_route(plain))

    def test_route_is_expressed_in_the_robot_frame(self) -> None:
        """A route point one metre along the robot's heading must come out as
        straight ahead, whatever the heading is."""
        for heading in (0.0, 90.0, 225.0):
            policy = AutonomousPolicy(0.0, 112)
            policy.pose_x_m = 5.0
            policy.pose_y_m = 5.0
            policy.pose_heading_deg = heading
            radians = math.radians(heading)
            ahead = (5.0 + math.sin(radians), 5.0 - math.cos(radians))
            further = (5.0 + 2.0 * math.sin(radians), 5.0 - 2.0 * math.cos(radians))
            policy.exploration = ExplorationState(
                active=True, path_xy_m=((5.0, 5.0), ahead, further)
            )
            route = policy._route_robot_frame()
            self.assertIsNotNone(route)
            self.assertAlmostEqual(float(route[-1][0]), 0.0, places=3)
            self.assertAlmostEqual(float(route[-1][1]), 2.0, places=3)


# --------------------------------------------------------------------------- memory clearing


class MemoryClearingTests(unittest.TestCase):
    def _memory_with(self, angle_deg: float, range_m: float) -> ObstacleMemory:
        memory = ObstacleMemory()
        memory.add_scan(
            np.array([angle_deg], dtype=np.float32),
            np.array([range_m], dtype=np.float32),
            1.0,
        )
        return memory

    def test_a_ghost_left_by_a_moving_object_is_cleared(self) -> None:
        memory = self._memory_with(10.0, 1.0)
        removed = memory.clear_seen_through(
            np.array([10.0], dtype=np.float32), np.array([3.0], dtype=np.float32)
        )
        self.assertEqual(removed, 1)
        self.assertEqual(memory.size, 0)

    def test_something_still_in_view_is_kept(self) -> None:
        memory = self._memory_with(10.0, 1.0)
        memory.clear_seen_through(
            np.array([10.0], dtype=np.float32), np.array([1.02], dtype=np.float32)
        )
        self.assertEqual(memory.size, 1)

    def test_missing_returns_do_not_clear_anything(self) -> None:
        memory = self._memory_with(10.0, 1.0)
        memory.clear_seen_through(
            np.array([200.0], dtype=np.float32), np.array([3.0], dtype=np.float32)
        )
        self.assertEqual(memory.size, 1)


# --------------------------------------------------------------------------- moving-object policy


class YieldTests(unittest.TestCase):
    def _clear(self, now: float) -> SectorClearance:
        return SectorClearance(2.5, 2.5, 2.5, True, 2.5, 2.5, rear_m=2.0, scan_at=now)

    def _status(self, now: float) -> ArduinoStatus:
        return ArduinoStatus(front_cm=None, motion="S", received_at=now)

    def test_yields_to_an_object_about_to_cross(self) -> None:
        policy = AutonomousPolicy(0.0, 112)
        # Half a metre ahead, crossing at walking pace: on a collision course.
        policy.observe_movers((TrackedObject(1, 0, 0, 0, 0, 0.12, True, 9, -0.7, 0.5, 0.9, 0.0),))
        now = time.monotonic()
        command = policy.decide(self._clear(now), self._status(now), False, now)
        self.assertEqual(command, "STOP")
        self.assertTrue(policy.reason.startswith("STOP:YIELD_MOVING_OBJECT"))
        self.assertEqual(policy.intent, "YIELDING")

    def test_does_not_yield_to_an_object_moving_away(self) -> None:
        policy = AutonomousPolicy(0.0, 112)
        policy.observe_movers((TrackedObject(1, 0, 0, 0, 0, 0.12, True, 9, 0.7, 0.8, 0.9, 0.0),))
        now = time.monotonic()
        command = policy.decide(self._clear(now), self._status(now), False, now)
        self.assertEqual(command, "F")

    def test_a_yield_is_bounded_in_time(self) -> None:
        """Someone standing in a doorway must not park the robot forever."""
        from robot_autonomy import YIELD_MAX_S

        policy = AutonomousPolicy(0.0, 112)
        threat = TrackedObject(1, 0, 0, 0, 0, 0.12, True, 9, -0.7, 0.5, 0.9, 0.0)
        now = time.monotonic()
        released = False
        for index in range(int((YIELD_MAX_S + 1.0) / 0.05)):
            tick = now + index * 0.05
            policy.observe_movers((threat,))
            policy.decide(self._clear(tick), self._status(tick), False, tick)
            if not policy.reason.startswith("STOP:YIELD"):
                released = True
                break
        self.assertTrue(released)


# --------------------------------------------------------------------------- exploration


def _room_map() -> LidarSlamLite:
    mapper = LidarSlamLite()
    for step in range(15):
        now = step * 0.1
        points = []
        for index, angle in enumerate(np.arange(0.0, 360.0, 0.8)):
            radians = math.radians(angle)
            distance = min(2.5 / max(abs(math.cos(radians)), 1e-6), 2.0 / max(abs(math.sin(radians)), 1e-6))
            points.append((index, LidarPoint(float(angle), int(distance * 1000), 210, now)))
        mapper._integrate_points(points, now)
        mapper._map_updates += 1
    return mapper


class ExplorationCommitmentTests(unittest.TestCase):
    def test_global_inflation_matches_what_the_arc_planner_accepts(self) -> None:
        """Two planners disagreeing about what fits is what produced plans the
        robot would never drive."""
        needed = ROBOT_FOOTPRINT.radius_m + PlannerLimits().safety_margin_m
        self.assertAlmostEqual(FrontierExplorer.ROBOT_CLEARANCE_M, needed, delta=0.01)

    def test_an_open_room_produces_a_purposeful_goal(self) -> None:
        mapper = _room_map()
        explorer = FrontierExplorer()
        state = explorer.update(
            mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
            mapper.heading, mapper.metres, mapper._map_updates, 10.0,
        )
        self.assertTrue(state.active)
        self.assertGreaterEqual(state.target_distance_m, FrontierExplorer.MIN_TARGET_DISTANCE_M)

    def test_goal_is_held_across_replans(self) -> None:
        mapper = _room_map()
        explorer = FrontierExplorer()
        first = explorer.update(
            mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
            mapper.heading, mapper.metres, mapper._map_updates, 10.0,
        )
        for step in range(1, 6):
            state = explorer.update(
                mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
                mapper.heading, mapper.metres, mapper._map_updates, 10.0 + step,
            )
            self.assertEqual(state.target_x_m, first.target_x_m)
            self.assertEqual(state.target_y_m, first.target_y_m)

    def test_a_goal_with_no_progress_is_abandoned_and_blacklisted(self) -> None:
        mapper = _room_map()
        explorer = FrontierExplorer()
        first = explorer.update(
            mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
            mapper.heading, mapper.metres, mapper._map_updates, 10.0,
        )
        later = None
        for step in range(1, 30):
            later = explorer.update(
                mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
                mapper.heading, mapper.metres, mapper._map_updates,
                10.0 + step * (FrontierExplorer.PROGRESS_TIMEOUT_S / 10.0),
            )
        self.assertGreater(later.blacklisted, 0)
        self.assertNotEqual(
            (later.target_x_m, later.target_y_m), (first.target_x_m, first.target_y_m)
        )

    def test_a_recentre_moves_the_goal_instead_of_dropping_it(self) -> None:
        mapper = _room_map()
        explorer = FrontierExplorer()
        state = explorer.update(
            mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
            mapper.heading, mapper.metres, mapper._map_updates, 10.0,
        )
        target_before = explorer._target_cell
        fine_scale = mapper.cells / mapper.metres
        explorer.shift(30, -12, fine_scale)
        self.assertIsNotNone(explorer._target_cell)
        self.assertEqual(explorer._target_cell[0] - target_before[0], 10)
        self.assertEqual(explorer._target_cell[1] - target_before[1], -4)
        self.assertTrue(state.active)

    def test_roaming_prefers_far_unvisited_space(self) -> None:
        reachable = np.ones((120, 120), dtype=bool)
        visits = np.zeros((120, 120), dtype=np.uint16)
        visits[:, :60] = 400
        candidates = FrontierExplorer._patrol_candidates(reachable, visits, (60, 60), 16.0)
        self.assertTrue(candidates)
        _score, (row, col) = candidates[0]
        distance = math.hypot(row - 60, col - 60) / 16.0
        self.assertGreaterEqual(distance, FrontierExplorer.ROAM_MIN_DISTANCE_M)
        self.assertGreaterEqual(col, 60)

    def test_straight_line_goals_skip_the_search(self) -> None:
        free = np.ones((100, 100), dtype=bool)
        cost = np.zeros((100, 100), dtype=np.float32)
        path = FrontierExplorer._astar(free, (50, 10), (50, 90), traversal_cost=cost)
        self.assertEqual(path, [(50, 10), (50, 90)])


class AsyncExplorerTests(unittest.TestCase):
    def test_plans_arrive_without_blocking_the_caller(self) -> None:
        mapper = _room_map()
        explorer = AsyncExplorer(FrontierExplorer())
        self.addCleanup(explorer.close)
        started = time.perf_counter()
        explorer.publish(
            mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
            mapper.heading, mapper.metres, mapper._map_updates, time.monotonic(), force=True,
        )
        self.assertLess(time.perf_counter() - started, 0.05)
        deadline = time.monotonic() + 5.0
        while not explorer.state().active and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(explorer.state().active)

    def test_a_shift_moves_the_published_route_with_the_map(self) -> None:
        mapper = _room_map()
        explorer = AsyncExplorer(FrontierExplorer())
        self.addCleanup(explorer.close)
        explorer.publish(
            mapper.grid, mapper.observed, mapper.visits, mapper.x, mapper.y,
            mapper.heading, mapper.metres, mapper._map_updates, time.monotonic(), force=True,
        )
        deadline = time.monotonic() + 5.0
        while not explorer.state().active and time.monotonic() < deadline:
            time.sleep(0.02)
        before = explorer.state()
        fine_scale = mapper.cells / mapper.metres
        explorer.shift(48, 0, fine_scale)
        after = explorer.state()
        self.assertAlmostEqual(after.target_y_m - before.target_y_m, 48 / fine_scale, places=6)

    def test_invalidate_clears_the_published_plan(self) -> None:
        explorer = AsyncExplorer(FrontierExplorer())
        self.addCleanup(explorer.close)
        explorer.invalidate()
        self.assertFalse(explorer.state().active)


if __name__ == "__main__":
    unittest.main()
