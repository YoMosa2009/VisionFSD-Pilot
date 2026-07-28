#!/usr/bin/env python3
"""Geometry-aware local navigation for the LD19 robot.

This module holds the parts of the robot runtime that can be reasoned about
and unit tested without any hardware attached: the 360-bin scan model, the
body-inflated clearance test, LiDAR-only yaw estimation, and the planner state
machine.

The central idea is that the robot is a rectangle, not a point.  For every
candidate heading the planner asks "how far can a body of my width travel that
way before something enters the swept corridor", which is what makes narrow
gaps passable and thin obstacles respected.  A camera classification never
enters that geometry; it can only veto or slow motion.

There is still no encoder and no IMU on this chassis.  Heading change is
estimated by correlating consecutive LiDAR range profiles, which is good enough
to notice "I have been turning the same way for a long time" and "I commanded
motion but the world did not change".  It is not odometry and it is not SLAM.

Angle convention throughout: degrees, zero straight ahead, positive to the
robot's right, matching the LD19 bins after the mounting offset is applied.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


BIN_COUNT = 360
# Beyond this the LD19 is still useful for display, but not for deciding how
# fast to drive a 20 cm robot across a room.  It is kept above typical indoor
# room dimensions so that the long axis of a room still outscores the short
# one; clipping too early makes every open heading tie and the robot shuttles.
PLANNING_HORIZON_M = 3.50
MAX_VALID_RANGE_M = 6.0
MIN_VALID_RANGE_M = 0.08


@dataclass(frozen=True)
class RobotGeometry:
    """Physical envelope of the chassis, in metres.

    ``lidar_forward_offset_m`` is how far the LD19 sits ahead of the middle of
    the robot; negative means it is mounted behind the middle.  Without it the
    corridor test measures from the wrong origin and the robot clips corners.
    """

    width_m: float = 0.14
    length_m: float = 0.15
    # Measured chassis, LiDAR assumed centred until measured.  Set
    # VISIONFSD_LIDAR_OFFSET_M if the LD19 is not over the middle of the robot.
    lidar_forward_offset_m: float = 0.0
    # Scaled down with the body: on a 0.14 m robot the previous 0.055 m was 79%
    # of the half-width, which quietly made every gap look narrower than it is.
    safety_margin_m: float = 0.040

    @property
    def half_width_m(self) -> float:
        return self.width_m / 2.0

    @property
    def corridor_half_width_m(self) -> float:
        return self.half_width_m + self.safety_margin_m

    @property
    def front_overhang_m(self) -> float:
        """Distance from the LiDAR origin to the front bumper."""
        return max(0.0, self.length_m / 2.0 - self.lidar_forward_offset_m)

    @property
    def rear_overhang_m(self) -> float:
        return max(0.0, self.length_m / 2.0 + self.lidar_forward_offset_m)

    @property
    def pivot_radius_m(self) -> float:
        """Radius swept by the corners when turning on the spot."""
        return math.hypot(self.length_m / 2.0, self.width_m / 2.0) + self.safety_margin_m


@dataclass(frozen=True)
class ScanFrame:
    """One planning-ready 360-bin polar scan in the robot's frame.

    ``ranges`` is metres with NaN for "no usable return in this direction".
    NaN deliberately means unknown rather than clear: the planner treats an
    unknown direction as explorable but never as measured free space.
    """

    ranges: np.ndarray
    confidence: np.ndarray
    fresh: bool
    stamp: float

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.ranges)))


def _filled(ranges: np.ndarray) -> np.ndarray:
    """Replace unknown directions with +inf so they never limit travel."""
    return np.where(np.isfinite(ranges), ranges, np.inf).astype(np.float32)


def scan_from_points(points, fresh: bool, stamp: float | None = None) -> ScanFrame:
    """Build a ScanFrame from ``LivePolarMap.fresh()`` output."""
    ranges = np.full(BIN_COUNT, np.nan, dtype=np.float32)
    confidence = np.zeros(BIN_COUNT, dtype=np.float32)
    for index, point in points:
        distance = point.distance_mm / 1000.0
        if not MIN_VALID_RANGE_M <= distance <= MAX_VALID_RANGE_M:
            continue
        slot = int(index) % BIN_COUNT
        # Keep the nearest return per direction: for collision avoidance the
        # closest surface is the one that matters.
        if not np.isfinite(ranges[slot]) or distance < ranges[slot]:
            ranges[slot] = distance
            confidence[slot] = float(point.confidence)
    return ScanFrame(ranges, confidence, fresh, time.monotonic() if stamp is None else stamp)


def scan_from_ranges(ranges_by_degree: dict[int, float], default: float | None = None,
                     confidence: float = 200.0, fresh: bool = True) -> ScanFrame:
    """Test and diagnostic helper: build a scan from degree -> metre pairs."""
    base = np.nan if default is None else float(default)
    ranges = np.full(BIN_COUNT, base, dtype=np.float32)
    conf = np.full(BIN_COUNT, confidence if default is not None else 0.0, dtype=np.float32)
    for angle, distance in ranges_by_degree.items():
        slot = int(round(angle)) % BIN_COUNT
        ranges[slot] = float(distance)
        conf[slot] = confidence
    return ScanFrame(ranges, conf, fresh, time.monotonic())


def despeckle(scan: ScanFrame, min_confidence: float = 55.0) -> ScanFrame:
    """Drop lone weak returns while preserving genuine thin obstacles.

    A chair or table leg at 2 m occupies barely one degree, so a rule that
    demands several neighbouring returns makes it invisible and the robot
    drives into it.  Only an isolated *and* low-confidence return is removed.
    """
    ranges = scan.ranges.copy()
    finite = np.isfinite(ranges)
    has_left = np.roll(finite, 1)
    has_right = np.roll(finite, -1)
    left_range = np.roll(ranges, 1)
    right_range = np.roll(ranges, -1)
    weak = scan.confidence < min_confidence

    isolated = finite & ~has_left & ~has_right & weak
    with np.errstate(invalid="ignore"):
        spike = (
            finite & has_left & has_right & weak
            & (ranges < left_range - 0.35)
            & (ranges < right_range - 0.35)
        )
    ranges[isolated | spike] = np.nan
    return ScanFrame(ranges, scan.confidence, scan.fresh, scan.stamp)


class CorridorModel:
    """Body-inflated travel limit for a set of candidate headings.

    Trigonometry for the candidate grid is built once, so each planning cycle
    is a couple of vectorised multiplies over a 106x360 table.  That keeps the
    Pi 3B's planning cost near a millisecond.
    """

    def __init__(self, geometry: RobotGeometry, span_deg: float = 104.0, step_deg: float = 2.0) -> None:
        self.geometry = geometry
        # The span must be a whole number of steps so that 0 degrees is itself a
        # candidate.  With an even split the two nearest options are -1 and +1,
        # they tie in open space, and argmax alternates between them: the robot
        # weaves while believing it is driving straight.
        steps = int(round(span_deg / step_deg))
        self.headings = (np.arange(-steps, steps + 1, dtype=np.float32) * step_deg)
        bins = np.arange(BIN_COUNT, dtype=np.float32)
        delta = np.radians(((bins[None, :] - self.headings[:, None]) + 180.0) % 360.0 - 180.0)
        self._sin = np.sin(delta).astype(np.float32)
        self._cos = np.cos(delta).astype(np.float32)
        self.straight_index = int(np.argmin(np.abs(self.headings)))

    def travel_limits(self, scan: ScanFrame) -> np.ndarray:
        """Metres of clear travel ahead of the bumper for every candidate heading."""
        ranges = _filled(scan.ranges)
        lateral = self._sin * ranges[None, :]
        along = self._cos * ranges[None, :]
        # cos > 0 keeps beams that are actually ahead of the candidate heading;
        # without it, returns behind the robot produce negative travel limits.
        inside = (self._cos > 0.02) & (np.abs(lateral) <= self.geometry.corridor_half_width_m)
        blocked = np.where(inside, along, np.inf)
        limits = blocked.min(axis=1) - self.geometry.front_overhang_m
        return np.clip(limits, 0.0, PLANNING_HORIZON_M).astype(np.float32)

    def limit_at(self, scan: ScanFrame, heading_deg: float) -> float:
        index = int(np.argmin(np.abs(self.headings - heading_deg)))
        return float(self.travel_limits(scan)[index])


def reverse_limit(scan: ScanFrame, geometry: RobotGeometry, span_deg: float = 30.0) -> float:
    """Clear travel behind the rear bumper, over a cone about straight back.

    The LD19 sees one horizontal plane only, so a shoe, a cable, or a step down
    is invisible behind the robot.  Reverse therefore always uses a larger
    margin than forward and is only ever commanded in short bursts.
    """
    ranges = _filled(scan.ranges)
    delta = np.radians(((np.arange(BIN_COUNT, dtype=np.float32) - 180.0) + 180.0) % 360.0 - 180.0)
    lateral = np.sin(delta) * ranges
    along = np.cos(delta) * ranges
    in_cone = np.abs(np.degrees(delta)) <= span_deg
    inside = in_cone & (np.abs(lateral) <= geometry.corridor_half_width_m + 0.02)
    if not np.any(inside):
        # Nothing measured behind the robot within LiDAR range.  That is good
        # evidence of open floor in the scan plane, but it is not a measurement,
        # so claim only enough room for one short burst rather than a clear run.
        confirmed = int(np.count_nonzero(np.isfinite(scan.ranges) & in_cone))
        return PLANNING_HORIZON_M if confirmed >= 4 else 0.30
    limit = float(np.min(np.where(inside, along, np.inf))) - geometry.rear_overhang_m
    return float(np.clip(limit, 0.0, PLANNING_HORIZON_M))


def pivot_clearance(scan: ScanFrame, geometry: RobotGeometry) -> float:
    """Smallest distance from the robot's centre to any return, in metres."""
    ranges = scan.ranges
    finite = np.isfinite(ranges)
    if not np.any(finite):
        return PLANNING_HORIZON_M
    angles = np.radians(np.arange(BIN_COUNT, dtype=np.float32))[finite]
    measured = ranges[finite]
    offset = geometry.lidar_forward_offset_m
    # Range measured from the LiDAR, re-expressed from the middle of the robot.
    centred = np.sqrt(np.maximum(0.0, measured ** 2 + offset ** 2 - 2.0 * measured * offset * np.cos(angles)))
    return float(np.min(centred))


class YawTracker:
    """Estimate heading change by correlating successive LiDAR range profiles.

    This is a 1-D scan match over a bounded shift window.  It gives the planner
    a measured turn rate instead of a commanded one, which is what makes both
    "am I circling" and "did the battery sag until the wheels stalled"
    answerable on a chassis with no encoders.
    """

    def __init__(self, max_shift_deg: int = 30, interval_s: float = 0.22) -> None:
        self.max_shift_deg = max_shift_deg
        self.interval_s = interval_s
        self._reference: np.ndarray | None = None
        self._reference_at = 0.0
        self.yaw_rate_dps = 0.0
        self.scene_change = 1.0
        self.heading_deg = 0.0

    def update(self, scan: ScanFrame, now: float) -> None:
        if not scan.fresh:
            return
        profile = np.where(np.isfinite(scan.ranges) & (scan.ranges <= 4.0), scan.ranges, np.nan)
        if np.count_nonzero(np.isfinite(profile)) < 40:
            return
        if self._reference is None:
            self._reference, self._reference_at = profile, now
            return
        elapsed = now - self._reference_at
        if elapsed < self.interval_s:
            return
        def cost_of(shift: int) -> float:
            difference = np.abs(profile - np.roll(self._reference, shift))
            valid = np.isfinite(difference)
            if np.count_nonzero(valid) < 30:
                return float("inf")
            return float(np.mean(difference[valid]))

        zero_cost = cost_of(0)
        best_shift, best_cost = 0, zero_cost
        # Visiting shifts in increasing magnitude makes a tie resolve to "no
        # rotation".  A featureless or highly symmetric room makes every shift
        # score alike, and picking an arbitrary winner there would invent a turn
        # rate out of nothing.
        for shift in sorted(range(-self.max_shift_deg, self.max_shift_deg + 1), key=abs):
            cost = cost_of(shift)
            if cost < best_cost - 1e-6:
                best_shift, best_cost = shift, cost
        self._reference, self._reference_at = profile, now
        if math.isinf(best_cost):
            return
        # Only accept a rotation that explains the scan materially better than
        # standing still, so sensor noise cannot accumulate into false heading.
        if math.isfinite(zero_cost) and zero_cost - best_cost < 0.01:
            best_shift = 0
            best_cost = zero_cost
        # Rolling the older profile forward by +shift aligns it with a scene
        # that moved that way past the sensor, so the robot itself turned the
        # other way.
        self.yaw_rate_dps = -best_shift / max(1e-3, elapsed)
        self.heading_deg = (self.heading_deg - best_shift) % 360.0
        self.scene_change = best_cost


class MotionMonitor:
    """Detect a commanded-but-not-moving robot, and persistent one-way turning.

    The weak motor battery on this kit sags under load, so "the Pi commanded
    55 PWM" and "the wheels turned" are genuinely different statements.
    """

    # A stationary LD19 still produces a few centimetres of range noise, and a
    # robot creeping at 0.2 m/s only shifts the profile a little more.  The
    # threshold and window are therefore deliberately forgiving: a missed stall
    # costs nothing (the Uno gates the bumper anyway) while a false one wastes a
    # reverse burst.
    SCENE_CHANGE_M = 0.02

    def __init__(self, stall_after_s: float = 2.5, circle_window_s: float = 25.0) -> None:
        self.stall_after_s = stall_after_s
        self.circle_window_s = circle_window_s
        self._moving_since: float | None = None
        self._progress_at = 0.0
        self.stalled = False
        self.turn_integral_deg = 0.0
        self._last_update = 0.0

    def update(self, left_pwm: int, right_pwm: int, yaw: YawTracker, now: float) -> None:
        elapsed = 0.0 if self._last_update == 0.0 else min(0.5, max(0.0, now - self._last_update))
        self._last_update = now
        if self.circle_window_s > 0:
            self.turn_integral_deg *= math.exp(-elapsed / self.circle_window_s)
        self.turn_integral_deg += yaw.yaw_rate_dps * elapsed

        if abs(left_pwm) + abs(right_pwm) == 0:
            self._moving_since = None
            self._progress_at = now
            self.stalled = False
            return
        if self._moving_since is None:
            self._moving_since = now
            self._progress_at = now
        # Either the world is shifting past the sensor, or the robot is turning.
        # Neither for over a second while driving means the wheels are not.
        if yaw.scene_change > self.SCENE_CHANGE_M or abs(yaw.yaw_rate_dps) > 5.0:
            self._progress_at = now
        self.stalled = now - self._progress_at > self.stall_after_s

    def reset_progress(self, now: float) -> None:
        self._progress_at = now
        self.stalled = False


class ExplorationMemory:
    """Coarse memory of recently travelled directions, to stop orbiting.

    The robot has no map it can trust, so this does not try to remember places.
    It remembers headings: a direction that has already absorbed a lot of
    driving time becomes progressively less attractive than an untried one.
    """

    def __init__(self, bins: int = 24, half_life_s: float = 30.0) -> None:
        self.bins = bins
        self.half_life_s = half_life_s
        self.visits = np.zeros(bins, dtype=np.float32)
        self._last_update = 0.0

    def update(self, world_heading_deg: float, moving: bool, now: float) -> None:
        elapsed = 0.0 if self._last_update == 0.0 else min(1.0, max(0.0, now - self._last_update))
        self._last_update = now
        if elapsed > 0.0 and self.half_life_s > 0:
            self.visits *= float(0.5 ** (elapsed / self.half_life_s))
        if moving:
            self.visits[self._slot(np.array([world_heading_deg]))[0]] += elapsed

    def _slot(self, headings_deg: np.ndarray) -> np.ndarray:
        return (np.round(headings_deg / (360.0 / self.bins)).astype(np.int32)) % self.bins

    def penalty(self, world_headings_deg: np.ndarray) -> np.ndarray:
        peak = float(self.visits.max())
        if peak < 2.0:
            return np.zeros(len(world_headings_deg), dtype=np.float32)
        return (self.visits[self._slot(world_headings_deg)] / peak).astype(np.float32)


class UltrasonicTrust:
    """Decide whether the Uno's single fixed ultrasonic agrees with the LiDAR.

    It exists because one static sensor aimed slightly low, or partly at the
    chassis, reads a constant short distance and silently vetoes everything.
    The LiDAR is the arbiter; the ultrasonic is a corroborating near-field
    input that may shorten a travel limit and nothing more.
    """

    def __init__(self, disagreement_s: float = 3.0, recovery_s: float = 4.0) -> None:
        self.disagreement_s = disagreement_s
        self.recovery_s = recovery_s
        self.trusted = True
        self._disagreeing_since: float | None = None
        self._agreeing_since: float | None = None

    def update(self, front_cm: float | None, lidar_front_m: float, now: float) -> None:
        if front_cm is None or not math.isfinite(lidar_front_m):
            self._disagreeing_since = None
            return
        ultrasonic_m = front_cm / 100.0
        # Ultrasonic reads much closer than the LiDAR sees anything: it is
        # looking at something the planner cannot confirm exists.
        disagrees = ultrasonic_m < 0.60 and lidar_front_m > ultrasonic_m + 0.30
        if disagrees:
            self._agreeing_since = None
            if self._disagreeing_since is None:
                self._disagreeing_since = now
            elif now - self._disagreeing_since >= self.disagreement_s:
                self.trusted = False
        else:
            self._disagreeing_since = None
            if not self.trusted:
                if self._agreeing_since is None:
                    self._agreeing_since = now
                elif now - self._agreeing_since >= self.recovery_s:
                    self.trusted = True


@dataclass
class DriveCommand:
    left_pwm: int
    right_pwm: int
    state: str
    reason: str
    heading_deg: float = 0.0
    target_speed: int = 0


@dataclass
class PlannerTuning:
    """Behaviour weights.  Distances are metres, penalties are metre-equivalent."""

    speed: int = 70
    min_move_pwm: int = 50
    pivot_pwm: int = 58
    reverse_pwm: int = 52
    ramp_per_cycle: int = 22

    creep_limit_m: float = 0.34
    cruise_limit_m: float = 1.10
    reverse_needed_m: float = 0.30
    reverse_burst_s: float = 0.85
    # Last resort before giving up: the LiDAR still measures this gap behind the
    # rear bumper, and backing away from whatever is in front is the safest
    # thing left to do.  Short and slow, re-checked every cycle.
    reverse_min_m: float = 0.12
    nudge_burst_s: float = 0.35
    pivot_max_s: float = 2.6

    # Extra clear distance stops being worth anything once there is a decent
    # run ahead.  Without this saturation the planner always rotates toward
    # whichever direction is roomiest, and a robot in the middle of a room
    # spins gently instead of crossing it.
    clearance_saturation_m: float = 1.60
    straight_cost_per_90deg: float = 0.90
    arc_max_deg: float = 42.0
    # A new heading must be this much better before the robot commits to it.
    # It exists to stop near-tied candidates swapping every cycle on scan noise.
    # Keep it small: the margin is also the largest standing heading bias the
    # robot can hold, since straight-line cost has to overcome it to recentre.
    # At 0.9 m per 90 degrees, 0.05 caps that bias at about 5 degrees.
    heading_switch_margin_m: float = 0.05
    circling_penalty: float = 0.80
    exploration_penalty: float = 0.55
    person_penalty: float = 1.20
    unknown_bonus: float = 0.25


class NavigationPlanner:
    """Reactive local navigator with escape, reverse and anti-orbit behaviour."""

    def __init__(self, geometry: RobotGeometry, tuning: PlannerTuning, standby_s: float,
                 started_at: float | None = None) -> None:
        self.geometry = geometry
        self.tuning = tuning
        self.standby_s = standby_s
        self.started_at = time.monotonic() if started_at is None else started_at
        self.corridor = CorridorModel(geometry)
        self.yaw = YawTracker()
        self.motion = MotionMonitor()
        self.memory = ExplorationMemory()
        self.ultrasonic = UltrasonicTrust()

        self.state = "STANDBY"
        self.reason = "BOOT_STANDBY"
        self.left_pwm = 0
        self.right_pwm = 0
        self.chosen_heading = 0.0
        self.forward_limit_m = 0.0
        self.rear_limit_m = 0.0
        self.recovery_count = 0
        self._manoeuvre: str | None = None
        self._manoeuvre_until = 0.0
        self._manoeuvre_dir = 1
        self._pivot_goal_deg = 0.0
        self._reverse_floor_m = 0.0
        self._speed_scale = 1.0
        self._commit_until = 0.0
        self._heading_index: int | None = None
        self._blocked_since: float | None = None
        self._last_escape_dir = 1
        self._unknown_kernel = np.ones(25, dtype=np.float32) / 25.0

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    def _end_manoeuvre(self, now: float) -> None:
        """Finish a turn and briefly forbid starting another one.

        Without this the robot pivots, sees a still-better heading behind it,
        pivots again, and shuffles on the spot without ever translating.
        """
        self._manoeuvre = None
        self._commit_until = now + 0.7

    # ---- output shaping -------------------------------------------------
    def _apply(self, left: float, right: float) -> None:
        """Respect the motor deadband, then ramp to limit battery current spikes."""
        floor = self.tuning.min_move_pwm
        step = max(1, self.tuning.ramp_per_cycle)

        def shape(value: float) -> int:
            magnitude = abs(value)
            # Below roughly half the deadband, asking for motion only makes the
            # motor buzz and sag the battery.  Idling that wheel instead is what
            # produces a genuinely tight arc or a one-wheel pivot.
            if magnitude < floor * 0.55:
                return 0
            return int(math.copysign(max(floor, min(105.0, magnitude)), value))

        def ramp(target: int, current: int) -> int:
            if target == 0:
                return 0
            value = int(np.clip(target, current - step, current + step))
            # Ramping must never leave a wheel parked inside its deadband.
            return value if abs(value) >= floor else int(math.copysign(floor, target))

        self.left_pwm = ramp(shape(left), self.left_pwm)
        self.right_pwm = ramp(shape(right), self.right_pwm)

    def _stop(self, state: str, reason: str) -> DriveCommand:
        self.state, self.reason = state, reason
        self.left_pwm = self.right_pwm = 0
        self._manoeuvre = None
        self._heading_index = None
        return DriveCommand(0, 0, state, reason, self.chosen_heading, 0)

    def _speed_for(self, limit_m: float) -> int:
        tuning = self.tuning
        span = max(0.05, tuning.cruise_limit_m - tuning.creep_limit_m)
        fraction = float(np.clip((limit_m - tuning.creep_limit_m) / span, 0.0, 1.0))
        slowest = max(tuning.min_move_pwm, int(tuning.speed * 0.62))
        value = (slowest + fraction * (tuning.speed - slowest)) * self._speed_scale
        # Any external slow-down still has to clear the motor deadband, or the
        # wheels only buzz and the stall detector fires on our own command.
        return int(round(max(tuning.min_move_pwm, value)))

    def _arc(self, heading_deg: float, speed: int) -> DriveCommand:
        # Curvature scales with heading error, so a distant obstacle is dodged
        # by a lazy arc and a near one by a tight arc, without ever stopping.
        turn = float(np.clip(heading_deg / self.tuning.arc_max_deg, -1.0, 1.0)) * 0.92
        self._apply(speed * (1.0 + turn), speed * (1.0 - turn))
        self.state = "ARC" if abs(heading_deg) > 4.0 else "CRUISE"
        return DriveCommand(self.left_pwm, self.right_pwm, self.state, self.reason, heading_deg, speed)

    def _pivot(self, direction: int) -> DriveCommand:
        """direction +1 turns right (clockwise seen from above), -1 turns left."""
        power = self.tuning.pivot_pwm
        self._apply(power * direction, -power * direction)
        self.state = "PIVOT"
        return DriveCommand(self.left_pwm, self.right_pwm, self.state, self.reason,
                            90.0 * direction, power)

    def _reverse(self, curve: int = 0) -> DriveCommand:
        """Back up; ``curve`` +1 swings the nose right, -1 swings it left."""
        power = self.tuning.reverse_pwm
        inner = int(power * 0.35)
        left = -power if curve >= 0 else -inner
        right = -inner if curve > 0 else -power
        self._apply(left, right)
        self.state = "REVERSE"
        return DriveCommand(self.left_pwm, self.right_pwm, self.state, self.reason, 180.0, power)

    # ---- candidate scoring ----------------------------------------------
    def _score(self, scan: ScanFrame, limits: np.ndarray, person_bearings: list[float]) -> np.ndarray:
        tuning = self.tuning
        headings = self.corridor.headings
        score = np.minimum(limits, tuning.clearance_saturation_m)
        score -= np.abs(headings) / 90.0 * tuning.straight_cost_per_90deg

        # Anti-orbit: after sustained one-way rotation, further turning the
        # same way costs more than turning back or driving straight.
        turn_integral = self.motion.turn_integral_deg
        if abs(turn_integral) > 240.0:
            same_way = (np.sign(headings) == np.sign(turn_integral)).astype(np.float32)
            strength = min(1.0, (abs(turn_integral) - 240.0) / 360.0)
            score -= same_way * (np.abs(headings) / 90.0) * tuning.circling_penalty * strength

        # Heading-diversity only influences which way to *turn*.  Applying it to
        # near-straight candidates penalises whatever heading the robot is
        # currently driving along, which curls a straight run into a slow arc.
        world = (self.yaw.heading_deg + headings) % 360.0
        turning = np.clip((np.abs(headings) - 10.0) / 30.0, 0.0, 1.0)
        score -= self.memory.penalty(world) * tuning.exploration_penalty * turning

        # An unmeasured direction is a candidate frontier, not a wall.
        unknown = (~np.isfinite(scan.ranges)).astype(np.float32)
        padded = np.concatenate((unknown[-12:], unknown, unknown[:12]))
        smoothed = np.convolve(padded, self._unknown_kernel, mode="valid")[:BIN_COUNT]
        score += smoothed[np.round(headings).astype(np.int32) % BIN_COUNT] * tuning.unknown_bonus

        for bearing in person_bearings:
            closeness = np.exp(-((headings - bearing) ** 2) / (2.0 * 22.0 ** 2))
            score -= closeness * tuning.person_penalty
        return score

    @staticmethod
    def _window(centre_deg: float, half_width_deg: int) -> np.ndarray:
        centre = int(round(centre_deg)) % BIN_COUNT
        return np.arange(centre - half_width_deg, centre + half_width_deg + 1) % BIN_COUNT

    # ---- main entry point ------------------------------------------------
    def decide(self, scan: ScanFrame | None, front_cm: float | None, uno_fresh: bool,
               camera_ready: bool, person_stop: bool, person_bearings: list[float],
               now: float, speed_scale: float = 1.0) -> DriveCommand:
        # A caller-supplied slow-down (for example the camera seeing clutter
        # close ahead) is applied as a speed cap inside the planner, so it goes
        # through the same deadband and ramp handling as every other output.
        self._speed_scale = float(np.clip(speed_scale, 0.5, 1.0))
        if now - self.started_at < self.standby_s:
            remaining = max(0, int(self.standby_s - (now - self.started_at)))
            return self._stop("STANDBY", f"STANDBY {remaining}s")
        if not uno_fresh:
            return self._stop("SAFETY_STOP", "STOP:UNO_STATUS_STALE")
        if scan is None or not scan.fresh:
            return self._stop("SAFETY_STOP", "STOP:LD19_STALE")
        if not camera_ready:
            return self._stop("SAFETY_STOP", "STOP:CAMERA_STALE")
        if person_stop:
            return self._stop("SAFETY_STOP", "STOP:CONFIRMED_PERSON")
        if scan.valid_count < 30:
            return self._stop("SAFETY_STOP", "STOP:LD19_TOO_FEW_RETURNS")

        clean = despeckle(scan)
        limits = self.corridor.travel_limits(clean)
        straight = self.corridor.straight_index

        self.yaw.update(clean, now)
        self.motion.update(self.left_pwm, self.right_pwm, self.yaw, now)
        self.memory.update(self.yaw.heading_deg, self.left_pwm + self.right_pwm > 40, now)
        self.ultrasonic.update(front_cm, float(limits[straight]) + self.geometry.front_overhang_m, now)

        if self.ultrasonic.trusted and front_cm is not None:
            # A trusted near return may only shorten travel, across the sensor's
            # own narrow cone.  It can no longer trigger a turn by itself, which
            # is what produced the pivot loop.
            ultrasonic_limit = max(0.0, front_cm / 100.0 - self.geometry.front_overhang_m)
            cone = np.abs(self.corridor.headings) <= 12.0
            limits = np.where(cone, np.minimum(limits, ultrasonic_limit), limits)

        self.forward_limit_m = float(limits[straight])
        self.rear_limit_m = reverse_limit(clean, self.geometry)

        active = self._continue_manoeuvre(clean, limits, now)
        if active is not None:
            return active
        if self.motion.stalled:
            return self._begin_recovery(clean, now, "STALLED")

        scores = self._score(clean, limits, person_bearings)
        # Only headings with real room to travel may be selected at all.
        usable = limits >= self.tuning.creep_limit_m
        if now < self._commit_until:
            # Just finished a turn: drive somewhere before considering another.
            usable = usable & (np.abs(self.corridor.headings) <= self.tuning.arc_max_deg)
        if not np.any(usable):
            if now < self._commit_until:
                self._commit_until = 0.0
                usable = limits >= self.tuning.creep_limit_m
            if not np.any(usable):
                return self._begin_recovery(clean, now, "BOXED_IN")

        self._blocked_since = None
        best = int(np.argmax(np.where(usable, scores, -np.inf)))
        # Stay on the previous heading unless a different one is clearly better.
        previous = self._heading_index
        if previous is not None and usable[previous] and best != previous:
            if scores[previous] + self.tuning.heading_switch_margin_m >= scores[best]:
                best = previous
        self._heading_index = best
        heading = float(self.corridor.headings[best])
        limit = float(limits[best])
        self.chosen_heading = heading
        if abs(heading) <= self.tuning.arc_max_deg:
            self.reason = f"DRIVE {heading:+.0f}deg CLEAR {limit:.2f}m"
            return self._arc(heading, self._speed_for(limit))
        # Too far off the nose to arc into: turn on the spot to face it first.
        return self._begin_pivot(clean, now, heading, f"ALIGN {heading:+.0f}deg")

    # ---- manoeuvres ------------------------------------------------------
    def _continue_manoeuvre(self, scan: ScanFrame, limits: np.ndarray, now: float) -> DriveCommand | None:
        if self._manoeuvre is None or now >= self._manoeuvre_until:
            if self._manoeuvre is not None:
                self._end_manoeuvre(now)
            return None
        if self._manoeuvre == "PIVOT":
            if self.motion.stalled or pivot_clearance(scan, self.geometry) < self.geometry.pivot_radius_m * 0.85:
                self._end_manoeuvre(now)
                return None
            # Turn until the robot actually faces where it decided to go, using
            # the measured heading rather than a fixed spin duration.
            error = self._wrap(self._pivot_goal_deg - self.yaw.heading_deg)
            reached = abs(error) <= 8.0 or (error != 0.0 and math.copysign(1, error) != self._manoeuvre_dir)
            if reached and float(limits[self.corridor.straight_index]) >= self.tuning.creep_limit_m:
                self._end_manoeuvre(now)
                return None
            return self._pivot(self._manoeuvre_dir)
        if self._manoeuvre == "REVERSE":
            if reverse_limit(scan, self.geometry) < self._reverse_floor_m:
                self._end_manoeuvre(now)
                self.reason = "REVERSE_ABORT:REAR_BLOCKED"
                return None
            return self._reverse(self._manoeuvre_dir)
        self._end_manoeuvre(now)
        return None

    def _begin_pivot(self, scan: ScanFrame, now: float, target_deg: float, reason: str) -> DriveCommand:
        if pivot_clearance(scan, self.geometry) < self.geometry.pivot_radius_m:
            return self._begin_recovery(scan, now, "PIVOT_TOO_TIGHT")
        direction = 1 if target_deg > 0 else -1
        self._manoeuvre = "PIVOT"
        self._heading_index = None
        self._manoeuvre_dir = direction
        self._manoeuvre_until = now + self.tuning.pivot_max_s
        self._pivot_goal_deg = (self.yaw.heading_deg + target_deg) % 360.0
        self._last_escape_dir = direction
        self.reason = f"{reason} PIVOT{'R' if direction > 0 else 'L'}"
        self.motion.reset_progress(now)
        return self._pivot(direction)

    def _begin_recovery(self, scan: ScanFrame, now: float, cause: str) -> DriveCommand:
        """Escape ladder, most useful first; HOLD only when nothing is safe."""
        self.recovery_count += 1
        rear = reverse_limit(scan, self.geometry)
        pivot_room = pivot_clearance(scan, self.geometry)

        if rear >= self.tuning.reverse_needed_m:
            # Curving the reverse swings the nose away from what blocked us, so
            # the robot usually does not need a separate pivot afterwards.
            return self._start_reverse(now, -self._last_escape_dir, self.tuning.reverse_burst_s,
                                       self.tuning.reverse_needed_m * 0.6,
                                       f"RECOVER {cause}: REVERSE {rear:.2f}m")
        if pivot_room >= self.geometry.pivot_radius_m:
            return self._start_escape_pivot(scan, now, cause)
        if rear >= self.tuning.reverse_min_m:
            # Too tight to turn and too tight for a full burst: shuffle straight
            # back a few centimetres and re-plan from there.
            return self._start_reverse(now, 0, self.tuning.nudge_burst_s, 0.06,
                                       f"RECOVER {cause}: NUDGE BACK {rear:.2f}m")
        if pivot_room >= self.geometry.pivot_radius_m * 0.88:
            return self._start_escape_pivot(scan, now, cause + "_TIGHT")
        if self._blocked_since is None:
            self._blocked_since = now
        held = now - self._blocked_since
        return self._stop("HOLD", f"HOLD {cause}: NO SAFE ESCAPE {held:.0f}s")

    def _start_reverse(self, now: float, curve: int, burst_s: float, abort_below_m: float,
                       reason: str) -> DriveCommand:
        self._manoeuvre = "REVERSE"
        self._heading_index = None
        self._manoeuvre_dir = curve
        self._manoeuvre_until = now + burst_s
        self._reverse_floor_m = abort_below_m
        self.reason = reason
        self.motion.reset_progress(now)
        return self._reverse(curve)

    def _start_escape_pivot(self, scan: ScanFrame, now: float, cause: str) -> DriveCommand:
        direction = self._best_pivot_direction(scan)
        self._manoeuvre = "PIVOT"
        self._heading_index = None
        self._manoeuvre_dir = direction
        self._manoeuvre_until = now + self.tuning.pivot_max_s
        # No specific target when escaping: sweep a quarter turn and re-plan.
        self._pivot_goal_deg = (self.yaw.heading_deg + 90.0 * direction) % 360.0
        self._last_escape_dir = direction
        self.reason = f"RECOVER {cause}: PIVOT{'R' if direction > 0 else 'L'}"
        self.motion.reset_progress(now)
        return self._pivot(direction)

    def _best_pivot_direction(self, scan: ScanFrame) -> int:
        """Turn toward the roomier half, breaking ties away from recent turning."""
        ranges = np.nan_to_num(scan.ranges, nan=PLANNING_HORIZON_M, posinf=PLANNING_HORIZON_M)
        right = float(np.mean(ranges[self._window(60.0, 40)]))
        left = float(np.mean(ranges[self._window(-60.0, 40)]))
        if abs(right - left) < 0.12:
            return -1 if self.motion.turn_integral_deg > 0 else 1
        return 1 if right > left else -1

    # ---- reporting -------------------------------------------------------
    def telemetry(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "heading": round(self.chosen_heading, 1),
            "forward_m": round(self.forward_limit_m, 2),
            "rear_m": round(self.rear_limit_m, 2),
            "yaw_rate_dps": round(self.yaw.yaw_rate_dps, 1),
            "turn_integral": round(self.motion.turn_integral_deg, 0),
            "stalled": self.motion.stalled,
            "ultrasonic_trusted": self.ultrasonic.trusted,
            "recoveries": self.recovery_count,
        }
