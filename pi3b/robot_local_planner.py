"""Short-horizon obstacle memory and arc selection for the Pi robot runtime.

Two problems in the previous planner motivated this module.

**No memory.** Every steering decision was made from the single most recent
LD19 revolution. An obstacle that passed out of view behind the chassis, or
was briefly occluded by something nearer, simply ceased to exist. That is
also why multi-stage maneuvers failed: backing out of a corner needs the
robot to still know about the thing it just drove away from.

**No lookahead.** The planner scored headings, not motion. It asked "which
direction has the most room right now" and re-answered from scratch every
tick, so it could commit to a heading, discover a metre later that the
heading was a dead end, stop, and pivot - the observed drive/stop/drive.

The approach here is the standard one for this class of robot: keep a small
rolling, motion-compensated local obstacle set (a local costmap in all but
name), then evaluate a dynamic window of feasible (speed, steering) arcs
against it over a short horizon and pick the best-scoring admissible one.
It is deliberately the cheap version - hundreds of arithmetic operations per
tick, no grid inflation pass, no global search - because it has to share a
Pi 3B with LiDAR parsing, optical flow and the dashboard.

Nothing here is a localisation claim. The memory is a decaying buffer in the
robot's own frame, dead-reckoned over fractions of a second, not a map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# Robot frame used throughout: +y is straight ahead, +x is to the robot's
# right, angles are degrees clockwise from straight ahead. This matches the
# convention corridor_profile() already uses for LD19 returns.

# How long a remembered return stays usable. Long enough to cover the blind
# arc behind the chassis during a turn and brief occlusions, short enough
# that dead-reckoning error over the window stays small and a moved object
# cannot haunt the map.
MEMORY_HORIZON_S = 1.5
MEMORY_MAX_POINTS = 1100
MEMORY_MAX_RANGE_M = 4.0
# Remembered returns are dead-reckoned, so they are less trustworthy than the
# live scan. They can only ever *reduce* clearance, and they are held slightly
# further away than measured so memory alone cannot manufacture a hard block.
MEMORY_RANGE_BIAS_M = 0.04
# Multiplier that packs an angular bin index and a range into one sortable
# float64 key. Comfortably larger than any usable range in metres.
_BIN_KEY_SCALE = 1000.0


@dataclass(frozen=True)
class ArcChoice:
    """One evaluated (speed, steering) candidate."""

    admissible: bool = False
    steering_deg: float = 0.0
    pwm: int = 0
    speed_mps: float = 0.0
    clearance_m: float = 0.0
    reachable_m: float = 0.0
    stopping_m: float = 0.0
    score: float = 0.0
    reason: str = "NONE"
    # Sampled arc in robot-frame metres, for the dashboard overlay.
    path_xy: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Footprint:
    """Two-circle cover of a rectangular chassis.

    A single circumscribed circle is the cheap way to model a rectangle, but
    for a 23 x 27 cm chassis its radius is 17.6 cm against a true half-width
    of 11.4 cm - the robot would refuse gaps it fits through easily. Two
    circles placed along the length cover the same rectangle with a radius of
    13.2 cm, which is both more accurate and less timid, at the cost of one
    extra distance evaluation per sampled pose.
    """

    radius_m: float
    offset_m: float

    @classmethod
    def from_rectangle(cls, width_m: float, length_m: float) -> "Footprint":
        offset = length_m / 4.0
        return cls(radius_m=math.hypot(width_m / 2.0, offset), offset_m=offset)


@dataclass(frozen=True)
class GapChoice:
    """One opening measured directly from the scan, in robot-frame degrees."""

    found: bool = False
    bearing_deg: float = 0.0
    width_deg: float = 0.0
    clearance_m: float = 0.0
    score: float = 0.0


@dataclass
class PlannerLimits:
    """Physical and timing limits the arc search has to respect.

    ``top_speed_mps`` is the chassis speed at full PWM. It is an estimate,
    not a measurement - this robot has no encoders - and it is deliberately
    configurable, because every distance the governor computes scales with
    it. Setting it too high only makes the robot more cautious.
    """

    top_speed_mps: float = 0.55
    # LD19 revolution + control tick + serial + motor response. The distance
    # covered during this is unavoidable: the robot is committed to it before
    # any new measurement can change the command.
    reaction_latency_s: float = 0.25
    # Deceleration once the wheels are commanded to stop or reverse. Low
    # because a light chassis on a hard floor coasts.
    braking_mps2: float = 0.85
    # Never plan to pass closer than this to a remembered or measured return.
    safety_margin_m: float = 0.10
    # Chassis footprint used for every collision check.
    footprint: Footprint = field(
        default_factory=lambda: Footprint.from_rectangle(0.2286, 0.2667)
    )
    # An opening has to be at least this deep before it is worth turning
    # toward; shallower than this is an alcove, not a route.
    min_gap_depth_m: float = 0.70
    # Headroom an arc must have beyond its own stopping distance before it
    # counts as drivable. Without it the planner will happily commit to a
    # level whose whole usable arc is braking distance.
    stop_buffer_m: float = 0.25
    # Lookahead is a distance, not a time. A time horizon collapses to
    # almost nothing at this chassis's speed and cannot plan around anything.
    min_lookahead_m: float = 1.60
    max_lookahead_m: float = 2.40
    # Six samples over the lookahead leaves 27 cm between poses, and each
    # pose covers footprint radius plus margin - about 23 cm - so consecutive
    # samples still overlap and nothing can slip between them. Eight was
    # simply paying for resolution the collision check does not need.
    horizon_steps: int = 6
    # Look-ahead past the end of each arc. The arc itself only reaches about
    # 1.6 m; what the robot will face when it gets there decides whether that
    # arc leads anywhere. Open space is measured down a corridor this wide,
    # out to this distance.
    openness_half_width_m: float = 0.26
    openness_range_m: float = 4.0
    # Cruise is only earned by open space: the faster drive level scores
    # nothing unless at least this much clear corridor lies ahead of the arc
    # and the arc is close to straight.
    cruise_open_depth_m: float = 3.0
    cruise_max_steer_deg: float = 10.0
    # Degrees per second of yaw at a full-scale steering command.
    #
    # This is not a free parameter: it follows from how the chassis actually
    # steers. _differential() applies at most MAX_TURN_SPLIT_PWM (28) of wheel
    # split, and the chassis turn model puts that at roughly
    # 28 / 255 * 130 = 14 deg/s. Overstating it would make the planner believe
    # it can dodge sideways far more sharply than the wheels allow, commit to
    # an arc it cannot follow, and drive into the thing it meant to avoid.
    yaw_rate_dps_at_full_steer: float = 14.0
    # Steering command that counts as full scale, matching
    # MAX_GENTLE_HEADING_DEG in the runtime.
    full_steer_deg: float = 40.0


class ObstacleMemory:
    """A decaying, motion-compensated set of returns in the robot frame.

    Points are stored as Cartesian robot-frame metres and shifted on every
    control tick by the motion the robot believes it just made. Because the
    window is short, the accumulated dead-reckoning error stays small even
    though the underlying motion estimate is only commanded PWM plus, when
    available, measured IMU yaw.
    """

    def __init__(
        self,
        horizon_s: float = MEMORY_HORIZON_S,
        max_points: int = MEMORY_MAX_POINTS,
        max_range_m: float = MEMORY_MAX_RANGE_M,
    ) -> None:
        self.horizon_s = horizon_s
        self.max_points = max_points
        self.max_range_m = max_range_m
        self._x = np.zeros(0, dtype=np.float32)
        self._y = np.zeros(0, dtype=np.float32)
        self._stamp = np.zeros(0, dtype=np.float64)
        self._last_scan_at: float | None = None

    def reset(self) -> None:
        self._x = np.zeros(0, dtype=np.float32)
        self._y = np.zeros(0, dtype=np.float32)
        self._stamp = np.zeros(0, dtype=np.float64)
        self._last_scan_at = None

    @property
    def size(self) -> int:
        return int(self._x.size)

    def integrate_motion(self, forward_m: float, yaw_deg: float) -> None:
        """Move stored points opposite to the robot's own motion."""
        if self._x.size == 0:
            return
        if yaw_deg:
            # yaw_deg is the robot's own rotation, clockwise-positive. In the
            # robot frame (+x right, +y ahead) the surrounding points rotate
            # by the same signed angle, not its negation: turning right by 90
            # degrees puts what was ahead onto the robot's left.
            radians = math.radians(yaw_deg)
            cos = math.cos(radians)
            sin = math.sin(radians)
            x = cos * self._x - sin * self._y
            y = sin * self._x + cos * self._y
            self._x, self._y = x.astype(np.float32), y.astype(np.float32)
        if forward_m:
            self._y = (self._y - forward_m).astype(np.float32)

    def add_scan(
        self, angles_deg: np.ndarray, ranges_m: np.ndarray, now: float
    ) -> None:
        """Fold one LD19 revolution into the memory."""
        if self._last_scan_at is not None and now <= self._last_scan_at:
            # The control loop runs faster than the LD19; adding one cached
            # revolution repeatedly would give it disproportionate weight.
            return
        self._last_scan_at = now
        usable = (ranges_m >= 0.08) & (ranges_m <= self.max_range_m)
        if not np.any(usable):
            self._expire(now)
            return
        radians = np.radians(angles_deg[usable].astype(np.float32))
        biased = ranges_m[usable].astype(np.float32) + MEMORY_RANGE_BIAS_M
        x = np.sin(radians) * biased
        y = np.cos(radians) * biased
        stamp = np.full(x.size, now, dtype=np.float64)
        self._x = np.concatenate((self._x, x))
        self._y = np.concatenate((self._y, y))
        self._stamp = np.concatenate((self._stamp, stamp))
        self._expire(now)

    def _expire(self, now: float) -> None:
        keep = self._stamp >= now - self.horizon_s
        if not np.all(keep):
            self._x = self._x[keep]
            self._y = self._y[keep]
            self._stamp = self._stamp[keep]
        if self._x.size > self.max_points:
            # Drop the oldest first; the newest returns describe where the
            # robot is about to be.
            surplus = self._x.size - self.max_points
            self._x = self._x[surplus:]
            self._y = self._y[surplus:]
            self._stamp = self._stamp[surplus:]

    def cartesian(self) -> tuple[np.ndarray, np.ndarray]:
        """Remembered returns as robot-frame (x, y) metres."""
        return self._x, self._y

    #: Angular resolution of the see-through test.
    CLEAR_BIN_DEG = 2.0
    #: A remembered return is cleared when the live beam in its direction
    #: reaches at least this much further. Covers LD19 range noise and the
    #: deliberate MEMORY_RANGE_BIAS_M on stored returns.
    CLEAR_MARGIN_M = 0.20

    def clear_seen_through(
        self, angles_deg: np.ndarray, ranges_m: np.ndarray
    ) -> int:
        """Forget remembered returns the live scan has just seen past.

        This is costmap raytrace clearing. Without it, a person walking across
        the robot's view left a trail of remembered returns behind them for
        the full memory window - a phantom wall the planner routed around, or
        stopped for, after the person had gone. If the LD19 now measures a
        return further out along the same bearing, the space where the old
        return sat is demonstrably empty.

        Bearings with no live return clear nothing: missing data is not
        evidence of free space.
        """
        if self._x.size == 0 or angles_deg.size == 0:
            return 0
        bins = int(round(360.0 / self.CLEAR_BIN_DEG))
        usable = (ranges_m >= 0.08) & (ranges_m <= self.max_range_m + 1.0)
        if not np.any(usable):
            return 0
        live_bin = (
            (angles_deg[usable] % 360.0) / self.CLEAR_BIN_DEG
        ).astype(np.int32) % bins
        live = np.full(bins, np.inf, dtype=np.float32)
        np.minimum.at(live, live_bin, ranges_m[usable].astype(np.float32))
        remembered_range = np.hypot(self._x, self._y)
        remembered_bin = (
            (np.degrees(np.arctan2(self._x, self._y)) % 360.0)
            / self.CLEAR_BIN_DEG
        ).astype(np.int32) % bins
        seen_past = live[remembered_bin]
        cleared = np.isfinite(seen_past) & (
            seen_past > remembered_range + self.CLEAR_MARGIN_M
        )
        removed = int(np.count_nonzero(cleared))
        if removed:
            keep = ~cleared
            self._x = self._x[keep]
            self._y = self._y[keep]
            self._stamp = self._stamp[keep]
        return removed

    def polar(self) -> tuple[np.ndarray, np.ndarray]:
        """Remembered returns as (angles_deg, ranges_m)."""
        if self._x.size == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        ranges = np.hypot(self._x, self._y)
        angles = np.degrees(np.arctan2(self._x, self._y)) % 360.0
        keep = ranges >= 0.08
        return angles[keep].astype(np.float32), ranges[keep].astype(np.float32)


def stopping_distance_m(speed_mps: float, limits: PlannerLimits) -> float:
    """Distance covered before the chassis can be stopped.

    Reaction distance is travelled at full speed because no decision made
    during it can change the outcome; braking distance follows.
    """
    if speed_mps <= 0.0:
        return 0.0
    reaction = speed_mps * limits.reaction_latency_s
    braking = (speed_mps * speed_mps) / (2.0 * max(0.05, limits.braking_mps2))
    return reaction + braking


def reduce_obstacles(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 144,
    max_range_m: float = 3.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the obstacle set to the nearest return per angular bin.

    Collision checking only ever cares about the closest thing in a given
    direction, so a few thousand raw returns carry no more information than
    one per two-degree bin. On a Pi 3B this is the difference between a
    planner that fits in the control loop and one that does not: it shrinks
    the distance computation by roughly an order of magnitude without
    changing any decision.
    """
    if x.size == 0:
        return x, y
    ranges = np.hypot(x, y)
    keep = (ranges >= 0.05) & (ranges <= max_range_m)
    if not np.any(keep):
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    ranges = ranges[keep]
    angles = np.arctan2(x[keep], y[keep])
    index = np.floor((angles + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
    np.clip(index, 0, bins - 1, out=index)
    # Per-bin minimum without a scatter. np.minimum.at is an unbuffered
    # scatter and np.lexsort makes two passes; packing the bin and the range
    # into one sortable key needs a single sort, which is the cheapest of the
    # three and the one that fits a Pi 3B control loop. The key is exact in
    # float64 for these magnitudes, and sorting it ascending puts each bin's
    # nearest return first.
    key = index.astype(np.float64) * _BIN_KEY_SCALE + ranges
    key.sort()
    sorted_index = (key * (1.0 / _BIN_KEY_SCALE)).astype(np.int32)
    first = np.empty(sorted_index.size, dtype=bool)
    first[0] = True
    np.not_equal(sorted_index[1:], sorted_index[:-1], out=first[1:])
    bin_index = sorted_index[first]
    reach = (key[first] - bin_index * _BIN_KEY_SCALE).astype(np.float32)
    centres = ((bin_index + 0.5) / bins * 2.0 * np.pi - np.pi).astype(np.float32)
    return (
        (np.sin(centres) * reach).astype(np.float32),
        (np.cos(centres) * reach).astype(np.float32),
    )


class ArcBank:
    """Precomputed candidate arcs for one fixed set of planner options.

    The arc geometry depends only on the steering options, the speed options
    and the lookahead - all constant from tick to tick - so building it once
    and reusing it removes the dominant cost of the search. Constructing the
    27 small arcs per call cost more than the collision check itself, which
    is the sort of overhead a Pi 3B cannot absorb inside a control loop.

    Sampling by distance rather than by time is what fixes the lookahead. A
    1.3 s window at this chassis's 0.25 m/s sees 33 cm ahead, which cannot
    plan around anything; a distance horizon looks the same distance ahead
    regardless of how fast the robot happens to be going. Speed still enters,
    through curvature: a differential chassis turning at a given yaw rate
    carves a tighter radius the slower it travels.
    """

    def __init__(
        self,
        limits: PlannerLimits,
        steering_options_deg: np.ndarray,
        speed_options: tuple[tuple[int, float], ...],
    ) -> None:
        self.limits = limits
        self.steering_options_deg = np.asarray(
            steering_options_deg, dtype=np.float32
        )
        self.speed_options = tuple(speed_options)
        self.candidates = [
            (float(steering), int(pwm), float(mps))
            for steering in self.steering_options_deg
            for pwm, mps in self.speed_options
            if mps > 0.0
        ]
        self.fastest_mps = max(
            (mps for _pwm, mps in self.speed_options), default=1.0
        )
        self.slowest_mps = min((mps for _pwm, mps in self.speed_options), default=1.0)
        self.speed_span_mps = max(self.fastest_mps - self.slowest_mps, 1e-6)
        self.horizon_m = float(
            np.clip(
                max(
                    limits.min_lookahead_m,
                    3.0 * stopping_distance_m(self.fastest_mps, limits),
                ),
                limits.min_lookahead_m,
                limits.max_lookahead_m,
            )
        )
        self.steps = max(2, limits.horizon_steps)
        self.distance = np.linspace(
            self.horizon_m / self.steps,
            self.horizon_m,
            self.steps,
            dtype=np.float32,
        )
        if not self.candidates:
            self.pose_x = np.zeros(0, dtype=np.float32)
            self.pose_y = np.zeros(0, dtype=np.float32)
            self.arc_x = np.zeros((0, self.steps), dtype=np.float32)
            self.arc_y = np.zeros((0, self.steps), dtype=np.float32)
            return
        steering = np.array(
            [item[0] for item in self.candidates], dtype=np.float32
        )
        speeds = np.array(
            [item[2] for item in self.candidates], dtype=np.float32
        )
        yaw_rate_dps = (
            steering / limits.full_steer_deg * limits.yaw_rate_dps_at_full_steer
        )
        yaw_per_metre = yaw_rate_dps / np.maximum(0.05, speeds)
        headings = np.radians(
            yaw_per_metre[:, None] * self.distance[None, :]
        )
        step_distance = np.diff(
            np.concatenate((np.zeros(1, dtype=np.float32), self.distance))
        )
        # Integrate along each arc rather than assuming a straight line at
        # the final heading: the near part of the trajectory is what collides.
        self.arc_x = np.cumsum(
            np.sin(headings) * step_distance[None, :], axis=1
        ).astype(np.float32)
        self.arc_y = np.cumsum(
            np.cos(headings) * step_distance[None, :], axis=1
        ).astype(np.float32)
        self.arc_heading = headings.astype(np.float32)
        # Two footprint circles per sampled pose, offset along the chassis
        # axis, so a turning robot's trailing corner is checked as well as its
        # leading one. Front circles first, then rear, so a reshape can take
        # the worse of the two per pose.
        offset = limits.footprint.offset_m
        lead_x = self.arc_x + np.sin(self.arc_heading) * offset
        lead_y = self.arc_y + np.cos(self.arc_heading) * offset
        trail_x = self.arc_x - np.sin(self.arc_heading) * offset
        trail_y = self.arc_y - np.cos(self.arc_heading) * offset
        self.pose_x = np.concatenate(
            (lead_x.reshape(-1), trail_x.reshape(-1))
        ).astype(np.float32)
        self.pose_y = np.concatenate(
            (lead_y.reshape(-1), trail_y.reshape(-1))
        ).astype(np.float32)

    def matches(
        self,
        steering_options_deg: np.ndarray,
        speed_options: tuple[tuple[int, float], ...],
    ) -> bool:
        return (
            self.speed_options == tuple(speed_options)
            and self.steering_options_deg.shape
            == np.shape(steering_options_deg)
            and bool(
                np.array_equal(
                    self.steering_options_deg,
                    np.asarray(steering_options_deg, dtype=np.float32),
                )
            )
        )


def route_progress(
    route_x: np.ndarray,
    route_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance made good along a route, and distance off it, per endpoint.

    This is the substance of Nav2 DWB's PathDist and PathAlign critics: score a
    candidate by where its end lands relative to the global route, rather than
    by how far it gets along the robot's current heading.

    Heading-based progress is what fails at a turn. With the route bending
    left a metre ahead, driving straight on makes excellent "forward progress"
    and lands well off the route; the arc that starts turning early makes less
    forward progress and lands on it. Measured along the route, the second arc
    is correctly the better one - which is the whole difference between
    reacting to the next waypoint and following the plan through a multi-turn
    space.
    """
    count = route_x.size
    if count < 2 or end_x.size == 0:
        return (
            np.zeros(end_x.size, dtype=np.float32),
            np.zeros(end_x.size, dtype=np.float32),
        )
    ax = route_x[:-1]
    ay = route_y[:-1]
    sx = route_x[1:] - ax
    sy = route_y[1:] - ay
    length_sq = np.maximum(sx * sx + sy * sy, 1e-9)
    length = np.sqrt(length_sq)
    cumulative = np.concatenate(([0.0], np.cumsum(length)))

    def project(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.clip(
            ((px[:, None] - ax) * sx + (py[:, None] - ay) * sy) / length_sq,
            0.0,
            1.0,
        )
        dx = px[:, None] - (ax + t * sx)
        dy = py[:, None] - (ay + t * sy)
        distance_sq = dx * dx + dy * dy
        nearest = np.argmin(distance_sq, axis=1)
        rows = np.arange(px.size)
        along = cumulative[nearest] + t[rows, nearest] * length[nearest]
        return along, np.sqrt(distance_sq[rows, nearest])

    origin_along, _origin_cross = project(
        np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    )
    along, cross = project(end_x, end_y)
    return (
        (along - origin_along[0]).astype(np.float32),
        cross.astype(np.float32),
    )


def arc_openness(
    bank: "ArcBank",
    obstacle_x: np.ndarray,
    obstacle_y: np.ndarray,
    any_blocked: np.ndarray,
    first_blocked: np.ndarray,
) -> np.ndarray:
    """Clear distance ahead of each arc's usable end, along its end heading.

    This is what gives the local planner a horizon beyond its own arcs. Two
    arcs can have identical clearance for their 1.6 m, while one ends pointing
    down an open corridor and the other ends squarely facing a wall a metre
    further on. Scoring only the arc prefers them equally; scoring what lies
    beyond prefers the one that actually leads somewhere, which is the
    difference between reacting to the next obstacle and heading for the
    opening.
    """
    limits = bank.limits
    count = len(bank.candidates)
    depth = np.full(count, limits.openness_range_m, dtype=np.float32)
    if count == 0 or obstacle_x.size == 0:
        return depth
    last = np.where(any_blocked, first_blocked - 1, bank.steps - 1)
    rows = np.arange(count)
    index = np.maximum(last, 0)
    end_x = np.where(last >= 0, bank.arc_x[rows, index], 0.0)
    end_y = np.where(last >= 0, bank.arc_y[rows, index], 0.0)
    heading = np.where(last >= 0, bank.arc_heading[rows, index], 0.0)
    ux = np.sin(heading)[:, None]
    uy = np.cos(heading)[:, None]
    rel_x = obstacle_x[None, :] - end_x[:, None]
    rel_y = obstacle_y[None, :] - end_y[:, None]
    along = rel_x * ux + rel_y * uy
    lateral = np.abs(rel_x * uy - rel_y * ux)
    in_corridor = (along > 0.0) & (lateral <= limits.openness_half_width_m)
    nearest = np.where(in_corridor, along, np.inf).min(axis=1)
    depth = np.minimum(depth, nearest).astype(np.float32)
    # An arc that is blocked outright leads nowhere, whatever it faces.
    depth[last < 0] = 0.0
    return depth


def evaluate_arcs(
    obstacle_x: np.ndarray,
    obstacle_y: np.ndarray,
    bank: ArcBank,
    goal_heading_deg: float | None,
    current_steering_deg: float,
    reduce: bool = True,
    direction_lock: int = 0,
    route_xy: np.ndarray | None = None,
) -> ArcChoice:
    """Pick the best admissible (speed, steering) arc from a prebuilt bank.

    Scoring balances four things the previous heading-only chooser could not
    trade off against each other at all:

    * how much room the whole predicted arc has, not just its final heading,
    * progress toward the exploration goal,
    * staying near the currently commanded steering, which is what removes
      the tick-to-tick weaving, and
    * preferring speed, but only among arcs that are already safe.

    An arc is admissible only when the chassis could still stop inside the
    clearance that arc actually has. That single rule ties commanded speed to
    measured room, and is why this cannot select a speed the sensing latency
    does not support.
    """
    best = ArcChoice(reason="NO_ADMISSIBLE_ARC")
    if not bank.candidates:
        return best
    if reduce:
        # Out to 5 m rather than 3.2 so the look-ahead past each arc can see
        # the far side of a room, not only what the arc itself might touch.
        obstacle_x, obstacle_y = reduce_obstacles(
            obstacle_x, obstacle_y, max_range_m=5.0
        )
    limits = bank.limits
    distance = bank.distance
    if obstacle_x.size:
        # One batched distance computation for every candidate pose against
        # the reduced obstacle set. Squared distances keep this out of
        # np.hypot, whose overflow-safe path costs several times more than
        # the arithmetic it protects; the square root is then taken over the
        # few hundred per-pose minima rather than the whole matrix.
        delta_x = obstacle_x[None, :] - bank.pose_x[:, None]
        delta_y = obstacle_y[None, :] - bank.pose_y[:, None]
        squared = delta_x * delta_x + delta_y * delta_y
        nearest = np.sqrt(squared.min(axis=1))
        # Lead and trail circles were stacked in that order; a pose is only as
        # clear as its worse circle.
        per_circle = nearest.reshape(2, len(bank.candidates), bank.steps)
        margins = per_circle.min(axis=0) - limits.footprint.radius_m
    else:
        margins = np.full(
            (len(bank.candidates), bank.steps),
            limits.safety_margin_m * 4.0,
            dtype=np.float32,
        )
    blocked = margins < limits.safety_margin_m
    any_blocked = blocked.any(axis=1)
    first_blocked = blocked.argmax(axis=1)
    reachable = np.where(
        any_blocked,
        np.where(first_blocked > 0, distance[first_blocked - 1], 0.0),
        distance[-1],
    )
    # Report clearance over the part of the arc the robot can actually use.
    # Including samples past the block point produced negative "gaps" for
    # arcs that were never going to travel that far.
    steps_index = np.arange(bank.steps)[None, :]
    usable = np.where(
        any_blocked[:, None], steps_index < first_blocked[:, None], True
    )
    worst_margin = np.where(usable, margins, np.inf).min(axis=1)
    openness = arc_openness(bank, obstacle_x, obstacle_y, any_blocked, first_blocked)
    # Progress is distance made good along the current heading, not arc
    # length. Scoring arc length rewards the arc that curls tightly away from
    # everything - it stays "clear" for its whole length while going nowhere -
    # which is what made the robot orbit local objects instead of crossing
    # open floor and driving through a gap.
    last_usable = np.where(any_blocked, first_blocked - 1, bank.steps - 1)
    rows = np.arange(len(bank.candidates))
    usable_index = np.maximum(last_usable, 0)
    progress = np.where(
        last_usable >= 0, bank.arc_y[rows, usable_index], 0.0
    )
    following_route = route_xy is not None and len(route_xy) >= 2
    if following_route:
        route_along, route_cross = route_progress(
            route_xy[:, 0].astype(np.float32),
            route_xy[:, 1].astype(np.float32),
            np.where(last_usable >= 0, bank.arc_x[rows, usable_index], 0.0),
            np.where(last_usable >= 0, bank.arc_y[rows, usable_index], 0.0),
        )

    best_score = -np.inf
    for index, (steering_deg, pwm, speed_mps) in enumerate(bank.candidates):
        if direction_lock and steering_deg * direction_lock < 0.0:
            # Oscillation was detected and this turn is against the locked
            # direction. Removing the option outright is the point: leaving
            # it available at a penalty is what let the tie keep flipping.
            continue
        reachable_m = float(reachable[index])
        required = stopping_distance_m(speed_mps, limits)
        # Require real headroom beyond the bare stopping distance. Bare
        # equality means committing to a drive level whose entire usable arc
        # is consumed by braking, with nothing left for the braking model
        # being optimistic - which on a smooth floor it will be.
        if reachable_m < required + limits.stop_buffer_m:
            continue
        clearance_m = float(worst_margin[index])
        if not np.isfinite(clearance_m):
            clearance_m = limits.safety_margin_m
        goal_term = 0.0
        if goal_heading_deg is not None:
            # Falls off over 45 degrees rather than 90. A gentler curve leaves
            # the goal nearly flat across the whole candidate set, so forward
            # progress decides everything and the robot drives straight past
            # the direction it meant to explore.
            error = abs(steering_deg - goal_heading_deg)
            goal_term = max(0.0, 1.0 - error / 45.0) * 1.40
        # An arc that runs into something is categorically worse than one that
        # does not, and more so the sooner it happens. Without this an arc
        # blocked at half a metre scores about the same as a clear one that
        # merely curves more, because progress and clearance alone are too
        # close together to separate them.
        obstruction = 0.0
        if bool(any_blocked[index]):
            obstruction = 0.35 + 0.45 * (
                1.0 - min(1.0, reachable_m / max(0.05, bank.horizon_m))
            )
        # Staying near the steering already applied is what removes the
        # tick-to-tick weaving; without it the search re-answers from scratch
        # every cycle and two near-tied arcs alternate.
        smoothness = -abs(steering_deg - current_steering_deg) / 90.0 * 1.05
        straightness = -abs(steering_deg) / 90.0 * 0.30
        if following_route:
            # Guided by the route: reward distance made good along it and
            # penalise ending up off it. Straight-ahead progress is kept only
            # as a small tie-breaker, and the single-heading goal term is
            # dropped - the route already says where to go, turn by turn.
            room = min(float(progress[index]), 2.0) * 0.30
            goal_term = (
                float(np.clip(route_along[index], -0.5, 2.5)) * 1.40
                - min(float(route_cross[index]), 1.5) * 1.10
            )
        else:
            room = min(float(progress[index]), 2.0) * 0.80
        margin_term = min(max(clearance_m, 0.0), 0.60) * 0.55
        # Where the arc leads: clear corridor ahead of its end, in the
        # direction the chassis will then face.
        depth = float(openness[index])
        ahead_term = min(depth, limits.openness_range_m) / limits.openness_range_m * (
            # A global route already looks past this horizon, across the whole
            # map, and knows which way to turn at a junction; a straight
            # corridor ahead must not outvote it. Without a route this is the
            # planner's only view of where an arc leads.
            0.0 if following_route else 1.20
        )
        # 0 for the slowest drive level, 1 for the fastest. Ranked across the
        # levels rather than as a fraction of top speed: 105 and 112 PWM are
        # only 6% apart, which made the preference too weak to matter.
        speed_rank = (speed_mps - bank.slowest_mps) / bank.speed_span_mps
        open_enough = (
            depth >= limits.cruise_open_depth_m
            and abs(steering_deg) <= limits.cruise_max_steer_deg
        )
        score = (
            room
            + margin_term
            + goal_term
            + smoothness
            + straightness
            - obstruction
            + ahead_term
            # Speed is earned by open space, not preferred by default. Outside
            # it, the slower level is actively favoured.
            + (
                # Small: enough to pick the faster level on the same arc, not
                # enough to pull the steering choice toward straight.
                speed_rank * 0.20
                if open_enough
                else -speed_rank * 0.60
            )
        )
        if score > best_score:
            best_score = score
            best = ArcChoice(
                admissible=True,
                steering_deg=steering_deg,
                pwm=pwm,
                speed_mps=speed_mps,
                clearance_m=clearance_m,
                reachable_m=reachable_m,
                stopping_m=required,
                score=float(score),
                reason="ARC",
                path_xy=tuple(
                    (float(px), float(py))
                    for px, py in zip(bank.arc_x[index], bank.arc_y[index])
                ),
            )
    return best


class ProgressWatchdog:
    """Detect turning without getting anywhere, then lock a turn direction.

    Two similar openings score almost identically, so a planner that
    re-decides every cycle can swap between them indefinitely: turn left,
    which makes the right gap look better, turn right, repeat. The robot is
    moving the whole time, so no stall detector fires, and it never leaves
    the spot. That is the classic local-planner limit cycle.

    The established remedies agree on the shape of the answer. Nav2's
    OscillationCritic keeps a watchdog and refuses the opposite sign of
    motion until the robot has actually travelled a minimum distance or
    turned a minimum angle. TEB detects the same condition and responds by
    weighting the optimiser toward "prefer the current turning direction".
    Both are: notice the indecision, then remove the choice until real
    progress happens.

    The signature used here is deliberately different from TEB's
    velocity-epsilon test, which assumes an oscillating robot is nearly
    stationary. This one spins briskly. What distinguishes it is that a lot
    of *absolute* yaw accumulates while *net* yaw and forward travel stay
    near zero.
    """

    def __init__(
        self,
        window_s: float = 3.0,
        abs_yaw_deg: float = 70.0,
        net_yaw_deg: float = 30.0,
        advance_m: float = 0.25,
        lock_s: float = 4.0,
        release_advance_m: float = 0.45,
    ) -> None:
        self.window_s = window_s
        self.abs_yaw_deg = abs_yaw_deg
        self.net_yaw_deg = net_yaw_deg
        self.advance_m = advance_m
        self.lock_s = lock_s
        self.release_advance_m = release_advance_m
        self._history: list[tuple[float, float, float]] = []
        self._locked_sign = 0
        self._lock_until = 0.0
        self._lock_advance_m = 0.0
        self.oscillations = 0

    def reset(self) -> None:
        self._history.clear()
        self._locked_sign = 0
        self._lock_until = 0.0
        self._lock_advance_m = 0.0

    @property
    def locked_sign(self) -> int:
        """-1 to allow only left turns, +1 only right, 0 for no lock."""
        return self._locked_sign

    def update(
        self,
        now: float,
        forward_step_m: float,
        yaw_step_deg: float,
        commanding: bool,
    ) -> bool:
        """Fold one control tick in. Returns True when oscillation is new."""
        if self._locked_sign:
            self._lock_advance_m += max(0.0, forward_step_m)
            released = (
                now >= self._lock_until
                or self._lock_advance_m >= self.release_advance_m
            )
            if released:
                self._locked_sign = 0
                self._lock_advance_m = 0.0
        if not commanding:
            self._history.clear()
            return False
        self._history.append((now, forward_step_m, yaw_step_deg))
        cutoff = now - self.window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.pop(0)
        if len(self._history) < 4:
            return False
        elapsed = now - self._history[0][0]
        if elapsed < self.window_s * 0.8:
            return False
        advance = sum(step for _t, step, _yaw in self._history)
        yaw_steps = [yaw for _t, _step, yaw in self._history]
        absolute_yaw = sum(abs(value) for value in yaw_steps)
        net_yaw = abs(sum(yaw_steps))
        oscillating = (
            absolute_yaw >= self.abs_yaw_deg
            and net_yaw <= self.net_yaw_deg
            and advance <= self.advance_m
        )
        if not oscillating or self._locked_sign:
            return False
        # Commit to the direction the robot has net-turned toward, or, with
        # no net preference at all, simply to the most recent one. Breaking a
        # symmetric tie arbitrarily and then sticking to it beats re-deciding
        # it every cycle; which way is chosen matters far less than that the
        # choice stops changing.
        total = sum(yaw_steps)
        if abs(total) > 1e-6:
            self._locked_sign = 1 if total > 0.0 else -1
        else:
            recent = next(
                (value for value in reversed(yaw_steps) if abs(value) > 1e-6),
                1.0,
            )
            self._locked_sign = 1 if recent > 0.0 else -1
        self._lock_until = now + self.lock_s
        self._lock_advance_m = 0.0
        self.oscillations += 1
        self._history.clear()
        return True


def find_gap(
    obstacle_x: np.ndarray,
    obstacle_y: np.ndarray,
    limits: PlannerLimits,
    goal_heading_deg: float | None = None,
    held_bearing_deg: float | None = None,
    bins: int = 72,
    max_range_m: float = 3.2,
    direction_lock: int = 0,
) -> GapChoice:
    """Find the widest usable opening anywhere around the robot.

    The arc search only considers headings the chassis can reach while still
    rolling forward, roughly plus or minus 36 degrees. When every one of
    those is blocked - which is what being surrounded by furniture looks like
    - the old answer was a fixed-size blind pivot, re-evaluated on arrival,
    which in a cluttered room produced repeated pivots that never committed
    to anything. This looks at the whole revolution instead and names a real
    measured opening to turn toward.

    This is the gap-selection stage of the Follow-The-Gap family. The known
    failure of that family is that two similar gaps swap rank between scans
    and the robot zigzags, so ``held_bearing_deg`` applies hysteresis toward
    the opening already being followed.
    """
    span_deg = 360.0 / bins
    clearance = np.full(bins, max_range_m, dtype=np.float32)
    if obstacle_x.size:
        ranges = np.hypot(obstacle_x, obstacle_y)
        keep = ranges <= max_range_m
        if np.any(keep):
            angles = np.degrees(
                np.arctan2(obstacle_x[keep], obstacle_y[keep])
            ) % 360.0
            index = np.minimum((angles / span_deg).astype(np.int32), bins - 1)
            np.minimum.at(clearance, index, ranges[keep].astype(np.float32))
    # A direction is usable when the chassis plus its margin fits, with
    # enough room beyond to be worth committing to.
    needed = limits.footprint.radius_m + limits.safety_margin_m
    passable = clearance >= max(limits.min_gap_depth_m, needed * 2.0)
    if not np.any(passable):
        return GapChoice()
    # Walk contiguous runs on the circle by rotating the mask so a run that
    # straddles the zero crossing is not split into two.
    start = 0
    if passable[0] and passable[-1]:
        blocked = np.nonzero(~passable)[0]
        if blocked.size == 0:
            # Every direction is open; straight ahead is as good as any.
            return GapChoice(
                found=True,
                bearing_deg=0.0,
                width_deg=360.0,
                clearance_m=float(clearance.min()),
                score=float(clearance.min()),
            )
        start = int(blocked[0])
    rolled = np.roll(passable, -start)
    rolled_clearance = np.roll(clearance, -start)
    best = GapChoice()
    index = 0
    while index < bins:
        if not rolled[index]:
            index += 1
            continue
        end = index
        while end + 1 < bins and rolled[end + 1]:
            end += 1
        run = slice(index, end + 1)
        width_deg = (end - index + 1) * span_deg
        depth_m = float(rolled_clearance[run].min())
        # Physical width is the chord between the two returns that bound the
        # opening - the classic Follow-The-Gap definition - not the arc length
        # at the gap's own depth.
        #
        # Measuring at the depth is wrong in the direction that matters: a
        # doorway reads as deep because the room beyond it is deep, so a 30
        # degree slot between walls half a metre away scored as if it were
        # 1.4 m wide when its actual mouth is 29 cm, and the robot committed
        # to turning toward openings it cannot fit through. The arc planner
        # then refused to drive, and the two disagreed forever.
        left_edge = (index - 1) % bins
        right_edge = (end + 1) % bins
        edge_left_m = float(rolled_clearance[left_edge])
        edge_right_m = float(rolled_clearance[right_edge])
        separation = math.radians(width_deg + span_deg)
        mouth_m = math.sqrt(
            max(
                0.0,
                edge_left_m * edge_left_m
                + edge_right_m * edge_right_m
                - 2.0 * edge_left_m * edge_right_m * math.cos(separation),
            )
        )
        if width_deg >= 180.0:
            # More than half the circle is open; there are no bounding edges
            # to take a chord between.
            mouth_m = max(mouth_m, depth_m)
        if mouth_m >= needed * 2.0:
            centre_bin = (start + index + end) / 2.0 + 0.5
            bearing = (centre_bin * span_deg + 180.0) % 360.0 - 180.0
            if direction_lock and bearing * direction_lock < 0.0:
                # Turning the other way is exactly what the lock exists to
                # stop; an opening behind that turn is not an option yet.
                index = end + 1
                continue
            score = (
                min(depth_m, 2.5) * 0.9
                + min(mouth_m, 1.5) / 1.5 * 1.1
                - abs(bearing) / 180.0 * 0.85
            )
            if goal_heading_deg is not None:
                error = abs(
                    (bearing - goal_heading_deg + 180.0) % 360.0 - 180.0
                )
                score += max(0.0, 1.0 - error / 90.0) * 0.9
            if held_bearing_deg is not None:
                held_error = abs(
                    (bearing - held_bearing_deg + 180.0) % 360.0 - 180.0
                )
                # Hysteresis toward the opening already being followed. Two
                # similar gaps swapping rank between scans is the classic
                # Follow-The-Gap zigzag.
                score += max(0.0, 1.0 - held_error / 45.0) * 0.75
            if score > best.score or not best.found:
                best = GapChoice(
                    found=True,
                    bearing_deg=float(bearing),
                    width_deg=float(width_deg),
                    clearance_m=depth_m,
                    score=float(score),
                )
        index = end + 1
    return best


@dataclass
class LocalPlanner:
    """Obstacle memory plus arc selection, with the state the UI needs."""

    limits: PlannerLimits = field(default_factory=PlannerLimits)
    memory: ObstacleMemory = field(default_factory=ObstacleMemory)
    last_choice: ArcChoice = field(default_factory=ArcChoice)
    _bank: ArcBank | None = field(default=None, repr=False)

    def bank(
        self,
        steering_options_deg: np.ndarray,
        speed_options: tuple[tuple[int, float], ...],
    ) -> ArcBank:
        """Cached candidate arcs, rebuilt only when the options change."""
        if self._bank is None or not self._bank.matches(
            steering_options_deg, speed_options
        ):
            self._bank = ArcBank(
                self.limits, steering_options_deg, speed_options
            )
        return self._bank

    def reset(self) -> None:
        self.memory.reset()
        self.last_choice = ArcChoice()

    def commanded_speed_mps(self, left_pwm: int, right_pwm: int) -> float:
        """Best available estimate of current travel speed.

        Commanded PWM is all there is without encoders. It over-reports
        during a stall, which is the safe direction here: it makes the
        governor demand more room, and the stuck detector owns the stall.
        """
        forward = (left_pwm + right_pwm) * 0.5
        return max(0.0, forward / 255.0) * self.limits.top_speed_mps

    def track_motion(
        self,
        left_pwm: int,
        right_pwm: int,
        elapsed_s: float,
        measured_yaw_delta_deg: float | None,
    ) -> None:
        elapsed_s = min(0.25, max(0.0, elapsed_s))
        if elapsed_s <= 0.0:
            return
        forward_m = self.commanded_speed_mps(left_pwm, right_pwm) * elapsed_s
        if measured_yaw_delta_deg is not None:
            yaw_deg = measured_yaw_delta_deg
        else:
            yaw_deg = (
                (left_pwm - right_pwm) / 255.0
                * self.limits.yaw_rate_dps_at_full_steer
                * 2.1
                * elapsed_s
            )
        self.memory.integrate_motion(forward_m, yaw_deg)
