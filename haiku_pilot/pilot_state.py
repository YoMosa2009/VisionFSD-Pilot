"""What the pilot remembers between commands: pivot timing and position.

Each pilot command is a separate run of robot.py, so anything learned or
tracked is kept in small JSON files next to the saved images.

* ``PivotModel`` learns how many degrees a timed pivot turns at each power,
  from LiDAR-measured pivots, so a turn can be cut into pivots that land
  close to the angle asked.
* ``Track`` adds up the measured moves since ``manual on`` into a position
  relative to where Manual Control started, and remembers marked targets
  (the capsule, the bucket) in the same frame, so they can be found again
  after they leave the camera's view. It drifts a little with every move;
  re-mark a target whenever it is seen.

Frame: x to the right of the starting pose, y straight ahead of it, metres;
heading in degrees, positive to the LEFT of the starting direction.
"""

from __future__ import annotations

import json
import math
import os

# Prior for a pivot at manual pivot power before anything is measured:
# degrees = rate * seconds + offset. Conservative (a fast rate means a short
# first pivot), then replaced by measurements.
PRIOR_RATE_DPS = 160.0
PRIOR_OFFSET_DEG = 2.0
RATE_LIMITS = (30.0, 400.0)
OFFSET_LIMITS = (-10.0, 30.0)
MIN_PIVOT_S = 0.03
MAX_PIVOT_S = 1.5
KEEP_OBSERVATIONS = 16


def _load(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(temporary, path)


class PivotModel:
    """degrees turned = rate x seconds + offset, fitted per power level."""

    def __init__(self, path: str, power: float) -> None:
        self.path = path
        self.key = f"{round(power, 1):.1f}"
        self._all = _load(path)
        raw = self._all.get(self.key) or []
        self.observations = [
            (float(s), float(d)) for s, d in raw
            if isinstance(s, (int, float)) and isinstance(d, (int, float))
        ][-KEEP_OBSERVATIONS:]

    def fit(self) -> tuple[float, float]:
        points = [(s, d) for s, d in self.observations if s > 0]
        if not points:
            return PRIOR_RATE_DPS, PRIOR_OFFSET_DEG
        seconds = [s for s, _d in points]
        if len(points) >= 3 and max(seconds) - min(seconds) >= 0.05:
            mean_s = sum(seconds) / len(points)
            mean_d = sum(d for _s, d in points) / len(points)
            spread = sum((s - mean_s) ** 2 for s in seconds)
            rate = sum((s - mean_s) * (d - mean_d) for s, d in points) / spread
            rate = min(RATE_LIMITS[1], max(RATE_LIMITS[0], rate))
            offset = mean_d - rate * mean_s
        else:
            offset = PRIOR_OFFSET_DEG
            rate = sum((d - offset) / s for s, d in points) / len(points)
            rate = min(RATE_LIMITS[1], max(RATE_LIMITS[0], rate))
        offset = min(OFFSET_LIMITS[1], max(OFFSET_LIMITS[0], offset))
        return rate, offset

    def predict(self, seconds: float) -> float:
        rate, offset = self.fit()
        return max(0.0, rate * seconds + offset)

    def seconds_for(self, degrees: float) -> float:
        rate, offset = self.fit()
        return min(MAX_PIVOT_S, max(MIN_PIVOT_S, (degrees - offset) / rate))

    def add(self, seconds: float, degrees: float) -> None:
        self.observations = (self.observations + [(round(seconds, 3), round(degrees, 1))])[
            -KEEP_OBSERVATIONS:]

    def save(self) -> None:
        self._all = _load(self.path)
        self._all[self.key] = self.observations
        _save(self.path, self._all)


class Track:
    """Position since `manual on`, and marked targets, from measured moves."""

    def __init__(self, path: str) -> None:
        self.path = path
        data = _load(path)
        self.x = float(data.get("x", 0.0))
        self.y = float(data.get("y", 0.0))
        self.heading = float(data.get("heading", 0.0))
        self.moves = int(data.get("moves", 0))
        self.unmeasured = int(data.get("unmeasured", 0))
        marks = data.get("marks") or {}
        self.marks = {
            str(name): (float(value[0]), float(value[1]), int(value[2]))
            for name, value in marks.items()
            if isinstance(value, list) and len(value) == 3
        }

    def reset(self) -> None:
        self.x = self.y = self.heading = 0.0
        self.moves = self.unmeasured = 0
        self.marks = {}

    def save(self) -> None:
        _save(self.path, {
            "x": round(self.x, 3), "y": round(self.y, 3), "heading": round(self.heading, 1),
            "moves": self.moves, "unmeasured": self.unmeasured,
            "marks": {name: [round(x, 3), round(y, 3), at] for name, (x, y, at) in self.marks.items()},
        })

    def _axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        radians = math.radians(self.heading)
        forward = (-math.sin(radians), math.cos(radians))
        right = (math.cos(radians), math.sin(radians))
        return forward, right

    def apply(self, turn_deg: float, forward_m: float, right_m: float, measured: bool) -> None:
        """Add one move, given in the robot's frame before the move."""
        forward, right = self._axes()
        self.x += forward_m * forward[0] + right_m * right[0]
        self.y += forward_m * forward[1] + right_m * right[1]
        self.heading = (self.heading + turn_deg + 180.0) % 360.0 - 180.0
        self.moves += 1
        if not measured:
            self.unmeasured += 1

    def mark(self, name: str, bearing_deg: float, distance_m: float) -> None:
        """Remember a target seen ``distance_m`` away at ``bearing_deg``
        (positive = to the right) from the robot's current pose."""
        forward, right = self._axes()
        radians = math.radians(bearing_deg)
        along, across = distance_m * math.cos(radians), distance_m * math.sin(radians)
        self.marks[name] = (
            self.x + along * forward[0] + across * right[0],
            self.y + along * forward[1] + across * right[1],
            self.moves,
        )

    def relative(self, name: str) -> tuple[float, float, int] | None:
        """(distance m, bearing deg - positive right, moves since marked)."""
        if name not in self.marks:
            return None
        x, y, at = self.marks[name]
        forward, right = self._axes()
        dx, dy = x - self.x, y - self.y
        along = dx * forward[0] + dy * forward[1]
        across = dx * right[0] + dy * right[1]
        return math.hypot(along, across), math.degrees(math.atan2(across, along)), self.moves - at

    def in_robot_frame(self) -> list[tuple[str, float, float]]:
        """Marks as (name, x right, y ahead) for drawing on the LiDAR map."""
        out = []
        for name in self.marks:
            distance, bearing, _since = self.relative(name)
            radians = math.radians(bearing)
            out.append((name, distance * math.sin(radians), distance * math.cos(radians)))
        return out
