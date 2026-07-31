"""Lightweight frontier exploration for the Pi 3B robot runtime.

The explorer supplies a desired heading only.  The live LD19 corridor planner,
camera veto, and Uno ultrasonic stop remain the motor-safety authorities.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ExplorationState:
    active: bool = False
    mode: str = "BUILDING_MAP"
    heading_error_deg: float = 0.0
    target_distance_m: float = 0.0
    frontier_count: int = 0
    coverage_ratio: float = 0.0
    target_x_m: float | None = None
    target_y_m: float | None = None
    waypoint_x_m: float | None = None
    waypoint_y_m: float | None = None
    replans: int = 0


class FrontierExplorer:
    """Select reachable unknown-space boundaries on a bounded occupancy map."""

    REPLAN_PERIOD_S = 0.75
    MIN_MAP_UPDATES = 8
    OCCUPIED_THRESHOLD = 28
    ROBOT_CLEARANCE_M = 0.14
    MIN_FRONTIER_CELLS = 4
    MIN_TARGET_DISTANCE_M = 0.45
    WAYPOINT_LOOKAHEAD_M = 0.55
    TARGET_REACHED_M = 0.30
    MAX_ASTAR_VISITS = 35_000

    def __init__(self) -> None:
        self._next_replan_at = 0.0
        self._target_cell: tuple[int, int] | None = None
        self._waypoint_cell: tuple[int, int] | None = None
        self._frontier_count = 0
        self._coverage_ratio = 0.0
        self._mode = "BUILDING_MAP"
        self._replans = 0

    @staticmethod
    def _relative_heading_deg(
        x_m: float,
        y_m: float,
        heading_deg: float,
        target_x_m: float,
        target_y_m: float,
    ) -> float:
        target_heading = math.degrees(
            math.atan2(target_x_m - x_m, -(target_y_m - y_m))
        )
        return (target_heading - heading_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _line_is_clear(
        free: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> bool:
        length = max(abs(goal[0] - start[0]), abs(goal[1] - start[1])) + 1
        rows = np.rint(np.linspace(start[0], goal[0], length)).astype(np.int32)
        cols = np.rint(np.linspace(start[1], goal[1], length)).astype(np.int32)
        return bool(np.all(free[rows, cols]))

    @classmethod
    def _astar(
        cls,
        free: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        if start == goal:
            return [start]
        if cls._line_is_clear(free, start, goal):
            return [start, goal]

        height, width = free.shape
        g_score = np.full((height, width), np.inf, dtype=np.float32)
        came_row = np.full((height, width), -1, dtype=np.int16)
        came_col = np.full((height, width), -1, dtype=np.int16)
        closed = np.zeros((height, width), dtype=bool)
        g_score[start] = 0.0
        queue: list[tuple[float, float, int, int]] = [
            (math.hypot(goal[0] - start[0], goal[1] - start[1]), 0.0, *start)
        ]
        neighbours = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        visited = 0
        found = False
        while queue and visited < cls.MAX_ASTAR_VISITS:
            _estimate, cost, row, col = heapq.heappop(queue)
            if closed[row, col]:
                continue
            closed[row, col] = True
            visited += 1
            if (row, col) == goal:
                found = True
                break
            for delta_row, delta_col, step_cost in neighbours:
                next_row = row + delta_row
                next_col = col + delta_col
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if not free[next_row, next_col] or closed[next_row, next_col]:
                    continue
                if delta_row != 0 and delta_col != 0:
                    if not free[row + delta_row, col] or not free[row, col + delta_col]:
                        continue
                next_cost = cost + step_cost
                if next_cost >= float(g_score[next_row, next_col]):
                    continue
                g_score[next_row, next_col] = next_cost
                came_row[next_row, next_col] = row
                came_col[next_row, next_col] = col
                heuristic = math.hypot(goal[0] - next_row, goal[1] - next_col)
                heapq.heappush(
                    queue,
                    (next_cost + heuristic, next_cost, next_row, next_col),
                )
        if not found:
            return None

        path = [goal]
        current = goal
        while current != start:
            row = int(came_row[current])
            col = int(came_col[current])
            if row < 0 or col < 0:
                return None
            current = (row, col)
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _known_coverage(known: np.ndarray) -> float:
        rows, cols = np.nonzero(known)
        if rows.size == 0:
            return 0.0
        area = (int(rows.max()) - int(rows.min()) + 1) * (
            int(cols.max()) - int(cols.min()) + 1
        )
        return float(rows.size / max(1, area))

    def _candidate_frontiers(
        self,
        reachable: np.ndarray,
        known: np.ndarray,
        visits: np.ndarray,
        robot: tuple[int, int],
        scale: float,
    ) -> list[tuple[float, tuple[int, int]]]:
        unknown_nearby = cv2.dilate(
            (~known).astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
        ) > 0
        frontier = reachable & unknown_nearby
        rows, cols = np.indices(frontier.shape)
        distance_m = np.hypot(rows - robot[0], cols - robot[1]) / scale
        frontier &= distance_m >= self.MIN_TARGET_DISTANCE_M
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier.astype(np.uint8), connectivity=8
        )
        candidates: list[tuple[float, tuple[int, int]]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.MIN_FRONTIER_CELLS:
                continue
            component_rows, component_cols = np.nonzero(labels == label)
            centre_col, centre_row = centroids[label]
            nearest = int(np.argmin(
                (component_rows - centre_row) ** 2
                + (component_cols - centre_col) ** 2
            ))
            target = (
                int(component_rows[nearest]),
                int(component_cols[nearest]),
            )
            distance = math.hypot(
                target[0] - robot[0], target[1] - robot[1]
            ) / scale
            visit_penalty = float(visits[target]) * 0.025
            persistence = 0.55 if self._target_cell == target else 0.0
            score = (
                math.log1p(area) * 0.85
                + min(distance, 3.0) * 0.18
                + persistence
                - visit_penalty
            )
            candidates.append((score, target))
        self._frontier_count = len(candidates)
        candidates.sort(reverse=True)
        return candidates

    @staticmethod
    def _patrol_candidates(
        reachable: np.ndarray,
        visits: np.ndarray,
        robot: tuple[int, int],
        scale: float,
    ) -> list[tuple[float, tuple[int, int]]]:
        rows, cols = np.nonzero(reachable)
        if rows.size == 0:
            return []
        sampled = np.arange(0, rows.size, 8, dtype=np.int32)
        candidates: list[tuple[float, tuple[int, int]]] = []
        for index in sampled:
            target = (int(rows[index]), int(cols[index]))
            distance = math.hypot(
                target[0] - robot[0], target[1] - robot[1]
            ) / scale
            if distance < 0.80:
                continue
            score = min(distance, 2.5) - float(visits[target]) * 0.08
            candidates.append((score, target))
        candidates.sort(reverse=True)
        return candidates[:12]

    def _state(
        self,
        x_m: float,
        y_m: float,
        heading_deg: float,
        scale: float,
    ) -> ExplorationState:
        if self._target_cell is None or self._waypoint_cell is None:
            return ExplorationState(
                mode=self._mode,
                frontier_count=self._frontier_count,
                coverage_ratio=self._coverage_ratio,
                replans=self._replans,
            )
        target_x = (self._target_cell[1] + 0.5) / scale
        target_y = (self._target_cell[0] + 0.5) / scale
        waypoint_x = (self._waypoint_cell[1] + 0.5) / scale
        waypoint_y = (self._waypoint_cell[0] + 0.5) / scale
        return ExplorationState(
            active=True,
            mode=self._mode,
            heading_error_deg=self._relative_heading_deg(
                x_m, y_m, heading_deg, waypoint_x, waypoint_y
            ),
            target_distance_m=math.hypot(target_x - x_m, target_y - y_m),
            frontier_count=self._frontier_count,
            coverage_ratio=self._coverage_ratio,
            target_x_m=target_x,
            target_y_m=target_y,
            waypoint_x_m=waypoint_x,
            waypoint_y_m=waypoint_y,
            replans=self._replans,
        )

    def update(
        self,
        grid: np.ndarray,
        observed: np.ndarray,
        visits: np.ndarray,
        x_m: float,
        y_m: float,
        heading_deg: float,
        metres: float,
        map_updates: int,
        now: float,
    ) -> ExplorationState:
        scale = grid.shape[0] / metres
        current = self._state(x_m, y_m, heading_deg, scale)
        if (
            map_updates < self.MIN_MAP_UPDATES
            or int(np.count_nonzero(observed)) < 120
        ):
            self._mode = "BUILDING_MAP"
            self._target_cell = None
            self._waypoint_cell = None
            return self._state(x_m, y_m, heading_deg, scale)
        if (
            now < self._next_replan_at
            and current.active
            and current.target_distance_m > self.TARGET_REACHED_M
        ):
            return current

        self._next_replan_at = now + self.REPLAN_PERIOD_S
        self._replans += 1
        known = observed > 0
        self._coverage_ratio = self._known_coverage(known)
        occupied = grid >= self.OCCUPIED_THRESHOLD
        clearance_cells = max(1, int(math.ceil(self.ROBOT_CLEARANCE_M * scale)))
        kernel_size = clearance_cells * 2 + 1
        inflated = cv2.dilate(
            occupied.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
        ) > 0
        free = (known & ~inflated).astype(np.uint8)
        robot = (
            int(np.clip(round(y_m * scale), 0, grid.shape[0] - 1)),
            int(np.clip(round(x_m * scale), 0, grid.shape[1] - 1)),
        )
        cv2.circle(free, (robot[1], robot[0]), 2, 1, -1)
        free_mask = free > 0
        component_count, labels = cv2.connectedComponents(
            free, connectivity=8
        )
        robot_label = int(labels[robot])
        if component_count <= 1 or robot_label == 0:
            self._mode = "NO_REACHABLE_SPACE"
            self._target_cell = None
            self._waypoint_cell = None
            return self._state(x_m, y_m, heading_deg, scale)
        reachable = (labels == robot_label) & free_mask

        candidates = self._candidate_frontiers(
            reachable, known, visits, robot, scale
        )
        self._mode = "FRONTIER"
        if not candidates:
            self._mode = "PATROL"
            candidates = self._patrol_candidates(
                reachable, visits, robot, scale
            )

        chosen_target: tuple[int, int] | None = None
        chosen_path: list[tuple[int, int]] | None = None
        for _score, target in candidates[:8]:
            path = self._astar(reachable, robot, target)
            if path:
                chosen_target = target
                chosen_path = path
                break
        if chosen_target is None or chosen_path is None:
            self._mode = "NO_REACHABLE_TARGET"
            self._target_cell = None
            self._waypoint_cell = None
            return self._state(x_m, y_m, heading_deg, scale)

        lookahead_cells = self.WAYPOINT_LOOKAHEAD_M * scale
        waypoint = chosen_path[-1]
        for cell in chosen_path[1:]:
            if math.hypot(cell[0] - robot[0], cell[1] - robot[1]) >= lookahead_cells:
                waypoint = cell
                break
        self._target_cell = chosen_target
        self._waypoint_cell = waypoint
        return self._state(x_m, y_m, heading_deg, scale)
