"""Lightweight frontier exploration for the Pi 3B robot runtime.

The explorer supplies a desired heading only.  The live LD19 corridor planner,
camera veto, and Uno ultrasonic stop remain the motor-safety authorities.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class ExplorationState:
    active: bool = False
    mode: str = "BUILDING_MAP"
    heading_error_deg: float = 0.0
    target_distance_m: float = 0.0
    waypoint_distance_m: float = 0.0
    frontier_count: int = 0
    coverage_ratio: float = 0.0
    target_x_m: float | None = None
    target_y_m: float | None = None
    waypoint_x_m: float | None = None
    waypoint_y_m: float | None = None
    replans: int = 0
    planning_ms: float = 0.0


class FrontierExplorer:
    """Select reachable unknown-space boundaries on a bounded occupancy map."""

    REPLAN_PERIOD_S = 0.60
    MIN_MAP_UPDATES = 8
    OCCUPIED_THRESHOLD = 28
    ROBOT_CLEARANCE_M = 0.14
    MIN_FRONTIER_CELLS = 4
    MIN_TARGET_DISTANCE_M = 0.45
    WAYPOINT_LOOKAHEAD_M = 0.85
    TARGET_REACHED_M = 0.30
    MAX_ASTAR_VISITS = 12_000
    PLAN_TIME_BUDGET_S = 0.045
    ROBOT_RECONNECT_M = 0.30
    ROUTE_CLEARANCE_WEIGHT = 0.80
    FRONTIER_CLEARANCE_WEIGHT = 0.55
    ROUTE_LENGTH_WEIGHT = 0.28
    ROUTE_DETOUR_WEIGHT = 0.45

    def __init__(self) -> None:
        self._next_replan_at = 0.0
        self._target_cell: tuple[int, int] | None = None
        self._waypoint_cell: tuple[int, int] | None = None
        self._frontier_count = 0
        self._coverage_ratio = 0.0
        self._mode = "BUILDING_MAP"
        self._replans = 0
        self._path_cells: list[tuple[int, int]] = []
        self._route_free: np.ndarray | None = None
        self._planning_ms = 0.0

    def invalidate(self) -> None:
        """Drop cached grid-index state and force an immediate replan.

        Call this whenever the caller's occupancy grid was recentred (rolled
        to keep the robot away from its edge): every cached target/waypoint
        cell and the cached route mask are indices into the grid *before* the
        shift, so reusing them would aim the robot at the wrong physical
        place until the next scheduled replan caught up.
        """
        self._target_cell = None
        self._waypoint_cell = None
        self._path_cells = []
        self._route_free = None
        self._next_replan_at = 0.0

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
        deadline: float | None = None,
        traversal_cost: np.ndarray | None = None,
    ) -> list[tuple[int, int]] | None:
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        if start == goal:
            return [start]
        if traversal_cost is None and cls._line_is_clear(free, start, goal):
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
            if visited % 64 == 0 and deadline is not None and time.perf_counter() >= deadline:
                return None
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
                next_cost = cost + step_cost * (
                    1.0
                    if traversal_cost is None
                    else 1.0 + float(traversal_cost[next_row, next_col])
                )
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
        obstacle_clearance: np.ndarray | None = None,
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
            persistence = 0.0
            if self._target_cell is not None:
                previous_distance = math.hypot(
                    self._target_cell[0] - target[0],
                    self._target_cell[1] - target[1],
                ) / scale
                persistence = max(0.0, 0.70 - previous_distance * 1.75)
            clearance_reward = (
                0.0
                if obstacle_clearance is None
                else min(float(obstacle_clearance[target]) / scale, 0.75)
                * self.FRONTIER_CLEARANCE_WEIGHT
            )
            score = (
                math.log1p(area) * 0.85
                + min(distance, 3.0) * 0.07
                + persistence
                + clearance_reward
                - visit_penalty
            )
            candidates.append((score, target))
        self._frontier_count = len(candidates)
        candidates.sort(reverse=True)
        return candidates

    @classmethod
    def _route_utility(
        cls,
        candidate_score: float,
        route_length_m: float,
        direct_distance_m: float,
    ) -> float:
        """Balance useful/open targets against unnecessary route detours."""
        detour_m = max(0.0, route_length_m - direct_distance_m)
        return (
            candidate_score
            - route_length_m * cls.ROUTE_LENGTH_WEIGHT
            - detour_m * cls.ROUTE_DETOUR_WEIGHT
        )

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

    @staticmethod
    def _nearest_free_cell(
        free: np.ndarray,
        robot: tuple[int, int],
        radius_cells: int,
    ) -> tuple[int, int] | None:
        if free[robot]:
            return robot
        row_min = max(0, robot[0] - radius_cells)
        row_max = min(free.shape[0], robot[0] + radius_cells + 1)
        col_min = max(0, robot[1] - radius_cells)
        col_max = min(free.shape[1], robot[1] + radius_cells + 1)
        rows, cols = np.nonzero(free[row_min:row_max, col_min:col_max])
        if rows.size == 0:
            return None
        rows = rows + row_min
        cols = cols + col_min
        distances = (rows - robot[0]) ** 2 + (cols - robot[1]) ** 2
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) > float(radius_cells * radius_cells):
            return None
        return int(rows[nearest]), int(cols[nearest])

    @classmethod
    def _select_waypoint(
        cls,
        path: list[tuple[int, int]],
        robot: tuple[int, int],
        scale: float,
        free: np.ndarray,
    ) -> tuple[int, int] | None:
        if not path:
            return None
        distances_sq = np.fromiter(
            (
                (cell[0] - robot[0]) ** 2 + (cell[1] - robot[1]) ** 2
                for cell in path
            ),
            dtype=np.float32,
            count=len(path),
        )
        closest = int(np.argmin(distances_sq))
        lookahead_cells = cls.WAYPOINT_LOOKAHEAD_M * scale
        waypoint = path[closest]
        for cell in path[closest + 1:]:
            distance = math.hypot(cell[0] - robot[0], cell[1] - robot[1])
            if distance > lookahead_cells:
                break
            if cls._line_is_clear(free, robot, cell):
                waypoint = cell
        if waypoint == path[closest] and closest + 1 < len(path):
            waypoint = path[closest + 1]
        return waypoint

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
                planning_ms=self._planning_ms,
            )
        target_x = (self._target_cell[1] + 0.5) / scale
        target_y = (self._target_cell[0] + 0.5) / scale
        waypoint_x = (self._waypoint_cell[1] + 0.5) / scale
        waypoint_y = (self._waypoint_cell[0] + 0.5) / scale
        waypoint_distance = math.hypot(waypoint_x - x_m, waypoint_y - y_m)
        return ExplorationState(
            active=True,
            mode=self._mode,
            heading_error_deg=self._relative_heading_deg(
                x_m, y_m, heading_deg, waypoint_x, waypoint_y
            ),
            target_distance_m=math.hypot(target_x - x_m, target_y - y_m),
            waypoint_distance_m=waypoint_distance,
            frontier_count=self._frontier_count,
            coverage_ratio=self._coverage_ratio,
            target_x_m=target_x,
            target_y_m=target_y,
            waypoint_x_m=waypoint_x,
            waypoint_y_m=waypoint_y,
            replans=self._replans,
            planning_ms=self._planning_ms,
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
        robot = (
            int(np.clip(round(y_m * scale), 0, grid.shape[0] - 1)),
            int(np.clip(round(x_m * scale), 0, grid.shape[1] - 1)),
        )
        if self._path_cells and self._route_free is not None:
            self._waypoint_cell = self._select_waypoint(
                self._path_cells, robot, scale, self._route_free
            )
        current = self._state(x_m, y_m, heading_deg, scale)
        if (
            map_updates < self.MIN_MAP_UPDATES
            or int(np.count_nonzero(observed)) < 120
        ):
            self._mode = "BUILDING_MAP"
            self._target_cell = None
            self._waypoint_cell = None
            self._path_cells = []
            self._route_free = None
            return self._state(x_m, y_m, heading_deg, scale)
        if (
            now < self._next_replan_at
            and current.active
            and current.target_distance_m > self.TARGET_REACHED_M
        ):
            return current

        self._next_replan_at = now + self.REPLAN_PERIOD_S
        self._replans += 1
        planning_started = time.perf_counter()
        deadline = planning_started + self.PLAN_TIME_BUDGET_S
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
        free_mask = free > 0
        # A binary inflated map prevents collision but gives every remaining
        # cell equal cost, so plain A* can scrape walls. Add a gentle graded
        # traversal penalty inside known free space to prefer the middle of
        # corridors while still allowing narrow passages when necessary.
        route_clearance = cv2.distanceTransform(free, cv2.DIST_L2, 3)
        comfort_cells = max(2.0, float(clearance_cells) * 2.5)
        traversal_cost = (
            np.clip((comfort_cells - route_clearance) / comfort_cells, 0.0, 1.0)
            ** 2
            * self.ROUTE_CLEARANCE_WEIGHT
        ).astype(np.float32)
        obstacle_clearance = cv2.distanceTransform(
            (~inflated).astype(np.uint8), cv2.DIST_L2, 3
        )
        planning_start = self._nearest_free_cell(
            free_mask,
            robot,
            max(1, int(math.ceil(self.ROBOT_RECONNECT_M * scale))),
        )
        if planning_start is None:
            self._planning_ms = (time.perf_counter() - planning_started) * 1000.0
            self._mode = "NO_REACHABLE_SPACE"
            self._target_cell = None
            self._waypoint_cell = None
            self._path_cells = []
            self._route_free = None
            return self._state(x_m, y_m, heading_deg, scale)
        component_count, labels = cv2.connectedComponents(
            free, connectivity=8
        )
        robot_label = int(labels[planning_start])
        if component_count <= 1 or robot_label == 0:
            self._mode = "NO_REACHABLE_SPACE"
            self._target_cell = None
            self._waypoint_cell = None
            self._path_cells = []
            self._route_free = None
            return self._state(x_m, y_m, heading_deg, scale)
        reachable = (labels == robot_label) & free_mask

        candidates = self._candidate_frontiers(
            reachable, known, visits, robot, scale, obstacle_clearance
        )
        self._mode = "FRONTIER"
        if not candidates:
            self._mode = "PATROL"
            candidates = self._patrol_candidates(
                reachable, visits, robot, scale
            )

        chosen_target: tuple[int, int] | None = None
        chosen_path: list[tuple[int, int]] | None = None
        chosen_utility = float("-inf")
        # Do not spend the bounded A* budget on a distant candidate merely
        # because it has a large frontier. Prefer high-value nearby openings,
        # then use actual route detour as a second efficiency penalty.
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                item[0]
                - math.hypot(
                    item[1][0] - planning_start[0],
                    item[1][1] - planning_start[1],
                ) / scale * self.ROUTE_LENGTH_WEIGHT
            ),
            reverse=True,
        )
        for candidate_score, target in ranked_candidates[:6]:
            path = self._astar(
                reachable,
                planning_start,
                target,
                deadline,
                traversal_cost,
            )
            if path:
                route_length_m = sum(
                    math.hypot(
                        current[0] - previous[0],
                        current[1] - previous[1],
                    )
                    for previous, current in zip(path, path[1:])
                ) / scale
                direct_distance_m = math.hypot(
                    target[0] - planning_start[0],
                    target[1] - planning_start[1],
                ) / scale
                utility = self._route_utility(
                    candidate_score,
                    route_length_m,
                    direct_distance_m,
                )
                if utility > chosen_utility:
                    chosen_utility = utility
                    chosen_target = target
                    chosen_path = path
            if time.perf_counter() >= deadline:
                break
        self._planning_ms = (time.perf_counter() - planning_started) * 1000.0
        if chosen_target is None or chosen_path is None:
            if time.perf_counter() >= deadline and current.active:
                self._next_replan_at = now + 0.15
                return current
            self._mode = "NO_REACHABLE_TARGET"
            self._target_cell = None
            self._waypoint_cell = None
            self._path_cells = []
            self._route_free = None
            return self._state(x_m, y_m, heading_deg, scale)

        self._target_cell = chosen_target
        self._path_cells = chosen_path
        self._route_free = reachable
        self._waypoint_cell = self._select_waypoint(
            chosen_path, robot, scale, reachable
        )
        return self._state(x_m, y_m, heading_deg, scale)
