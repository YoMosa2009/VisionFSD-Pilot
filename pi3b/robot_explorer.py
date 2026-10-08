"""Lightweight frontier exploration for the Pi 3B robot runtime.

The explorer supplies a desired heading only.  The live LD19 corridor planner,
camera veto, and Uno ultrasonic stop remain the motor-safety authorities.
"""

from __future__ import annotations

from array import array
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
    # The planned route in map metres, thinned for drawing. This is what the
    # robot currently intends to do, so the dashboard can show the plan rather
    # than only the single next waypoint.
    path_xy_m: tuple[tuple[float, float], ...] = ()
    # How long the current goal has been held, and how far the route to it has
    # shrunk since it was chosen. A goal that stops shrinking is abandoned.
    goal_age_s: float = 0.0
    goal_progress_m: float = 0.0
    blacklisted: int = 0
    route_length_m: float = 0.0


class FrontierExplorer:
    """Select reachable unknown-space boundaries on a bounded occupancy map."""

    REPLAN_PERIOD_S = 0.60
    MIN_MAP_UPDATES = 8
    OCCUPIED_THRESHOLD = 28
    # Obstacle inflation for the global route, matched to what the local arc
    # planner will actually accept: the chassis footprint radius (13.2 cm for
    # the measured 9 x 10.5 inch body) plus its 10 cm safety margin.
    #
    # This was 0.14 m, inherited from a chassis model 60% too narrow. The
    # global planner therefore routed through gaps about 28 cm wide that the
    # local planner, correctly, refused to enter - so the robot committed to a
    # plan through a space it would never drive, reached it, and had to
    # improvise. Two planners disagreeing about what fits is the same failure
    # that caused the spinning fixed in v1.9.21, one level up.
    ROBOT_CLEARANCE_M = 0.23
    MIN_FRONTIER_CELLS = 4
    # A frontier this close is not worth a goal: reaching it reveals almost
    # nothing and immediately needs another, which from outside reads as the
    # robot shuffling around one spot.
    MIN_TARGET_DISTANCE_M = 0.80
    # Pure-pursuit lookahead. Raised from 0.85 m after testing showed the
    # robot turning continuously all the way along a curved route.
    #
    # The waypoint is chased, so the lookahead sets how hard the chassis is
    # asked to turn: a short one keeps the target close and off to the side,
    # which holds a large heading error the whole way round a bend and reads
    # as one endless sharp turn. It also has to be large relative to the turn
    # radius - about 1.0 m here - or the robot is being asked to cut inside a
    # circle it cannot physically follow, and simply saturates.
    WAYPOINT_LOOKAHEAD_M = 1.30
    TARGET_REACHED_M = 0.35
    MAX_ASTAR_VISITS = 12_000
    # Planning runs on its own thread (AsyncExplorer), so this bounds how
    # long one replan may hold the interpreter lock rather than how long the
    # control loop waits. It was 45 ms when planning ran inline; that was too
    # short for a 12 m map on a Pi 3B, and it failed to route in open rooms.
    # The planner runs on its own thread, so its budget costs plan freshness,
    # not control latency. At 0.12 s it timed out in ~85% of plans on the
    # 2026-09-23 run: a Pi 3B needs seconds for what a desktop does in 0.2 s,
    # and without a plan the robot has nowhere to go and turns in place.
    PLAN_TIME_BUDGET_S = 0.40
    # Candidates searched with A* per plan, after ranking by straight-line
    # utility. Every candidate is already known to be reachable; the search is
    # for the route's length and shape, and six searches per plan was most of
    # the cost.
    ROUTE_SHORTLIST = 3
    # Weighted A*: routes at most this factor longer than optimal (in practice
    # a few percent), for far fewer expanded cells.
    ASTAR_HEURISTIC_WEIGHT = 1.6
    # Plan on a coarser grid than the map is stored at. The 12 m map has
    # 2 cm cells for mapping accuracy; routing at that resolution spent most of
    # the planning budget on cells finer than the chassis can steer between.
    # Inflation, frontiers and A* all run on cells about 6 cm across.
    PLAN_CELL_M = 0.0625
    # Goal commitment, following explore_lite: keep a goal until it is reached
    # or stops getting closer, rather than re-choosing every replan.
    GOAL_SWITCH_MARGIN = 0.45
    PROGRESS_TIMEOUT_S = 10.0
    PROGRESS_MIN_M = 0.20
    BLACKLIST_S = 45.0
    REACHED_MEMORY_S = 30.0
    BLACKLIST_RADIUS_M = 0.55
    ROAM_MIN_DISTANCE_M = 1.30
    ROAM_MAX_DISTANCE_M = 5.00
    ROBOT_RECONNECT_M = 0.40
    ROUTE_CLEARANCE_WEIGHT = 0.80
    FRONTIER_CLEARANCE_WEIGHT = 0.55
    # "Been there" memory. A 360-degree LiDAR sees a whole room within seconds,
    # so what the robot has *seen* says little about where it has *been*.
    # Novelty is the distance from a cell to the nearest place the chassis has
    # physically driven; goals prefer it, capped so that very far does not
    # outweigh reachable and open.
    NOVELTY_CAP_M = 2.5
    FRONTIER_NOVELTY_WEIGHT = 0.45
    PATROL_NOVELTY_WEIGHT = 1.00
    PATROL_DISTANCE_WEIGHT = 0.25
    # Where the robot got stuck, and the goal it was chasing then, are avoided
    # for this long: retrying the same thing the same way is what it looked
    # like it had forgotten.
    TROUBLE_MEMORY_S = 90.0
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
        self._goal_started_at = 0.0
        self._goal_best_route_m = math.inf
        self._goal_initial_route_m = 0.0
        self._goal_progress_at = 0.0
        self._route_length_m = 0.0
        # (map x metres, map y metres, expires_at)
        self._blacklist: list[tuple[float, float, float]] = []
        self._route_searches = 0
        self._scale = 1.0
        self._now = 0.0

    def shift(self, shift_rows: int, shift_cols: int, fine_scale: float) -> None:
        """Follow a map recentre instead of discarding the plan.

        The mapper scrolls its grid to keep the robot away from the edge. In a
        large room that happens every metre or two, and the old response was
        to throw away the goal and route entirely - so a robot crossing open
        floor kept dropping its destination and choosing a new one, which is
        what driving around the same place looked like. The goal is a physical
        place; moving the stored cells by the same shift keeps it one.
        """
        if fine_scale <= 0.0 or self._scale <= 0.0:
            self.invalidate()
            return
        dy = shift_rows / fine_scale
        dx = shift_cols / fine_scale
        rows = int(round(dy * self._scale))
        cols = int(round(dx * self._scale))

        def moved(cell):
            return None if cell is None else (cell[0] + rows, cell[1] + cols)

        self._target_cell = moved(self._target_cell)
        self._waypoint_cell = moved(self._waypoint_cell)
        self._path_cells = [(row + rows, col + cols) for row, col in self._path_cells]
        self._route_free = None
        self._blacklist = [(x + dx, y + dy, until) for x, y, until in self._blacklist]
        self._next_replan_at = 0.0

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
        self._goal_best_route_m = math.inf

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
        if not np.all(free[rows, cols]):
            return False
        # Match A*'s no-corner-cut rule when smoothing the route. A diagonal
        # shortcut touches both neighbouring cells, not just its centre line.
        diagonal = (rows[1:] != rows[:-1]) & (cols[1:] != cols[:-1])
        return bool(
            np.all(free[rows[:-1][diagonal], cols[1:][diagonal]])
            and np.all(free[rows[1:][diagonal], cols[:-1][diagonal]])
        )

    #: Straight segments are accepted without search when no cell on them has
    #: more than this traversal penalty, i.e. they do not skim an obstacle.
    LINE_COST_SHORTCUT_MAX = 0.20

    @classmethod
    def _line_cost_is_low(
        cls,
        traversal_cost: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> bool:
        length = max(abs(goal[0] - start[0]), abs(goal[1] - start[1])) + 1
        rows = np.rint(np.linspace(start[0], goal[0], length)).astype(np.int32)
        cols = np.rint(np.linspace(start[1], goal[1], length)).astype(np.int32)
        return float(traversal_cost[rows, cols].max()) <= cls.LINE_COST_SHORTCUT_MAX

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
        if cls._line_is_clear(free, start, goal) and (
            traversal_cost is None
            or cls._line_cost_is_low(traversal_cost, start, goal)
        ):
            # Line of sight through low-cost space is already the best route.
            # This was only taken when no traversal cost was supplied - which
            # is never, in the runtime - so every goal in an open room paid
            # for a full search and three fresh grid-sized arrays.
            return [start, goal]

        # The search runs over flat Python arrays rather than indexing numpy
        # arrays one element at a time: per expanded cell that was eight
        # numpy scalar reads, and it made each search ~20 ms on a desktop and
        # seconds on the Pi. g-scores are stored as float32, exactly as before,
        # so every comparison - and therefore every route - is unchanged.
        height, width = free.shape
        size = height * width
        passable = free.astype(np.uint8).tobytes()
        extra = (
            None
            if traversal_cost is None
            else traversal_cost.astype(np.float32).ravel().tolist()
        )
        g_score = array("f", [math.inf]) * size
        came = array("i", [-1]) * size
        closed = bytearray(size)
        start_index = start[0] * width + start[1]
        goal_index = goal[0] * width + goal[1]
        goal_row, goal_col = goal
        weight = cls.ASTAR_HEURISTIC_WEIGHT
        g_score[start_index] = 0.0
        queue: list[tuple[float, float, int, int]] = [
            (
                math.hypot(goal_row - start[0], goal_col - start[1]) * weight,
                0.0,
                *start,
            )
        ]
        diagonal = math.sqrt(2.0)
        neighbours = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, diagonal),
            (-1, 1, diagonal),
            (1, -1, diagonal),
            (1, 1, diagonal),
        )
        hypot = math.hypot
        heappush = heapq.heappush
        heappop = heapq.heappop
        visited = 0
        found = False
        while queue and visited < cls.MAX_ASTAR_VISITS:
            if visited % 64 == 0 and deadline is not None and time.perf_counter() >= deadline:
                return None
            _estimate, cost, row, col = heappop(queue)
            index = row * width + col
            if closed[index]:
                continue
            closed[index] = 1
            visited += 1
            if index == goal_index:
                found = True
                break
            for delta_row, delta_col, step_cost in neighbours:
                next_row = row + delta_row
                next_col = col + delta_col
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                next_index = next_row * width + next_col
                if not passable[next_index] or closed[next_index]:
                    continue
                if delta_row != 0 and delta_col != 0:
                    if (
                        not passable[(row + delta_row) * width + col]
                        or not passable[row * width + col + delta_col]
                    ):
                        continue
                next_cost = cost + step_cost * (
                    1.0 if extra is None else 1.0 + extra[next_index]
                )
                if next_cost >= g_score[next_index]:
                    continue
                g_score[next_index] = next_cost
                came[next_index] = index
                heappush(
                    queue,
                    (
                        next_cost + hypot(goal_row - next_row, goal_col - next_col) * weight,
                        next_cost,
                        next_row,
                        next_col,
                    ),
                )
        if not found:
            return None

        path = [goal]
        current = goal_index
        while current != start_index:
            current = came[current]
            if current < 0:
                return None
            path.append((current // width, current % width))
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
        novelty_m: np.ndarray | None = None,
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
        if novelty_m is None:
            novelty_m = self._novelty_m(visits, scale)
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
            novelty = float(novelty_m[target])
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
                + min(distance, 3.0) * 0.22
                + persistence
                + clearance_reward
                + novelty * self.FRONTIER_NOVELTY_WEIGHT
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

    @classmethod
    def _patrol_candidates(
        cls,
        reachable: np.ndarray,
        visits: np.ndarray,
        robot: tuple[int, int],
        scale: float,
        obstacle_clearance: np.ndarray | None = None,
        novelty_m: np.ndarray | None = None,
    ) -> list[tuple[float, tuple[int, int]]]:
        """Roaming goals for when there is nothing left to discover.

        With no frontiers the old patrol picked nearby cells and re-picked
        them every replan, so in a large open room the robot drove small loops
        in one area. Goals here are deliberately far, deliberately where the
        robot has spent little time, and preferably in open space - so the
        robot crosses the room with purpose, and the goal commitment in
        update() keeps it going there until it arrives.
        """
        rows, cols = np.nonzero(reachable)
        if rows.size == 0:
            return []
        distance = np.hypot(rows - robot[0], cols - robot[1]) / scale
        wanted = distance >= cls.ROAM_MIN_DISTANCE_M
        if not np.any(wanted):
            wanted = distance >= min(0.8, float(distance.max()))
        rows = rows[wanted]
        cols = cols[wanted]
        distance = distance[wanted]
        if rows.size == 0:
            return []
        stride = max(1, rows.size // 400)
        rows = rows[::stride]
        cols = cols[::stride]
        distance = distance[::stride]
        openness = (
            np.zeros(rows.size, dtype=np.float32)
            if obstacle_clearance is None
            else np.minimum(obstacle_clearance[rows, cols] / scale, 0.8)
        )
        if novelty_m is None:
            novelty_m = cls._novelty_m(visits, scale)
        novelty = novelty_m[rows, cols]
        # Where the robot has not been leads; open enough to fit, and far
        # enough to be worth the trip, follow. Candidates are already
        # reachable with the chassis inflation, so every one of them fits.
        score = (
            novelty * cls.PATROL_NOVELTY_WEIGHT
            + np.minimum(distance, cls.ROAM_MAX_DISTANCE_M) * cls.PATROL_DISTANCE_WEIGHT
            + openness * 0.6
        )
        order = np.argsort(score)[::-1][:12]
        return [
            (float(score[index]), (int(rows[index]), int(cols[index])))
            for index in order
        ]

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
        if not free[robot]:
            # Preserve bounded reconnection for an approximate pose inside
            # map inflation. Live LD19 remains the physical motion authority.
            anchor = cls._nearest_free_cell(
                free, robot, max(1, int(math.ceil(cls.ROBOT_RECONNECT_M * scale)))
            )
            if anchor is None:
                return None
            robot = anchor
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
            candidate = path[closest + 1]
            if cls._line_is_clear(free, robot, candidate):
                waypoint = candidate
        return waypoint if cls._line_is_clear(free, robot, waypoint) else None

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
            path_xy_m=self._path_world(scale),
            goal_age_s=max(0.0, self._now - self._goal_started_at),
            goal_progress_m=max(
                0.0, self._goal_initial_route_m - self._route_length_m
            ),
            blacklisted=len(self._blacklist),
            route_length_m=self._route_length_m,
        )

    def _path_world(self, scale: float) -> tuple[tuple[float, float], ...]:
        """Route cells as map metres, thinned to a drawable polyline.

        A* returns one cell per step, which at roughly 2 cm per cell is far
        more vertices than a dashboard needs. Keeping every eighth cell plus
        the endpoint preserves the shape of the route at a fraction of the
        drawing cost.
        """
        if not self._path_cells:
            return ()
        stride = 8
        cells = self._path_cells[::stride]
        if cells[-1] != self._path_cells[-1]:
            cells.append(self._path_cells[-1])
        return tuple(
            ((col + 0.5) / scale, (row + 0.5) / scale) for row, col in cells
        )

    def _planning_grids(
        self,
        grid: np.ndarray,
        observed: np.ndarray,
        visits: np.ndarray,
        metres: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Occupancy, knowledge and visits on the planning grid, and its scale.

        Obstacles are max-pooled, so a coarse cell is occupied if any map cell
        inside it is: coarsening can only make a route more conservative, never
        let one pass through a wall. A coarse cell counts as known when most of
        it has been observed.
        """
        fine_scale = grid.shape[0] / metres
        factor = max(1, int(round(self.PLAN_CELL_M * fine_scale)))
        occupied = grid >= self.OCCUPIED_THRESHOLD
        known = observed > 0
        if factor == 1:
            return occupied, known, visits, fine_scale
        size = (grid.shape[0] // factor) * factor
        coarse = size // factor
        # Area-averaging in OpenCV instead of numpy reshape reductions: the
        # latter spent several milliseconds per plan on a 576-cell map.
        area = cv2.resize(
            np.ascontiguousarray(
                np.stack(
                    (occupied[:size, :size], known[:size, :size]), axis=-1
                ).astype(np.uint8)
                * 255
            ),
            (coarse, coarse),
            interpolation=cv2.INTER_AREA,
        )
        return (
            area[:, :, 0] > 0,
            area[:, :, 1] >= 128,
            # Visit counts are smooth blobs a chassis wide, so sampling
            # rather than pooling loses nothing the roaming score can see.
            visits[:size:factor, :size:factor],
            fine_scale / factor,
        )

    @classmethod
    def _novelty_m(cls, visits: np.ndarray, scale: float) -> np.ndarray:
        """Metres from each planning cell to the nearest cell driven through.

        A single-cell visit lookup, as before, let a goal 20 cm beside a path
        the robot had driven ten times score as never visited.
        """
        unvisited = (visits == 0).astype(np.uint8)
        if not np.any(unvisited == 0):
            # Nowhere driven yet: everything is equally new.
            return np.full(visits.shape, cls.NOVELTY_CAP_M, dtype=np.float32)
        distance = cv2.distanceTransform(unvisited, cv2.DIST_L2, 3) / scale
        return np.minimum(distance, cls.NOVELTY_CAP_M).astype(np.float32)

    def note_trouble(self, x_m: float, y_m: float, now: float) -> None:
        """The robot got stuck here: avoid this spot, and drop the goal.

        explore_lite's rule for a failed goal, applied to getting stuck: the
        goal being chased and the place it went wrong are both kept out of
        consideration for a while, and a new goal is chosen now.
        """
        until = now + self.TROUBLE_MEMORY_S
        self._blacklist.append((x_m, y_m, until))
        if self._target_cell is not None and self._scale > 0.0:
            self._remember(self._target_cell, self._scale, until)
        self._drop_goal()
        self._next_replan_at = 0.0

    def _prune_blacklist(self, now: float) -> None:
        self._blacklist = [entry for entry in self._blacklist if entry[2] > now]

    def _blacklisted(self, cell: tuple[int, int], scale: float) -> bool:
        x_m = (cell[1] + 0.5) / scale
        y_m = (cell[0] + 0.5) / scale
        return any(
            math.hypot(x_m - bx, y_m - by) <= self.BLACKLIST_RADIUS_M
            for bx, by, _until in self._blacklist
        )

    def _remember(self, cell: tuple[int, int], scale: float, until: float) -> None:
        self._blacklist.append(
            ((cell[1] + 0.5) / scale, (cell[0] + 0.5) / scale, until)
        )

    @staticmethod
    def _route_length(path: list[tuple[int, int]], scale: float) -> float:
        return sum(
            math.hypot(current[0] - previous[0], current[1] - previous[1])
            for previous, current in zip(path, path[1:])
        ) / scale

    def _drop_goal(self) -> None:
        self._target_cell = None
        self._waypoint_cell = None
        self._path_cells = []
        self._route_free = None
        self._goal_best_route_m = math.inf

    def _best_route(
        self,
        ranked_candidates,
        reachable,
        planning_start,
        traversal_cost,
        scale,
        deadline,
        skip=None,
        must_beat: float = float("-inf"),
    ):
        """Best candidate by route utility, searching as few routes as possible.

        Candidates arrive sorted by straight-line utility, an upper bound on
        their true utility. Once the next bound cannot beat the best route
        found - or ``must_beat``, the bar an alternative has to clear to
        replace the goal already being driven toward - no remaining candidate
        can win and the search stops. In steady driving that is usually before
        the first A*. ``ROUTE_SHORTLIST`` caps the searches per plan.
        """
        chosen_target = None
        chosen_path = None
        chosen_utility = float("-inf")
        searches = 0
        for candidate_score, target in ranked_candidates:
            if skip is not None and target == skip:
                continue
            direct_m = math.hypot(
                target[0] - planning_start[0],
                target[1] - planning_start[1],
            ) / scale
            bound = self._route_utility(candidate_score, direct_m, direct_m)
            if bound <= chosen_utility or bound < must_beat:
                break
            if searches >= self.ROUTE_SHORTLIST:
                break
            path = self._astar(
                reachable, planning_start, target, deadline, traversal_cost
            )
            searches += 1
            if path:
                utility = self._route_utility(
                    candidate_score, self._route_length(path, scale), direct_m
                )
                if utility > chosen_utility:
                    chosen_utility = utility
                    chosen_target = target
                    chosen_path = path
            if deadline is not None and time.perf_counter() >= deadline:
                break
        self._route_searches = searches
        return chosen_target, chosen_path, chosen_utility

    def _reuse_route(
        self,
        reachable: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        """The current route from where the robot is now, if still clear."""
        path = self._path_cells
        if not path or path[-1] != goal:
            return None
        height, width = reachable.shape
        for row, col in path:
            if not (0 <= row < height and 0 <= col < width) or not reachable[row, col]:
                return None
        nearest = min(
            range(len(path)),
            key=lambda index: (path[index][0] - start[0]) ** 2
            + (path[index][1] - start[1]) ** 2,
        )
        remaining = path[nearest:]
        if remaining[0] == start:
            return remaining
        if not self._line_is_clear(reachable, start, remaining[0]):
            return None
        return [start] + remaining

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
        self._now = now
        fine_scale = grid.shape[0] / metres
        factor = max(1, int(round(self.PLAN_CELL_M * fine_scale)))
        scale = fine_scale / factor
        self._scale = scale
        coarse_size = grid.shape[0] // factor
        robot = (
            int(np.clip(round(y_m * scale), 0, coarse_size - 1)),
            int(np.clip(round(x_m * scale), 0, coarse_size - 1)),
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
            self._drop_goal()
            return self._state(x_m, y_m, heading_deg, scale)
        reached = (
            current.active and current.target_distance_m <= self.TARGET_REACHED_M
        )
        if reached and self._target_cell is not None:
            # Remember where it just arrived so the very next choice is not
            # the same place again.
            self._remember(self._target_cell, scale, now + self.REACHED_MEMORY_S)
            self._drop_goal()
        if now < self._next_replan_at and current.active and not reached:
            return current

        self._next_replan_at = now + self.REPLAN_PERIOD_S
        self._replans += 1
        self._prune_blacklist(now)
        planning_started = time.perf_counter()
        deadline = planning_started + self.PLAN_TIME_BUDGET_S
        occupied, known, plan_visits, scale = self._planning_grids(
            grid, observed, visits, metres
        )
        self._coverage_ratio = self._known_coverage(known)
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
            self._drop_goal()
            return self._state(x_m, y_m, heading_deg, scale)
        component_count, labels = cv2.connectedComponents(free, connectivity=8)
        robot_label = int(labels[planning_start])
        if component_count <= 1 or robot_label == 0:
            self._mode = "NO_REACHABLE_SPACE"
            self._drop_goal()
            return self._state(x_m, y_m, heading_deg, scale)
        reachable = (labels == robot_label) & free_mask

        novelty_m = self._novelty_m(plan_visits, scale)
        candidates = self._candidate_frontiers(
            reachable, known, plan_visits, robot, scale, obstacle_clearance,
            novelty_m,
        )
        self._mode = "FRONTIER"
        if not candidates:
            self._mode = "PATROL"
            candidates = self._patrol_candidates(
                reachable, plan_visits, robot, scale, obstacle_clearance,
                novelty_m,
            )
        candidates = [
            item for item in candidates if not self._blacklisted(item[1], scale)
        ]

        committed = self._target_cell
        committed_valid = (
            committed is not None
            and 0 <= committed[0] < reachable.shape[0]
            and 0 <= committed[1] < reachable.shape[1]
            and bool(reachable[committed])
            and not self._blacklisted(committed, scale)
        )
        # Rank by straight-line utility. A real route is never shorter than
        # the straight line and never has negative detour, so this is an upper
        # bound on each candidate's true utility - which is what lets
        # _best_route stop searching as soon as nothing left can win.
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

        # The goal being driven toward: reuse its route while that route is
        # still clear, and search only when it is not. Re-searching an
        # unchanged route every replan was a large part of the planner's cost.
        committed_path = None
        if committed_valid:
            committed_path = self._reuse_route(reachable, planning_start, committed)
            if committed_path is None:
                committed_path = self._astar(
                    reachable, planning_start, committed, deadline, traversal_cost
                )

        committed_utility = None
        must_beat = float("-inf")
        skip = None
        if committed_path:
            route_m = self._route_length(committed_path, scale)
            if route_m < self._goal_best_route_m - self.PROGRESS_MIN_M:
                self._goal_best_route_m = route_m
                self._goal_progress_at = now
            skip = committed
            if now - self._goal_progress_at > self.PROGRESS_TIMEOUT_S:
                # explore_lite's rule: a goal the robot has stopped getting
                # closer to is abandoned, and kept out of consideration for a
                # while rather than chosen again the moment it is dropped.
                self._remember(committed, scale, now + self.BLACKLIST_S)
            else:
                committed_score = next(
                    (score for score, target in candidates if target == committed),
                    None,
                )
                if committed_score is None:
                    # A frontier that has since been seen is no longer a
                    # frontier; keep it only while nothing else is on offer.
                    committed_score = min(
                        (score for score, _target in candidates), default=0.0
                    )
                committed_utility = self._route_utility(
                    committed_score,
                    route_m,
                    math.hypot(
                        committed[0] - planning_start[0],
                        committed[1] - planning_start[1],
                    ) / scale,
                )
                # Hysteresis. A slightly better goal appearing is not a reason
                # to turn around; with a one-metre turning radius, changing
                # goals is itself expensive.
                must_beat = committed_utility + self.GOAL_SWITCH_MARGIN
        chosen_target, chosen_path, chosen_utility = self._best_route(
            ranked_candidates, reachable, planning_start, traversal_cost,
            scale, deadline, skip=skip, must_beat=must_beat,
        )
        if committed_utility is not None and (
            chosen_target is None or chosen_utility < must_beat
        ):
            chosen_target = committed
            chosen_path = committed_path
            chosen_utility = committed_utility

        self._planning_ms = (time.perf_counter() - planning_started) * 1000.0
        if chosen_target is None or chosen_path is None:
            timed_out = time.perf_counter() >= deadline
            if timed_out:
                self._next_replan_at = now + 0.15
                # Reuse only when the complete reachable mask is unchanged.
                # This bounded array comparison preserves a safe commitment
                # without starting another unbudgeted path search.
                if (current.active and committed_valid
                        and now - self._goal_progress_at <= self.PROGRESS_TIMEOUT_S
                        and self._route_free is not None
                        and np.array_equal(self._route_free, reachable)):
                    return current
            # Never resurrect the old route after new obstacles invalidated it.
            self._mode = "PLAN_TIMEOUT" if timed_out else "NO_REACHABLE_TARGET"
            self._drop_goal()
            return self._state(x_m, y_m, heading_deg, scale)

        if chosen_target != committed:
            self._goal_started_at = now
            self._goal_progress_at = now
            self._goal_best_route_m = self._route_length(chosen_path, scale)
            self._goal_initial_route_m = self._goal_best_route_m
        self._route_length_m = self._route_length(chosen_path, scale)
        self._target_cell = chosen_target
        self._path_cells = chosen_path
        self._route_free = reachable
        self._waypoint_cell = self._select_waypoint(
            chosen_path, robot, scale, reachable
        )
        return self._state(x_m, y_m, heading_deg, scale)


def _shifted_state(
    state: ExplorationState, dx_m: float, dy_m: float
) -> ExplorationState:
    """An ExplorationState with every map coordinate moved by (dx, dy)."""
    from dataclasses import replace

    def moved(value: float | None, delta: float) -> float | None:
        return None if value is None else value + delta

    return replace(
        state,
        target_x_m=moved(state.target_x_m, dx_m),
        target_y_m=moved(state.target_y_m, dy_m),
        waypoint_x_m=moved(state.waypoint_x_m, dx_m),
        waypoint_y_m=moved(state.waypoint_y_m, dy_m),
        path_xy_m=tuple((x + dx_m, y + dy_m) for x, y in state.path_xy_m),
    )


class AsyncExplorer:
    """Run the frontier explorer on its own thread, off the control loop.

    Global planning used to run inline, once per control tick, with a 45 ms
    budget. Every replan therefore stole up to 45 ms from the loop that reads
    the LD19 and refreshes the motor command - a periodic latency spike right
    in the drive path. Planning is advisory: the local arc planner and the
    live safety layers decide what the wheels do on every tick regardless. So
    it can run behind, on its own schedule, without the control loop ever
    waiting for it.

    The map is copied when it is handed over. The mapper scrolls and rewrites
    its arrays in place, so planning on live references could read a grid
    halfway through an update. Copies are taken at roughly the replan rate,
    not every tick.

    Every shift or invalidate bumps an epoch. A plan computed from a map
    snapshot older than the latest epoch describes a grid that has since
    moved, and is discarded rather than published.
    """

    SNAPSHOT_PERIOD_S = 0.45
    MAX_SNAPSHOT_AGE_S = 1.5
    WORKER_PERIOD_S = 0.10

    def __init__(self, explorer: FrontierExplorer | None = None) -> None:
        import threading

        self.explorer = explorer or FrontierExplorer()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._epoch = 0
        self._commands: list[tuple] = []
        self._snapshot: tuple | None = None
        self._pose: tuple[float, float, float] | None = None
        self._state = ExplorationState()
        self._state_snapshot_at = 0.0
        self._next_snapshot_at = 0.0
        self.last_error: str | None = None
        self.plans = 0
        self._thread = threading.Thread(
            target=self._run, name="explorer", daemon=True
        )
        self._thread.start()

    def publish(
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
        force: bool = False,
    ) -> None:
        """Hand over the latest pose, and a map copy when one is due."""
        snapshot = None
        if force or now >= self._next_snapshot_at:
            snapshot = (grid.copy(), observed.copy(), visits.copy(), metres, map_updates)
            self._next_snapshot_at = now + self.SNAPSHOT_PERIOD_S
        with self._lock:
            self._pose = (x_m, y_m, heading_deg)
            if snapshot is not None:
                self._snapshot = (self._epoch, now, *snapshot)
        self._wake.set()

    def shift(self, shift_rows: int, shift_cols: int, fine_scale: float) -> None:
        with self._lock:
            self._epoch += 1
            self._commands.append(("shift", shift_rows, shift_cols, fine_scale))
            self._next_snapshot_at = 0.0
            # Move the published plan with the map, so the arc planner is not
            # handed a route in the pre-shift frame for the moment it takes
            # the worker to replan.
            if fine_scale > 0.0:
                self._state = _shifted_state(
                    self._state, shift_cols / fine_scale, shift_rows / fine_scale
                )

    def note_trouble(self, x_m: float, y_m: float) -> None:
        """The robot got stuck at this map position; see FrontierExplorer."""
        with self._lock:
            self._commands.append(("trouble", float(x_m), float(y_m)))
        self._wake.set()

    def invalidate(self) -> None:
        with self._lock:
            self._epoch += 1
            self._commands.append(("invalidate",))
            self._state = ExplorationState()
            self._next_snapshot_at = 0.0

    def state(self) -> ExplorationState:
        with self._lock:
            if (self._state.active and
                    time.monotonic() - self._state_snapshot_at > self.MAX_SNAPSHOT_AGE_S):
                return ExplorationState(mode="STALE_MAP")
            return self._state

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.WORKER_PERIOD_S)
            self._wake.clear()
            with self._lock:
                commands, self._commands = self._commands, []
                snapshot = self._snapshot
                pose = self._pose
                epoch = self._epoch
            for command in commands:
                if command[0] == "shift":
                    self.explorer.shift(command[1], command[2], command[3])
                elif command[0] == "trouble":
                    self.explorer.note_trouble(command[1], command[2], time.monotonic())
                else:
                    self.explorer.invalidate()
            if snapshot is None or pose is None or snapshot[0] < epoch:
                continue
            _epoch, snapshot_at, grid, observed, visits, metres, map_updates = snapshot
            if time.monotonic() - snapshot_at > self.MAX_SNAPSHOT_AGE_S:
                continue
            try:
                state = self.explorer.update(
                    grid, observed, visits, pose[0], pose[1], pose[2],
                    metres, map_updates, time.monotonic(),
                )
            except Exception as exc:  # planning must never kill the robot
                self.last_error = str(exc)
                with self._lock:
                    if self._epoch == epoch:
                        self._state = ExplorationState(mode="PLANNER_ERROR")
                continue
            self.plans += 1
            with self._lock:
                if self._epoch == epoch:
                    self._state = state
                    self._state_snapshot_at = snapshot_at
                    self.last_error = None

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
