"""Regressions for the planner's cost (v1.9.28).

On the 2026-09-23 run the global planner timed out in about 85% of plans:
up to seven Python A* searches per plan, against a 0.12 s budget on a Pi that
is roughly eleven times slower than a desktop. Without a plan the robot had
nowhere to go and turned in place. These pin down the changes that cut a plan
on real recorded maps from a median 437 ms to 36 ms (desktop), without
changing which goal wins.
"""

from __future__ import annotations

import heapq
import math
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_explorer import FrontierExplorer


def _dijkstra_length(free: np.ndarray, start, goal) -> float:
    """Reference shortest 8-connected route length, no corner cutting."""
    height, width = free.shape
    best = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, (row, col) = heapq.heappop(queue)
        if (row, col) == goal:
            return cost
        if cost > best.get((row, col), math.inf):
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width) or not free[nr, nc]:
                continue
            if dr and dc and (not free[row + dr, col] or not free[row, col + dc]):
                continue
            step = math.sqrt(2.0) if dr and dc else 1.0
            if cost + step < best.get((nr, nc), math.inf):
                best[(nr, nc)] = cost + step
                heapq.heappush(queue, (cost + step, (nr, nc)))
    return math.inf


def _length(path) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def _maze(seed: int, size: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    free = np.ones((size, size), dtype=bool)
    for _ in range(12):
        row, col = rng.integers(2, size - 2, 2)
        if rng.random() < 0.5:
            free[row, max(0, col - 15):col + 15] = False
        else:
            free[max(0, row - 15):row + 15, col] = False
    return free


class AStarTests(unittest.TestCase):
    def test_unweighted_search_is_still_optimal(self) -> None:
        """The flat-array rewrite must find shortest routes, like before."""
        with mock.patch.object(FrontierExplorer, "ASTAR_HEURISTIC_WEIGHT", 1.0):
            for seed in range(12):
                free = _maze(seed)
                cells = np.argwhere(free)
                start = tuple(int(v) for v in cells[0])
                goal = tuple(int(v) for v in cells[-1])
                path = FrontierExplorer._astar(free, start, goal)
                reference = _dijkstra_length(free, start, goal)
                if path is None:
                    self.assertEqual(reference, math.inf)
                    continue
                # A straight-line shortcut is taken when clear, which can be
                # shorter than any 8-connected route; never longer.
                self.assertLessEqual(_length(path), reference + 1e-6)

    def test_weighted_search_stays_close_to_optimal(self) -> None:
        for seed in range(12):
            free = _maze(seed)
            cells = np.argwhere(free)
            start = tuple(int(v) for v in cells[0])
            goal = tuple(int(v) for v in cells[-1])
            path = FrontierExplorer._astar(free, start, goal)
            reference = _dijkstra_length(free, start, goal)
            if path is None:
                continue
            self.assertLessEqual(
                _length(path), reference * FrontierExplorer.ASTAR_HEURISTIC_WEIGHT + 1e-6
            )

    def test_routes_never_cross_blocked_cells(self) -> None:
        free = _maze(3)
        cells = np.argwhere(free)
        path = FrontierExplorer._astar(
            free, tuple(int(v) for v in cells[0]), tuple(int(v) for v in cells[-1])
        )
        self.assertIsNotNone(path)
        for a, b in zip(path, path[1:]):
            self.assertTrue(FrontierExplorer._line_is_clear(free, a, b))


class GoalChoiceCostTests(unittest.TestCase):
    def _room(self):
        grid = np.zeros((120, 120), dtype=np.uint8)
        grid[[0, -1], :] = 255
        grid[:, [0, -1]] = 255
        grid[40, 10:90] = 255
        observed = np.full_like(grid, 255)
        visits = np.zeros_like(grid, dtype=np.uint16)
        return (grid, observed, visits, 1.5, 4.5, 0.0, 6.0, 20)

    def test_a_still_clear_route_is_not_searched_again(self) -> None:
        explorer = FrontierExplorer()
        args = self._room()
        first = explorer.update(*args, 10.0)
        self.assertTrue(first.active)
        with mock.patch.object(explorer, "_astar", wraps=explorer._astar) as search:
            second = explorer.update(*args, 11.0)
        self.assertTrue(second.active)
        self.assertEqual(second.target_x_m, first.target_x_m)
        self.assertEqual(search.call_count, 0)

    def test_bounded_choice_matches_an_exhaustive_search(self) -> None:
        """Stopping early must never change which candidate wins."""
        rng = np.random.default_rng(5)
        for trial in range(20):
            free = _maze(trial, 80)
            cells = np.argwhere(free)
            start = tuple(int(v) for v in cells[len(cells) // 2])
            picks = rng.choice(len(cells), 8, replace=False)
            candidates = [(float(rng.uniform(0, 3)), tuple(int(v) for v in cells[i])) for i in picks]
            explorer = FrontierExplorer()
            ranked = sorted(
                candidates,
                key=lambda item: item[0]
                - math.hypot(item[1][0] - start[0], item[1][1] - start[1]) / 16.0
                * FrontierExplorer.ROUTE_LENGTH_WEIGHT,
                reverse=True,
            )
            with mock.patch.object(FrontierExplorer, "ROUTE_SHORTLIST", 100):
                bounded = explorer._best_route(ranked, free, start, None, 16.0, None)
            exhaustive_utility = -math.inf
            for score, target in candidates:
                path = FrontierExplorer._astar(free, start, target)
                if not path:
                    continue
                utility = FrontierExplorer._route_utility(
                    score,
                    FrontierExplorer._route_length(path, 16.0),
                    math.hypot(target[0] - start[0], target[1] - start[1]) / 16.0,
                )
                if utility > exhaustive_utility:
                    exhaustive_utility = utility
            self.assertAlmostEqual(bounded[2], exhaustive_utility, places=9)

    def test_an_alternative_is_not_searched_when_it_cannot_beat_the_goal(self) -> None:
        explorer = FrontierExplorer()
        free = np.ones((60, 60), dtype=bool)
        ranked = [(0.1, (50, 50)), (0.05, (10, 50))]
        with mock.patch.object(explorer, "_astar", wraps=explorer._astar) as search:
            target, _path, _utility = explorer._best_route(
                ranked, free, (30, 30), None, 16.0, None, must_beat=5.0
            )
        self.assertIsNone(target)
        self.assertEqual(search.call_count, 0)


if __name__ == "__main__":
    unittest.main()
