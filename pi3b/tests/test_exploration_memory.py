"""Regressions for exploration memory (v1.9.28).

From driving: the robot did not seem to remember what it had done or where it
had been, and did not pick out open space it had not yet driven that it could
fit into. Three causes were found and are covered here and in
test_robot_motion.DisplacementIsNotTurningTests:

* turning was mistaken for being picked up, wiping the map every few seconds;
* visit memory was a single-cell lookup, so a goal beside a well-driven path
  scored as never visited;
* getting stuck left no trace, so the same goal was chased the same way again.
"""

from __future__ import annotations

import math
import pathlib
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_explorer import AsyncExplorer, FrontierExplorer


class NoveltyTests(unittest.TestCase):
    def test_being_beside_a_driven_path_is_not_novel(self) -> None:
        """The old single-cell visit lookup scored a cell 20 cm beside a path
        driven ten times the same as one across the room."""
        visits = np.zeros((120, 120), dtype=np.uint16)
        visits[58:63, :] = 10          # a strip the robot has driven along
        novelty = FrontierExplorer._novelty_m(visits, 16.0)
        beside = float(novelty[66, 60])   # 25 cm off the strip, never visited itself
        across = float(novelty[110, 60])
        self.assertEqual(int(visits[66, 60]), 0)
        self.assertLess(beside, 0.4)
        self.assertGreater(across, 2.0)

    def test_nowhere_driven_means_everywhere_is_new(self) -> None:
        novelty = FrontierExplorer._novelty_m(np.zeros((40, 40), dtype=np.uint16), 16.0)
        self.assertTrue(np.all(novelty == FrontierExplorer.NOVELTY_CAP_M))

    def test_roaming_heads_for_space_it_has_not_driven(self) -> None:
        """Nothing left to discover: the next goal is away from everywhere the
        robot has been, not merely far from where it is now."""
        reachable = np.ones((120, 120), dtype=bool)
        visits = np.zeros((120, 120), dtype=np.uint16)
        # It has driven a loop around the room's middle and left side.
        visits[20:100, 15:25] = 5
        visits[20:30, 15:70] = 5
        visits[90:100, 15:70] = 5
        candidates = FrontierExplorer._patrol_candidates(reachable, visits, (60, 40), 16.0)
        _score, (row, col) = candidates[0]
        novelty = FrontierExplorer._novelty_m(visits, 16.0)
        self.assertGreater(float(novelty[row, col]), 1.5)
        self.assertGreater(col, 80)

    def test_roaming_goals_must_fit(self) -> None:
        """Only reachable cells are candidates, and reachability is computed
        with the chassis inflation, so novelty cannot pull a goal into a gap
        the robot does not fit through."""
        reachable = np.zeros((120, 120), dtype=bool)
        reachable[:, :70] = True       # the part it fits into
        visits = np.zeros((120, 120), dtype=np.uint16)
        visits[:, :40] = 5
        candidates = FrontierExplorer._patrol_candidates(reachable, visits, (60, 20), 16.0)
        self.assertTrue(candidates)
        for _score, (row, col) in candidates:
            self.assertTrue(reachable[row, col])


class TroubleMemoryTests(unittest.TestCase):
    def test_getting_stuck_drops_the_goal_and_avoids_the_spot(self) -> None:
        explorer = FrontierExplorer()
        explorer._scale = 16.0
        explorer._target_cell = (40, 90)
        explorer.note_trouble(2.0, 3.0, now=100.0)
        self.assertIsNone(explorer._target_cell)
        # The spot where it got stuck ...
        self.assertTrue(explorer._blacklisted((48, 32), 16.0))
        # ... and the goal it was chasing are both avoided.
        self.assertTrue(explorer._blacklisted((40, 90), 16.0))
        # Only for a while: memory is not a permanent ban.
        explorer._prune_blacklist(100.0 + FrontierExplorer.TROUBLE_MEMORY_S + 1.0)
        self.assertFalse(explorer._blacklisted((48, 32), 16.0))

    def test_trouble_reaches_the_planner_thread(self) -> None:
        explorer = AsyncExplorer(FrontierExplorer())
        self.addCleanup(explorer.close)
        explorer.note_trouble(1.5, 2.5)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not explorer.explorer._blacklist:
            time.sleep(0.01)
        self.assertTrue(any(
            math.isclose(x, 1.5) and math.isclose(y, 2.5)
            for x, y, _until in explorer.explorer._blacklist
        ))

    def test_the_runtime_reports_confirmed_stuck_to_the_planner(self) -> None:
        source = (pathlib.Path(__file__).resolve().parents[1] / "robot_autonomy.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('policy.stuck_phase == "RECOVER" and last_stuck_phase != "RECOVER"', source)
        self.assertIn("explorer.note_trouble(slam_lite.x_m, slam_lite.y_m)", source)


if __name__ == "__main__":
    unittest.main()
