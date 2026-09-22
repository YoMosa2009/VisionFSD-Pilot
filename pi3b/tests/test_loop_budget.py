"""Regressions for loop timing and advisory-work budgeting (v1.9.27).

The 2026-09-21 field run measured the control loop at about 1 Hz against its
25 ms design period, which expired the command lease 663 times and left the
frontier planner timing out in 62% of samples. These cover the two mechanisms
added in response: measuring where loop time goes, and shedding advisory work
while the loop is late.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import (
    LOOP_LATE_MS,
    LOOP_RECOVERED_MS,
    UNO_CONTROL_LEASE_S,
)
from robot_loop_budget import AdvisoryBudget, StageTimer


class StageTimerTests(unittest.TestCase):
    def test_stages_are_reported_busiest_first(self) -> None:
        timer = StageTimer()
        for _ in range(4):
            timer.note("render", 0.050)
            timer.note("sense", 0.002)
            timer.tick()
        summary = timer.summary()
        self.assertEqual(list(summary), ["render", "sense"])
        self.assertAlmostEqual(summary["render"]["mean_ms"], 50.0, places=1)
        self.assertAlmostEqual(summary["sense"]["mean_ms"], 2.0, places=1)

    def test_the_worst_tick_is_kept_not_only_the_mean(self) -> None:
        """A stage that is usually fast and occasionally 300 ms is exactly the
        kind that expires the command lease; a mean alone hides it."""
        timer = StageTimer()
        for _ in range(9):
            timer.note("slam", 0.005)
            timer.tick()
        timer.note("slam", 0.300)
        timer.tick()
        self.assertAlmostEqual(timer.summary()["slam"]["max_ms"], 300.0, places=1)
        self.assertLess(timer.summary()["slam"]["mean_ms"], 40.0)

    def test_the_context_manager_records_a_stage(self) -> None:
        timer = StageTimer()
        with timer.stage("decide"):
            pass
        timer.tick()
        self.assertIn("decide", timer.summary())

    def test_reset_clears_the_window(self) -> None:
        timer = StageTimer()
        timer.note("render", 0.05)
        timer.tick()
        timer.reset()
        self.assertEqual(timer.ticks, 0)
        self.assertEqual(timer.summary(), {})

    def test_the_log_line_names_the_rate_and_the_stages(self) -> None:
        timer = StageTimer()
        timer.note("render", 0.12)
        timer.tick()
        line = timer.format_line()
        self.assertTrue(line.startswith("LOOP "))
        self.assertIn("render=", line)
        self.assertIn("hz=", line)


class AdvisoryBudgetTests(unittest.TestCase):
    def test_a_healthy_loop_runs_every_advisory_job(self) -> None:
        budget = AdvisoryBudget()
        for tick in range(50):
            budget.observe(25.0)
            self.assertTrue(budget.allows(tick * 0.025))
        self.assertFalse(budget.behind)
        self.assertEqual(budget.skipped, 0)

    def test_a_late_loop_sheds_advisory_work(self) -> None:
        budget = AdvisoryBudget()
        budget.observe(25.0)
        budget.allows(0.0)
        for _ in range(20):
            budget.observe(900.0)
        self.assertTrue(budget.behind)
        self.assertFalse(budget.allows(0.1))

    def test_shedding_starts_before_the_command_lease_can_expire(self) -> None:
        """Shedding after the lease has already expired would be shedding
        after the motors were cut, which is too late to be worth anything."""
        self.assertLess(LOOP_LATE_MS / 1000.0, UNO_CONTROL_LEASE_S)
        self.assertLess(LOOP_RECOVERED_MS, LOOP_LATE_MS)

    def test_the_view_still_refreshes_while_the_robot_is_busy(self) -> None:
        budget = AdvisoryBudget(max_skip_s=2.0)
        budget.observe(25.0)
        self.assertTrue(budget.allows(9.5))
        for _ in range(20):
            budget.observe(900.0)
        self.assertFalse(budget.allows(10.0))
        self.assertFalse(budget.allows(11.0))
        # Forced through so a viewer can tell a busy robot from a dead one.
        self.assertTrue(budget.allows(12.5))
        self.assertEqual(budget.forced, 1)

    def test_recovery_needs_a_faster_loop_than_shedding_needed(self) -> None:
        """One threshold for both directions would flap between rendering and
        not rendering, which is its own source of jitter."""
        budget = AdvisoryBudget()
        for _ in range(20):
            budget.observe(900.0)
        self.assertTrue(budget.behind)
        for _ in range(3):
            budget.observe(120.0)
        self.assertTrue(budget.behind)
        for _ in range(30):
            budget.observe(25.0)
        self.assertFalse(budget.behind)

    def test_state_is_reportable(self) -> None:
        budget = AdvisoryBudget()
        budget.observe(40.0)
        state = budget.state()
        self.assertIn("gap_ms", state)
        self.assertIn("behind", state)


if __name__ == "__main__":
    unittest.main()
