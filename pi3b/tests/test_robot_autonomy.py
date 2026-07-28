from __future__ import annotations

import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import ArduinoStatus, AutonomousPolicy, SectorClearance


class AutonomousPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status = ArduinoStatus(front_cm=80.0, motion="S", received_at=time.monotonic())

    def test_boot_standby_never_moves(self) -> None:
        policy = AutonomousPolicy(25.0, 70)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        self.assertEqual(policy.decide(clear, self.status, False, policy.started_at + 3.0), "STOP")

    def test_stale_lidar_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        stale = SectorClearance(None, None, None, False)
        self.assertEqual(policy.decide(stale, self.status, False, time.monotonic()), "STOP")

    def test_ultrasonic_wins_over_clear_lidar(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        blocked = ArduinoStatus(front_cm=15.0, motion="S", received_at=time.monotonic())
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        self.assertEqual(policy.decide(clear, blocked, False, time.monotonic()), "STOP")

    def test_confirmed_person_stops(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(2.0, 2.0, 2.0, True)
        self.assertEqual(policy.decide(clear, self.status, True, time.monotonic()), "STOP")

    def test_front_obstacle_turns_toward_clearer_side(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        obstacle = SectorClearance(0.25, 0.7, 1.6, True)
        self.assertEqual(policy.decide(obstacle, self.status, False, time.monotonic()), "R")

    def test_clear_path_moves_forward(self) -> None:
        policy = AutonomousPolicy(0.0, 70)
        clear = SectorClearance(1.2, 1.0, 1.0, True)
        self.assertEqual(policy.decide(clear, self.status, False, time.monotonic()), "F")


if __name__ == "__main__":
    unittest.main()
