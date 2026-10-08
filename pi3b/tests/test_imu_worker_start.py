"""The IMU adapter worker must start without re-importing the runtime (v1.9.28).

On 2026-09-23 every attempt failed with "adapter worker did not start within
8.0 s": a spawned child re-imports the program's main module - the whole robot
runtime - before it can open the adapter.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import robot_imu


class WorkerStartTests(unittest.TestCase):
    def test_linux_forks_instead_of_re_importing_the_runtime(self) -> None:
        with mock.patch.object(robot_imu.os, "name", "posix"), mock.patch.object(
            robot_imu.multiprocessing, "get_all_start_methods",
            return_value=["fork", "spawn", "forkserver"],
        ), mock.patch.object(robot_imu.multiprocessing, "get_context") as context:
            robot_imu._usb_worker_context()
        context.assert_called_once_with("fork")

    def test_platforms_without_fork_keep_spawn(self) -> None:
        with mock.patch.object(robot_imu.os, "name", "nt"), mock.patch.object(
            robot_imu.multiprocessing, "get_all_start_methods", return_value=["spawn"],
        ), mock.patch.object(robot_imu.multiprocessing, "get_context") as context:
            robot_imu._usb_worker_context()
        context.assert_called_once_with("spawn")

    def test_startup_allows_for_importing_blinka_on_a_pi(self) -> None:
        self.assertGreaterEqual(robot_imu.USB_WORKER_STARTUP_S, 10.0)


if __name__ == "__main__":
    unittest.main()
