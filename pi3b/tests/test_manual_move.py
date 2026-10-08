"""Precise manual moves for an AI pilot (robot_manual_move.py, v1.9.32).

2026-10-08: the pilot's turns were held buttons sampled by a ~5 Hz control
loop, so the same 0.1-0.5 s turn rotated anywhere from 0 to 178 degrees.
These tests drive the executor with a simulated clock, chassis and gyro.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_autonomy import ArduinoLink, AutonomousPolicy, SectorClearance, ArduinoStatus  # noqa: E402
from robot_manual_move import (  # noqa: E402
    MOVE_PERIOD_S,
    SETTLE_S,
    TURN_TIMEOUT_S,
    ManualMoveExecutor,
    parse_move,
)
from robot_web import RobotControl  # noqa: E402


class _Status:
    blocked = False


class _ImuState:
    def __init__(self, yaw: float, rate: float, fresh: bool = True) -> None:
        self.yaw_deg = yaw
        self.gyro_z_dps = rate
        self.fresh = fresh
        self.calibrated = True


class SimChassis:
    """Uno link + gyro. Pivots at ``rate_dps`` and coasts ``coast_s`` of
    rotation after a stop; ``sign`` flips the gyro like an inverted mount."""

    def __init__(self, rate_dps: float = 150.0, coast_s: float = 0.08, sign: float = 1.0,
                 stuck: bool = False) -> None:
        self.differential_ready = True
        self.rate_dps = rate_dps
        self.coast_s = coast_s
        self.sign = sign
        self.stuck = stuck
        self.output = (0, 0)
        self.writes: list[tuple[int, int]] = []
        self.yaw = 0.0
        self.rate = 0.0
        self.imu_live = True
        self.status_value = _Status()
        self.released = False

    def advance(self, dt: float) -> None:
        left, right = self.output
        if left * right < 0 and not self.stuck:
            self.rate = self.rate_dps * (1 if right > left else -1)
        elif self.rate:
            # Exponential coast: total extra rotation = rate * coast_s.
            decay = min(1.0, dt / self.coast_s)
            self.rate -= self.rate * decay
            if abs(self.rate) < 1.0:
                self.rate = 0.0
        self.yaw = (self.yaw + self.rate * dt + 180.0) % 360.0 - 180.0

    def publish_drive(self, left: int, right: int, owner=None) -> None:
        self.output = (left, right)
        self.writes.append((left, right))

    def claim(self, owner, until) -> None:
        pass

    def release(self, owner) -> None:
        self.released = True

    def status(self):
        return self.status_value

    def state(self):
        return _ImuState(self.sign * self.yaw, self.sign * self.rate, self.imu_live)


def wheels(command: str, magnitude: float) -> tuple[int, int]:
    level = int(105 + 40 * magnitude)
    return {"F": (level, level), "B": (-level, -level),
            "L": (-level, level), "R": (level, -level)}[command]


def run_move(executor: ManualMoveExecutor, chassis: SimChassis, control: RobotControl,
             limit_s: float = 6.0, during=None) -> dict:
    now = 100.0
    steps = int(limit_s / MOVE_PERIOD_S)
    for index in range(steps):
        executor.step(now)
        if during is not None:
            during(index * MOVE_PERIOD_S)
        now += MOVE_PERIOD_S
        chassis.advance(MOVE_PERIOD_S)
        if index > 0 and not executor.active and control.state().get("move"):
            return control.state()["move"]
    raise AssertionError("the move never finished")


def setup(**chassis_options):
    control = RobotControl()
    control.set_manual(True)
    chassis = SimChassis(**chassis_options)
    executor = ManualMoveExecutor(control, chassis, chassis, wheels, start_thread=False)
    return control, chassis, executor


class TurnTests(unittest.TestCase):
    def test_a_turn_ends_at_the_angle_asked(self) -> None:
        control, chassis, executor = setup()
        for asked in (10, 30, 90, 180):
            start = chassis.yaw
            self.assertTrue(control.request_move(
                {"kind": "turn", "dir": "L", "amount": asked, "mag": 0.3, "id": str(asked)}))
            result = run_move(executor, chassis, control)
            actual = abs((chassis.yaw - start + 180.0) % 360.0 - 180.0) if asked < 180 else None
            self.assertEqual(result["state"], "done")
            self.assertTrue(result["measured"])
            if actual is not None:
                self.assertAlmostEqual(actual, asked, delta=6.0)
            self.assertAlmostEqual(result["turned_deg"], asked, delta=6.0)
        self.assertEqual(chassis.output, (0, 0))
        self.assertTrue(chassis.released)

    def test_the_coast_is_learned(self) -> None:
        control, chassis, executor = setup(coast_s=0.15)
        errors = []
        for index in range(6):
            control.request_move({"kind": "turn", "dir": "R", "amount": 45, "id": str(index)})
            errors.append(abs(run_move(executor, chassis, control)["turned_deg"] - 45))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 5.0)

    def test_an_inverted_gyro_still_ends_the_turn(self) -> None:
        control, chassis, executor = setup(sign=-1.0)
        control.request_move({"kind": "turn", "dir": "L", "amount": 60, "id": "x"})
        result = run_move(executor, chassis, control)
        self.assertEqual(result["state"], "done")
        self.assertAlmostEqual(result["turned_deg"], 60, delta=6.0)

    def test_a_stuck_turn_times_out(self) -> None:
        control, chassis, executor = setup(stuck=True)
        control.request_move({"kind": "turn", "dir": "L", "amount": 90, "id": "x"})
        result = run_move(executor, chassis, control)
        self.assertEqual(result["state"], "stopped")
        self.assertIn("did not reach", result["note"])
        self.assertAlmostEqual(result["drove_s"], TURN_TIMEOUT_S, delta=0.05)

    def test_without_the_gyro_a_turn_is_timed_and_says_so(self) -> None:
        control, chassis, executor = setup()
        chassis.imu_live = False
        control.request_move({"kind": "turn", "dir": "L", "amount": 70, "id": "x"})
        result = run_move(executor, chassis, control)
        self.assertFalse(result["measured"])
        self.assertAlmostEqual(result["drove_s"], 70 / 140.0, delta=0.03)

    def test_a_dashboard_stop_ends_the_turn_at_once(self) -> None:
        control, chassis, executor = setup()
        control.request_move({"kind": "turn", "dir": "L", "amount": 180, "id": "x"})

        def press(at: float) -> None:
            if 0.2 <= at < 0.2 + MOVE_PERIOD_S:
                control.halt()

        result = run_move(executor, chassis, control, during=press)
        self.assertEqual(result["state"], "stopped")
        self.assertLess(result["drove_s"], 0.3)

    def test_pressing_a_pad_button_takes_over(self) -> None:
        control, chassis, executor = setup()
        control.request_move({"kind": "turn", "dir": "L", "amount": 180, "id": "x"})

        def press(at: float) -> None:
            if 0.2 <= at < 0.2 + MOVE_PERIOD_S:
                control.drive("F", 0.5)

        result = run_move(executor, chassis, control, during=press)
        self.assertEqual(result["state"], "stopped")


class DriveTests(unittest.TestCase):
    def test_a_drive_lasts_the_seconds_asked(self) -> None:
        control, chassis, executor = setup()
        control.request_move({"kind": "drive", "dir": "F", "amount": 0.3, "mag": 0.5, "id": "d"})
        result = run_move(executor, chassis, control)
        self.assertEqual(result["state"], "done")
        self.assertAlmostEqual(result["drove_s"], 0.3, delta=MOVE_PERIOD_S + 1e-9)
        driving = [w for w in chassis.writes if w != (0, 0)]
        self.assertAlmostEqual(len(driving) * MOVE_PERIOD_S, 0.3, delta=MOVE_PERIOD_S + 1e-9)
        self.assertEqual(chassis.writes[-1], (0, 0))

    def test_the_arduinos_forward_stop_ends_a_drive(self) -> None:
        control, chassis, executor = setup()
        chassis.status_value.blocked = True
        control.request_move({"kind": "drive", "dir": "F", "amount": 1.0, "id": "d"})
        result = run_move(executor, chassis, control)
        self.assertIn("18 cm", result["note"])

    def test_reversing_ignores_the_forward_stop(self) -> None:
        control, chassis, executor = setup()
        chassis.status_value.blocked = True
        control.request_move({"kind": "drive", "dir": "B", "amount": 0.2, "id": "d"})
        self.assertEqual(run_move(executor, chassis, control)["state"], "done")

    def test_the_result_arrives_after_the_settle(self) -> None:
        control, chassis, executor = setup()
        control.request_move({"kind": "drive", "dir": "B", "amount": 0.1, "id": "d"})
        now = 0.0
        while now < 0.1 + SETTLE_S - 0.05:
            executor.step(now)
            now += MOVE_PERIOD_S
        self.assertNotIn("move", control.state())


class RequestTests(unittest.TestCase):
    def test_requests_are_validated_and_capped(self) -> None:
        self.assertIsNone(parse_move({"kind": "turn", "dir": "F", "amount": 10}))
        self.assertIsNone(parse_move({"kind": "spin", "dir": "L", "amount": 10}))
        self.assertIsNone(parse_move({"kind": "turn", "dir": "L", "amount": "nan"}))
        self.assertEqual(parse_move({"kind": "turn", "dir": "l", "amount": 900}).amount, 180.0)
        self.assertEqual(parse_move({"kind": "drive", "dir": "B", "amount": 9}).amount, 2.0)

    def test_no_move_outside_manual_control_or_while_halted(self) -> None:
        control = RobotControl()
        self.assertFalse(control.request_move({"kind": "turn", "dir": "L", "amount": 10}))
        control.set_manual(True)
        control.halt()
        self.assertFalse(control.request_move({"kind": "turn", "dir": "L", "amount": 10}))

    def test_the_control_state_advertises_precise_moves(self) -> None:
        self.assertTrue(RobotControl().state()["moves"])

    def test_a_move_is_refused_without_the_differential_firmware(self) -> None:
        control, chassis, executor = setup()
        chassis.differential_ready = False
        control.request_move({"kind": "turn", "dir": "L", "amount": 10, "id": "r"})
        executor.step(0.0)
        self.assertEqual(control.state()["move"]["state"], "refused")
        self.assertEqual(chassis.writes, [])


class UnoClaimTests(unittest.TestCase):
    def _link(self) -> ArduinoLink:
        link = object.__new__(ArduinoLink)
        link._drive_lock = threading.Lock()
        link._drive_command = "STOP"
        link._drive_lease_until = 0.0
        link._drive_last_write = 0.0
        link._drive_expired = True
        link._owner = None
        link._owner_until = 0.0
        link._write = mock.Mock(return_value=True)
        return link

    def test_the_control_loop_is_ignored_while_a_move_owns_the_link(self) -> None:
        link = self._link()
        mover = object()
        with mock.patch("robot_autonomy.time.monotonic", return_value=10.0):
            link.claim(mover, 10.2)
            link.publish_drive(-120, 120, owner=mover)
            link.publish_drive(0, 0)                 # the loop's own decision
        link._write.assert_called_once_with("DRIVE -120 120")

    def test_a_lapsed_claim_returns_the_link_to_the_loop(self) -> None:
        link = self._link()
        link.claim(object(), 10.2)
        with mock.patch("robot_autonomy.time.monotonic", return_value=10.5):
            link.publish_drive(110, 110)
        link._write.assert_called_once_with("DRIVE 110 110")


class PolicyMirrorTests(unittest.TestCase):
    def test_the_policy_reports_the_move_and_does_not_drive_over_it(self) -> None:
        policy = AutonomousPolicy(0.0, 120, 105)
        control = RobotControl()
        control.set_manual(True)
        policy.control = control
        move = mock.Mock(active=True, output=(-117, 117), label="MOVE L 30deg")
        policy.manual_move = move
        lidar = SectorClearance(2.0, 2.0, 2.0, True)
        command = policy._apply_control_mode(lidar, ArduinoStatus(None, "S", 0.0), 1.0)
        self.assertEqual(command, "MANUAL_MOVE")
        self.assertEqual((policy.left_pwm, policy.right_pwm), (-117, 117))
        self.assertEqual(policy.reason, "MANUAL:MOVE L 30deg")


if __name__ == "__main__":
    unittest.main()
