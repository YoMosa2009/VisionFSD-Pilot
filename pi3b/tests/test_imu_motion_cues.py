"""Regression tests for the LSM6DS3 attitude and motion cues.

Before these, the only thing the runtime extracted from the accelerometer was
a single scalar (|a| - 1 g), which cannot distinguish a chassis resting on a
rug lip from one on flat floor, nor a quiet stall from a robot being carried.

Everything here is a first-order observation of the present moment. Nothing
is integrated into a velocity or a position: with no wheel encoders and no
absolute reference, integrating this sensor would drift within seconds.
"""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_imu import LSM6DS3MCP2221Link

# 1 g in LSM6DS3 +/-2 g raw counts (0.000061 g/LSB).
_ONE_G = 16393
_DPS = 1.0 / 0.00875


def _little_word(value: int) -> list[int]:
    if value < 0:
        value += 65536
    return [value & 0xFF, (value >> 8) & 0xFF]


def _sample(gx=0, gy=0, gz=0, ax=0, ay=0, az=_ONE_G, temperature=0) -> list[int]:
    data: list[int] = []
    for value in (temperature, gx, gy, gz, ax, ay, az):
        data.extend(_little_word(value))
    return data


class _FakeBus:
    def __init__(self) -> None:
        self.sample = _sample()
        self.registers: dict[int, int] = {}

    def read_byte_data(self, _address: int, register: int) -> int:
        if register == 0x1E:
            return 0x03
        if register == 0x0F:
            return 0x6A
        return self.registers.get(register, 0)

    def write_byte_data(self, _address: int, register: int, value: int) -> None:
        self.registers[register] = value

    def read_i2c_block_data(self, _address, _register, _length) -> list[int]:
        return list(self.sample)

    def close(self) -> None:
        return None


def _calibrated_link(samples: int = 20):
    bus = _FakeBus()
    link = LSM6DS3MCP2221Link(
        address=0x6A, calibration_samples=samples, bus_factory=lambda _n: bus
    )
    now = 1.0
    for _ in range(samples + 4):
        state = link.tick(now, stationary=True)
        now += 0.02
    assert state.calibrated, "fixture failed to calibrate"
    return link, bus, now


def _run(link, bus, now, ticks, stationary, sample):
    bus.sample = sample
    state = None
    for _ in range(ticks):
        state = link.tick(now, stationary=stationary)
        now += 0.02
    return state, now


class AttitudeTests(unittest.TestCase):
    def test_flat_mounting_reports_near_zero_tilt(self) -> None:
        link, _bus, _now = _calibrated_link()
        self.assertLess(link.state().tilt_deg, 3.0)

    def test_tilting_the_chassis_is_measured(self) -> None:
        """Driving onto a rug lip or a threshold changes the gravity
        direction relative to the board, whatever the mount yaw is."""
        link, bus, now = _calibrated_link()
        angle = math.radians(25.0)
        tilted = _sample(
            ax=int(_ONE_G * math.sin(angle)),
            az=int(_ONE_G * math.cos(angle)),
        )
        state, _now = _run(link, bus, now, 400, True, tilted)

        self.assertGreater(state.tilt_deg, 18.0)
        self.assertLess(state.tilt_deg, 32.0)

    def test_tilt_is_independent_of_which_horizontal_axis_is_forward(self) -> None:
        """The angle between gravity and the board Z axis does not depend on
        the mount-yaw correction, so it cannot be wrong because the forward
        axis convention was guessed."""
        angle = math.radians(20.0)
        readings = []
        for mount_yaw in (0.0, 90.0, 180.0):
            bus = _FakeBus()
            link = LSM6DS3MCP2221Link(
                address=0x6A,
                mount_yaw_deg=mount_yaw,
                calibration_samples=20,
                bus_factory=lambda _n, bus=bus: bus,
            )
            now = 1.0
            for _ in range(24):
                link.tick(now, stationary=True)
                now += 0.02
            state, _now = _run(
                link, bus, now, 400, True,
                _sample(
                    ax=int(_ONE_G * math.sin(angle)),
                    az=int(_ONE_G * math.cos(angle)),
                ),
            )
            readings.append(state.tilt_deg)
        self.assertLess(max(readings) - min(readings), 1.0)


class MotionEnergyTests(unittest.TestCase):
    def test_stationary_floor_is_learned_during_calibration(self) -> None:
        link, _bus, _now = _calibrated_link()
        state = link.state()
        self.assertGreater(state.still_energy_g, 0.0)
        self.assertGreaterEqual(
            state.still_energy_g, LSM6DS3MCP2221Link.MIN_STILL_ENERGY_G
        )

    def test_quiet_chassis_stays_at_the_stationary_floor(self) -> None:
        link, bus, now = _calibrated_link()
        state, _now = _run(link, bus, now, 60, True, _sample())
        self.assertLessEqual(state.motion_energy_g, state.still_energy_g * 1.6)

    def test_shaking_raises_motion_energy_well_above_the_floor(self) -> None:
        link, bus, now = _calibrated_link()
        shaken = None
        for index in range(80):
            bus.sample = _sample(
                ax=int(_ONE_G * (0.25 if index % 2 else -0.25))
            )
            shaken = link.tick(now, stationary=False)
            now += 0.02
        self.assertGreater(shaken.motion_energy_g, shaken.still_energy_g * 4.0)


class HandlingTests(unittest.TestCase):
    def test_a_quiet_stationary_chassis_is_not_flagged_as_handled(self) -> None:
        link, bus, now = _calibrated_link()
        state, _now = _run(link, bus, now, 80, True, _sample())
        self.assertFalse(state.handled)

    def test_being_shaken_while_stopped_is_flagged_as_handled(self) -> None:
        """With the motors commanded stopped, anything the accelerometer sees
        is by definition something the robot did not do."""
        link, bus, now = _calibrated_link()
        state = None
        for index in range(120):
            bus.sample = _sample(
                ax=int(_ONE_G * (0.45 if index % 2 else -0.45))
            )
            state = link.tick(now, stationary=True)
            now += 0.02
        self.assertTrue(state.handled)

    def test_a_commanded_drive_is_never_read_as_handling(self) -> None:
        link, bus, now = _calibrated_link()
        state = None
        for index in range(120):
            bus.sample = _sample(
                ax=int(_ONE_G * (0.45 if index % 2 else -0.45))
            )
            state = link.tick(now, stationary=False)
            now += 0.02
        self.assertFalse(state.handled)

    def test_resting_at_a_new_attitude_is_flagged_as_handled(self) -> None:
        link, bus, now = _calibrated_link()
        angle = math.radians(30.0)
        state, _now = _run(
            link, bus, now, 500, True,
            _sample(
                ax=int(_ONE_G * math.sin(angle)),
                az=int(_ONE_G * math.cos(angle)),
            ),
        )
        self.assertTrue(state.handled)


class YawHistoryTests(unittest.TestCase):
    def test_a_sustained_turn_accumulates_yaw_change(self) -> None:
        """Answers "did that commanded pivot actually turn?" - which neither
        the instantaneous rate nor the wrapped yaw angle can."""
        link, bus, now = _calibrated_link()
        state, _now = _run(
            link, bus, now, 60, False, _sample(gz=int(60.0 * _DPS))
        )
        self.assertGreater(abs(state.yaw_delta_1s_deg), 20.0)

    def test_a_still_chassis_reports_no_recent_turn(self) -> None:
        link, bus, now = _calibrated_link()
        state, _now = _run(link, bus, now, 60, True, _sample())
        self.assertLess(abs(state.yaw_delta_1s_deg), 3.0)

    def test_yaw_delta_survives_the_plus_minus_180_wrap(self) -> None:
        """Unwrapping matters: a turn through the wrap point must not read as
        a 360-degree jump."""
        link, bus, now = _calibrated_link()
        state, now = _run(
            link, bus, now, 400, False, _sample(gz=int(90.0 * _DPS))
        )
        self.assertLess(abs(state.yaw_delta_1s_deg), 180.0)


if __name__ == "__main__":
    unittest.main()
