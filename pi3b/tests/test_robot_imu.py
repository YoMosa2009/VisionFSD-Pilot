from __future__ import annotations

import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_imu import (
    AsyncIMULink,
    IMUState,
    LSM6DS3MCP2221Link,
)


def _little_word(value: int) -> list[int]:
    if value < 0:
        value += 65536
    return [value & 0xFF, (value >> 8) & 0xFF]


def _lsm_sample(
    temperature: int = 0,
    gx: int = 0,
    gy: int = 0,
    gz: int = 0,
    ax: int = 0,
    ay: int = 0,
    az: int = 16393,
) -> list[int]:
    data: list[int] = []
    for value in (temperature, gx, gy, gz, ax, ay, az):
        data.extend(_little_word(value))
    return data


class _FakeLSMBus:
    def __init__(self) -> None:
        self.sample = _lsm_sample()
        self.writes: list[tuple[int, int, int]] = []
        self.closed = False
        self.probed: list[int] = []
        self.registers: dict[int, int] = {}
        self.valid_config_readback = True

    def read_byte_data(self, address: int, register: int) -> int:
        if register == 0x1E:
            return 0x03
        if register == 0x0F:
            self.probed.append(address)
            if address == 0x6A:
                raise OSError("no response")
            return 0x6A
        value = self.registers.get(register, 0)
        return value if self.valid_config_readback else value ^ 0x01

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self.writes.append((address, register, value))
        self.registers[register] = value

    def read_i2c_block_data(
        self, _address: int, _register: int, _length: int
    ) -> list[int]:
        return list(self.sample)

    def close(self) -> None:
        self.closed = True


class _ProgressLink:
    def __init__(self) -> None:
        self.samples = 0
        self.closed = False

    def tick(self, _now: float, _stationary: bool) -> IMUState:
        self.samples += 1
        return IMUState(
            connected=True,
            calibrated=self.samples >= 4,
            fresh=True,
            calibration_progress=min(1.0, self.samples / 4.0),
            source="LSM6DS3 USB",
        )

    def state(self, _now: float | None = None) -> IMUState:
        return IMUState(connected=True, source="LSM6DS3 USB")

    def close(self) -> None:
        self.closed = True


class AsyncIMULinkTests(unittest.TestCase):
    def test_async_sampler_calibrates_without_main_loop_ticks(self) -> None:
        link = _ProgressLink()
        imu = AsyncIMULink(link, sample_period_s=0.005)
        try:
            deadline = time.monotonic() + 0.5
            while not imu.state().calibrated and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(imu.state().calibrated)
            self.assertGreaterEqual(link.samples, 4)
        finally:
            imu.close()
        self.assertTrue(link.closed)


class LSM6DS3Tests(unittest.TestCase):
    def test_rejected_aggregate_window_retains_progress_and_recovers(self) -> None:
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(calibration_samples=20, bus_factory=lambda _number: bus)

        for index in range(20):
            # +/-6.0 dps alternating, population std 6.0 dps: outside the
            # aggregate CALIBRATION_GYRO_STD_MAX_DPS (2.5) bar.
            bus.sample = _lsm_sample(gz=686 if index % 2 else -686)
            state = imu.tick(1.0 + index * 0.03, stationary=True)

        self.assertFalse(state.calibrated)
        self.assertGreaterEqual(state.calibration_progress, 0.90)
        for index in range(30):
            bus.sample = _lsm_sample()
            state = imu.tick(2.0 + index * 0.03, stationary=True)
            if state.calibrated:
                break
        self.assertTrue(state.calibrated)

    def test_moderate_gyro_variance_converges_within_the_calibration_bar(self) -> None:
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(calibration_samples=20, bus_factory=lambda _number: bus)
        state = None
        for index in range(20):
            # +/-2.1875 dps alternating, population std ~2.19 dps: inside the
            # 2.5 dps aggregate bar.
            bus.sample = _lsm_sample(gz=250 if index % 2 else -250)
            state = imu.tick(1.0 + index * 0.03, stationary=True)
        self.assertTrue(state.calibrated)

    def test_motion_pauses_unfinished_calibration_without_resetting_progress(self) -> None:
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(calibration_samples=20, bus_factory=lambda _number: bus)
        for index in range(10):
            imu.tick(1.0 + index * 0.03, stationary=True)
        moved = imu.tick(1.4, stationary=False)
        self.assertFalse(moved.calibrated)
        self.assertEqual(moved.calibration_progress, 0.5)
        for index in range(10):
            moved = imu.tick(1.5 + index * 0.03, stationary=True)
        self.assertTrue(moved.calibrated)

    def test_handling_jolt_is_rejected_without_erasing_still_samples(self) -> None:
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(calibration_samples=20, bus_factory=lambda _number: bus)
        state = None
        for index in range(10):
            state = imu.tick(1.0 + index * 0.03, stationary=True)
        self.assertEqual(state.calibration_progress, 0.5)

        # 52.5 dps exceeds the per-sample CALIBRATION_MAX_GYRO_DPS (35.0)
        # stillness filter, simulating a handling/USB-plug jolt.
        bus.sample = _lsm_sample(gz=6000)
        for index in range(4):
            state = imu.tick(1.4 + index * 0.03, stationary=True)
        self.assertEqual(state.calibration_progress, 0.5)

        bus.sample = _lsm_sample()
        for index in range(10):
            state = imu.tick(1.6 + index * 0.03, stationary=True)
        self.assertTrue(state.calibrated)

    def test_missing_bus_degrades_without_raising(self) -> None:
        def missing(_number: int):
            raise OSError("missing")

        imu = LSM6DS3MCP2221Link(bus_factory=missing)
        state = imu.tick(1.0)
        self.assertFalse(state.connected)
        self.assertFalse(state.fresh)
        self.assertIn("missing", state.error)

    def test_usb_imu_uses_short_startup_calibration_window(self) -> None:
        imu = LSM6DS3MCP2221Link(bus_factory=lambda _number: _FakeLSMBus())
        self.assertEqual(imu.calibration_samples, 40)

    def test_stationary_tilt_and_moderate_zero_rate_bias_calibrate(self) -> None:
        bus = _FakeLSMBus()
        # One g total acceleration on a slightly tilted mount, plus a stable
        # 10.5 dps zero-rate bias that the previous absolute 8 dps gate rejected.
        bus.sample = _lsm_sample(gx=1200, ax=8197, az=14197)
        imu = LSM6DS3MCP2221Link(
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )

        for index in range(20):
            state = imu.tick(1.0 + index * 0.03, stationary=True)

        self.assertTrue(state.calibrated)
        self.assertEqual(state.calibration_progress, 1.0)

    def test_tr_c_identity_auto_probes_second_address_and_decodes_little_endian(self) -> None:
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(
            mount_yaw_deg=0.0,
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )
        state = None
        for index in range(20):
            state = imu.tick(1.0 + index * 0.03, stationary=True)
        self.assertIsNotNone(state)
        self.assertTrue(state.connected)
        self.assertTrue(state.calibrated)
        self.assertEqual(state.source, "LSM6DS3 USB")
        self.assertEqual(bus.probed[:2], [0x6A, 0x6B])
        self.assertEqual(imu.address, 0x6B)
        self.assertAlmostEqual(state.accel_z_g, 1.0, places=3)
        self.assertEqual(
            bus.writes,
            [(0x6B, 0x10, 0x40), (0x6B, 0x11, 0x40), (0x6B, 0x12, 0x44)],
        )

    def test_actual_mount_rotates_board_xy_180_and_scales_temperature(self) -> None:
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )
        for index in range(20):
            imu.tick(1.0 + index * 0.03, stationary=True)
        # On the installed board +X points rearward and +Y points rightward.
        bus.sample = _lsm_sample(temperature=256, ax=-16393, ay=-8197, az=16393)
        state = imu.tick(1.7, stationary=False)
        self.assertAlmostEqual(state.accel_x_g, 1.0, places=3)
        self.assertAlmostEqual(state.accel_y_g, 0.5, places=3)
        self.assertAlmostEqual(state.accel_z_g, 1.0, places=3)
        self.assertAlmostEqual(state.temperature_c, 26.0, places=3)

    def test_not_ready_sample_does_not_advance_calibration(self) -> None:
        bus = _FakeLSMBus()
        original_read = bus.read_byte_data

        def not_ready(address: int, register: int) -> int:
            return 0x00 if register == 0x1E else original_read(address, register)

        bus.read_byte_data = not_ready
        imu = LSM6DS3MCP2221Link(
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )
        state = imu.tick(1.0, stationary=True)
        self.assertTrue(state.connected)
        self.assertEqual(state.calibration_progress, 0.0)
        self.assertFalse(state.fresh)
        timed_out = imu.tick(1.8, stationary=True)
        self.assertFalse(timed_out.connected)
        self.assertIn("stopped producing fresh data", timed_out.error)

    def test_configuration_readback_failure_rejects_sensor(self) -> None:
        bus = _FakeLSMBus()
        bus.valid_config_readback = False
        imu = LSM6DS3MCP2221Link(bus_factory=lambda _number: bus)
        state = imu.tick(1.0)
        self.assertFalse(state.connected)
        self.assertIn("config verify failed", state.error)

    def test_confirmed_stationary_period_tracks_small_gyro_bias_drift(self) -> None:
        bus = _FakeLSMBus()
        imu = LSM6DS3MCP2221Link(
            mount_yaw_deg=0.0,
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )
        for index in range(20):
            imu.tick(1.0 + index * 0.03, stationary=True)
        bus.sample = _lsm_sample(gz=57)
        initial = imu.tick(1.7, stationary=True)
        state = initial
        for index in range(100):
            state = imu.tick(1.73 + index * 0.03, stationary=True)
        self.assertGreater(initial.gyro_z_dps, 0.1)
        self.assertLess(abs(state.gyro_z_dps), abs(initial.gyro_z_dps) * 0.6)


if __name__ == "__main__":
    unittest.main()
