from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from robot_imu import AutoIMULink, IMUState, LSM6DS3MCP2221Link, MPU6050Link


def _word(value: int) -> list[int]:
    if value < 0:
        value += 65536
    return [(value >> 8) & 0xFF, value & 0xFF]


def _sample(
    ax: int = 0,
    ay: int = 0,
    az: int = 16384,
    temperature: int = 0,
    gx: int = 0,
    gy: int = 0,
    gz: int = 131,
) -> list[int]:
    data: list[int] = []
    for value in (ax, ay, az, temperature, gx, gy, gz):
        data.extend(_word(value))
    return data


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


class _FakeBus:
    def __init__(self) -> None:
        self.identity = 0x68
        self.sample = _sample()
        self.writes: list[tuple[int, int, int]] = []
        self.closed = False

    def read_byte_data(self, _address: int, _register: int) -> int:
        return self.identity

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self.writes.append((address, register, value))

    def read_i2c_block_data(
        self, _address: int, _register: int, _length: int
    ) -> list[int]:
        return list(self.sample)

    def close(self) -> None:
        self.closed = True


class _FakeLSMBus(_FakeBus):
    def __init__(self) -> None:
        super().__init__()
        self.sample = _lsm_sample()
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
        super().write_byte_data(address, register, value)
        self.registers[register] = value


class _FakeLink:
    def __init__(self, state: IMUState) -> None:
        self.current = state
        self.closed = False

    def tick(self, _now: float, _stationary: bool) -> IMUState:
        return self.current

    def state(self, _now: float | None = None) -> IMUState:
        return self.current

    def close(self) -> None:
        self.closed = True


class MPU6050Tests(unittest.TestCase):
    def test_stationary_samples_calibrate_and_report_live(self) -> None:
        bus = _FakeBus()
        imu = MPU6050Link(
            mount_yaw_deg=180.0,
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )
        state = None
        for index in range(20):
            state = imu.tick(1.0 + index * 0.03, stationary=True)
        self.assertIsNotNone(state)
        self.assertTrue(state.connected)
        self.assertTrue(state.calibrated)
        self.assertTrue(state.fresh)
        self.assertAlmostEqual(state.accel_z_g, 1.0, places=3)
        self.assertGreaterEqual(len(bus.writes), 5)

    def test_180_degree_mount_reverses_xy_but_retains_z_yaw(self) -> None:
        bus = _FakeBus()
        imu = MPU6050Link(
            mount_yaw_deg=180.0,
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )
        for index in range(20):
            imu.tick(1.0 + index * 0.03, stationary=True)
        bus.sample = _sample(ax=16384, ay=8192, az=16384, gz=1310)
        state = imu.tick(1.7, stationary=False)
        self.assertAlmostEqual(state.accel_x_g, -1.0, places=3)
        self.assertAlmostEqual(state.accel_y_g, -0.5, places=3)
        self.assertAlmostEqual(state.accel_z_g, 1.0, places=3)
        self.assertGreater(state.gyro_z_dps, 0.0)
        self.assertGreater(state.yaw_deg, 0.0)

    def test_motion_restarts_unfinished_calibration(self) -> None:
        bus = _FakeBus()
        imu = MPU6050Link(
            calibration_samples=20,
            bus_factory=lambda _number: bus,
        )
        for index in range(10):
            imu.tick(1.0 + index * 0.03, stationary=True)
        moved = imu.tick(1.4, stationary=False)
        self.assertFalse(moved.calibrated)
        self.assertEqual(moved.calibration_progress, 0.0)

    def test_missing_bus_degrades_without_raising(self) -> None:
        def missing(_number: int):
            raise OSError("missing")

        imu = MPU6050Link(bus_factory=missing)
        state = imu.tick(1.0)
        self.assertFalse(state.connected)
        self.assertFalse(state.fresh)
        self.assertIn("missing", state.error)


class LSM6DS3Tests(unittest.TestCase):
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

    def test_auto_link_prefers_usb_then_falls_back_to_gpio(self) -> None:
        lsm = _FakeLink(IMUState(connected=True, source="LSM6DS3 USB"))
        mpu = _FakeLink(IMUState(connected=True, source="MPU-6050"))
        auto = AutoIMULink(lsm_link=lsm, mpu_link=mpu)
        self.assertEqual(auto.tick(1.0).source, "LSM6DS3 USB")

        lsm.current = IMUState(error="unplugged", source="LSM6DS3 USB")
        self.assertEqual(auto.tick(2.0).source, "MPU-6050")

        mpu.current = IMUState(error="missing", source="MPU-6050")
        missing = auto.tick(3.0)
        self.assertFalse(missing.connected)
        self.assertEqual(missing.source, "AUTO")
        self.assertIn("unplugged", missing.error)
        self.assertIn("missing", missing.error)


if __name__ == "__main__":
    unittest.main()
