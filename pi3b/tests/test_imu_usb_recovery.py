from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_imu import _MCP2221Bus, AsyncIMULink, IMUState, LSM6DS3MCP2221Link
from test_robot_imu import _FakeLSMBus


class WorkingBus:
    def read_byte_data(self, address, register):
        return 106
    def close(self):
        pass


class BlockedBus(WorkingBus):
    def read_byte_data(self, address, register):
        time.sleep(60)


class BlockedOpen(WorkingBus):
    def __init__(self):
        time.sleep(60)


class MissingBus:
    def __init__(self):
        raise OSError("USB permission denied")


class USBRecoveryTests(unittest.TestCase):
    def test_blocked_usb_read_terminates_owner_and_new_owner_reads(self):
        bus = _MCP2221Bus(BlockedBus, transaction_timeout_s=0.1)
        started = time.monotonic()
        with self.assertRaisesRegex(OSError, "USB timeout"):
            bus.read_byte_data(106, 15)
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(bus._process.is_alive())
        replacement = _MCP2221Bus(WorkingBus)
        try:
            self.assertEqual(replacement.read_byte_data(106, 15), 106)
        finally:
            replacement.close()

    def test_usb_proxy_keeps_second_address_probe_working(self):
        imu = LSM6DS3MCP2221Link(calibration_samples=20,
            bus_factory=lambda _: _MCP2221Bus(_FakeLSMBus))
        try:
            for i in range(20):
                state = imu.tick(1 + i * 0.03)
            self.assertEqual(imu.address, 0x6B)
            self.assertTrue(state.calibrated)
        finally:
            imu.close()

    def test_open_error_keeps_actionable_diagnostic(self):
        with self.assertRaisesRegex(OSError, "USB permission denied"):
            _MCP2221Bus(MissingBus)

    def test_blocked_open_is_bounded(self):
        started = time.monotonic()
        with self.assertRaisesRegex(OSError, "USB timeout"):
            _MCP2221Bus(BlockedOpen, startup_timeout_s=0.2)
        self.assertLess(time.monotonic() - started, 2)

    def test_calibration_resumes_after_timeout_without_bypassing_gate(self):
        first, second = _FakeLSMBus(), _FakeLSMBus()
        buses = iter((first, second))
        imu = LSM6DS3MCP2221Link(calibration_samples=20, bus_factory=lambda _: next(buses))
        for i in range(10):
            state = imu.tick(1 + i * 0.03)
        self.assertEqual(state.calibration_progress, 0.5)
        def timeout(*args):
            raise OSError("MCP2221 USB timeout; reopening adapter")
        first.read_i2c_block_data = timeout
        state = imu.tick(1.4)
        self.assertFalse(state.calibrated)
        self.assertFalse(state.fresh)
        for i in range(10):
            state = imu.tick(2.5 + i * 0.03)
        self.assertTrue(state.calibrated)
        self.assertTrue(first.closed)

    def test_recognized_sensor_with_config_fault_is_not_absent(self):
        bus = _FakeLSMBus()
        bus.valid_config_readback = False
        imu = LSM6DS3MCP2221Link(bus_factory=lambda _: bus)
        state = imu.tick(1)
        self.assertTrue(state.detected)
        self.assertFalse(state.connected)
        self.assertFalse(state.calibrated)

    def test_stale_calibration_never_says_accepting(self):
        imu = AsyncIMULink.__new__(AsyncIMULink)
        imu._lock = threading.Lock()
        imu._state = IMUState(connected=True, fresh=True, updated_at=1, calibration_progress=0.5)
        state = imu.state(2)
        self.assertFalse(state.fresh)
        self.assertEqual(state.calibration_hold, "USB WAIT")
        self.assertFalse(state.calibrated)


if __name__ == "__main__":
    unittest.main()
