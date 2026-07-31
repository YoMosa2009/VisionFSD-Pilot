"""Low-cost MPU-6050 sampling for the Raspberry Pi robot runtime.

The IMU improves short-term yaw prediction and turn-rate limiting.  It does
not provide absolute heading or position because the MPU-6050 has no
magnetometer and the chassis has no wheel encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Callable, Protocol


class _SMBusLike(Protocol):
    def read_byte_data(self, address: int, register: int) -> int: ...
    def write_byte_data(self, address: int, register: int, value: int) -> None: ...
    def read_i2c_block_data(self, address: int, register: int, length: int) -> list[int]: ...
    def close(self) -> None: ...


def _default_bus_factory(bus_number: int) -> _SMBusLike:
    # smbus2 is installed by the Pi requirements.  Keep compatibility with
    # Raspberry Pi OS's python3-smbus package because existing installations
    # may already provide that implementation through system-site-packages.
    try:
        from smbus2 import SMBus
    except ImportError:
        from smbus import SMBus
    return SMBus(bus_number)


@dataclass(frozen=True)
class IMUState:
    connected: bool = False
    calibrated: bool = False
    fresh: bool = False
    calibration_progress: float = 0.0
    accel_x_g: float = 0.0
    accel_y_g: float = 0.0
    accel_z_g: float = 0.0
    gyro_x_dps: float = 0.0
    gyro_y_dps: float = 0.0
    gyro_z_dps: float = 0.0
    yaw_deg: float = 0.0
    temperature_c: float = 0.0
    updated_at: float = 0.0
    error: str | None = None


class MPU6050Link:
    """Poll an MPU-6050 and expose robot-frame, bias-corrected measurements."""

    WHO_AM_I = 0x75
    PWR_MGMT_1 = 0x6B
    CONFIG = 0x1A
    SMPLRT_DIV = 0x19
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    DATA_START = 0x3B
    EXPECTED_IDS = (0x68, 0x69)
    RETRY_SECONDS = 1.0
    MAX_SAMPLE_AGE_S = 0.25

    def __init__(
        self,
        bus_number: int = 1,
        address: int = 0x68,
        mount_yaw_deg: float = 180.0,
        calibration_samples: int = 80,
        bus_factory: Callable[[int], _SMBusLike] | None = None,
    ) -> None:
        self.bus_number = bus_number
        self.address = address
        self.mount_yaw_deg = mount_yaw_deg
        self.calibration_samples = max(20, calibration_samples)
        self._bus_factory = bus_factory or _default_bus_factory
        self._bus: _SMBusLike | None = None
        self._next_connect_at = 0.0
        self._error: str | None = None
        self._calibration: list[tuple[float, float, float, float]] = []
        self._gyro_bias = (0.0, 0.0, 0.0)
        self._calibrated = False
        self._last_sample_at = 0.0
        self._accel = (0.0, 0.0, 0.0)
        self._gyro = (0.0, 0.0, 0.0)
        self._filtered_gyro = (0.0, 0.0, 0.0)
        self._temperature_c = 0.0
        self._yaw_deg = 0.0

    @staticmethod
    def _signed16(high: int, low: int) -> int:
        value = (high << 8) | low
        return value - 65536 if value >= 32768 else value

    @classmethod
    def _decode_block(
        cls, data: list[int]
    ) -> tuple[float, float, float, float, float, float, float]:
        if len(data) != 14:
            raise OSError(f"MPU-6050 returned {len(data)} bytes, expected 14")
        values = [cls._signed16(data[index], data[index + 1]) for index in range(0, 14, 2)]
        ax, ay, az, temperature, gx, gy, gz = values
        return (
            ax / 16384.0,
            ay / 16384.0,
            az / 16384.0,
            temperature / 340.0 + 36.53,
            gx / 131.0,
            gy / 131.0,
            gz / 131.0,
        )

    def _robot_frame(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        radians = math.radians(self.mount_yaw_deg)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        return cosine * x - sine * y, sine * x + cosine * y, z

    def _connect(self, now: float) -> bool:
        bus: _SMBusLike | None = None
        try:
            bus = self._bus_factory(self.bus_number)
            identity = bus.read_byte_data(self.address, self.WHO_AM_I)
            if identity not in self.EXPECTED_IDS:
                bus.close()
                raise OSError(f"unexpected WHO_AM_I 0x{identity:02X}")
            # PLL clock, 44 Hz digital low-pass, 100 Hz internal sampling,
            # +/-250 deg/s gyro and +/-2 g accelerometer.
            bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x01)
            bus.write_byte_data(self.address, self.CONFIG, 0x03)
            bus.write_byte_data(self.address, self.SMPLRT_DIV, 0x09)
            bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
            bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)
            self._bus = bus
            self._error = None
            return True
        except Exception as exc:
            if bus is not None:
                try:
                    bus.close()
                except Exception:
                    pass
            self._bus = None
            self._error = str(exc)
            self._next_connect_at = now + self.RETRY_SECONDS
            return False

    def _disconnect(self, now: float, error: Exception) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass
        self._error = str(error)
        self._next_connect_at = now + self.RETRY_SECONDS
        self._last_sample_at = 0.0

    def _finish_calibration(self) -> bool:
        gyro_axes = list(zip(*(sample[:3] for sample in self._calibration)))
        gyro_std = max(statistics.pstdev(axis) for axis in gyro_axes)
        mean_accel_norm = statistics.fmean(sample[3] for sample in self._calibration)
        if gyro_std > 1.5 or not 0.75 <= mean_accel_norm <= 1.25:
            self._calibration.clear()
            return False
        self._gyro_bias = tuple(statistics.fmean(axis) for axis in gyro_axes)
        self._calibrated = True
        self._yaw_deg = 0.0
        self._filtered_gyro = (0.0, 0.0, 0.0)
        self._calibration.clear()
        return True

    def tick(self, now: float | None = None, stationary: bool = True) -> IMUState:
        now = time.monotonic() if now is None else now
        if self._bus is None and now >= self._next_connect_at and not self._connect(now):
            return self.state(now)
        if self._bus is None:
            return self.state(now)
        try:
            raw = self._bus.read_i2c_block_data(self.address, self.DATA_START, 14)
            ax, ay, az, temperature, gx, gy, gz = self._decode_block(raw)
        except Exception as exc:
            self._disconnect(now, exc)
            return self.state(now)

        ax, ay, az = self._robot_frame(ax, ay, az)
        gx, gy, gz = self._robot_frame(gx, gy, gz)
        previous_at = self._last_sample_at
        self._last_sample_at = now
        self._accel = (ax, ay, az)
        self._temperature_c = temperature

        if not self._calibrated:
            self._gyro = (gx, gy, gz)
            if not stationary:
                self._calibration.clear()
                return self.state(now)
            accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
            self._calibration.append((gx, gy, gz, accel_norm))
            if len(self._calibration) >= self.calibration_samples:
                self._finish_calibration()
            return self.state(now)

        corrected = tuple(
            value - bias for value, bias in zip((gx, gy, gz), self._gyro_bias)
        )
        alpha = 0.35
        self._filtered_gyro = tuple(
            previous * (1.0 - alpha) + current * alpha
            for previous, current in zip(self._filtered_gyro, corrected)
        )
        self._gyro = self._filtered_gyro
        if previous_at > 0.0:
            elapsed = min(0.10, max(0.0, now - previous_at))
            self._yaw_deg = (
                self._yaw_deg + self._gyro[2] * elapsed + 180.0
            ) % 360.0 - 180.0
        return self.state(now)

    def state(self, now: float | None = None) -> IMUState:
        now = time.monotonic() if now is None else now
        connected = self._bus is not None
        fresh = connected and now - self._last_sample_at <= self.MAX_SAMPLE_AGE_S
        progress = 1.0 if self._calibrated else min(
            1.0, len(self._calibration) / self.calibration_samples
        )
        return IMUState(
            connected=connected,
            calibrated=self._calibrated,
            fresh=fresh,
            calibration_progress=progress,
            accel_x_g=self._accel[0],
            accel_y_g=self._accel[1],
            accel_z_g=self._accel[2],
            gyro_x_dps=self._gyro[0],
            gyro_y_dps=self._gyro[1],
            gyro_z_dps=self._gyro[2],
            yaw_deg=self._yaw_deg,
            temperature_c=self._temperature_c,
            updated_at=self._last_sample_at,
            error=self._error,
        )

    def close(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass
