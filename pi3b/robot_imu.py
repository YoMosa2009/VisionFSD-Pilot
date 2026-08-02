"""Optional USB LSM6DS3 and GPIO MPU-6050 sampling for the Pi runtime.

The IMU improves short-term yaw prediction and turn-rate limiting.  It does
not provide absolute heading or position because neither configured IMU has a
magnetometer and the chassis has no wheel encoders.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
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
    source: str = "NONE"


class _MCP2221Bus:
    """SMBus-shaped adapter over Blinka's MCP2221 USB-I2C transport."""

    def __init__(self) -> None:
        os.environ["BLINKA_MCP2221"] = "1"
        import board

        self._i2c = board.I2C()
        deadline = time.monotonic() + 1.0
        while not self._i2c.try_lock():
            if time.monotonic() >= deadline:
                if hasattr(self._i2c, "deinit"):
                    self._i2c.deinit()
                raise OSError("MCP2221 I2C lock timed out")
            time.sleep(0.01)
        self._closed = False

    def read_byte_data(self, address: int, register: int) -> int:
        return self.read_i2c_block_data(address, register, 1)[0]

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self._i2c.writeto(address, bytes((register & 0xFF, value & 0xFF)))

    def read_i2c_block_data(
        self, address: int, register: int, length: int
    ) -> list[int]:
        result = bytearray(length)
        self._i2c.writeto_then_readfrom(
            address, bytes((register & 0xFF,)), result
        )
        return list(result)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._i2c.unlock()


def _mcp2221_bus_factory(_bus_number: int) -> _SMBusLike:
    return _MCP2221Bus()


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
    DATA_READY_TIMEOUT_S = 0.75
    FILTER_CUTOFF_HZ = 5.0
    BIAS_TRACK_MIN_SAMPLES = 20
    BIAS_TRACK_ALPHA = 0.01
    CALIBRATION_ACCEL_MIN_G = 0.88
    CALIBRATION_ACCEL_MAX_G = 1.12
    CALIBRATION_MAX_GYRO_DPS = 8.0
    CALIBRATION_MAX_HORIZONTAL_G = 0.45
    SENSOR_NAME = "MPU-6050"

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
        self._calibration: list[
            tuple[float, float, float, float, float, float, float]
        ] = []
        self._gyro_bias = (0.0, 0.0, 0.0)
        self._stationary_bias_samples = 0
        self._calibrated = False
        self._last_sample_at = 0.0
        self._connected_at = 0.0
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
            self._connected_at = now
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
        self._connected_at = 0.0

    def _sample_ready(self, bus: _SMBusLike) -> bool:
        return True

    def _finish_calibration(self) -> bool:
        gyro_axes = list(zip(*(sample[:3] for sample in self._calibration)))
        gyro_std = max(statistics.pstdev(axis) for axis in gyro_axes)
        mean_accel_norm = statistics.fmean(sample[3] for sample in self._calibration)
        mean_accel_z = statistics.fmean(sample[6] for sample in self._calibration)
        mean_horizontal = math.hypot(
            statistics.fmean(sample[4] for sample in self._calibration),
            statistics.fmean(sample[5] for sample in self._calibration),
        )
        if (
            gyro_std > 1.5
            or not 0.80 <= mean_accel_norm <= 1.20
            or mean_accel_z < 0.70
            or mean_horizontal > 0.45
        ):
            self._calibration.clear()
            return False
        # A trimmed mean rejects an isolated USB/I2C or handling spike without
        # biasing every later yaw integration step.
        trim = max(1, len(self._calibration) // 10)
        self._gyro_bias = tuple(
            statistics.fmean(sorted(axis)[trim:-trim]) for axis in gyro_axes
        )
        self._calibrated = True
        self._stationary_bias_samples = 0
        self._yaw_deg = 0.0
        self._filtered_gyro = (0.0, 0.0, 0.0)
        self._calibration.clear()
        return True

    def _calibration_sample_is_still(
        self,
        gx: float,
        gy: float,
        gz: float,
        accel_norm: float,
        ax: float,
        ay: float,
        az: float,
    ) -> bool:
        """Reject handling/USB-plug jolts before they enter the bias window."""
        return (
            self.CALIBRATION_ACCEL_MIN_G
            <= accel_norm
            <= self.CALIBRATION_ACCEL_MAX_G
            and max(abs(gx), abs(gy), abs(gz))
            <= self.CALIBRATION_MAX_GYRO_DPS
            and math.hypot(ax, ay) <= self.CALIBRATION_MAX_HORIZONTAL_G
            and az >= 0.70
        )

    def tick(self, now: float | None = None, stationary: bool = True) -> IMUState:
        now = time.monotonic() if now is None else now
        if self._bus is None and now >= self._next_connect_at and not self._connect(now):
            return self.state(now)
        if self._bus is None:
            return self.state(now)
        try:
            if not self._sample_ready(self._bus):
                last_data = self._last_sample_at or self._connected_at
                if last_data > 0.0 and now - last_data > self.DATA_READY_TIMEOUT_S:
                    self._disconnect(now, OSError("sensor stopped producing fresh data"))
                return self.state(now)
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
            # A commanded movement pauses calibration, but does not throw away
            # earlier valid still samples. Unplugging another USB device can
            # momentarily delay the loop or jolt the chassis; neither should
            # make a nearly complete calibration visibly restart at zero.
            if not stationary:
                return self.state(now)
            accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
            if not self._calibration_sample_is_still(
                gx, gy, gz, accel_norm, ax, ay, az
            ):
                return self.state(now)
            self._calibration.append((gx, gy, gz, accel_norm, ax, ay, az))
            if len(self._calibration) >= self.calibration_samples:
                self._finish_calibration()
            return self.state(now)

        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        bias_delta = tuple(
            value - bias for value, bias in zip((gx, gy, gz), self._gyro_bias)
        )
        if (
            stationary
            and 0.92 <= accel_norm <= 1.08
            and max(abs(value) for value in bias_delta) <= 0.75
        ):
            self._stationary_bias_samples += 1
            if self._stationary_bias_samples >= self.BIAS_TRACK_MIN_SAMPLES:
                beta = self.BIAS_TRACK_ALPHA
                self._gyro_bias = tuple(
                    bias * (1.0 - beta) + value * beta
                    for bias, value in zip(self._gyro_bias, (gx, gy, gz))
                )
                bias_delta = tuple(
                    value - bias
                    for value, bias in zip((gx, gy, gz), self._gyro_bias)
                )
        else:
            self._stationary_bias_samples = 0

        previous_filtered = self._filtered_gyro
        elapsed = min(0.10, max(0.0, now - previous_at)) if previous_at > 0.0 else 0.0
        alpha = (
            min(0.85, max(0.10, 1.0 - math.exp(-2.0 * math.pi * self.FILTER_CUTOFF_HZ * elapsed)))
            if elapsed > 0.0
            else 0.35
        )
        self._filtered_gyro = tuple(
            previous * (1.0 - alpha) + current * alpha
            for previous, current in zip(previous_filtered, bias_delta)
        )
        self._gyro = self._filtered_gyro
        if elapsed > 0.0:
            yaw_rate = (previous_filtered[2] + self._gyro[2]) * 0.5
            self._yaw_deg = (
                self._yaw_deg + yaw_rate * elapsed + 180.0
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
            source=self.SENSOR_NAME,
        )

    def close(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass


class LSM6DS3MCP2221Link(MPU6050Link):
    """LSM6DS3 sampled through the MCP2221A USB-I2C adapter."""

    WHO_AM_I = 0x0F
    CTRL1_XL = 0x10
    CTRL2_G = 0x11
    CTRL3_C = 0x12
    STATUS_REG = 0x1E
    DATA_START = 0x20
    # LSM6DS3TR-C identifies as 0x6A. The older non-C LSM6DS3 uses 0x69.
    EXPECTED_IDS = (0x6A,)
    SENSOR_NAME = "LSM6DS3 USB"

    def __init__(
        self,
        address: int | None = None,
        mount_yaw_deg: float = 180.0,
        calibration_samples: int = 80,
        bus_factory: Callable[[int], _SMBusLike] | None = None,
    ) -> None:
        super().__init__(
            bus_number=0,
            address=0x6A if address is None else address,
            mount_yaw_deg=mount_yaw_deg,
            calibration_samples=calibration_samples,
            bus_factory=bus_factory or _mcp2221_bus_factory,
        )
        self._candidate_addresses = (
            (0x6A, 0x6B) if address is None else (address,)
        )

    @classmethod
    def _decode_block(
        cls, data: list[int]
    ) -> tuple[float, float, float, float, float, float, float]:
        if len(data) != 14:
            raise OSError(f"LSM6DS3 returned {len(data)} bytes, expected 14")

        def signed16_le(offset: int) -> int:
            value = data[offset] | (data[offset + 1] << 8)
            return value - 65536 if value >= 32768 else value

        temperature = signed16_le(0)
        gx, gy, gz = (signed16_le(offset) for offset in (2, 4, 6))
        ax, ay, az = (signed16_le(offset) for offset in (8, 10, 12))
        return (
            ax * 0.000061,
            ay * 0.000061,
            az * 0.000061,
            25.0 + temperature / 256.0,
            gx * 0.00875,
            gy * 0.00875,
            gz * 0.00875,
        )

    def _connect(self, now: float) -> bool:
        bus: _SMBusLike | None = None
        try:
            bus = self._bus_factory(self.bus_number)
            errors: list[str] = []
            for address in self._candidate_addresses:
                try:
                    identity = bus.read_byte_data(address, self.WHO_AM_I)
                except Exception as exc:
                    errors.append(f"0x{address:02X}: {exc}")
                    continue
                if identity in self.EXPECTED_IDS:
                    self.address = address
                    break
                errors.append(f"0x{address:02X}: WHO_AM_I 0x{identity:02X}")
            else:
                raise OSError("LSM6DS3 not found (" + "; ".join(errors) + ")")
            # 104 Hz, +/-2 g accelerometer; 104 Hz, +/-245 dps gyro;
            # block-data update and automatic register increment enabled.
            configuration = (
                (self.CTRL1_XL, 0x40),
                (self.CTRL2_G, 0x40),
                (self.CTRL3_C, 0x44),
            )
            for register, value in configuration:
                bus.write_byte_data(self.address, register, value)
            for register, expected in configuration:
                actual = bus.read_byte_data(self.address, register)
                if actual != expected:
                    raise OSError(
                        f"LSM6DS3 config verify failed at 0x{register:02X}: "
                        f"wrote 0x{expected:02X}, read 0x{actual:02X}"
                    )
            self._bus = bus
            self._connected_at = now
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

    def _sample_ready(self, bus: _SMBusLike) -> bool:
        # Only count and integrate complete new gyro+accelerometer samples.
        # This prevents a fast control loop from repeatedly integrating the
        # same 104 Hz hardware output.
        return bus.read_byte_data(self.address, self.STATUS_REG) & 0x03 == 0x03


class AutoIMULink:
    """Prefer USB LSM6DS3, falling back to the existing GPIO MPU-6050."""

    def __init__(
        self,
        mpu_bus: int = 1,
        mpu_address: int = 0x68,
        mount_yaw_deg: float = 180.0,
        lsm_address: int | None = None,
        lsm_link: LSM6DS3MCP2221Link | None = None,
        mpu_link: MPU6050Link | None = None,
    ) -> None:
        self._lsm = lsm_link or LSM6DS3MCP2221Link(
            address=lsm_address, mount_yaw_deg=mount_yaw_deg
        )
        self._mpu = mpu_link or MPU6050Link(
            bus_number=mpu_bus,
            address=mpu_address,
            mount_yaw_deg=mount_yaw_deg,
        )
        self._using_lsm = False

    def tick(self, now: float | None = None, stationary: bool = True) -> IMUState:
        now = time.monotonic() if now is None else now
        lsm_state = self._lsm.tick(now, stationary)
        if lsm_state.connected:
            self._using_lsm = True
            return lsm_state
        self._using_lsm = False
        mpu_state = self._mpu.tick(now, stationary)
        if mpu_state.connected:
            return mpu_state
        return replace(
            mpu_state,
            error=(
                f"LSM6DS3 USB: {lsm_state.error or 'not found'}; "
                f"MPU-6050 GPIO: {mpu_state.error or 'not found'}"
            ),
            source="AUTO",
        )

    def state(self, now: float | None = None) -> IMUState:
        return (self._lsm if self._using_lsm else self._mpu).state(now)

    def close(self) -> None:
        self._lsm.close()
        self._mpu.close()
