"""USB LSM6DS3 sampling for the Pi runtime, via an MCP2221A USB-I2C adapter.

The IMU improves short-term yaw prediction and turn-rate limiting.  It does
not provide absolute heading or position because the LSM6DS3 has no
magnetometer and the chassis has no wheel encoders.  Non-IMU navigation
(camera + LD19 pose prediction) remains supported when no IMU is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import statistics
import threading
import time
from typing import Callable, Protocol


class _SMBusLike(Protocol):
    def read_byte_data(self, address: int, register: int) -> int: ...
    def write_byte_data(self, address: int, register: int, value: int) -> None: ...
    def read_i2c_block_data(self, address: int, register: int, length: int) -> list[int]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class IMUState:
    connected: bool = False
    calibrated: bool = False
    fresh: bool = False
    calibration_progress: float = 0.0
    accel_x_g: float = 0.0
    accel_y_g: float = 0.0
    accel_z_g: float = 0.0
    accel_deviation_g: float = 0.0
    gyro_x_dps: float = 0.0
    gyro_y_dps: float = 0.0
    gyro_z_dps: float = 0.0
    yaw_deg: float = 0.0
    temperature_c: float = 0.0
    updated_at: float = 0.0
    error: str | None = None
    source: str = "NONE"
    calibration_hold: str = ""


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


class LSM6DS3MCP2221Link:
    """Poll a USB LSM6DS3 (via MCP2221A) and expose robot-frame measurements."""

    WHO_AM_I = 0x0F
    CTRL1_XL = 0x10
    CTRL2_G = 0x11
    CTRL3_C = 0x12
    STATUS_REG = 0x1E
    DATA_START = 0x20
    # LSM6DS3TR-C identifies as 0x6A. The older non-C LSM6DS3 uses 0x69.
    EXPECTED_IDS = (0x6A,)
    RETRY_SECONDS = 1.0
    MAX_SAMPLE_AGE_S = 0.25
    DATA_READY_TIMEOUT_S = 0.75
    FILTER_CUTOFF_HZ = 5.0
    BIAS_TRACK_MIN_SAMPLES = 20
    BIAS_TRACK_ALPHA = 0.01
    CALIBRATION_ACCEL_MIN_G = 0.70
    CALIBRATION_ACCEL_MAX_G = 1.30
    CALIBRATION_MAX_GYRO_DPS = 35.0
    # Aggregate acceptance bar for the whole trimmed calibration window (see
    # _finish_calibration), distinct from the CALIBRATION_MAX_GYRO_DPS/ACCEL_*
    # per-sample stillness filter above. v1.9.10 briefly tightened these
    # below the values here based on datasheet noise specs alone; physical
    # testing showed the real sensor + MCP2221 USB path could not reliably
    # settle inside that tighter bar, so calibration_progress stalled short
    # of 100% forever and the robot never gained drive authority. Do not
    # retighten these without hardware-in-the-loop verification.
    CALIBRATION_GYRO_STD_MAX_DPS = 2.5
    CALIBRATION_ACCEL_NORM_MIN = 0.75
    CALIBRATION_ACCEL_NORM_MAX = 1.25
    SENSOR_NAME = "LSM6DS3 USB"

    def __init__(
        self,
        address: int | None = None,
        mount_yaw_deg: float = 180.0,
        calibration_samples: int = 40,
        bus_factory: Callable[[int], _SMBusLike] | None = None,
    ) -> None:
        self.address = 0x6A if address is None else address
        self._candidate_addresses = (
            (0x6A, 0x6B) if address is None else (address,)
        )
        self.mount_yaw_deg = mount_yaw_deg
        self.calibration_samples = max(20, calibration_samples)
        self._bus_factory = bus_factory or _mcp2221_bus_factory
        self._bus: _SMBusLike | None = None
        self._next_connect_at = 0.0
        self._error: str | None = None
        self._calibration: list[
            tuple[float, float, float, float, float, float, float]
        ] = []
        self._gyro_bias = (0.0, 0.0, 0.0)
        self._stationary_bias_samples = 0
        self._calibrated = False
        self._calibration_hold = "WAITING"
        self._last_sample_at = 0.0
        self._connected_at = 0.0
        self._accel = (0.0, 0.0, 0.0)
        self._accel_deviation_g = 0.0
        self._gyro = (0.0, 0.0, 0.0)
        self._filtered_gyro = (0.0, 0.0, 0.0)
        self._temperature_c = 0.0
        self._yaw_deg = 0.0

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

    def _robot_frame(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        radians = math.radians(self.mount_yaw_deg)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        return cosine * x - sine * y, sine * x + cosine * y, z

    def _connect(self, now: float) -> bool:
        bus: _SMBusLike | None = None
        try:
            bus = self._bus_factory(0)
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
        # Only count and integrate complete new gyro+accelerometer samples.
        # This prevents a fast control loop from repeatedly integrating the
        # same 104 Hz hardware output.
        return bus.read_byte_data(self.address, self.STATUS_REG) & 0x03 == 0x03

    def _finish_calibration(self) -> bool:
        gyro_axes = list(zip(*(sample[:3] for sample in self._calibration)))
        trim = max(1, len(self._calibration) // 10)
        trimmed_axes = [sorted(axis)[trim:-trim] for axis in gyro_axes]
        gyro_std = max(statistics.pstdev(axis) for axis in trimmed_axes)
        mean_accel_norm = statistics.fmean(sample[3] for sample in self._calibration)
        if (
            gyro_std > self.CALIBRATION_GYRO_STD_MAX_DPS
            or not self.CALIBRATION_ACCEL_NORM_MIN <= mean_accel_norm <= self.CALIBRATION_ACCEL_NORM_MAX
        ):
            # Keep a rolling window instead of throwing valid progress back to
            # zero. A cable insertion or mild chassis jolt can spoil one
            # aggregate window even though per-sample stillness filtering
            # rejected the obvious motion. The next stable sample replaces the
            # oldest candidate and calibration completes once the full window
            # is internally consistent.
            self._calibration.pop(0)
            self._calibration_hold = "UNSTABLE"
            return False
        # A trimmed mean rejects an isolated USB/I2C or handling spike without
        # biasing every later yaw integration step.
        self._gyro_bias = tuple(
            statistics.fmean(axis) for axis in trimmed_axes
        )
        self._calibrated = True
        self._calibration_hold = ""
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
    ) -> bool:
        """Reject handling/USB-plug jolts before they enter the bias window."""
        return (
            self.CALIBRATION_ACCEL_MIN_G
            <= accel_norm
            <= self.CALIBRATION_ACCEL_MAX_G
            and max(abs(gx), abs(gy), abs(gz))
            <= self.CALIBRATION_MAX_GYRO_DPS
        )

    def tick(self, now: float | None = None, stationary: bool = True) -> IMUState:
        now = time.monotonic() if now is None else now
        if self._bus is None and now >= self._next_connect_at and not self._connect(now):
            return self.state(now)
        if self._bus is None:
            return self.state(now)
        try:
            if not self._sample_ready(self._bus):
                if not self._calibrated:
                    self._calibration_hold = "WAIT DATA"
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
        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        deviation = abs(accel_norm - 1.0)
        self._accel_deviation_g = (
            self._accel_deviation_g * 0.82 + deviation * 0.18
        )
        self._temperature_c = temperature

        if not self._calibrated:
            self._gyro = (gx, gy, gz)
            # A commanded movement pauses calibration, but does not throw away
            # earlier valid still samples. Unplugging another USB device can
            # momentarily delay the loop or jolt the chassis; neither should
            # make a nearly complete calibration visibly restart at zero.
            if not stationary:
                self._calibration_hold = "MOTION"
                return self.state(now)
            if not self._calibration_sample_is_still(
                gx, gy, gz, accel_norm
            ):
                self._calibration_hold = "SAMPLE"
                return self.state(now)
            self._calibration_hold = ""
            self._calibration.append((gx, gy, gz, accel_norm, ax, ay, az))
            if len(self._calibration) >= self.calibration_samples:
                self._finish_calibration()
            return self.state(now)

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
            accel_deviation_g=self._accel_deviation_g,
            gyro_x_dps=self._gyro[0],
            gyro_y_dps=self._gyro[1],
            gyro_z_dps=self._gyro[2],
            yaw_deg=self._yaw_deg,
            temperature_c=self._temperature_c,
            updated_at=self._last_sample_at,
            error=self._error,
            source=self.SENSOR_NAME,
            calibration_hold=self._calibration_hold,
        )

    def close(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass


class AsyncIMULink:
    """Sample an IMU independently of camera, display, and planner latency."""

    def __init__(self, link: LSM6DS3MCP2221Link, sample_period_s: float = 0.02) -> None:
        self._link = link
        self._sample_period_s = max(0.005, sample_period_s)
        self._lock = threading.Lock()
        self._stationary = True
        self._state = link.state()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="imu-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                stationary = self._stationary
            now = time.monotonic()
            try:
                state = self._link.tick(now, stationary)
            except Exception as exc:
                state = IMUState(error=f"IMU sampler error: {exc}", source=LSM6DS3MCP2221Link.SENSOR_NAME)
            with self._lock:
                self._state = state
            self._stop.wait(self._sample_period_s)

    def tick(self, _now: float | None = None, stationary: bool = True) -> IMUState:
        with self._lock:
            self._stationary = stationary
            return self._state

    def state(self, _now: float | None = None) -> IMUState:
        with self._lock:
            return self._state

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._link.close()
