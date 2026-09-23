"""USB LSM6DS3 sampling for the Pi runtime, via an MCP2221A USB-I2C adapter.

The IMU improves short-term yaw prediction and turn-rate limiting.  It does
not provide absolute heading or position because the LSM6DS3 has no
magnetometer and the chassis has no wheel encoders.  Non-IMU navigation
(camera + LD19 pose prediction) remains supported when no IMU is connected.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, replace
import math
import multiprocessing
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
    sample_interval_s: float = 0.0
    sample_gaps: int = 0
    detected: bool = False
    # Gravity-referenced chassis attitude. The angle between measured gravity
    # and the board's own Z axis is independent of which horizontal axis
    # happens to point forward, so it stays correct regardless of the mount
    # yaw correction. Useful for driving onto a rug lip or a threshold, and
    # for noticing the chassis being picked up.
    tilt_deg: float = 0.0
    tilt_rate_dps: float = 0.0
    # Magnitude of the acceleration left after removing the estimated gravity
    # vector, split into the component in the horizontal plane and the total.
    # These are motion *cues*, never integrated into a position.
    planar_accel_g: float = 0.0
    motion_energy_g: float = 0.0
    # Accelerometer energy measured while stationary during calibration. It
    # is the reference that makes motion_energy_g interpretable on this
    # particular chassis, floor and mounting instead of an absolute guess.
    still_energy_g: float = 0.0
    # True while the accelerometer sees sustained energy or an attitude change
    # that the robot never commanded: the signature of being lifted, shoved or
    # otherwise repositioned by hand.
    handled: bool = False
    # Integrated yaw change over roughly the last second. This is a bounded
    # short-term record of what the chassis just did, not absolute heading.
    yaw_delta_1s_deg: float = 0.0


class _BlinkaMCP2221Bus:
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


def _usb_bus_worker(connection, bus_factory) -> None:
    """Own Blinka and its HID handle in a disposable process.

    Blinka can block inside hid.read or an I2C status loop. A thread timeout
    cannot release that handle safely; only this process ever touches it.
    """
    bus = None
    try:
        bus = bus_factory()
        connection.send((True, None))
        while True:
            method, args = connection.recv()
            if method == "close":
                break
            try:
                result = getattr(bus, method)(*args)
                connection.send((True, result))
            except Exception as exc:
                connection.send((False, str(exc)))
    except Exception as exc:
        try:
            connection.send((False, str(exc)))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if bus is not None:
            bus.close()
        connection.close()


class _MCP2221Bus:
    """Bound USB operations without abandoning a thread holding the adapter."""

    def __init__(self, bus_factory=None, startup_timeout_s=8.0,
                 transaction_timeout_s=0.75) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._timeout = transaction_timeout_s
        self._process = context.Process(
            target=_usb_bus_worker,
            args=(child, bus_factory or _BlinkaMCP2221Bus),
            name="imu-usb", daemon=True,
        )
        self._process.start()
        child.close()
        try:
            self._receive(startup_timeout_s, stage="startup")
        except Exception:
            self.close()
            raise

    def _receive(self, timeout_s, stage="transaction"):
        try:
            if not self._connection.poll(timeout_s):
                # Say which step timed out. The same text for both is why the
                # 2026-09 field runs could not tell a worker that never
                # started (software, CPU, USB enumeration) from a sensor that
                # never answered (wiring, pull-ups, power).
                if stage == "startup":
                    raise OSError(
                        "MCP2221 USB timeout: adapter worker did not start "
                        f"within {timeout_s:.1f} s; reopening adapter"
                    )
                raise OSError(
                    "MCP2221 USB timeout: no reply to an I2C transaction "
                    f"within {timeout_s:.2f} s; reopening adapter"
                )
            ok, result = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close()
            raise OSError(str(exc) or "MCP2221 USB worker exited") from exc
        # A sensor NACK is a completed transaction, not a dead USB reader.
        # Keep the owner alive so automatic probing can try address 0x6B.
        if not ok:
            raise OSError(result)
        return result

    def _call(self, method, *args):
        try:
            self._connection.send((method, args))
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close()
            raise OSError("MCP2221 USB worker unavailable") from exc
        return self._receive(self._timeout)

    def read_byte_data(self, address, register):
        return self._call("read_byte_data", address, register)

    def write_byte_data(self, address, register, value):
        return self._call("write_byte_data", address, register, value)

    def read_i2c_block_data(self, address, register, length):
        return self._call("read_i2c_block_data", address, register, length)

    def close(self):
        # Termination releases the OS HID handle even if Blinka cannot return.
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=0.5)
        self._connection.close()


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
    #: Longest wait between failed connection attempts. Each attempt spawns a
    #: Python process that imports the USB stack; retrying every second
    #: against an adapter that never answers kept one of the Pi 3B's four
    #: cores starting interpreters for the whole of both 2026-09 field runs.
    RETRY_MAX_SECONDS = 30.0
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
    # Gravity tracks slowly so real chassis acceleration is not absorbed into
    # the gravity estimate; motion energy tracks quickly so a stall or a jolt
    # is visible within a few samples.
    GRAVITY_ALPHA = 0.02
    ENERGY_ALPHA = 0.18
    TILT_RATE_ALPHA = 0.20
    # Lower bound on the learned stationary noise floor. Without it, an
    # unusually quiet calibration would make ordinary sensor noise look like
    # motion for the rest of the run.
    MIN_STILL_ENERGY_G = 0.004
    # Being handled: sustained energy well above the stationary floor, or a
    # clear attitude change, while nothing is being commanded.
    HANDLED_ENERGY_G = 0.09
    HANDLED_TILT_DEG = 12.0
    HANDLED_CONFIRM_S = 0.25
    YAW_HISTORY_S = 1.0

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
        self._connect_failures = 0
        self._error: str | None = None
        self._calibration: list[
            tuple[float, float, float, float, float, float, float]
        ] = []
        self._gyro_bias = (0.0, 0.0, 0.0)
        self._stationary_bias_samples = 0
        self._calibrated = False
        self._detected = False
        self._calibration_hold = "WAITING"
        self._last_sample_at = 0.0
        self._connected_at = 0.0
        self._accel = (0.0, 0.0, 0.0)
        self._accel_deviation_g = 0.0
        self._gyro = (0.0, 0.0, 0.0)
        self._filtered_gyro = (0.0, 0.0, 0.0)
        self._temperature_c = 0.0
        self._yaw_deg = 0.0
        self._sample_interval_s = 0.0
        self._sample_gaps = 0
        self._gravity: tuple[float, float, float] | None = None
        self._tilt_deg = 0.0
        self._tilt_rate_dps = 0.0
        self._planar_accel_g = 0.0
        self._motion_energy_g = 0.0
        self._still_energy_g = self.MIN_STILL_ENERGY_G
        self._calibration_tilt_deg: float | None = None
        self._calibration_energy: list[float] = []
        self._handled = False
        self._handled_since: float | None = None
        self._yaw_history: collections.deque[tuple[float, float, float]] = (
            collections.deque(maxlen=256)
        )
        self._yaw_delta_1s_deg = 0.0

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
                    self._detected = True
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
            self._connect_failures = 0
            return True
        except Exception as exc:
            if bus is not None:
                try:
                    bus.close()
                except Exception:
                    pass
            self._bus = None
            self._error = str(exc)
            # 1, 2, 4, 8, 16, then every 30 s. A link that drops while running
            # still retries after one second (see _disconnect); only repeated
            # failure to connect backs off.
            delay = min(
                self.RETRY_MAX_SECONDS,
                self.RETRY_SECONDS * (2.0 ** min(self._connect_failures, 5)),
            )
            self._connect_failures += 1
            self._next_connect_at = now + delay
            return False

    def _disconnect(self, now: float, error: Exception) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass
        self._error = str(error)
        if not self._calibrated:
            self._calibration_hold = "USB RETRY"
        self._next_connect_at = now + self.RETRY_SECONDS
        self._last_sample_at = 0.0
        self._connected_at = 0.0

    def _update_attitude(
        self,
        ax: float,
        ay: float,
        az: float,
        now: float,
        previous_at: float,
        stationary: bool,
    ) -> None:
        """Derive attitude and motion-energy cues from the accelerometer.

        Splitting the measurement into a slowly-tracked gravity vector and the
        residual gives two genuinely useful signals that the raw
        accel_deviation_g scalar cannot express:

        * the angle between gravity and the board's Z axis (chassis tilt), and
        * the size of the acceleration that is *not* gravity (motion energy).

        Neither is integrated into a velocity or a position. Without wheel
        encoders or an absolute reference, integrating this sensor would drift
        within seconds; these stay first-order observations of the present
        moment.
        """
        if self._gravity is None:
            self._gravity = (ax, ay, az)
        else:
            alpha = self.GRAVITY_ALPHA
            self._gravity = tuple(
                previous * (1.0 - alpha) + current * alpha
                for previous, current in zip(self._gravity, (ax, ay, az))
            )
        gravity_norm = math.sqrt(sum(value * value for value in self._gravity))
        if gravity_norm < 0.30:
            # Free fall or a disconnected sensor: no usable attitude.
            return
        unit = tuple(value / gravity_norm for value in self._gravity)
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, abs(unit[2])))))
        elapsed = now - previous_at if previous_at > 0.0 else 0.0
        if 0.0 < elapsed <= self.MAX_SAMPLE_AGE_S:
            rate = (tilt_deg - self._tilt_deg) / elapsed
            beta = self.TILT_RATE_ALPHA
            self._tilt_rate_dps = self._tilt_rate_dps * (1.0 - beta) + rate * beta
        self._tilt_deg = tilt_deg

        linear = tuple(
            measured - gravity
            for measured, gravity in zip((ax, ay, az), self._gravity)
        )
        along_gravity = sum(value * axis for value, axis in zip(linear, unit))
        planar = tuple(
            value - along_gravity * axis for value, axis in zip(linear, unit)
        )
        planar_magnitude = math.sqrt(sum(value * value for value in planar))
        total_magnitude = math.sqrt(sum(value * value for value in linear))
        gamma = self.ENERGY_ALPHA
        self._planar_accel_g = (
            self._planar_accel_g * (1.0 - gamma) + planar_magnitude * gamma
        )
        self._motion_energy_g = (
            self._motion_energy_g * (1.0 - gamma) + total_magnitude * gamma
        )
        if not self._calibrated:
            self._calibration_energy.append(total_magnitude)
            if len(self._calibration_energy) > 200:
                self._calibration_energy.pop(0)
            return
        self._update_handled(now, stationary)

    def _update_handled(self, now: float, stationary: bool) -> None:
        """Flag external handling while nothing is being commanded.

        Every other motion signal this runtime has is defined relative to a
        commanded drive, so none of them can notice the chassis being picked
        up and carried while it sits latched at zero PWM. Accelerometer energy
        and a changed resting attitude can: with the motors idle, anything the
        accelerometer sees is by definition something the robot did not do.
        """
        if not stationary:
            self._handled_since = None
            self._handled = False
            return
        tilt_changed = (
            self._calibration_tilt_deg is not None
            and abs(self._tilt_deg - self._calibration_tilt_deg)
            >= self.HANDLED_TILT_DEG
        )
        energetic = self._motion_energy_g >= max(
            self.HANDLED_ENERGY_G, self._still_energy_g * 4.0
        )
        if not (tilt_changed or energetic):
            self._handled_since = None
            self._handled = False
            return
        if self._handled_since is None:
            self._handled_since = now
        self._handled = now - self._handled_since >= self.HANDLED_CONFIRM_S

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
        # Calibration is by definition a stationary window, so it is the one
        # chance to measure what "not moving" actually looks like on this
        # chassis, floor and USB path. Everything downstream compares against
        # this instead of an absolute threshold guessed from a datasheet.
        if self._calibration_energy:
            spread = (
                statistics.pstdev(self._calibration_energy)
                if len(self._calibration_energy) > 1
                else 0.0
            )
            self._still_energy_g = max(
                self.MIN_STILL_ENERGY_G,
                statistics.fmean(self._calibration_energy) + 2.0 * spread,
            )
            self._calibration_energy.clear()
        self._calibration_tilt_deg = self._tilt_deg
        self._yaw_history.clear()
        self._yaw_delta_1s_deg = 0.0
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
        self._update_attitude(ax, ay, az, now, previous_at, stationary)

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
        elapsed = max(0.0, now - previous_at) if previous_at > 0.0 else 0.0
        self._sample_interval_s = elapsed
        if elapsed > self.MAX_SAMPLE_AGE_S:
            # There is no measured yaw history inside a USB outage. Resume
            # from the new rate without extrapolating across the missing time.
            self._sample_gaps += 1
            elapsed = 0.0
            previous_filtered = bias_delta
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
        self._update_yaw_history(now)
        return self.state(now)

    def _update_yaw_history(self, now: float) -> None:
        """Keep a bounded record of how far the chassis has just turned.

        A single yaw rate says what is happening right now and the wrapped
        yaw angle says where the estimate has drifted to; neither answers
        "did that commanded pivot actually turn 60 degrees?". Differencing
        unwrapped yaw across a short window does, without pretending to be an
        absolute heading.
        """
        history = self._yaw_history
        if history:
            previous = history[-1][1]
            step = (self._yaw_deg - previous + 180.0) % 360.0 - 180.0
            unwrapped = history[-1][2] + step
        else:
            unwrapped = 0.0
        history.append((now, self._yaw_deg, unwrapped))
        cutoff = now - self.YAW_HISTORY_S
        while len(history) > 2 and history[0][0] < cutoff:
            history.popleft()
        self._yaw_delta_1s_deg = unwrapped - history[0][2]

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
            sample_interval_s=self._sample_interval_s,
            sample_gaps=self._sample_gaps,
            detected=self._detected,
            tilt_deg=self._tilt_deg,
            tilt_rate_dps=self._tilt_rate_dps,
            planar_accel_g=self._planar_accel_g,
            motion_energy_g=self._motion_energy_g,
            still_energy_g=self._still_energy_g,
            handled=self._handled,
            yaw_delta_1s_deg=self._yaw_delta_1s_deg,
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
        try:
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
                self._stop.wait(max(0.0, self._sample_period_s - (time.monotonic() - now)))
        finally:
            self._link.close()

    def tick(self, _now: float | None = None, stationary: bool = True) -> IMUState:
        with self._lock:
            self._stationary = stationary
        return self.state(_now)

    def state(self, _now: float | None = None) -> IMUState:
        now = time.monotonic() if _now is None else _now
        with self._lock:
            state = self._state
        # Age completed measurements even when a USB read blocks the sampler.
        fresh = (state.fresh and state.connected
                 and now - state.updated_at <= LSM6DS3MCP2221Link.MAX_SAMPLE_AGE_S)
        return replace(
            state, fresh=fresh,
            calibration_hold=("USB WAIT" if state.connected and not fresh
                              and not state.calibrated else state.calibration_hold),
        )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
