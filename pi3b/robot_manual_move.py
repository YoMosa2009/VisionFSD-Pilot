"""Precise manual moves for an AI pilot (haiku_pilot/).

A held dashboard button is sampled by the control loop, which runs at about
5 Hz on a Pi 3B. A short move therefore cannot start or end more precisely
than one loop tick - roughly 0.2 s, or 30 degrees of a pivot - and a loop
stall makes it worse. The first AI-pilot sessions (2026-10-08) measured
exactly that: the same 0.1-0.5 s turn rotated anywhere from 0 to 178 degrees.

This executor runs one requested move on its own 50 Hz thread instead. It
writes the wheel command straight to the Uno link (which the control loop
yields while a move runs), ends a drive on its own clock, and ends a turn on
the IMU's measured rotation, stopping early by the coast it has learned. The
IMU is used only for how far this one turn has gone, never as a heading.
Without a fresh, calibrated IMU a turn falls back to time at a learned rate.

Every move is bounded in time, ends on a dashboard STOP, on any manual drive
command, on leaving Manual Control, and - if this thread itself stalls - on
the Uno link's 250 ms lease and the firmware's 350 ms timeout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

MOVE_PERIOD_S = 0.02
TURN_MAX_DEG = 180.0
TURN_MIN_DEG = 1.0
DRIVE_MAX_S = 2.0
DRIVE_MIN_S = 0.05
# A turn the IMU cannot confirm in this long ends anyway: a wheel stuck on a
# chair leg must not spin for ever.
TURN_TIMEOUT_S = 3.0
# After the stop the chassis coasts and the gyro filter catches up. Keep
# measuring this long so the reported angle is the final one.
SETTLE_S = 0.35
# Stop a turn this many seconds of current rotation before the target.
# Learned from each IMU-measured turn's actual coast.
COAST_LEAD_S = 0.06
COAST_LEAD_MAX_S = 0.25
# Pivot rate assumed for a turn without the IMU (no measurement yet). The
# operator's turns on 2026-10-08 implied ~130-155 deg/s at manual pivot PWM.
TURN_RATE_GUESS_DPS = 140.0
# Hold the Uno link for this long past every write; the control loop's own
# commands are ignored until then.
CLAIM_S = 0.2


@dataclass(frozen=True)
class MoveRequest:
    kind: str          # "turn" or "drive"
    direction: str     # L/R for a turn, F/B for a drive
    amount: float      # degrees for a turn, seconds for a drive
    magnitude: float   # 0..1 of the manual PWM range
    move_id: str


def parse_move(payload: object) -> MoveRequest | None:
    """A validated move from a dashboard/pilot message, or None."""
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    direction = str(payload.get("dir", "")).upper()
    try:
        amount = float(payload.get("amount"))
        magnitude = float(payload.get("mag", 0.0))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(amount) and math.isfinite(magnitude)):
        return None
    magnitude = min(1.0, max(0.0, magnitude))
    if kind == "turn" and direction in ("L", "R"):
        amount = min(TURN_MAX_DEG, max(TURN_MIN_DEG, amount))
    elif kind == "drive" and direction in ("F", "B"):
        amount = min(DRIVE_MAX_S, max(DRIVE_MIN_S, amount))
    else:
        return None
    return MoveRequest(kind, direction, amount, magnitude, str(payload.get("id", ""))[:40])


def _wrap(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


class ManualMoveExecutor:
    """Runs one precise move at a time; see the module docstring."""

    def __init__(
        self,
        control,
        link,
        imu,
        wheels_for: Callable[[str, float], tuple[int, int]],
        clock: Callable[[], float] = time.monotonic,
        start_thread: bool = True,
    ) -> None:
        self._control = control
        self._link = link
        self._imu = imu
        self._wheels_for = wheels_for
        self._clock = clock
        self._lock = threading.Lock()
        self._move: MoveRequest | None = None
        self._phase = "IDLE"
        self._output = (0, 0)
        self._started = 0.0
        self._stopped_at: float | None = None
        self._note = ""
        self._yaw_last: float | None = None
        self._signed = 0.0
        self._turned = 0.0
        self._imu_used = False
        self._turned_at_stop = 0.0
        self._rate_at_stop = 0.0
        self.coast_lead_s = COAST_LEAD_S
        self.turn_rate_dps = TURN_RATE_GUESS_DPS
        self._stop = threading.Event()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, name="manual-move", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ state

    @property
    def active(self) -> bool:
        with self._lock:
            return self._phase != "IDLE"

    @property
    def output(self) -> tuple[int, int]:
        with self._lock:
            return self._output

    @property
    def label(self) -> str:
        with self._lock:
            move = self._move
            if move is None:
                return "MOVE"
            amount = f"{move.amount:.0f}deg" if move.kind == "turn" else f"{move.amount:.2f}s"
            return f"MOVE {move.direction} {amount}" + (" settling" if self._phase == "SETTLING" else "")

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        while not self._stop.is_set():
            started = self._clock()
            try:
                self.step(started)
            except Exception as exc:  # never leave the wheels driving
                print(f"Manual move error: {exc}")
                self._abandon(f"stopped: internal error ({exc})")
            self._stop.wait(max(0.0, MOVE_PERIOD_S - (self._clock() - started)))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._abandon("stopped: robot shutting down")

    def _imu_state(self):
        if self._imu is None:
            return None
        state = self._imu.state()
        return state if state.fresh and state.calibrated else None

    def step(self, now: float) -> None:
        with self._lock:
            phase = self._phase
        if phase == "IDLE":
            request = self._control.take_move()
            if request is not None:
                self._begin(request, now)
            return
        if phase == "RUNNING":
            self._drive_step(now)
        else:
            self._settle_step(now)

    def _begin(self, request: MoveRequest, now: float) -> None:
        refusal = None
        if self._control.move_cancelled():
            refusal = "refused: Manual Control is off or STOP is pressed"
        elif not getattr(self._link, "differential_ready", False):
            refusal = "refused: the Arduino is not ready"
        if refusal:
            self._control.finish_move(self._result(request, "refused", 0.0, 0.0, False, refusal))
            return
        imu = self._imu_state()
        command = request.direction
        left, right = self._wheels_for(command, request.magnitude)
        with self._lock:
            self._move = request
            self._phase = "RUNNING"
            self._output = (left, right)
            self._started = now
            self._stopped_at = None
            self._note = ""
            self._yaw_last = None if imu is None else imu.yaw_deg
            self._turned = 0.0
            self._signed = 0.0
            self._imu_used = imu is not None
            self._turned_at_stop = 0.0
            self._rate_at_stop = 0.0
        self._write(now)

    def _measure(self) -> float:
        """Rotation rate (deg/s) now; accumulates this turn's |yaw| change."""
        imu = self._imu_state()
        if imu is None or self._yaw_last is None:
            # Lost mid-turn: finish on time instead (see _drive_step).
            if self._phase == "RUNNING":
                self._imu_used = False
            return 0.0
        # Magnitude only: the commanded direction is known, and an inverted
        # mount sign must not turn "reached" into "never reached".
        self._signed += _wrap(imu.yaw_deg - self._yaw_last)
        self._yaw_last = imu.yaw_deg
        self._turned = abs(self._signed)
        return abs(imu.gyro_z_dps)

    def _drive_step(self, now: float) -> None:
        move = self._move
        elapsed = now - self._started
        rate = self._measure() if move.kind == "turn" else 0.0
        reason = None
        if self._control.move_cancelled():
            reason = "stopped: STOP, a manual command or Manual Control off"
        elif move.kind == "drive":
            if elapsed >= move.amount:
                reason = "done"
            elif move.direction == "F" and getattr(self._link.status(), "blocked", False):
                reason = "stopped: the Arduino's 18 cm ultrasonic stop"
        elif self._imu_used:
            if self._turned + rate * self.coast_lead_s >= move.amount:
                reason = "done"
            elif elapsed >= TURN_TIMEOUT_S:
                reason = "stopped: the turn did not reach its angle in time (blocked?)"
        elif elapsed >= move.amount / max(30.0, self.turn_rate_dps):
            reason = "done"
        if reason is None:
            self._write(now)
            return
        with self._lock:
            self._phase = "SETTLING"
            self._output = (0, 0)
            self._stopped_at = now
            self._note = "" if reason == "done" else reason
            self._turned_at_stop = self._turned
            self._rate_at_stop = rate
        self._write(now)

    def _settle_step(self, now: float) -> None:
        move = self._move
        if move.kind == "turn":
            self._measure()
        if now - self._stopped_at < SETTLE_S:
            self._write(now)
            return
        driven = self._stopped_at - self._started
        if move.kind == "turn" and self._imu_used:
            coast = self._turned - self._turned_at_stop
            if self._rate_at_stop >= 20.0 and not self._note:
                sample = min(COAST_LEAD_MAX_S, max(0.0, coast / self._rate_at_stop))
                self.coast_lead_s = 0.7 * self.coast_lead_s + 0.3 * sample
            if driven >= 0.15 and self._turned_at_stop >= 10.0:
                measured_rate = self._turned_at_stop / driven
                self.turn_rate_dps = 0.7 * self.turn_rate_dps + 0.3 * measured_rate
            turned = self._turned
        elif move.kind == "turn":
            turned = driven * self.turn_rate_dps
        else:
            turned = 0.0
        result = self._result(move, "stopped" if self._note else "done", turned, driven,
                              self._imu_used and move.kind == "turn", self._note)
        with self._lock:
            self._phase = "IDLE"
            self._move = None
            self._output = (0, 0)
        release = getattr(self._link, "release", None)
        if release is not None:
            release(self)
        self._control.finish_move(result)

    def _write(self, now: float) -> None:
        left, right = self.output
        claim = getattr(self._link, "claim", None)
        if claim is not None:
            claim(self, now + CLAIM_S)
        self._link.publish_drive(left, right, owner=self)

    def _abandon(self, note: str) -> None:
        with self._lock:
            move, phase = self._move, self._phase
            self._phase = "IDLE"
            self._move = None
            self._output = (0, 0)
        if phase == "IDLE":
            return
        try:
            self._link.publish_drive(0, 0, owner=self)
        finally:
            release = getattr(self._link, "release", None)
            if release is not None:
                release(self)
            self._control.finish_move(self._result(move, "stopped", 0.0, 0.0, False, note))

    @staticmethod
    def _result(move: MoveRequest, state: str, turned: float, driven: float,
                measured: bool, note: str) -> dict:
        result = {
            "id": move.move_id,
            "kind": move.kind,
            "dir": move.direction,
            "asked": round(move.amount, 2),
            "state": state,
            "drove_s": round(driven, 2),
            "note": note,
        }
        if move.kind == "turn":
            result["turned_deg"] = round(turned, 1)
            result["measured"] = measured
        return result
