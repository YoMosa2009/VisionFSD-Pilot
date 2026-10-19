"""Loop timing and advisory-work budgeting for the Pi 3B control loop.

The field run of 2026-09-21 measured the control loop at roughly 1 Hz against
its 25 ms design period: median gap 484-889 ms, worst 1.9 s. Everything else
observed that day followed from it - 663 expired command leases and the
resulting stop/go pulsing, a frontier planner that timed out in 62% of samples,
and a robot that circulated inside one room because it had a destination only
3.5% of the time. See FIELD_REPORT_2026-09-21.md.

Two things are needed to fix that, and both live here.

**Measurement.** Nobody knew which part of the loop was slow, because nothing
timed the parts. ``StageTimer`` costs a perf_counter call per stage and
summarises once a second, so the next run reports where the time actually goes
instead of inviting another guess.

**Priority.** Drawing the HDMI panel, encoding the map and serialising every
LiDAR return for the phone are *advisory*: they inform a person, they never
decide where the robot drives. They also run on the control thread, so when
they overrun, the safety decision waits behind them and the Uno command lease
expires. That is a priority inversion, and the honest fix is at the source:
``AdvisoryBudget`` sheds that work while the loop is late and restores it once
the loop catches up. Lengthening the lease would only have hidden it while the
robot kept acting on second-old decisions.

Neither class touches what the robot decides. They decide when optional work
is allowed to run.
"""

from __future__ import annotations

import contextlib
import time


class StageTimer:
    """Wall time per named loop stage, summarised over a window.

    Kept deliberately cheap: two clock reads and a dict update per stage, no
    allocation per tick, and a bounded set of names. A profiler that costs
    enough to change the thing it measures is worse than no profiler.
    """

    def __init__(self) -> None:
        self._total: dict[str, float] = {}
        self._worst: dict[str, float] = {}
        self._ticks = 0
        self._started_at = time.monotonic()

    @contextlib.contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.note(name, time.perf_counter() - started)

    def note(self, name: str, seconds: float) -> None:
        self._total[name] = self._total.get(name, 0.0) + seconds
        if seconds > self._worst.get(name, 0.0):
            self._worst[name] = seconds

    def tick(self) -> None:
        """Mark the end of one pass through the loop."""
        self._ticks += 1

    @property
    def ticks(self) -> int:
        return self._ticks

    def elapsed(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self._started_at

    def summary(self, now: float | None = None) -> dict[str, dict[str, float]]:
        """Mean and worst milliseconds per stage, busiest first."""
        ticks = max(1, self._ticks)
        rows = {
            name: {
                "mean_ms": round(total / ticks * 1000.0, 1),
                "max_ms": round(self._worst.get(name, 0.0) * 1000.0, 1),
            }
            for name, total in self._total.items()
        }
        return dict(
            sorted(rows.items(), key=lambda item: -item[1]["mean_ms"])
        )

    def format_line(self, now: float | None = None) -> str:
        elapsed = max(1e-6, self.elapsed(now))
        rate = self._ticks / elapsed
        parts = " ".join(
            f"{name}={row['mean_ms']:.1f}/{row['max_ms']:.1f}"
            for name, row in self.summary(now).items()
        )
        return (
            f"LOOP ticks={self._ticks} hz={rate:.1f} "
            f"period_ms={elapsed / max(1, self._ticks) * 1000.0:.0f} "
            f"mean/max_ms: {parts}"
        )

    def reset(self, now: float | None = None) -> None:
        self._total.clear()
        self._worst.clear()
        self._ticks = 0
        self._started_at = time.monotonic() if now is None else now


class AdvisoryBudget:
    """Permit optional work only while the control loop is keeping up.

    Hysteresis rather than a single threshold: shedding at the same number
    that restores would flap between rendering and not rendering every tick,
    which is its own kind of jitter. A forced pass every ``max_skip_s``
    guarantees the screen and the phone still refresh - a viewer must be able
    to tell a busy robot from a crashed one.
    """

    def __init__(
        self,
        late_ms: float = 150.0,
        recovered_ms: float = 80.0,
        max_skip_s: float = 2.0,
        smoothing: float = 0.3,
    ) -> None:
        self.late_ms = late_ms
        self.recovered_ms = recovered_ms
        self.max_skip_s = max_skip_s
        self.smoothing = smoothing
        self.average_gap_ms = 0.0
        self.behind = False
        self.skipped = 0
        self.forced = 0
        # None until the first pass runs; 0.0 would collide with a clock
        # that legitimately starts at zero.
        self._last_allowed_at: float | None = None

    def observe(self, gap_ms: float) -> None:
        if gap_ms <= 0.0:
            return
        if self.average_gap_ms == 0.0:
            self.average_gap_ms = gap_ms
        else:
            self.average_gap_ms += self.smoothing * (gap_ms - self.average_gap_ms)
        if self.behind:
            if self.average_gap_ms <= self.recovered_ms:
                self.behind = False
        elif self.average_gap_ms >= self.late_ms:
            self.behind = True

    def allows(self, now: float) -> bool:
        """True when advisory work may run on this tick."""
        if not self.behind:
            self._last_allowed_at = now
            return True
        if self._last_allowed_at is None:
            # Nothing has run yet, so there is nothing to be overdue against;
            # measuring against the clock's own zero would force a pass on the
            # first busy tick after start-up.
            self._last_allowed_at = now
            return True
        if now - self._last_allowed_at >= self.max_skip_s:
            self._last_allowed_at = now
            self.forced += 1
            return True
        self.skipped += 1
        return False

    def state(self) -> dict[str, float | bool | int]:
        return {
            "gap_ms": round(self.average_gap_ms, 1),
            "behind": self.behind,
            "skipped": self.skipped,
            "forced": self.forced,
        }
