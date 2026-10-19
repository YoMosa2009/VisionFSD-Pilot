"""Run the occupancy map on its own thread, off the control loop.

The 2026-09-23 field run measured the map stage at 83-203 ms of every control
tick on the Pi, the second-largest cost in a loop running at 3 Hz against a
40 Hz design. Nothing on the safety path needs the map: the arc planner, the
clearance check and the emergency brake all work from the live scan and the
short-term obstacle memory. The map feeds pose, exploration and the view, all
of which can use a pose one scan old. So the control decision no longer waits
for it.

Two properties matter more than speed:

* **Dead reckoning is unchanged.** The map predicts motion from wheel
  commands with each step's elapsed time capped at 0.2 s. Every control tick
  submits its motion sample, and the worker replays them in order, so the
  map integrates exactly the sequence it always did. (At the measured 3 Hz
  the cap was silently discarding about 40% of commanded motion; a faster
  loop fixes that too.)
* **Scans land where they happened.** A scan is integrated after the motion
  of the tick it arrived on, not after later ticks the worker has queued. If
  the worker falls behind, older scans in a batch are skipped in favour of the
  newest, but no motion is.

The worker also owns what must stay in step with the map: recentre shifts go
to the explorer in order, and the explorer receives the map straight from the
worker. A pickup reset is a request applied before the next batch, and a
pre-reset map is never published after the reset was asked for.
"""

from __future__ import annotations

import collections
import contextlib
import threading
import time
from dataclasses import dataclass

from robot_slam_lite import LidarSlamLite, SlamLiteState


@dataclass(frozen=True)
class MotionSample:
    """One control tick's input to the map."""

    left_pwm: int
    right_pwm: int
    now: float
    imu_yaw_rate_dps: float | None = None
    camera_yaw_rate_dps: float | None = None
    camera_translation_scale: float = 1.0
    imu_yaw_deg: float | None = None
    points: object = None
    scan_stamp_hint: float | None = None


class MapWorker:
    #: Pending ticks held for the worker. At 40 Hz this is over six seconds;
    #: if the worker is that far behind, the oldest motion is dropped rather
    #: than letting memory grow.
    MAX_PENDING = 256

    def __init__(
        self,
        local_map: LidarSlamLite,
        explorer=None,
        threaded: bool = True,
    ) -> None:
        self.local_map = local_map
        self.explorer = explorer
        self.threaded = threaded
        self._map_lock = threading.Lock()
        self._queue_lock = threading.Condition()
        self._pending: collections.deque[MotionSample] = collections.deque()
        self._reset_requested = False
        self._force_publish = True
        self._state = local_map.state()
        self._stop = False
        self.processed = 0
        self.scans_skipped = 0
        self.samples_dropped = 0
        self.last_batch_ms = 0.0
        self._thread: threading.Thread | None = None
        if threaded:
            self._thread = threading.Thread(
                target=self._run, name="map-worker", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------ control side
    def submit(self, sample: MotionSample) -> None:
        with self._queue_lock:
            if len(self._pending) >= self.MAX_PENDING:
                self._pending.popleft()
                self.samples_dropped += 1
            self._pending.append(sample)
            self._queue_lock.notify()
        if not self.threaded:
            self._drain()

    def state(self) -> SlamLiteState:
        return self._state

    def request_reset(self) -> None:
        """Discard the map before the next batch; queued motion is stale too."""
        with self._queue_lock:
            self._reset_requested = True
            self._pending.clear()
            self._queue_lock.notify()
        if not self.threaded:
            self._drain()

    @contextlib.contextmanager
    def borrow(self, blocking: bool = False):
        """The map for reading, or None if the worker is using it.

        For the view and telemetry, which are advisory: rather than make the
        control thread wait out a map update, they skip a frame.
        """
        acquired = self._map_lock.acquire(blocking=blocking)
        try:
            yield self.local_map if acquired else None
        finally:
            if acquired:
                self._map_lock.release()

    def close(self) -> None:
        with self._queue_lock:
            self._stop = True
            self._queue_lock.notify()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------ worker side
    def _run(self) -> None:
        while True:
            with self._queue_lock:
                while not self._pending and not self._reset_requested and not self._stop:
                    self._queue_lock.wait(timeout=0.5)
                if self._stop:
                    return
            self._drain()

    def _take(self) -> tuple[list[MotionSample], bool]:
        with self._queue_lock:
            batch = list(self._pending)
            self._pending.clear()
            reset = self._reset_requested
            self._reset_requested = False
        return batch, reset

    def _drain(self) -> None:
        batch, reset = self._take()
        if not batch and not reset:
            return
        started = time.perf_counter()
        local_map = self.local_map
        with self._map_lock:
            if reset:
                # The control side invalidates the explorer itself, at the
                # moment the displacement is detected.
                local_map.reset()
                self._force_publish = True
            # Only the newest scan in a batch is integrated; skipping older
            # ones costs detail, never motion.
            last_scan = max(
                (index for index, item in enumerate(batch) if item.points is not None),
                default=-1,
            )
            self.scans_skipped += sum(
                1 for index, item in enumerate(batch)
                if item.points is not None and index != last_scan
            )
            state = None
            for index, item in enumerate(batch):
                # Read the counter directly: state() counts the whole observed
                # grid, which is not something to do once per tick.
                recentres = local_map._recenter_count
                local_map.integrate_motion(
                    item.left_pwm,
                    item.right_pwm,
                    item.now,
                    item.imu_yaw_rate_dps,
                    item.camera_yaw_rate_dps,
                    item.camera_translation_scale,
                    item.imu_yaw_deg,
                )
                if local_map._recenter_count != recentres:
                    # The grid scrolled. Move the goal and route by the same
                    # shift rather than discarding them: in a large room this
                    # happens every metre or two, and dropping the goal each
                    # time is what kept the robot re-choosing destinations.
                    if self.explorer is not None:
                        shift_rows, shift_cols = local_map.last_shift_cells
                        self.explorer.shift(
                            shift_rows, shift_cols, local_map.cells / local_map.metres
                        )
                    self._force_publish = True
                if index == last_scan:
                    state = local_map.integrate_scan(
                        item.points,
                        item.left_pwm,
                        item.right_pwm,
                        item.now,
                        item.imu_yaw_rate_dps,
                        item.scan_stamp_hint,
                    )
            if state is None:
                state = local_map.state()
            with self._queue_lock:
                # A reset asked for while this batch ran makes this map stale:
                # do not hand it to the planner.
                publish = not self._reset_requested
            if publish and self.explorer is not None and batch:
                self.explorer.publish(
                    local_map.grid,
                    local_map.observed,
                    local_map.visits,
                    local_map.x,
                    local_map.y,
                    local_map.heading,
                    local_map.metres,
                    state.map_updates,
                    batch[-1].now,
                    force=self._force_publish,
                )
                self._force_publish = False
            self._state = state
        self.processed += len(batch)
        self.last_batch_ms = (time.perf_counter() - started) * 1000.0
