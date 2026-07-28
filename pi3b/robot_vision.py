#!/usr/bin/env python3
"""Camera-derived hazards that the LD19 structurally cannot see.

A 2D LiDAR measures one horizontal plane at its own mounting height.  A shoe, a
cable, a book, a door threshold, or a low step are all invisible to it and are
exactly what a small indoor robot drives into.  The camera can see them.

What it cannot do is measure them.  Without a calibrated camera height and tilt
there is no honest way to turn "dark blob low in the frame" into metres, so
this module never produces a distance, never produces a bearing to steer by,
and never enters the corridor geometry.  Its only output is a forward speed
cap, plus a flag on the dashboard so a misfire is visible rather than
mysterious.  That keeps the failure mode "drove a bit slowly for no reason"
instead of the earlier "stopped and would not move".

The floor reference is bootstrapped from the region directly in front of the
robot, which is the one part of the image that is floor whenever the robot is
driving safely.  A patterned rug will still upset it, hence the sustain
requirement and the disable flag.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised on the Pi, not in unit tests
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


ANALYSIS_WIDTH = 160
ANALYSIS_HEIGHT = 120


class LowObstacleGuard:
    """Flag floor-level clutter ahead and slow down for it."""

    def __init__(self, enabled: bool = True, sustain_frames: int = 3,
                 coverage_trigger: float = 0.16, slow_scale: float = 0.62) -> None:
        self.enabled = enabled
        self.sustain_frames = sustain_frames
        self.coverage_trigger = coverage_trigger
        self.slow_scale = slow_scale
        self.coverage = 0.0
        self.blocked = False
        self._streak = 0
        self._floor_lab: np.ndarray | None = None

    def _floor_reference(self, lab: np.ndarray) -> np.ndarray:
        """Median colour of the strip immediately in front of the robot."""
        height, width = lab.shape[:2]
        patch = lab[int(height * 0.86):, int(width * 0.32):int(width * 0.68)]
        sample = np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)
        if self._floor_lab is None:
            self._floor_lab = sample
        else:
            # Track slowly, so an obstacle filling the patch cannot instantly
            # redefine what "floor" looks like.
            self._floor_lab = 0.92 * self._floor_lab + 0.08 * sample
        return self._floor_lab

    def update(self, frame: np.ndarray | None) -> float:
        """Return the forward speed scale to apply (1.0 when nothing is seen)."""
        if not self.enabled or frame is None or cv2 is None:
            self.coverage, self.blocked, self._streak = 0.0, False, 0
            return 1.0
        small = cv2.resize(frame, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference = self._floor_reference(lab)
        # Chrominance separates a dark object from a shadow far better than
        # brightness does, so luminance is weighted down rather than ignored.
        difference = lab - reference[None, None, :]
        distance = np.sqrt(0.25 * difference[..., 0] ** 2
                           + difference[..., 1] ** 2
                           + difference[..., 2] ** 2)
        # Only the lower-central wedge matters: that is the floor the robot is
        # about to drive over, not the far wall or the ceiling.
        region = distance[int(ANALYSIS_HEIGHT * 0.60):, int(ANALYSIS_WIDTH * 0.25):int(ANALYSIS_WIDTH * 0.75)]
        mask = (region > 26.0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.coverage = float(np.count_nonzero(mask)) / float(mask.size)

        if self.coverage >= self.coverage_trigger:
            self._streak = min(self.sustain_frames, self._streak + 1)
        else:
            self._streak = max(0, self._streak - 1)
        self.blocked = self._streak >= self.sustain_frames
        return self.slow_scale if self.blocked else 1.0

    def annotate(self, panel: np.ndarray) -> None:
        if not self.enabled or cv2 is None:
            return
        height, width = panel.shape[:2]
        top = int(height * 0.60)
        left, right = int(width * 0.25), int(width * 0.75)
        colour = (60, 170, 255) if self.blocked else (90, 110, 120)
        cv2.rectangle(panel, (left, top), (right, height - 2), colour, 1, cv2.LINE_AA)
        if self.blocked:
            cv2.putText(panel, "LOW OBSTACLE", (left + 6, top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colour, 1, cv2.LINE_AA)
