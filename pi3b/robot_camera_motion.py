"""Bounded, non-neural motion cues from a forward USB camera.

These are image-motion observations, not metric depth or wheel odometry.
Only well-distributed, round-trip-consistent tracks contribute to navigation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionState:
    fresh: bool = False
    confidence: float = 0.0
    tracked_features: int = 0
    yaw_rate_dps: float | None = None
    translation_scale: float = 1.0
    motion_observed: bool = False
    captured_at: float = 0.0
    quality: str = "WARMUP"
    coverage: float = 0.0
    expansion_rate_s: float = 0.0
    processing_ms: float = 0.0


def estimate_motion(previous: np.ndarray, gray: np.ndarray, elapsed: float,
                    fov_deg: float, left_pwm: int, right_pwm: int,
                    captured_at: float) -> CameraMotionState:
    """Track at most 80 features on a 160x120 image; abstain on weak evidence."""
    def unknown(reason: str) -> CameraMotionState:
        return CameraMotionState(captured_at=captured_at, quality=reason)

    if not 0.035 <= elapsed <= 0.35 or previous.shape != gray.shape:
        return unknown("TIMING")
    features = cv2.goodFeaturesToTrack(
        previous, maxCorners=80, qualityLevel=0.025, minDistance=7, blockSize=7
    )
    if features is None or len(features) < 12:
        return unknown("LOW_TEXTURE")
    params = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))
    following, status, errors = cv2.calcOpticalFlowPyrLK(previous, gray, features, None, **params)
    if following is None or status is None:
        return unknown("TRACK_LOST")
    valid = status.reshape(-1) == 1
    valid &= np.all(np.isfinite(following.reshape(-1, 2)), axis=1)
    if errors is not None:
        valid &= errors.reshape(-1) < 24.0
    old, new = features.reshape(-1, 2)[valid], following.reshape(-1, 2)[valid]
    if len(old) < 12:
        return unknown("TRACK_LOST")
    # A one-way LK match can look convincing even after occlusion or blur.
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous, new.reshape(-1, 1, 2), None, **params
    )
    if back is None or back_status is None:
        return unknown("TRACK_LOST")
    consistent = ((back_status.reshape(-1) == 1)
                  & (np.linalg.norm(back.reshape(-1, 2) - old, axis=1) <= 0.8))
    old, new = old[consistent], new[consistent]
    if len(old) < 12:
        return unknown("INCONSISTENT")
    transform, inliers = cv2.estimateAffinePartial2D(
        old, new, method=cv2.RANSAC, ransacReprojThreshold=1.25,
        maxIters=80, confidence=0.95, refineIters=3,
    )
    if transform is None or inliers is None or not np.all(np.isfinite(transform)):
        return unknown("INCONSISTENT")
    keep = inliers.reshape(-1) != 0
    inlier_ratio = float(np.mean(keep))
    old, new = old[keep], new[keep]
    if len(old) < 12 or inlier_ratio < 0.65:
        return unknown("INCONSISTENT")
    height, width = gray.shape
    tiles = (np.clip((old[:, 0] * 4 / width).astype(int), 0, 3)
             + 4 * np.clip((old[:, 1] * 3 / height).astype(int), 0, 2))
    coverage = np.unique(tiles).size / 12.0
    if coverage < 0.5 or np.ptp(old[:, 0]) < width * 0.4 or np.ptp(old[:, 1]) < height * 0.35:
        return unknown("LOCAL_FEATURES")
    predicted = old @ transform[:, :2].T + transform[:, 2]
    residual = float(np.median(np.linalg.norm(new - predicted, axis=1)))
    confidence = float(np.clip(
        min(1.0, len(old) / 45.0) * inlier_ratio * min(1.0, coverage / 0.75)
        * max(0.0, 1.0 - residual / 1.5), 0.0, 1.0
    ))
    flow = new - old
    magnitude = float(np.median(np.linalg.norm(flow, axis=1)))
    moving = magnitude >= 0.35
    turning = abs(left_pwm - right_pwm) >= 18
    translating = left_pwm * right_pwm > 0
    # Pinhole bearing differences are a bounded yaw cue during commanded turns.
    # Unknown camera intrinsics/scene translation prevent absolute pose claims.
    focal = width / (2.0 * math.tan(math.radians(np.clip(fov_deg, 20., 140.)) / 2.0))
    bearing_change = (np.arctan((new[:, 0] - width / 2) / focal)
                      - np.arctan((old[:, 0] - width / 2) / focal))
    yaw_rate = (float(np.clip(-np.degrees(np.median(bearing_change)) / elapsed, -180., 180.))
                if turning and confidence >= 0.35 else None)
    scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
    expansion = math.log(max(scale, 1e-6)) / elapsed
    # Scene expansion is advisory only. Turning and weak matches abstain.
    expansion = (float(np.clip(expansion, 0., 4.))
                 if left_pwm > 0 and right_pwm > 0 and not turning and confidence >= 0.55 else 0.0)
    return CameraMotionState(
        fresh=True, confidence=confidence, tracked_features=len(old),
        yaw_rate_dps=yaw_rate,
        translation_scale=0.20 if translating and not moving and confidence >= 0.55 else 1.0,
        motion_observed=moving, captured_at=captured_at,
        quality="TRACKING" if confidence >= 0.35 else "WEAK",
        coverage=coverage, expansion_rate_s=expansion,
    )
