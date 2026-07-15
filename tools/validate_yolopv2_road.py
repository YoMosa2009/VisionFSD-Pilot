"""Offline learned-road regression on the saved highway frame."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_perception import YOLOPv2RoadEngine  # noqa: E402
from visionfsd import RoadGeometryTracker, draw_camera_view  # noqa: E402


def main() -> None:
    frame_path = ROOT / "logs" / "youtube-source-frame.png"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(frame_path)
    engine = YOLOPv2RoadEngine(
        ROOT / "models" / "yolopv2" / "openvino_fp16" / "yolopv2_road.xml",
        "GPU",
    )
    result = engine.infer(frame, 1)
    geometry = RoadGeometryTracker().update(
        frame, result.lane_probability, result.drivable_mask,
    )
    overlay = draw_camera_view(frame, [], True, road_geometry=geometry)
    preview_path = ROOT / "logs" / "yolopv2-road-preview.png"
    cv2.imwrite(str(preview_path), overlay)
    fallback_geometry = RoadGeometryTracker().update(
        frame, np.zeros_like(result.lane_probability), result.drivable_mask,
    )
    fallback_overlay = draw_camera_view(frame, [], True, road_geometry=fallback_geometry)
    fallback_path = ROOT / "logs" / "yolopv2-drivable-fallback-preview.png"
    cv2.imwrite(str(fallback_path), fallback_overlay)
    crop_width = min(frame.shape[1], int(round(frame.shape[0] * 4.0 / 3.0)))
    crop_x = (frame.shape[1] - crop_width) // 2
    four_by_three = frame[:, crop_x:crop_x + crop_width]
    aspect_result = engine.infer(four_by_three, 2)
    aspect_geometry = RoadGeometryTracker().update(
        four_by_three, aspect_result.lane_probability, aspect_result.drivable_mask,
    )
    print({
        "inference_ms": round(result.inference_ms, 2),
        "lane_pixel_ratio": round(float(np.count_nonzero(result.lane_probability >= 0.32)) / result.lane_probability.size, 5),
        "drivable_pixel_ratio": round(float(np.count_nonzero(result.drivable_mask)) / result.drivable_mask.size, 5),
        "geometry": geometry.topology,
        "confidence": round(geometry.confidence, 3),
        "boundaries": geometry.left_points is not None and geometry.right_points is not None,
        "preview": str(preview_path),
        "fallback_source": fallback_geometry.source,
        "fallback_boundaries": fallback_geometry.left_points is not None and fallback_geometry.right_points is not None,
        "fallback_center_endpoints": (None if fallback_geometry.center_points is None else
                                      [fallback_geometry.center_points[0].tolist(),
                                       fallback_geometry.center_points[-1].tolist()]),
        "fallback_preview": str(fallback_path),
        "four_by_three_boundaries": (aspect_geometry.left_points is not None and
                                      aspect_geometry.right_points is not None),
        "four_by_three_source": aspect_geometry.source,
    })
    engine.close()


if __name__ == "__main__":
    main()
