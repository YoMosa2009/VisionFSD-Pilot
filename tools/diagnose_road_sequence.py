"""Sample a local driving clip and report why learned-road fitting succeeds or fails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_perception import YOLOPv2RoadEngine  # noqa: E402
from visionfsd import estimate_road_geometry_from_masks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=str(ROOT / "logs" / "highway-test-clip.mp4"))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--device", default="GPU")
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {args.source}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = np.linspace(0, max(0, frame_count - 1), max(1, args.samples)).astype(int)
    engine = YOLOPv2RoadEngine(
        ROOT / "models" / "yolopv2" / "openvino_fp16" / "yolopv2_road.xml",
        args.device,
    )
    records: list[dict[str, object]] = []
    try:
        for sequence, index in enumerate(indices, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            result = engine.infer(frame, sequence)
            geometry = estimate_road_geometry_from_masks(result.lane_probability, result.drivable_mask)
            lane = result.lane_probability
            records.append({
                "frame": int(index),
                "lane_max": round(float(np.max(lane)), 4),
                "lane_p99": round(float(np.percentile(lane, 99.0)), 4),
                "lane_p995": round(float(np.percentile(lane, 99.5)), 4),
                "lane_nonzero_ratio": round(float(np.count_nonzero(lane)) / lane.size, 5),
                "drivable_ratio": round(float(np.count_nonzero(result.drivable_mask)) / result.drivable_mask.size, 5),
                "fit": geometry.center_points is not None,
                "topology": geometry.topology,
                "confidence": round(geometry.confidence, 3),
                "inference_ms": round(result.inference_ms, 2),
            })
    finally:
        capture.release()
        engine.close()
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
