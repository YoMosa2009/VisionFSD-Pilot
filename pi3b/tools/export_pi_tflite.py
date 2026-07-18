"""Export the intended Pi model on a desktop build machine, not on the Pi."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Pi3B TFLite vehicle detector")
    parser.add_argument("--weights", default="../../yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--output", default="../models/vehicle_yolo11n_320_int8.tflite")
    args = parser.parse_args()
    from ultralytics import YOLO

    here = Path(__file__).resolve().parent
    weights = (here / args.weights).resolve()
    output = (here / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = Path(YOLO(str(weights)).export(format="tflite", imgsz=args.imgsz, int8=True, nms=False))
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics did not produce an artifact: {exported}")
    shutil.copy2(exported, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
