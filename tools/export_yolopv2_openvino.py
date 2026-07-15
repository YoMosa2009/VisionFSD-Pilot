"""Export only YOLOPv2's learned road and lane heads to FP16 OpenVINO IR."""

from __future__ import annotations

from pathlib import Path

import openvino as ov
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "yolopv2" / "yolopv2.pt"
OUTPUT = ROOT / "models" / "yolopv2" / "openvino_fp16" / "yolopv2_road.xml"


class RoadHeads(torch.nn.Module):
    """Prune the detector output; YOLO11 remains the application's object model."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _detector, drivable_area, lane_lines = self.model(image)
        return drivable_area, lane_lines


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(SOURCE), map_location="cpu").eval()
    wrapper = RoadHeads(model).eval()
    example = torch.zeros((1, 3, 384, 640), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, strict=False)
        converted = ov.convert_model(traced, example_input=example)
    ov.save_model(converted, OUTPUT, compress_to_fp16=True)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
