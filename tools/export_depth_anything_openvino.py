"""Export Depth Anything V2 Small to FP16 OpenVINO IR for VisionFSD Pilot.

Requires: torch, openvino, transformers (installed into the project venv).

Usage:
  .venv\\Scripts\\python.exe tools\\export_depth_anything_openvino.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "models" / "depth_anything_v2_small" / "openvino_fp16"
OUTPUT_XML = OUTPUT_DIR / "depth_anything_v2_small.xml"
# Match DepthAnythingEngine input (multiples of 14).
INPUT_H = 252
INPUT_W = 336
HF_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class _DepthExportWrapper(nn.Module):
    """Thin wrapper so OpenVINO sees a single NCHW float tensor in/out."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # HF Depth Anything returns predicted_depth as (N, H, W).
        out = self.model(pixel_values=pixel_values)
        depth = out.predicted_depth
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        return depth


def main() -> int:
    try:
        from transformers import AutoModelForDepthEstimation
    except ImportError:
        print("Installing transformers (one-time for export)...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "transformers>=4.40,<5",
            "huggingface_hub>=0.23", "--quiet",
        ])
        from transformers import AutoModelForDepthEstimation

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {HF_ID} (CPU)...")
    model = AutoModelForDepthEstimation.from_pretrained(HF_ID)
    model.eval()
    wrapper = _DepthExportWrapper(model).eval()
    example = torch.zeros((1, 3, INPUT_H, INPUT_W), dtype=torch.float32)
    print(f"Tracing / converting at {INPUT_H}x{INPUT_W}...")
    with torch.inference_mode():
        # Prefer convert_model from the PyTorch model directly (OV 2024+).
        try:
            converted = ov.convert_model(wrapper, example_input=example)
        except Exception:
            traced = torch.jit.trace(wrapper, example, strict=False)
            converted = ov.convert_model(traced, example_input=example)
    ov.save_model(converted, str(OUTPUT_XML), compress_to_fp16=True)
    print(f"Saved {OUTPUT_XML}")
    # Quick sanity: compile and run once.
    core = ov.Core()
    compiled = core.compile_model(str(OUTPUT_XML), "CPU")
    dummy = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
    result = compiled([dummy])
    out = list(result.values())[0]
    print(f"Sanity output shape: {np.asarray(out).shape}")
    print("Depth Anything V2 Small OpenVINO export ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
