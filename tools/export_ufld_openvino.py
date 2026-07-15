"""Export Ultra-Fast-Lane-Detection (Tusimple ResNet18) to FP16 OpenVINO IR.

Downloads/uses models/ufld/tusimple_18.pth (Google Drive id 1WCYyur5ZaWczH15ecmeDowrW30xcLrCn)
and writes models/ufld/openvino_fp16/ufld_tusimple_18.xml

Usage:
  .venv\\Scripts\\python.exe tools\\export_ufld_openvino.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "ufld" / "tusimple_18.pth"
OUTPUT_DIR = ROOT / "models" / "ufld" / "openvino_fp16"
OUTPUT_XML = OUTPUT_DIR / "ufld_tusimple_18.xml"
INPUT_H, INPUT_W = 288, 800
# Tusimple: (griding_num+1, cls_num_per_lane, num_lanes)
CLS_DIM = (101, 56, 4)
GDRIVE_ID = "1WCYyur5ZaWczH15ecmeDowrW30xcLrCn"


class _ResNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = torchvision.models.resnet18(weights=None)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ParsingNetExport(nn.Module):
    """UFLD parsingNet without aux head — single NCHW tensor out (N,101,56,4)."""

    def __init__(self) -> None:
        super().__init__()
        self.cls_dim = CLS_DIM
        self.total_dim = int(np.prod(CLS_DIM))
        self.model = _ResNetBackbone()
        self.pool = nn.Conv2d(512, 8, 1)
        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.model(x)
        fea = self.pool(fea).reshape(fea.shape[0], 1800)
        return self.cls(fea).reshape(-1, *self.cls_dim)


def _ensure_weights() -> Path:
    if WEIGHTS.exists() and WEIGHTS.stat().st_size > 1_000_000:
        return WEIGHTS
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading tusimple_18.pth via gdown...")
    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "--quiet"])
        import gdown
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", str(WEIGHTS), quiet=False)
    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)
    return WEIGHTS


def _load_state(net: ParsingNetExport, path: Path) -> None:
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
        elif "state_dict" in raw:
            state = raw["state_dict"]
        else:
            state = raw
    else:
        state = raw
    cleaned = {}
    for key, value in state.items():
        name = key
        if name.startswith("module."):
            name = name[len("module."):]
        # Drop aux head weights if present.
        if name.startswith("aux_"):
            continue
        cleaned[name] = value
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    print(f"Loaded weights. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:8])


def main() -> int:
    weights = _ensure_weights()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    net = ParsingNetExport().eval()
    _load_state(net, weights)
    example = torch.zeros((1, 3, INPUT_H, INPUT_W), dtype=torch.float32)
    print(f"Converting UFLD Tusimple ResNet18 @ {INPUT_H}x{INPUT_W}...")
    with torch.inference_mode():
        try:
            converted = ov.convert_model(net, example_input=example)
        except Exception:
            traced = torch.jit.trace(net, example, strict=False)
            converted = ov.convert_model(traced, example_input=example)
    ov.save_model(converted, str(OUTPUT_XML), compress_to_fp16=True)
    print(f"Saved {OUTPUT_XML}")
    core = ov.Core()
    compiled = core.compile_model(str(OUTPUT_XML), "CPU")
    out = compiled([np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)])
    arr = np.asarray(list(out.values())[0])
    print(f"Sanity output shape: {arr.shape} (expect 1x101x56x4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
