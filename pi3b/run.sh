#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
PRIMARY_MODEL="$ROOT/models/vehicle_efficientdet_lite0_int8.tflite"
FALLBACK_MODEL="$ROOT/models/vehicle_ssd_mobilenet_v1.tflite"

if [[ ! -x "$PYTHON" ]]; then
  echo "Pi runtime is not installed. Run ./install.sh first." >&2
  exit 1
fi
if [[ -f "$PRIMARY_MODEL" ]]; then
  MODEL="$PRIMARY_MODEL"
elif [[ -f "$FALLBACK_MODEL" ]]; then
  MODEL="$FALLBACK_MODEL"
  echo "EfficientDet is unavailable; using the SSD fallback." >&2
else
  echo "Missing both Pi detector models." >&2
  echo "Re-run install.sh to download the verified default model." >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/visionfsd_pi.py" \
  --model "$MODEL" --fallback-model "$FALLBACK_MODEL" "$@"
