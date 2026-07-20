#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
MODEL="$ROOT/models/vehicle_ssd_mobilenet_v1.tflite"

if [[ ! -x "$PYTHON" ]]; then
  echo "Pi runtime is not installed. Run ./install.sh first." >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Missing vehicle model: $MODEL" >&2
  echo "Re-run install.sh to download the verified default model." >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/visionfsd_pi.py" --model "$MODEL" "$@"
