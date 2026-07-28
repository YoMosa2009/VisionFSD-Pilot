#!/usr/bin/env bash
# Pi-only robot launcher.  The Uno must be flashed with
# robot/firmware/visionfsd_pi_autonomy/visionfsd_pi_autonomy.ino first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Pi runtime is not installed. Run ./install.sh first." >&2
  exit 1
fi

PRIMARY_MODEL="$ROOT/models/vehicle_efficientdet_lite0_int8.tflite"
FALLBACK_MODEL="$ROOT/models/vehicle_ssd_mobilenet_v1.tflite"
MODEL="$PRIMARY_MODEL"
if [[ ! -f "$MODEL" ]]; then MODEL="$FALLBACK_MODEL"; fi
if [[ ! -f "$MODEL" ]]; then
  echo "No verified Pi detector model is installed. Re-run ./install.sh." >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/robot_autonomy.py" \
  --model "$MODEL" --fallback-model "$FALLBACK_MODEL" \
  --camera "${VISIONFSD_CAMERA:-0}" \
  --standby-seconds "${VISIONFSD_STANDBY_SECONDS:-25}" \
  --speed "${VISIONFSD_ROBOT_SPEED:-70}" \
  "$@"
