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

# The desktop autostart entry runs with no terminal, so anything printed here
# is lost and a failed start looks identical to "nothing happened".  Keep the
# last run's output on disk so it can be read afterwards.
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/robot.log"
if [[ -f "$LOG" ]]; then mv -f "$LOG" "$LOG_DIR/robot.previous.log"; fi

# USB devices are not always enumerated by the time the desktop session starts.
# Waiting briefly turns a boot-race failure into a normal start.
for _ in $(seq 1 20); do
  if compgen -G "/dev/ttyACM*" >/dev/null || compgen -G "/dev/ttyUSB*" >/dev/null; then break; fi
  sleep 0.5
done

{
  echo "=== VisionFSD robot start: $(date -Is) ==="
  echo "version: $(tr -d '\r\n' < "$ROOT/VERSION" 2>/dev/null)"
  echo "serial devices: $(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | tr '\n' ' ')"
} >> "$LOG"

ARGS=(
  "$ROOT/robot_autonomy.py"
  --model "$MODEL" --fallback-model "$FALLBACK_MODEL"
  --camera "${VISIONFSD_CAMERA:-0}"
  --standby-seconds "${VISIONFSD_STANDBY_SECONDS:-25}"
  --speed "${VISIONFSD_ROBOT_SPEED:-70}"
  --lidar-front-offset-deg "${VISIONFSD_LIDAR_FRONT_OFFSET_DEG:-0}"
  "$@"
)

# Run normally when a terminal is attached so errors are visible immediately;
# capture to the log only when there is nowhere for them to go.
if [[ -t 1 ]]; then
  exec "$PYTHON" "${ARGS[@]}"
fi
exec >>"$LOG" 2>&1
exec "$PYTHON" "${ARGS[@]}"
