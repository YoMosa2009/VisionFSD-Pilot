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

# The desktop autostart entry runs with no terminal, so anything printed here
# is lost and a failed start is indistinguishable from "nothing happened".
# Keep the last two runs on disk so a boot failure can be read afterwards.
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
if ! command -v flock >/dev/null 2>&1; then
  echo "Required single-instance utility is missing: flock" >&2
  exit 1
fi
# Raspberry Pi desktop releases may execute both XDG and compositor autostart
# entries.  Only one process may own the Uno, LD19, and webcam.
exec 9>"$LOG_DIR/robot.lock"
if ! flock -n 9; then
  exit 0
fi
LOG="$LOG_DIR/robot.log"
# Skipped on the re-exec below (see auto-update block just after) so this
# rotation reflects the previous boot's run, not the few auto-update log
# lines the pre-re-exec process just wrote.
if [[ "${VISIONFSD_POST_UPDATE_REEXEC:-0}" != "1" && -f "$LOG" ]]; then
  mv -f "$LOG" "$LOG_DIR/robot.previous.log"
fi

# Check for a newer installed version before autostarting, so the robot stays
# current without a manual update.sh run. Bounded and fail-safe: see
# auto_update.sh for what "fail-safe" covers (offline, dirty tree, a failed
# update). Skipped on the re-exec below so this can only run once per boot.
if [[ "${VISIONFSD_POST_UPDATE_REEXEC:-0}" != "1" ]]; then
  {
    echo "=== VisionFSD auto-update: $(date -Is) ==="
    set +e
    bash "$ROOT/auto_update.sh"
    auto_update_status=$?
    set -e
    echo "auto-update exit status: $auto_update_status"
  } >>"$LOG" 2>&1
  if [[ "${auto_update_status:-0}" -eq 2 ]]; then
    echo "auto-update applied a new version; restarting run_robot.sh" >>"$LOG"
    export VISIONFSD_POST_UPDATE_REEXEC=1
    exec "$ROOT/run_robot.sh" "$@"
  fi
fi

# USB devices are not always enumerated by the time the desktop session starts,
# and the runtime exits when it cannot find the Uno.  Waiting turns a boot race
# into a normal start instead of a silent failure.
for _ in $(seq 1 30); do
  if compgen -G "/dev/ttyACM*" >/dev/null && compgen -G "/dev/ttyUSB*" >/dev/null; then break; fi
  sleep 0.5
done

# OpenCV's Qt Wayland backend can ignore fullscreen requests made by an
# autostarted application. Raspberry Pi OS exposes XWayland as DISPLAY=:0, so
# use Qt's X11 backend where the fullscreen window property is reliable.
if [[ -n "${DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi

{
  echo "=== VisionFSD robot start: $(date -Is) ==="
  echo "version: $(tr -d '\r\n' < "$ROOT/VERSION" 2>/dev/null)"
  echo "serial: $(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | tr '\n' ' ')"
  echo "display: DISPLAY=${DISPLAY:-unset} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-unset} QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-unset}"
  if command -v vcgencmd >/dev/null 2>&1; then
    echo "power: $(vcgencmd get_throttled 2>/dev/null || echo unavailable)"
  fi
} >> "$LOG"

# 118 is a cautious clear-space ceiling.  The planner scales down only to the
# loaded-wheel movement floor, never into the buzzing/no-motion PWM band.
ARGS=(
  "$ROOT/robot_autonomy.py"
  --camera "${VISIONFSD_CAMERA:-auto}"
  --standby-seconds "${VISIONFSD_STANDBY_SECONDS:-25}"
  --speed "${VISIONFSD_ROBOT_SPEED:-118}"
  --lidar-front-offset-deg "${VISIONFSD_LIDAR_FRONT_OFFSET_DEG:-0}"
  --imu-mount-yaw-deg "${VISIONFSD_IMU_MOUNT_YAW_DEG:-180}"
)

# Automatic USB identity discovery is the stable default because Linux ttyACM
# numbers can change after a USB reset. Explicit overrides remain available for
# nonstandard hardware without forcing this robot to a stale device node.
if [[ -n "${VISIONFSD_ARDUINO_PORT:-}" ]]; then
  ARGS+=(--arduino-port "$VISIONFSD_ARDUINO_PORT")
fi
if [[ -n "${VISIONFSD_LIDAR_PORT:-}" ]]; then
  ARGS+=(--lidar-port "$VISIONFSD_LIDAR_PORT")
fi
ARGS+=("$@")
unset VISIONFSD_POST_UPDATE_REEXEC

# With a terminal attached, print straight to it so errors are visible now.
if [[ -t 1 ]]; then
  exec "$PYTHON" -u "${ARGS[@]}"
fi
exec >>"$LOG" 2>&1
exec "$PYTHON" -u "${ARGS[@]}"
