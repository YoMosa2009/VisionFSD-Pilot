#!/usr/bin/env bash
# Pi-only installer: never installs desktop VisionFSD dependencies.
set -euo pipefail

REPO_URL="https://github.com/YoMosa2009/VisionFSD-Pilot.git"
REF="main"
INSTALL_ROOT="${VISIONFSD_PI_HOME:-$HOME/visionfsd-pi}"
MODEL_URL="${VISIONFSD_PI_MODEL_URL:-https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite}"
MODEL_SHA256="${VISIONFSD_PI_MODEL_SHA256:-CBDECD08B44C5DEA3821F77C5468E2936ECFBF43CDE0795A2729FDB43401E58B}"
ROBOT_AUTOSTART=true

usage() {
  cat <<'EOF'
Usage: install.sh [--dir PATH] [--ref GIT_REF] [--no-robot-autostart]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) INSTALL_ROOT="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --no-robot-autostart) ROBOT_AUTOSTART=false; shift ;;
    --model-url) MODEL_URL="$2"; MODEL_SHA256=""; shift 2 ;;
    --model-sha256) MODEL_SHA256="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MODEL_URL" != https://* ]]; then
  echo "The model URL must use HTTPS." >&2
  exit 2
fi
if [[ ! "$MODEL_SHA256" =~ ^[A-Fa-f0-9]{64}$ ]]; then
  echo "A 64-character SHA-256 is required for a custom model URL." >&2
  exit 2
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "VisionFSD Pi requires 64-bit Raspberry Pi OS (aarch64)." >&2
  echo "Reflash the Pi 3B with Raspberry Pi OS (64-bit), then re-run this command." >&2
  exit 2
fi

configure_sparse_checkout() {
  # A Pi never needs Windows OpenVINO models, desktop source, CAD files, or
  # generated tooling.  Keep only the Pi runtime and the exact Uno sketch it
  # communicates with.  This also trims old full checkouts during update.
  git -C "$INSTALL_ROOT" sparse-checkout init --cone
  git -C "$INSTALL_ROOT" sparse-checkout set pi3b robot/firmware/visionfsd_pi_autonomy
}

clone_sparse_checkout() {
  git clone --depth 1 --filter=blob:none --sparse --branch "$REF" "$REPO_URL" "$INSTALL_ROOT"
}

preserve_and_reclone() {
  local backup_root="${INSTALL_ROOT}.git-recovery-$(date +%Y%m%d-%H%M%S)"
  # Never delete a user's models, logs, virtual environment, or local edits.
  # A damaged shallow/partial Git database cannot safely be repaired in place.
  mv "$INSTALL_ROOT" "$backup_root"
  echo "Preserved the incomplete Pi checkout at: $backup_root"
  clone_sparse_checkout
}

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git python3 python3-venv python3-pip python3-opencv curl
if command -v raspi-config >/dev/null 2>&1; then
  sudo raspi-config nonint do_i2c 0
fi

if [[ -e "$INSTALL_ROOT/.git" ]]; then
  if ! git -C "$INSTALL_ROOT" fetch --depth 1 origin "$REF" \
    || ! git -C "$INSTALL_ROOT" checkout --detach FETCH_HEAD; then
    echo "Existing Pi checkout has incomplete Git objects; rebuilding a clean Pi-only checkout."
    preserve_and_reclone
  fi
else
  clone_sparse_checkout
fi
configure_sparse_checkout

PI_ROOT="$INSTALL_ROOT/pi3b"
# Install persistent MCP2221 permissions and native build prerequisites before
# pip may need to build hidapi. A marker makes this a one-time system step.
bash "$PI_ROOT/setup_mcp2221.sh"
python3 -m venv --system-site-packages "$PI_ROOT/.venv"
"$PI_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$PI_ROOT/.venv/bin/python" -m pip install -r "$PI_ROOT/requirements.txt"
printf '%s\n' "$REF" > "$PI_ROOT/.install-ref"

mkdir -p "$PI_ROOT/models" "$PI_ROOT/logs"
TMP_MODEL="$PI_ROOT/models/.vehicle_model.download"
curl --fail --location --proto '=https' --tlsv1.2 --output "$TMP_MODEL" "$MODEL_URL"
test -s "$TMP_MODEL"
mv -f "$TMP_MODEL" "$PI_ROOT/models/vehicle_yolo11n_320_int8.tflite"
actual_hash="$(sha256sum "$PI_ROOT/models/vehicle_yolo11n_320_int8.tflite" | awk '{print toupper($1)}')"
if [[ "$actual_hash" != "${MODEL_SHA256^^}" ]]; then
  rm -f "$PI_ROOT/models/vehicle_yolo11n_320_int8.tflite"
  echo "Downloaded model failed SHA-256 verification." >&2
  exit 1
fi
mv -f "$PI_ROOT/models/vehicle_yolo11n_320_int8.tflite" "$PI_ROOT/models/vehicle_ssd_mobilenet_v1.tflite"

chmod +x \
  "$PI_ROOT/install.sh" \
  "$PI_ROOT/run.sh" \
  "$PI_ROOT/update.sh" \
  "$PI_ROOT/recover-update.sh" \
  "$PI_ROOT/sync_primary_model.sh" \
  "$PI_ROOT/run_lidar.sh" \
  "$PI_ROOT/run_robot.sh" \
  "$PI_ROOT/setup_mcp2221.sh"
bash "$PI_ROOT/sync_primary_model.sh" "$PI_ROOT"

if [[ "$ROBOT_AUTOSTART" == true ]]; then
  autostart_dir="$HOME/.config/autostart"
  mkdir -p "$autostart_dir"
  sed "s|__VISIONFSD_RUN_ROBOT__|$PI_ROOT/run_robot.sh|" \
    "$PI_ROOT/visionfsd-robot.desktop" > "$autostart_dir/visionfsd-robot.desktop"
  echo "Enabled Pi Desktop autostart: $autostart_dir/visionfsd-robot.desktop"
fi
version="$(tr -d '\r\n' < "$PI_ROOT/VERSION")"
echo "Installed at $PI_ROOT"
echo "VisionFSD Pi version: $version"
echo "Run: $PI_ROOT/run.sh --camera 0 --fps 25 --threads 3"
echo "Robot: flash $INSTALL_ROOT/robot/firmware/visionfsd_pi_autonomy/visionfsd_pi_autonomy.ino to the Uno, then run $PI_ROOT/run_robot.sh"
echo "Update later: bash $PI_ROOT/update.sh"
