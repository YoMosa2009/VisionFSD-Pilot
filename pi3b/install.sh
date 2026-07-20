#!/usr/bin/env bash
# Pi-only installer: never installs desktop VisionFSD dependencies.
set -euo pipefail

REPO_URL="https://github.com/YoMosa2009/VisionFSD-Pilot.git"
REF="main"
INSTALL_ROOT="${VISIONFSD_PI_HOME:-$HOME/visionfsd-pi}"
MODEL_URL="${VISIONFSD_PI_MODEL_URL:-https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite}"
MODEL_SHA256="${VISIONFSD_PI_MODEL_SHA256:-CBDECD08B44C5DEA3821F77C5468E2936ECFBF43CDE0795A2729FDB43401E58B}"

usage() {
  cat <<'EOF'
Usage: install.sh [--dir PATH] [--ref GIT_REF] [--model-url HTTPS_URL --model-sha256 SHA256]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) INSTALL_ROOT="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
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

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git python3 python3-venv python3-pip python3-opencv curl

if [[ -e "$INSTALL_ROOT/.git" ]]; then
  git -C "$INSTALL_ROOT" fetch --depth 1 origin "$REF"
  git -C "$INSTALL_ROOT" checkout --detach FETCH_HEAD
else
  git clone --depth 1 --branch "$REF" "$REPO_URL" "$INSTALL_ROOT"
fi

PI_ROOT="$INSTALL_ROOT/pi3b"
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
  "$PI_ROOT/recover-update.sh"
version="$(tr -d '\r\n' < "$PI_ROOT/VERSION")"
echo "Installed at $PI_ROOT"
echo "VisionFSD Pi version: $version"
echo "Run: $PI_ROOT/run.sh --camera 0 --fps 25 --threads 3"
echo "Update later: bash $PI_ROOT/update.sh"
