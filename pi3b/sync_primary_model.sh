#!/usr/bin/env bash
# Download and verify the Pi neural stack's primary detector atomically.
set -euo pipefail

PI_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODEL_URL="${VISIONFSD_PI_PRIMARY_MODEL_URL:-https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite}"
MODEL_SHA256="${VISIONFSD_PI_PRIMARY_MODEL_SHA256:-2E04C53BFEAC0AC2A30C057C7E2A777594CE39BAAAC35A92F74FB1E8C4FC4E0B}"
MODEL_PATH="$PI_ROOT/models/vehicle_efficientdet_lite0_int8.tflite"
TMP_MODEL="$PI_ROOT/models/.efficientdet_lite0.download"

if [[ "$MODEL_URL" != https://* ]]; then
  echo "The primary model URL must use HTTPS." >&2
  exit 2
fi
if [[ ! "$MODEL_SHA256" =~ ^[A-Fa-f0-9]{64}$ ]]; then
  echo "The primary model requires a 64-character SHA-256." >&2
  exit 2
fi

mkdir -p "$PI_ROOT/models"
if [[ -s "$MODEL_PATH" ]]; then
  current_hash="$(sha256sum "$MODEL_PATH" | awk '{print toupper($1)}')"
  if [[ "$current_hash" == "${MODEL_SHA256^^}" ]]; then
    echo "EfficientDet-Lite0 INT8 model already verified."
    exit 0
  fi
fi

trap 'rm -f "$TMP_MODEL"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 \
  --retry 3 --retry-delay 2 --output "$TMP_MODEL" "$MODEL_URL"
test -s "$TMP_MODEL"
actual_hash="$(sha256sum "$TMP_MODEL" | awk '{print toupper($1)}')"
if [[ "$actual_hash" != "${MODEL_SHA256^^}" ]]; then
  echo "Downloaded EfficientDet model failed SHA-256 verification." >&2
  exit 1
fi
mv -f "$TMP_MODEL" "$MODEL_PATH"
trap - EXIT
echo "Installed verified EfficientDet-Lite0 INT8 model: $MODEL_PATH"
