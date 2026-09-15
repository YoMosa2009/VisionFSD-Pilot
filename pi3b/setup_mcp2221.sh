#!/usr/bin/env bash
# One-time persistent Linux setup for the MCP2221A USB-I2C adapter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="$ROOT/.mcp2221-system-v1"
RULE_FILE="/etc/udev/rules.d/99-visionfsd-mcp2221.rules"
BLACKLIST_FILE="/etc/modprobe.d/visionfsd-mcp2221.conf"

if [[ -f "$MARKER" ]]; then
  echo "MCP2221 system setup already complete."
  exit 0
fi

packages_missing=false
for package in libusb-1.0-0-dev libudev-dev; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null \
    | grep -q '^install ok installed$'; then
    packages_missing=true
  fi
done
if [[ "$packages_missing" == true ]]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends libusb-1.0-0-dev libudev-dev
fi

# Blinka uses hidraw directly.  Give the desktop robot user persistent access
# and prevent Linux's optional native driver from claiming the same USB HID
# interface first.  Both files are harmless on kernels without hid_mcp2221.
printf '%s\n' \
  'SUBSYSTEMS=="usb", ACTION=="add", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="00dd", MODE="0666"' \
  | sudo tee "$RULE_FILE" >/dev/null
printf '%s\n' 'blacklist hid_mcp2221' \
  | sudo tee "$BLACKLIST_FILE" >/dev/null

if lsmod | grep -q '^hid_mcp2221 '; then
  sudo modprobe -r hid_mcp2221 || true
fi
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb

touch "$MARKER"
echo "MCP2221 persistent USB setup complete; runtime auto-detection enabled."
