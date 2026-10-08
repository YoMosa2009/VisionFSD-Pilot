#!/usr/bin/env bash
# Persistent setup plus bounded, noninteractive repair before robot startup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="$ROOT/.mcp2221-system-v1"
RULE_FILE="/etc/udev/rules.d/99-visionfsd-mcp2221.rules"
BLACKLIST_FILE="/etc/modprobe.d/visionfsd-mcp2221.conf"
REPAIR_ONLY=false
SUDO=(sudo)
if [[ "${1:-}" == "--repair" ]]; then
  REPAIR_ONLY=true
  SUDO=(sudo -n)
  # No password prompt, apt/network work or initramfs rebuild during boot.
  if ! "${SUDO[@]}" true; then
    echo "MCP2221 repair needs noninteractive sudo; run setup_mcp2221.sh manually." >&2
    exit 1
  fi
elif [[ $# -ne 0 ]]; then
  echo "Usage: setup_mcp2221.sh [--repair]" >&2
  exit 2
fi

if [[ "$REPAIR_ONLY" == false ]]; then
  packages_missing=false
  for package in libusb-1.0-0-dev libudev-dev; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null \
      | grep -q '^install ok installed$'; then
      packages_missing=true
    fi
  done
  if [[ "$packages_missing" == true ]]; then
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends libusb-1.0-0-dev libudev-dev
  fi
fi

# A marker records an earlier run; it cannot prove current rules or driver
# state. Always reconcile them, including hidraw nodes that already exist.
rules='SUBSYSTEMS=="usb", ACTION=="add", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="00dd", MODE="0666"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="04d8", ATTRS{idProduct}=="00dd", MODE="0666"'
if [[ ! -f "$RULE_FILE" ]] || [[ "$(cat "$RULE_FILE")" != "$rules" ]]; then
  printf '%s\n' "$rules" | "${SUDO[@]}" tee "$RULE_FILE" >/dev/null
fi
blacklist_changed=false
if [[ ! -f "$BLACKLIST_FILE" ]] || ! grep -qx 'blacklist hid_mcp2221' "$BLACKLIST_FILE"; then
  printf '%s\n' 'blacklist hid_mcp2221' | "${SUDO[@]}" tee "$BLACKLIST_FILE" >/dev/null
  blacklist_changed=true
fi
if lsmod | grep -q '^hid_mcp2221 '; then
  "${SUDO[@]}" modprobe -r hid_mcp2221
fi
"${SUDO[@]}" udevadm control --reload-rules
"${SUDO[@]}" udevadm trigger --action=add --subsystem-match=usb
"${SUDO[@]}" udevadm trigger --action=add --subsystem-match=hidraw
if [[ "$REPAIR_ONLY" == false && "$blacklist_changed" == true ]] && command -v update-initramfs >/dev/null 2>&1; then
  "${SUDO[@]}" update-initramfs -u
fi

touch "$MARKER"
echo "MCP2221 rules checked and driver conflict cleared; sensor connection still requires a successful identity read."
