#!/usr/bin/env bash
# Update an existing Pi deployment without re-imaging or reconfiguring it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${1:-main}"
PI_ROOT="$ROOT/pi3b"

if [[ ! -d "$ROOT/.git" ]]; then
  echo "Not a VisionFSD Pi installation: $ROOT" >&2
  exit 1
fi

git -C "$ROOT" fetch --depth 1 origin "$REF"
git -C "$ROOT" checkout --detach FETCH_HEAD
"$PI_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$PI_ROOT/.venv/bin/python" -m pip install -r "$PI_ROOT/requirements.txt"
chmod +x "$PI_ROOT/run.sh" "$PI_ROOT/update.sh"
echo "Updated VisionFSD Pi from $REF"
