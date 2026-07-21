#!/usr/bin/env bash
# Update an existing Pi deployment without re-imaging or reconfiguring it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_ROOT="$ROOT/pi3b"
REF_FILE="$PI_ROOT/.install-ref"

if [[ $# -gt 0 ]]; then
  REF="$1"
elif [[ -s "$REF_FILE" ]]; then
  REF="$(<"$REF_FILE")"
else
  REF="main"
fi

if [[ ! -d "$ROOT/.git" ]]; then
  echo "Not a VisionFSD Pi installation: $ROOT" >&2
  exit 1
fi

old_requirements=""
if [[ -f "$PI_ROOT/requirements.txt" ]]; then
  old_requirements="$(sha256sum "$PI_ROOT/requirements.txt" | awk '{print $1}')"
fi

# Installed Pi checkouts are read-only in normal use, but preserve any tracked
# local edits instead of making checkout fail or deleting them.
if ! git -C "$ROOT" diff --quiet -- pi3b; then
  backup_name="visionfsd-pi-update-$(date +%Y%m%d-%H%M%S)"
  git -C "$ROOT" stash push -m "$backup_name" -- pi3b
  echo "Backed up local Pi changes in git stash: $backup_name"
fi

git -C "$ROOT" fetch --depth 1 origin "$REF"
git -C "$ROOT" checkout --detach FETCH_HEAD
chmod +x \
  "$PI_ROOT/install.sh" \
  "$PI_ROOT/run.sh" \
  "$PI_ROOT/update.sh" \
  "$PI_ROOT/recover-update.sh" \
  "$PI_ROOT/sync_primary_model.sh"

if [[ ! -x "$PI_ROOT/.venv/bin/python" ]]; then
  echo "Pi virtual environment is missing. Re-run pi3b/install.sh." >&2
  exit 1
fi

new_requirements="$(sha256sum "$PI_ROOT/requirements.txt" | awk '{print $1}')"
dependencies_ok=true
if ! "$PI_ROOT/.venv/bin/python" -c 'import cv2, numpy; from ai_edge_litert.interpreter import Interpreter' >/dev/null 2>&1; then
  dependencies_ok=false
fi
if [[ "$old_requirements" != "$new_requirements" || "$dependencies_ok" != true ]]; then
  "$PI_ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$PI_ROOT/.venv/bin/python" -m pip install -r "$PI_ROOT/requirements.txt"
else
  echo "Python requirements unchanged; skipping package download."
fi
bash "$PI_ROOT/sync_primary_model.sh" "$PI_ROOT"
printf '%s\n' "$REF" > "$REF_FILE"
version="$(tr -d '\r\n' < "$PI_ROOT/VERSION")"
echo "Updated VisionFSD Pi to v$version from $REF"
