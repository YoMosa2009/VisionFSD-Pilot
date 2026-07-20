#!/usr/bin/env bash
# Recover an older Pi installation whose local edits block update.sh.
set -euo pipefail

INSTALL_ROOT="${VISIONFSD_PI_HOME:-$HOME/visionfsd-pi}"
REF="codex/pi3b-runtime"

usage() {
  cat <<'EOF'
Usage: recover-update.sh [--dir PATH] [--ref GIT_REF]

Preserves tracked local edits in a Git stash, then installs the requested
VisionFSD Pi revision without relying on the updater currently on disk.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) INSTALL_ROOT="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "$INSTALL_ROOT/.git" ]]; then
  echo "Not a VisionFSD Pi installation: $INSTALL_ROOT" >&2
  exit 1
fi

PI_ROOT="$INSTALL_ROOT/pi3b"
old_requirements=""
if [[ -f "$PI_ROOT/requirements.txt" ]]; then
  old_requirements="$(sha256sum "$PI_ROOT/requirements.txt" | awk '{print $1}')"
fi

# Older releases could not switch revisions when any tracked file was edited.
# Preserve every tracked edit so checkout is safe; untracked models and the
# virtual environment are intentionally left in place.
if [[ -n "$(git -C "$INSTALL_ROOT" status --porcelain --untracked-files=no)" ]]; then
  backup_name="visionfsd-pi-recovery-$(date +%Y%m%d-%H%M%S)"
  git -C "$INSTALL_ROOT" stash push -m "$backup_name"
  echo "Backed up tracked local changes in git stash: $backup_name"
fi

git -C "$INSTALL_ROOT" fetch --depth 1 origin "$REF"
git -C "$INSTALL_ROOT" checkout --detach FETCH_HEAD

PI_ROOT="$INSTALL_ROOT/pi3b"
chmod +x \
  "$PI_ROOT/install.sh" \
  "$PI_ROOT/run.sh" \
  "$PI_ROOT/update.sh" \
  "$PI_ROOT/recover-update.sh"

if [[ ! -x "$PI_ROOT/.venv/bin/python" ]]; then
  echo "Pi virtual environment is missing. Re-run pi3b/install.sh." >&2
  exit 1
fi

new_requirements="$(sha256sum "$PI_ROOT/requirements.txt" | awk '{print $1}')"
dependencies_ok=true
if ! "$PI_ROOT/.venv/bin/python" -c \
  'import cv2, numpy; from ai_edge_litert.interpreter import Interpreter' \
  >/dev/null 2>&1; then
  dependencies_ok=false
fi

if [[ "$old_requirements" != "$new_requirements" || "$dependencies_ok" != true ]]; then
  "$PI_ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$PI_ROOT/.venv/bin/python" -m pip install -r "$PI_ROOT/requirements.txt"
else
  echo "Python requirements unchanged; skipping package download."
fi

printf '%s\n' "$REF" > "$PI_ROOT/.install-ref"
version="$(tr -d '\r\n' < "$PI_ROOT/VERSION")"
echo "Recovered VisionFSD Pi to v$version from $REF"
echo "Run: bash $PI_ROOT/run.sh --camera 0 --fps 25 --threads 3"
