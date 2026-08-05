#!/usr/bin/env bash
# Lightweight boot-time update check, run once by run_robot.sh before the
# robot autostarts. Deliberately conservative: a slow/absent network, local
# edits, or a failed update must never prevent the robot from launching on
# whatever is already installed. Set VISIONFSD_AUTO_UPDATE=0 to disable.
#
# Exit codes (read by run_robot.sh):
#   0  no update needed, or update skipped/failed and rolled back -- launch
#      the currently installed version as-is.
#   2  a new version was checked out successfully -- the caller should
#      re-exec itself so a fresh process picks up the new code.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_ROOT="$ROOT/pi3b"
REF_FILE="$PI_ROOT/.install-ref"
FETCH_TIMEOUT_S="${VISIONFSD_AUTO_UPDATE_TIMEOUT_S:-12}"

if [[ "${VISIONFSD_AUTO_UPDATE:-1}" == "0" ]]; then
  echo "auto-update: disabled (VISIONFSD_AUTO_UPDATE=0)"
  exit 0
fi
if [[ ! -d "$ROOT/.git" ]]; then
  echo "auto-update: not a git checkout, skipping" >&2
  exit 0
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "auto-update: 'timeout' utility missing, skipping to avoid an unbounded network wait" >&2
  exit 0
fi

if [[ ! -s "$REF_FILE" ]]; then
  echo "auto-update: no $REF_FILE, skipping (run update.sh once to set it)" >&2
  exit 0
fi
REF="$(<"$REF_FILE")"

# Installed Pi checkouts are normally read-only. Auto-update never stashes on
# a person's behalf without them asking for it; a dirty tree defers to a
# manually-run update.sh instead.
if ! git -C "$ROOT" diff --quiet -- pi3b robot/firmware/visionfsd_pi_autonomy 2>/dev/null; then
  echo "auto-update: local Pi edits present, skipping (run update.sh manually)"
  exit 0
fi

old_head="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
if ! timeout "$FETCH_TIMEOUT_S" git -C "$ROOT" fetch --depth 1 origin "$REF" 2>&1; then
  echo "auto-update: fetch failed or timed out after ${FETCH_TIMEOUT_S}s, continuing with installed version"
  exit 0
fi
new_head="$(git -C "$ROOT" rev-parse FETCH_HEAD 2>/dev/null || true)"
if [[ -z "$new_head" || "$new_head" == "$old_head" ]]; then
  echo "auto-update: already up to date (${old_head:-unknown})"
  exit 0
fi

echo "auto-update: ${old_head:-unknown} -> $new_head, applying"
if bash "$PI_ROOT/update.sh" "$REF"; then
  echo "auto-update: applied successfully"
  exit 2
fi

echo "auto-update: update.sh failed, rolling back to ${old_head:-previous commit}" >&2
if [[ -n "$old_head" ]]; then
  git -C "$ROOT" checkout --detach "$old_head" 2>&1
  git -C "$ROOT" sparse-checkout init --cone 2>&1
  git -C "$ROOT" sparse-checkout set pi3b robot/firmware/visionfsd_pi_autonomy 2>&1
  echo "auto-update: rolled back to ${old_head}"
else
  echo "auto-update: no previous commit recorded, leaving checkout as update.sh left it" >&2
fi
exit 0
