#!/usr/bin/env bash
# Run from Python so checking out a new release cannot rewrite a running shell.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/robot_update.py"
