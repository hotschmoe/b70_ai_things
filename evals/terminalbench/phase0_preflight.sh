#!/usr/bin/env bash
# Run the non-GPU H01-H03 campaign preflight against exact Pi 0.84.3.
set -euo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
HARBOR_PYTHON="${HARBOR_PYTHON:-/home/hotschmoe/.local/share/uv/tools/harbor/bin/python}"
PI_ROOT="${PI_ORACLE_ROOT:-$ROOT/tooling/pi-0.84.3}"
PI_BINARY="${PI_0843_BINARY:-$PI_ROOT/node_modules/.bin/pi}"

if [ ! -x "$PI_BINARY" ] || [ "$("$PI_BINARY" --version 2>/dev/null || true)" != 0.84.3 ]; then
  mkdir -p "$PI_ROOT"
  npm install --prefix "$PI_ROOT" --ignore-scripts --no-audit --no-fund \
    @earendil-works/pi-coding-agent@0.84.3
fi

PYTHONPATH="$REPO" "$HARBOR_PYTHON" -m unittest discover \
  -s "$REPO/evals/terminalbench/tests" -v
PYTHONPATH="$REPO" "$HARBOR_PYTHON" \
  "$REPO/evals/terminalbench/pi_payload_oracle.py" --pi "$PI_BINARY"
