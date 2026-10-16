#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
[[ -n ${B70_AGENT:-} ]]
python3 "$HERE/audit.py"
bash "$HERE/decode.sh"
