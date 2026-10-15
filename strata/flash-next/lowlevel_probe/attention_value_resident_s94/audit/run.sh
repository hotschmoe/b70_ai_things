#!/usr/bin/env bash
# ROOT CPU-lease only after actual build; no Docker/GPU/ISA decode/build.
set -euo pipefail
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "$P/audit.py" "${1:?root actual frozen S94 main SHA256}" > "$P/audit.stdout" 2> "$P/audit.stderr"
