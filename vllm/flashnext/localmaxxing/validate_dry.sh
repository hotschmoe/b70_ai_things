#!/usr/bin/env bash
# Validate a real local measurement at the dry-run API. Never submits a run.
# The CLI reads its own saved authentication; keys never enter this command line.
set -euo pipefail
test "$#" = 1 || { echo 'usage: validate_dry.sh /path/to/suite/payload.json' >&2; exit 2; }
python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
payload = Path(sys.argv[1])
validation = json.loads((payload.parent / 'validation.json').read_text())
if validation.get('valid') is not True or validation.get('submitted') is not False:
    raise SystemExit('Refusing validation: local measurement is invalid or status is unknown')
PY
LMX_BIN="${LMX_BIN:-$HOME/.local/bin/lmx}"
"$LMX_BIN" speed-test validate-local "$1"
exec "$LMX_BIN" speed-test dry-run "$1" --json
