#!/usr/bin/env bash
# HTTP client only. Run in the server owner's agreed window; never takes its GPU lease.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/promo_gate.py" "$@"
