#!/usr/bin/env bash
# Only plans: lmx_vllm.py rejects --measure-local when --dry-run is present.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/lmx_vllm.py" --dry-run "$@"
