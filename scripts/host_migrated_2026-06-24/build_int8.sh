#!/bin/bash
set -uo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
export CCACHE_DIR=/mnt/vm_8tb/b70/.ccache
export TMPDIR=/mnt/vm_8tb/b70/.tmp
export TORCH_EXTENSIONS_DIR=/mnt/vm_8tb/b70/.torch_ext
export PIP_CACHE_DIR=/mnt/vm_8tb/b70/.pipcache
export MAX_JOBS=24
mkdir -p "$CCACHE_DIR" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$PIP_CACHE_DIR"
cd /src
echo "=== START $(date) ==="
pip install --no-build-isolation -e . -v
RC=$?
echo "=== PIP_RC=$RC $(date) ==="
