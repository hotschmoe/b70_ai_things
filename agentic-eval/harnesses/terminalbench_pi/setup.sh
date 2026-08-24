#!/usr/bin/env bash
# Install the pinned local Harbor runner. Task environments and Pi are installed
# by Harbor inside each benchmark container when the job runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
HARBOR_VERSION="${HARBOR_VERSION:-0.22.0}"

command -v uv >/dev/null 2>&1 || { echo "setup: uv is required" >&2; exit 1; }
uv venv --python python3 "$VENV"
uv pip install --python "$VENV/bin/python" "harbor==$HARBOR_VERSION"
"$VENV/bin/harbor" --version
