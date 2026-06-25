#!/usr/bin/env bash
# agentic-eval/harnesses/bfcl/setup.sh
# Idempotent: build a python3.12 uv venv and install the PINNED BFCL harness + deps.
# The BFCL multi_turn dataset SHIPS INSIDE the pip package (bfcl_eval/data/BFCL_v4_multi_turn_*.json),
# so there is NO separate dataset download step. No GPU / no live endpoint needed to install.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv not found on PATH (~/.local/bin)"; exit 1; }

# Avoid hardlink warnings when cache + target are on different filesystems.
export UV_LINK_MODE=copy

# ---- pinned versions (recorded in README) -----------------------------------------------------
BFCL_VER="2026.3.23"     # bfcl-eval on PyPI
SOUNDFILE_VER="0.13.1"   # transitive-but-lazy dep of qwen-agent (BFCL imports qwen at startup)

# 1) venv. `uv venv` errors if .venv already exists, so only create when missing (idempotent).
if [ ! -x ".venv/bin/python" ]; then
  uv venv --python 3.12 .venv
else
  echo "venv already present, reusing .venv"
fi

PY=".venv/bin/python"

# 2) install BFCL harness, pinned. We deliberately do NOT install the oss-eval-vllm extra
#    (that pulls vllm==0.8.5 for LOCAL serving). We use --skip-server-setup against our OWN
#    already-running vLLM endpoint, so the heavy serving extra is unnecessary.
uv pip install --python "$PY" "bfcl-eval==${BFCL_VER}"

# 3) BFCL imports qwen_agent at CLI startup; qwen_agent imports soundfile lazily and crashes
#    the whole CLI if it is absent. Install it so `bfcl --help`/generate work.
uv pip install --python "$PY" "soundfile==${SOUNDFILE_VER}"

# 4) sanity: CLI imports and the multi_turn dataset is present in the package.
"$PY" - <<'PYCHECK'
import importlib.util, sys, pathlib
spec = importlib.util.find_spec("bfcl_eval")
root = pathlib.Path(spec.submodule_search_locations[0])
need = ["BFCL_v4_multi_turn_base.json","BFCL_v4_multi_turn_miss_func.json",
        "BFCL_v4_multi_turn_miss_param.json","BFCL_v4_multi_turn_long_context.json"]
missing = [n for n in need if not (root/"data"/n).exists()]
if missing:
    print("FATAL: missing bundled multi_turn data:", missing); sys.exit(1)
print("bundled multi_turn dataset OK (4 subcategories present)")
PYCHECK

# 5) verify the CLI actually starts (catches the soundfile/qwen import chain).
#    NB: `bfcl version` is broken upstream (it looks up metadata for "bfcl", but the dist is
#    "bfcl-eval", raising PackageNotFoundError), so we use `test-categories` for the liveness
#    probe -- it runs the SAME heavy import chain and exits 0.
.venv/bin/bfcl test-categories >/dev/null 2>&1 || { echo "FATAL: 'bfcl test-categories' failed (CLI import chain broken)"; exit 1; }

# Record the resolved version for the README / reproducibility (read from dist metadata).
RESOLVED="$("$PY" -c "import importlib.metadata as m; print(m.version('bfcl-eval'))" 2>/dev/null || echo "?")"

echo "DONE: bfcl-eval==${RESOLVED} (pinned ${BFCL_VER}) installed in $HERE/.venv ; multi_turn dataset bundled ; CLI OK"
