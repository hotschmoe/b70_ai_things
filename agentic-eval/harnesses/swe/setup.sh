#!/usr/bin/env bash
# agentic-eval/harnesses/swe/setup.sh
# Idempotent installer for the SWE harness: mini-swe-agent (trajectory generator)
# + the official swebench grader (per-instance docker scoring).
#
# Creates a python-3.12 uv venv with PINNED versions, optionally pre-fetches the
# SWE-bench Verified dataset (cheap; ~a few MB of metadata, NOT the docker images),
# and prints DONE + the resolved versions. Does NOT need a GPU or the vLLM endpoint.
#
# Per HARNESS_CONTRACT.md: no system python is touched; the venv is gitignored
# (repo .gitignore covers harnesses/*/.venv/).
set -euo pipefail

# ---- pinned upstream versions (record these in README.md) -------------------
MINI_VER="2.4.2"      # mini-swe-agent (PyPI). bash/text agent, no tool-parser dependency.
SWEBENCH_VER="4.1.0"  # swebench official grader (PyPI). per-instance docker scoring.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "FATAL: uv not on PATH (expected ~/.local/bin/uv)"; exit 1; }

VENV="$HERE/.venv"
PY="$VENV/bin/python"

# 1) venv (idempotent: uv venv is a no-op-ish reuse; only (re)create if missing)
if [ ! -x "$PY" ]; then
  echo "[swe-setup] creating uv venv (python 3.12) at $VENV"
  uv venv --python 3.12 "$VENV"
fi

# 2) pinned installs. uv pip is idempotent (already-satisfied -> fast no-op).
echo "[swe-setup] installing mini-swe-agent==$MINI_VER swebench==$SWEBENCH_VER (pinned)"
uv pip install --python "$PY" \
  "mini-swe-agent==$MINI_VER" \
  "swebench==$SWEBENCH_VER"

# 3) sanity: both CLIs resolve (offline, no endpoint needed).
echo "[swe-setup] verifying CLIs resolve ..."
OPENAI_API_KEY=dummy "$VENV/bin/mini-extra" swebench --help >/dev/null
"$PY" -m swebench.harness.run_evaluation --help >/dev/null 2>&1
echo "[swe-setup] CLIs OK (mini-extra swebench + swebench.harness.run_evaluation)"

# 4) pre-fetch the SWE-bench Verified dataset metadata (cheap; cached under HF cache).
#    This is just the parquet of problem statements / gold patches, NOT the eval
#    docker images (those pull lazily at grade time). Best-effort: needs internet,
#    skip-tolerant so an offline box still finishes setup.
DATA_DIR="$HERE/data"
mkdir -p "$DATA_DIR"
export HF_HOME="${HF_HOME:-$DATA_DIR/hf}"
echo "[swe-setup] pre-fetching princeton-nlp/SWE-bench_Verified (test split) into $HF_HOME ..."
if "$PY" - <<'PY'
import sys
try:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"[swe-setup] dataset cached: {len(ds)} instances (expect ~500)")
except Exception as e:
    sys.stderr.write(f"[swe-setup] WARN: dataset prefetch skipped ({type(e).__name__}: {e})\n")
    sys.stderr.write("[swe-setup] (not fatal -- mini will fetch it at run time)\n")
PY
then :; fi

echo
echo "DONE"
echo "  mini-swe-agent == $("$PY" -c 'import importlib.metadata as m; print(m.version("mini-swe-agent"))')"
echo "  swebench       == $("$PY" -c 'import importlib.metadata as m; print(m.version("swebench"))')"
echo "  python         == $("$PY" -c 'import platform; print(platform.python_version())')"
echo "  venv           == $VENV"
