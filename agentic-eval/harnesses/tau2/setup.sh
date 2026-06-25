#!/usr/bin/env bash
# agentic-eval/harnesses/tau2/setup.sh -- idempotent installer for the tau2-bench harness.
#
# Creates a uv python-3.12 venv, clones tau2-bench at a PINNED commit, and installs it editable.
# Does NOT touch the GPU or require a live endpoint. Safe to re-run.
#
# tau2-bench: multi-turn tool-use benchmark with a SEPARATE user-simulator LLM.
#   upstream: https://github.com/sierra-research/tau2-bench  (the older tau-bench redirects here)
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/src/tau2-bench"
VENV="$HERE/.venv"
PY="$VENV/bin/python"

# PINNED upstream commit (recorded in README). Repeatability is the whole point.
TAU2_REPO="https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT="8ebb7499622fc2be9b9d510d6f7a7653461f4f29"   # 2026-06-22

echo "[tau2-setup] PATH ok: $(command -v uv)"

# --- venv (python 3.12; tau2 requires >=3.12,<3.14) ---------------------------------------------
if [ ! -x "$PY" ]; then
  echo "[tau2-setup] creating uv venv (python 3.12) at $VENV"
  uv venv --python 3.12 "$VENV" || { echo "[tau2-setup] FAILED to create venv" >&2; exit 1; }
else
  echo "[tau2-setup] venv exists: $VENV"
fi

# --- clone tau2-bench at the pinned commit ------------------------------------------------------
mkdir -p "$HERE/src"
if [ ! -d "$SRC_DIR/.git" ]; then
  echo "[tau2-setup] cloning $TAU2_REPO"
  git clone "$TAU2_REPO" "$SRC_DIR" || { echo "[tau2-setup] FAILED to clone" >&2; exit 1; }
fi
echo "[tau2-setup] pinning to $TAU2_COMMIT"
git -C "$SRC_DIR" fetch --quiet origin "$TAU2_COMMIT" 2>/dev/null || git -C "$SRC_DIR" fetch --quiet --all
git -C "$SRC_DIR" checkout --quiet "$TAU2_COMMIT" || { echo "[tau2-setup] FAILED to checkout pinned commit" >&2; exit 1; }

# --- install tau2 editable into the venv --------------------------------------------------------
# (-e so the bundled domain task data under src/tau2-bench/data/ is used in-place via TAU2_DATA_DIR.)
echo "[tau2-setup] installing tau2 (editable) into venv"
uv pip install --python "$PY" -e "$SRC_DIR" || { echo "[tau2-setup] FAILED to install tau2" >&2; exit 1; }

# --- verify the CLI + flags resolve (offline; no endpoint needed) -------------------------------
TAU2_DATA_DIR="$SRC_DIR/data" "$VENV/bin/tau2" --help >/dev/null 2>&1 \
  || { echo "[tau2-setup] WARN: 'tau2 --help' did not resolve" >&2; }
RUN_HELP="$(TAU2_DATA_DIR="$SRC_DIR/data" "$VENV/bin/tau2" run --help 2>&1 || true)"
for flag in --agent-llm --agent-llm-args --user-llm --user-llm-args --num-tasks --num-trials --task-ids; do
  if printf '%s' "$RUN_HELP" | grep -q -- "$flag"; then
    echo "[tau2-setup] flag confirmed: $flag"
  else
    echo "[tau2-setup] WARN: flag NOT found in 'tau2 run --help': $flag" >&2
  fi
done

VER="$("$PY" -c 'import tau2, importlib.metadata as m; print(m.version("tau2"))' 2>/dev/null || echo unknown)"
echo "DONE  tau2 version=$VER  pinned_commit=$TAU2_COMMIT"
