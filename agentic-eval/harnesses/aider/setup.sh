#!/usr/bin/env bash
# agentic-eval/harnesses/aider/setup.sh -- idempotent setup for the Aider polyglot
# benchmark (single-shot codegen CONTROL).
#
# What it does (all idempotent; safe to re-run):
#   1. uv venv (python 3.12) at ./.venv with PINNED aider-chat + driver deps. This venv is
#      ONLY used to drive setup/dataset/parse on the HOST; the benchmark itself runs INSIDE
#      the aider benchmark Docker image (which bundles the 6 language toolchains).
#   2. Clone+pin the aider repo to v0.86.2 at ./vendor/aider (the benchmark/ code that the
#      Docker image is built from -- it is NOT in the pip wheel).
#   3. Clone+pin the polyglot-benchmark dataset (225 exercises) to ./data/polyglot-benchmark.
#   4. Build the aider benchmark Docker image (tag: aider-benchmark) from the pinned repo.
#
# Pins (recorded in README.md):
#   aider repo + benchmark image : tag v0.86.2  commit 253f0368b873ba30d8ee26e463718f0c03614ddf
#   aider-chat (host driver venv): 0.86.2
#   polyglot-benchmark dataset   : commit 7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PATH="$HOME/.local/bin:$PATH"

# ---- pins --------------------------------------------------------------------------------
AIDER_CHAT_VERSION="0.86.2"
AIDER_REPO="https://github.com/Aider-AI/aider.git"
AIDER_COMMIT="253f0368b873ba30d8ee26e463718f0c03614ddf"   # tag v0.86.2
POLY_REPO="https://github.com/Aider-AI/polyglot-benchmark.git"
POLY_COMMIT="7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
IMAGE_TAG="aider-benchmark"

VENDOR_AIDER="$HERE/vendor/aider"
DATA_POLY="$HERE/data/polyglot-benchmark"

echo "[aider-setup] uv: $(uv --version)"

# ---- 1. host driver venv (pinned) --------------------------------------------------------
if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo "[aider-setup] creating .venv (python 3.12)"
  uv venv --python 3.12 "$HERE/.venv"
fi
echo "[aider-setup] installing pinned driver deps into .venv"
# aider-chat already pins pyyaml (==6.0.3 for 0.86.2) and brings litellm/openai; we only need
# aider-chat itself in the HOST driver venv (dataset/select/parse). The benchmark runs in Docker.
uv pip install --python "$HERE/.venv/bin/python" \
  "aider-chat==${AIDER_CHAT_VERSION}"

# ---- 2. vendored aider repo (benchmark code + Dockerfile), pinned -------------------------
if [ ! -d "$VENDOR_AIDER/.git" ]; then
  echo "[aider-setup] cloning aider repo -> $VENDOR_AIDER"
  mkdir -p "$(dirname "$VENDOR_AIDER")"
  git clone --quiet "$AIDER_REPO" "$VENDOR_AIDER"
fi
git -C "$VENDOR_AIDER" fetch --quiet origin || true
git -C "$VENDOR_AIDER" checkout --quiet "$AIDER_COMMIT"
echo "[aider-setup] aider repo pinned at $(git -C "$VENDOR_AIDER" rev-parse --short HEAD)"

# ---- 3. polyglot-benchmark dataset, pinned -----------------------------------------------
if [ ! -d "$DATA_POLY/.git" ]; then
  echo "[aider-setup] cloning polyglot-benchmark -> $DATA_POLY"
  mkdir -p "$(dirname "$DATA_POLY")"
  git clone --quiet "$POLY_REPO" "$DATA_POLY"
fi
git -C "$DATA_POLY" fetch --quiet origin || true
git -C "$DATA_POLY" checkout --quiet "$POLY_COMMIT"
N_EX=$(find "$DATA_POLY" -path '*/exercises/practice/*' -mindepth 4 -maxdepth 4 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "[aider-setup] polyglot dataset pinned at $(git -C "$DATA_POLY" rev-parse --short HEAD) ($N_EX exercise dirs)"

# ---- 4. build the aider benchmark Docker image (bundles g++/go/java/node/rust/python) -----
# benchmark/docker_build.sh builds 'aider-benchmark' FROM the repo root (COPY . /aider).
# We build only if the image is missing (toolchain install is slow). Force a rebuild by
# removing the image: docker rmi aider-benchmark
if docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "[aider-setup] docker image '$IMAGE_TAG' already present -- skipping build (docker rmi $IMAGE_TAG to force)"
else
  echo "[aider-setup] building docker image '$IMAGE_TAG' (this pulls the 6 language toolchains; slow) ..."
  ( cd "$VENDOR_AIDER" && docker build --file benchmark/Dockerfile -t "$IMAGE_TAG" . )
fi

# ---- versions / DONE ---------------------------------------------------------------------
echo "[aider-setup] versions:"
echo "  aider-chat (driver venv): $("$HERE/.venv/bin/python" -c 'import aider; print(aider.__version__)')"
echo "  aider repo commit       : $(git -C "$VENDOR_AIDER" rev-parse HEAD)  (tag v0.86.2)"
echo "  polyglot dataset commit : $(git -C "$DATA_POLY" rev-parse HEAD)"
echo "  docker image            : $IMAGE_TAG ($(docker image inspect "$IMAGE_TAG" --format '{{.Id}}' 2>/dev/null || echo MISSING))"
echo "  pyyaml (via aider-chat) : $("$HERE/.venv/bin/python" -c 'import yaml; print(yaml.__version__)')"
echo "DONE"
