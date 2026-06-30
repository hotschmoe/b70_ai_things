#!/usr/bin/env bash
# zml/build.sh -- compile the ZML oneAPI examples for the dual B70 (GPU-free; bazelisk BUILD, not run).
#
# ZML is Zig + MLIR/XLA/PJRT. It runs HF safetensors in bf16/f16 via XLA. It does NOT consume our
# compressed-tensors W8A8/W4A8/W4A16 artifacts and has no oneDNN int8 XPU kernel -- so on zml,
# "W8A8 TP=2 / W4A16 DP=2" only maps to "bf16/f16 dense, sharded TP=2 / replicated DP=2".
# See REVIEW_intel_arch.md for the full re-validation (PR #592 merged -> oneAPI multi-device is mainline).
#
# Toolchain: bazelisk (reads .bazelversion -> Bazel 9.1.1) at ~/.local/bin/bazelisk; zig at ~/.local/bin.
# The oneAPI PJRT plugin + 2026.0 runtime are fetched HERMETICALLY by bazel (pinned http_archive,
# amd64-only -- matches this x86_64 box). No system oneAPI install needed.
#
# This BUILD step is pure compilation (XLA/MLIR/Zig) -- heavy (tens of minutes cold) but GPU-free.
# The actual GPU runs are in test_sharding.sh / serve_llama_tp2.sh (under the gpu-run lease).
set -euo pipefail
ZML="${ZML:-/mnt/vm_8tb/b70/zml}"
BAZELISK="${BAZELISK:-$HOME/.local/bin/bazelisk}"
TARGETS=("${@:-//examples/sharding //examples/llm}")

command -v "$BAZELISK" >/dev/null 2>&1 || { echo "[!] bazelisk not found at $BAZELISK"; exit 2; }
cd "$ZML"
echo "=== zml oneAPI BUILD  $(cat .bazelversion 2>/dev/null)  targets: ${TARGETS[*]}  $(date) ==="
echo "    (compile-only; no GPU. Run targets later via test_sharding.sh / serve_llama_tp2.sh under gpu-run.)"
# shellcheck disable=SC2086
"$BAZELISK" build ${TARGETS[*]} \
  --config=release \
  --@zml//platforms:cpu=false \
  --@zml//platforms:oneapi=true
echo "=== zml oneAPI build done $(date) ==="
