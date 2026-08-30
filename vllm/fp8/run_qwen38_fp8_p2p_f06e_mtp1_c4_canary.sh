#!/usr/bin/env bash
# F06e: short concurrent exact-answer gate for MTP1 direct P2P.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SOURCE="${SOURCE:-$ROOT/steve-repro/qwen38-fp8-neural-20260829/source}"
F06D_STAMP="${F06D_STAMP:-20260830T123000Z}"
F06D_ROOT="${F06D_ROOT:-$ROOT/results/f06d_qwen38_fp8_neural_p2p/$F06D_STAMP}"
F06D_CACHE="${F06D_CACHE:-$ROOT/cache/f06d_qwen38_fp8_neural_p2p/$F06D_STAMP}"
QUALITY_SCRIPT="$SOURCE/experiments/qwen38-27b-b70/scripts/qwen38-concurrent-quality-canary.py"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F06D_ROOT/verdict.txt" "$F06D_ROOT/smoke.json" "$F06D_CACHE" "$QUALITY_SCRIPT"; do
  [ -e "$required" ] || { echo "missing frozen F06d input: $required" >&2; exit 1; }
done
grep -Fq 'VERDICT -> F06d PASS:' "$F06D_ROOT/verdict.txt" || {
  echo "frozen F06d evidence is not a pass" >&2
  exit 1
}

exec env \
  SPECULATIVE_TOKENS=1 \
  CAMPAIGN_LABEL=F06e \
  SEED_CACHE="$F06D_CACHE" \
  MAX_MODEL_LEN=1024 \
  MAX_NUM_SEQS=4 \
  MAX_NUM_BATCHED_TOKENS=1024 \
  EXTRA_SMOKE="$QUALITY_SCRIPT" \
  NAME="qwen38-fp8-neural-f06e-mtp1-p2p1-c4-$STAMP" \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f06e-c4 \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f06e_qwen38_fp8_neural_p2p/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f06e_qwen38_fp8_neural_p2p/$STAMP}" \
  "$SCRIPT_DIR/run_qwen38_fp8_p2p_f06c_model_smoke.sh" "$@"
