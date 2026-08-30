#!/usr/bin/env bash
# F06d: advance the F06c direct-P2P model smoke from MTP0 to MTP1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F06C_STAMP="${F06C_STAMP:-20260830T121000Z}"
F06C_ROOT="${F06C_ROOT:-$ROOT/results/f06c_qwen38_fp8_neural_p2p/$F06C_STAMP}"
F06C_CACHE="${F06C_CACHE:-$ROOT/cache/f06c_qwen38_fp8_neural_p2p/$F06C_STAMP}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in "$F06C_ROOT/verdict.txt" "$F06C_ROOT/smoke.json" "$F06C_CACHE"; do
  [ -e "$required" ] || { echo "missing frozen F06c evidence: $required" >&2; exit 1; }
done
grep -Fq 'VERDICT -> F06c PASS:' "$F06C_ROOT/verdict.txt" || {
  echo "frozen F06c evidence is not a pass" >&2
  exit 1
}

exec env \
  SPECULATIVE_TOKENS=1 \
  CAMPAIGN_LABEL=F06d \
  SEED_CACHE="$F06C_CACHE" \
  NAME="qwen38-fp8-neural-f06d-mtp1-p2p1-$STAMP" \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f06d \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f06d_qwen38_fp8_neural_p2p/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f06d_qwen38_fp8_neural_p2p/$STAMP}" \
  "$SCRIPT_DIR/run_qwen38_fp8_p2p_f06c_model_smoke.sh" "$@"
