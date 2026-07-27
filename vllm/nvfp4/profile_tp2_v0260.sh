#!/usr/bin/env bash
# Profile the qualified vLLM 0.26.0 NVFP4 TP=2 200K configuration.
#
# Run one mode per fresh server because repeated Kineto start/stop in one XPU
# worker has previously failed during profiler destruction:
#   ./bin/gpu-run bash vllm/nvfp4/profile_tp2_v0260.sh decode
#   ./bin/gpu-run bash vllm/nvfp4/profile_tp2_v0260.sh prefill
#
# Traces are runtime artifacts under /mnt/vm_8tb/b70/profiles. The parsed
# summaries are printed to stdout for capture in an experiment log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHELF="$REPO/rdy_to_serve/vllm/qwen36-27b-nvfp4/serve.sh"
MODE="${1:-decode}"
case "$MODE" in
  decode|prefill) ;;
  *) echo "usage: $0 decode|prefill" >&2; exit 2 ;;
esac

PORT="${PORT:-18079}"
NAME="${NAME:-nvfp4_tp2_profile_${MODE}}"
MODEL="${SERVED_FORCE:-qwen3.6-27b-NVFP4-modelopt-fused-graph-mtp5}"
PROFILE_ROOT="${PROFILE_ROOT:-/mnt/vm_8tb/b70/profiles}"
mkdir -p "$PROFILE_ROOT"
OUTDIR="${OUTDIR:-$(mktemp -d "$PROFILE_ROOT/v0260_tp2_${MODE}_XXXXXX")}"
mkdir -p "$OUTDIR"

export TP=2
export IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
export MAXLEN="${MAXLEN:-200000}"
export UTIL="${UTIL:-0.85}"
export MAXSEQS="${MAXSEQS:-8}"
export MAXBATCH="${MAXBATCH:-16384}"
export CAPSIZES="${CAPSIZES:-1,2,4,8}"
export MTPTOK="${MTPTOK:-5}"
export PREFIXCACHE="${PREFIXCACHE:-1}"
export PUSH_AR="${PUSH_AR:-1}"
export PUSH_AR_GRAPH="${PUSH_AR_GRAPH:-1}"
export PUSH_AR_MAXB="${PUSH_AR_MAXB:-268435456}"
export KV_FP8="${KV_FP8:-1}"
export KV_SCALES="${KV_SCALES:-$REPO/vllm/nvfp4/kv_scales_nvfp4_27b.json}"
export FUSED_SO="${FUSED_SO:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so}"
export GDN_LIB="${GDN_LIB:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so}"
export B70_EXTRA_ENV="${B70_EXTRA_ENV:-B70_PC_EAGLE_KEEP=1 B70_PC_CHUNK_ALIGN=1 B70_NVFP4_F8_SCALE_M_MAX=8}"
export B70_PROFILER_DIR=/prof
export B70_EXTRA_MOUNTS="$OUTDIR:/prof"
export PORT NAME SERVED_FORCE="$MODEL"

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker stop -t 60 "$NAME" >/dev/null 2>&1 || true
    docker rm "$NAME" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

echo "CONFIG -> mode=$MODE img=$IMG tp=$TP maxlen=$MAXLEN outdir=$OUTDIR"
bash "$SHELF" start

echo -n "WAIT -> health"
healthy=0
for _ in $(seq 1 240); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  echo -n "."
  sleep 5
done
echo
[ "$healthy" = 1 ] || {
  echo "RESULT -> FAIL: server did not become healthy"
  docker logs "$NAME" 2>&1 | tail -200
  exit 1
}

served="$(curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "IDENTITY -> id=$served"
[ "$served" = "$MODEL" ] || {
  echo "RESULT -> FAIL: served model id mismatch"
  exit 1
}

HOST=http://127.0.0.1 PORT="$PORT" MODEL="$MODEL" MODE="$MODE" \
  bash "$REPO/research/profiling/trace_driver.sh"

mapfile -t traces < <(find "$OUTDIR" -maxdepth 1 -type f \
  \( -name 'rank*.pt.trace.json' -o -name 'rank*.pt.trace.json.gz' \) | sort)
[ "${#traces[@]}" -ge 2 ] || {
  echo "RESULT -> FAIL: expected one trace per TP rank, found ${#traces[@]}"
  find "$OUTDIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
  exit 1
}

for trace in "${traces[@]}"; do
  python3 "$REPO/research/profiling/parse_trace.py" "$trace" 35
done
echo "VERDICT -> PASS: parsed ${#traces[@]} $MODE traces in $OUTDIR"
