#!/usr/bin/env bash
# Candidate-only C4 mechanism gate for M<=11 W8A16 routing.
# Run through the external dual-card lease:
#   ./bin/gpu-run bash sglang/06_c4_m11_w8a16_mechanism.sh
# Both known endpoints remain down after every exit; production is not restored.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
ANALYZER="$REPO/sglang/analyze_c4_m11_w8a16_mechanism.py"
SELF="$REPO/sglang/06_c4_m11_w8a16_mechanism.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c4_m11_w8a16_mechanism_$STAMP}"
PROFILE_NAME="c4_m11_w8a16_mechanism_$STAMP"
PROFILE_DIR="$ROOT/sgl_cache/$PROFILE_NAME"
NAME="c4_m11_w8a16_mechanism"
PORT=31006
SERVED="qwen36-27b-W8A8-sqgptq-mtp-c4-m11-w8a16-mechanism"
MODEL="/models/qwen3.6-27b/w8a8-sqgptq"
MODEL_HOST="$REPO/models/files/qwen3.6-27b/w8a8-sqgptq"
PROD_NAME="qwen38_unsloth_ud_q4k_xl_tp2"
PROD_PORT=18080
KDIR="$ROOT/w8a8_kernel"
PUSHDIR="$REPO/vllm/contrib/vllm_push_allreduce/prebuilt"
LOCK_BASE="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
LEASE_CHECK_ONLY="${B70_C4_LEASE_CHECK_ONLY:-0}"
lease_proven=0

case "$LEASE_CHECK_ONLY" in
  0|1) ;;
  *) echo "B70_C4_LEASE_CHECK_ONLY must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$PROFILE_DIR"
exec > >(tee -a "$OUT/gate.log") 2>&1
say() { echo "[c4-m11-mechanism $(date -u +%H:%M:%S)] $*"; }

require_external_lease() {
  local actual0 actual1
  actual0="$(readlink /proc/$$/fd/8 2>/dev/null || true)"
  actual1="$(readlink /proc/$$/fd/9 2>/dev/null || true)"
  if [ "$actual0" != "$LOCK_BASE.0" ] || [ "$actual1" != "$LOCK_BASE.1" ]; then
    echo "refusing GPU work: invoke through ./bin/gpu-run" >&2
    return 2
  fi
  flock -n 8 || return 2
  flock -n 9 || return 2
  {
    echo "LEASE_CHECK PASS cards=0,1"
    echo "fd8=$actual0"
    echo "fd9=$actual1"
  } >"$OUT/lease_check.txt"
}

save_candidate() {
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker logs "$NAME" >"$OUT/server.log" 2>&1 || true
    docker inspect "$NAME" >"$OUT/container_inspect.json" 2>/dev/null || true
  fi
}

stop_candidate() {
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker stop -t 60 "$NAME" >"$OUT/stop.log" 2>&1 || true
    docker rm "$NAME" >>"$OUT/stop.log" 2>&1 || true
  fi
}

ensure_down() {
  local failed=0
  if docker ps --filter "name=^/${NAME}$" --format '{{.Names}}' | rg -qx "$NAME"; then failed=1; fi
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' | rg -qx "$PROD_NAME"; then failed=1; fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then failed=1; fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PROD_PORT/health" >/dev/null 2>&1; then failed=1; fi
  if [ "$failed" = 0 ]; then echo down >"$OUT/endpoint_state.txt"; else echo not-down >"$OUT/endpoint_state.txt"; fi
  return "$failed"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$lease_proven" != 1 ]; then exit "$rc"; fi
  save_candidate
  stop_candidate
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' | rg -qx "$PROD_NAME"; then
    docker stop -t 60 "$PROD_NAME" >/dev/null 2>&1 || true
    docker rm "$PROD_NAME" >/dev/null 2>&1 || true
  fi
  if [ ! -s "$OUT/health_post.log" ]; then
    "$REPO/bin/xpu-health" >"$OUT/health_post.log" 2>&1 || rc=1
  fi
  ensure_down || rc=1
  say "exit rc=$rc artifacts=$OUT endpoint=down"
  exit "$rc"
}
trap cleanup EXIT INT TERM

require_external_lease
lease_proven=1
if [ "$LEASE_CHECK_ONLY" = 1 ]; then
  trap - EXIT INT TERM
  say "lease-check-only PASS; no GPU/container work attempted"
  exit 0
fi

artifacts=(
  "$SELF"
  "$ANALYZER"
  "$SHELF"
  "$REPO/sglang/patches/w8a8_shim.py"
  "$REPO/sglang/patches/woq_shim.py"
  "$REPO/sglang/patches/mtp_replicated_embedding.py"
  "$REPO/sglang/patches/push_ar_xpu.py"
  "$KDIR/_xpu_C.abi3.so"
  "$PUSHDIR/libxpu_push_ar_graph.so"
  "$MODEL_HOST/config.json"
  "$MODEL_HOST/model.safetensors.index.json"
)
for artifact in "${artifacts[@]}"; do
  [ -s "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 2; }
done
for op in int8_gemm_w8a16 int8_gemm_w8a8 dynamic_per_token_int8_quant; do
  rg -a -q "$op" "$KDIR/_xpu_C.abi3.so" || { echo "kernel SO missing op: $op" >&2; exit 2; }
done
sha256sum "${artifacts[@]}" >"$OUT/artifacts.sha256"
IMAGE_ID="$(docker image inspect sglang-xpu:mtp --format '{{.Id}}')"
{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "claim=c4_m11_w8a16_mechanism"
  echo "image=sglang-xpu:mtp"
  echo "image_id=$IMAGE_ID"
  echo "model=$MODEL"
  echo "served=$SERVED"
  echo "w8a16_m_max=11"
  echo "route_debug=1"
  echo "profile_steps=5"
  echo "profile_dir=$PROFILE_DIR"
  echo "endpoint_policy=down_after_gate_no_restore"
} >"$OUT/manifest.txt"

docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' >"$OUT/docker_ps_before.txt"
if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' | rg -qx "$PROD_NAME"; then
  docker inspect "$PROD_NAME" >"$OUT/production_inspect_before.json"
  docker stop -t 60 "$PROD_NAME" >"$OUT/production_stop.log"
  docker rm "$PROD_NAME" >>"$OUT/production_stop.log"
elif curl -fsS --max-time 3 "http://127.0.0.1:$PROD_PORT/health" >/dev/null 2>&1; then
  echo "unknown endpoint on production port" >&2
  exit 2
fi
curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && {
  echo "experiment port is already serving" >&2
  exit 2
}
docker rm -f "$NAME" >/dev/null 2>&1 || true
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_pre.log"

say "start base checkpoint with exact M<=11 W8A16 threshold"
W8A16_M_MAX=11 W8A16_ROUTE_DEBUG=1 \
  CTX=131072 RADIX=0 MAXREQ=4 PORT="$PORT" NAME="$NAME" \
  SERVED="$SERVED" CKPT="$MODEL" API_KEY= MEMFRAC=0.90 \
  TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3 METRICS=1 \
  THINKCAP=4096 KDIR="$KDIR" PUSHDIR="$PUSHDIR" \
  SPEC_STEPS=10 SPEC_DRAFT=11 REPLICATE_MTP_EMBED=1 \
  PUSH_AR=1 PUSH_AR_MIN_NUMEL=0 DELAY_MLP_AR=0 \
  FUSED_MLP_AR_NORM=0 LMHEAD_INT8=0 \
  bash "$SHELF" start 2>&1 | tee "$OUT/start.log"

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$OUT/models.json"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/get_server_info" \
  >"$OUT/server_info.json"
docker inspect "$NAME" >"$OUT/container_inspect.json"

say "same-process deterministic and mixed coherence gates"
for pass in 1 2; do
  python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --max-tokens 64 --out "$OUT/deterministic_$pass.json" \
    2>&1 | tee "$OUT/deterministic_$pass.log"
done
cmp "$OUT/deterministic_1.json" "$OUT/deterministic_2.json"
python3 "$REPO/vllm/gate_concurrent_coherence.py" \
  "http://127.0.0.1:$PORT/v1" "$SERVED" 4 6 128 \
  2>&1 | tee "$OUT/mixed.log"

say "profile exactly five M11 decode iterations"
curl -fsS -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"output_dir\":\"/sgl_cache/$PROFILE_NAME\",\"num_steps\":5,\"activities\":[\"CPU\",\"XPU\"],\"profile_by_stage\":true,\"record_shapes\":true,\"with_stack\":false,\"profile_prefix\":\"m11_w8a16\"}" \
  >"$OUT/start_profile_response.txt"
python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
  "http://127.0.0.1:$PORT/v1" "$SERVED" 1 32 2048 1 \
  2>&1 | tee "$OUT/profile_trigger.log"
found=0
for _ in $(seq 1 90); do
  if [ "$(find "$PROFILE_DIR" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' | wc -l)" = 2 ]; then
    found=1
    break
  fi
  sleep 1
done
[ "$found" = 1 ] || { echo "did not receive two decode traces" >&2; exit 1; }
find "$PROFILE_DIR" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' -print | sort >"$OUT/trace_files.txt"
mapfile -t traces <"$OUT/trace_files.txt"
python3 "$REPO/sglang/parse_tp2_math_census.py" "${traces[@]}" --steps 5 \
  2>&1 | tee "$OUT/math_census.log"

save_candidate
sha256sum "${artifacts[@]}" >"$OUT/artifacts_after.sha256"
cmp "$OUT/artifacts.sha256" "$OUT/artifacts_after.sha256"
stop_candidate
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_post.log"
ensure_down
python3 "$ANALYZER" "$OUT"
say "mechanism gate complete; endpoints remain down"
