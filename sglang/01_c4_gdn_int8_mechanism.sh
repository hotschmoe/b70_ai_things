#!/usr/bin/env bash
# Candidate-only C4 mechanism gate for target GDN RTN-INT8 projections.
#
# This script MUST be invoked through the external dual-card lease:
#   ./bin/gpu-run bash sglang/01_c4_gdn_int8_mechanism.sh
#
# It never restores production. Both the experiment and production endpoints
# are down after every exit, including failures and signals.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
PREPARE="$REPO/sglang/prepare_c4_gdn_int8_candidate.py"
ANALYZER="$REPO/sglang/analyze_c4_gdn_int8_mechanism.py"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c4_gdn_int8_mechanism_$STAMP}"
PROFILE_NAME="c4_gdn_int8_mechanism_$STAMP"
PROFILE_DIR="$ROOT/sgl_cache/$PROFILE_NAME"
NAME="c4_gdn_int8_mechanism"
PORT="31003"
SERVED="qwen36-27b-W8A8-sqgptq-GDNRTN-mtp-c4-mechanism"
MODEL_CONTAINER="/models/qwen3.6-27b/w8a8-sqgptq-gdnint8"
MODEL_HOST="$REPO/models/files/qwen3.6-27b/w8a8-sqgptq-gdnint8"
BASE_HOST="$REPO/models/files/qwen3.6-27b/w8a8-sqgptq"
PROD_NAME="qwen38_unsloth_ud_q4k_xl_tp2"
PROD_PORT="18080"
CTX=131072
MAXREQ=4
KDIR="/mnt/vm_8tb/b70/w8a8_kernel"
PUSHDIR="$REPO/vllm/contrib/vllm_push_allreduce/prebuilt"
PUSH_AR_SO="/work/push_ar/libxpu_push_ar_graph.so"
PUSH_AR_MAXB=536870912
LOCK_BASE="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
lease_proven=0
LEASE_CHECK_ONLY="${B70_C4_LEASE_CHECK_ONLY:-0}"

case "$LEASE_CHECK_ONLY" in
  0|1) ;;
  *) echo "B70_C4_LEASE_CHECK_ONLY must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$PROFILE_DIR"
exec > >(tee -a "$OUT/gate.log") 2>&1

say() { echo "[c4-gdn-mechanism $(date -u +%H:%M:%S)] $*"; }

require_external_dual_card_lease() {
  local actual0 actual1
  actual0="$(readlink "/proc/$$/fd/8" 2>/dev/null || true)"
  actual1="$(readlink "/proc/$$/fd/9" 2>/dev/null || true)"
  if [ "$actual0" != "$LOCK_BASE.0" ] || [ "$actual1" != "$LOCK_BASE.1" ]; then
    echo "refusing GPU work: invoke through ./bin/gpu-run without --card" >&2
    echo "fd8=$actual0 expected=$LOCK_BASE.0" >&2
    echo "fd9=$actual1 expected=$LOCK_BASE.1" >&2
    return 2
  fi
  flock -n 8 || { echo "inherited card-0 lease fd is not locked" >&2; return 2; }
  flock -n 9 || { echo "inherited card-1 lease fd is not locked" >&2; return 2; }
  {
    echo "LEASE_CHECK PASS cards=0,1"
    echo "fd8=$actual0"
    echo "fd9=$actual1"
    echo "owner0=$(cat "$LOCK_BASE.0.owner" 2>/dev/null || true)"
    echo "owner1=$(cat "$LOCK_BASE.1.owner" 2>/dev/null || true)"
  } | tee "$OUT/lease_check.txt"
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

ensure_endpoints_down() {
  local failed=0
  if docker ps --filter "name=^/${NAME}$" --format '{{.Names}}' | rg -qx "$NAME"; then
    failed=1
  fi
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
    | rg -qx "$PROD_NAME"; then
    failed=1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    failed=1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PROD_PORT/health" >/dev/null 2>&1; then
    failed=1
  fi
  if [ "$failed" = 0 ]; then
    echo "down" >"$OUT/endpoint_state.txt"
  else
    echo "not-down" >"$OUT/endpoint_state.txt"
  fi
  return "$failed"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$lease_proven" != 1 ]; then
    say "exit rc=$rc before lease proof; no GPU or container cleanup attempted"
    exit "$rc"
  fi
  save_candidate
  stop_candidate
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
    | rg -qx "$PROD_NAME"; then
    docker stop -t 60 "$PROD_NAME" >/dev/null 2>&1 || true
    docker rm "$PROD_NAME" >/dev/null 2>&1 || true
  fi
  if [ ! -s "$OUT/health_post.log" ]; then
    "$REPO/bin/xpu-health" >"$OUT/health_post.log" 2>&1 || rc=1
  fi
  ensure_endpoints_down || rc=1
  cat "$OUT/health_post.log" || true
  say "exit rc=$rc artifacts=$OUT endpoint=down"
  exit "$rc"
}
trap cleanup EXIT INT TERM

require_external_dual_card_lease
lease_proven=1
if [ "$LEASE_CHECK_ONLY" = 1 ]; then
  trap - EXIT INT TERM
  say "lease-check-only PASS; no GPU or container work attempted"
  exit 0
fi

artifacts=(
  "$SHELF"
  "$PREPARE"
  "$ANALYZER"
  "$REPO/sglang/01_c4_gdn_int8_mechanism.sh"
  "$REPO/sglang/configs/qwen36_w8a8_sqgptq_gdnrtn_quantization.json"
  "$REPO/sglang/patches/w8a8_shim.py"
  "$REPO/sglang/patches/woq_shim.py"
  "$REPO/sglang/patches/mtp_replicated_embedding.py"
  "$REPO/sglang/patches/push_ar_xpu.py"
  "$REPO/kernels/int8_gemm_w8a16.h"
  "$REPO/kernels/int8_gemm_w8a8.h"
  "$REPO/kernels/int8_quant_common.hpp"
  "$KDIR/_xpu_C.abi3.so"
  "$PUSHDIR/libxpu_push_ar_graph.so"
  "$BASE_HOST/config.json"
  "$BASE_HOST/model.safetensors.index.json"
  "$MODEL_HOST/config.json"
  "$MODEL_HOST/model.safetensors.index.json"
  "$MODEL_HOST/GDN_INT8_NOTE.txt"
)
for artifact in "${artifacts[@]}"; do
  [ -s "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 2; }
done
for op in int8_gemm_w8a16 int8_gemm_w8a8 dynamic_per_token_int8_quant; do
  rg -a -q "$op" "$KDIR/_xpu_C.abi3.so" || {
    echo "kernel SO is missing required op: $op" >&2
    exit 2
  }
done

say "audit checkpoint and materialize exact candidate config"
python3 "$PREPARE" \
  --base-dir "$BASE_HOST" \
  --candidate-dir "$MODEL_HOST" \
  --overlay-out "$OUT/candidate_config.json" \
  --report-out "$OUT/checkpoint_audit.json"
artifacts+=("$OUT/candidate_config.json" "$OUT/checkpoint_audit.json")
sha256sum "${artifacts[@]}" >"$OUT/artifacts.sha256"

IMAGE_ID="$(docker image inspect sglang-xpu:mtp --format '{{.Id}}')"
{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "claim=c4_target_gdn_int8_mechanism"
  echo "image=sglang-xpu:mtp"
  echo "image_id=$IMAGE_ID"
  echo "model=$MODEL_CONTAINER"
  echo "served=$SERVED"
  echo "ctx=$CTX"
  echo "maxreq=$MAXREQ"
  echo "spec_steps=10"
  echo "spec_draft=11"
  echo "push_ar_min_numel=0"
  echo "replicate_mtp_embed=1"
  echo "lmhead_int8=0"
  echo "ccl_topo_p2p_access=0"
  echo "profile_steps=5"
  echo "profile_dir=$PROFILE_DIR"
  echo "endpoint_policy=down_after_gate_no_restore"
} >"$OUT/manifest.txt"

say "snapshot and stop the known production endpoint if present"
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' >"$OUT/docker_ps_before.txt"
if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
  | rg -qx "$PROD_NAME"; then
  curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
    >"$OUT/production_models_before.json" || true
  docker inspect "$PROD_NAME" >"$OUT/production_inspect_before.json"
  docker stop -t 60 "$PROD_NAME" >"$OUT/production_stop.log"
  docker rm "$PROD_NAME" >>"$OUT/production_stop.log"
elif curl -fsS --max-time 3 "http://127.0.0.1:$PROD_PORT/health" >/dev/null 2>&1; then
  echo "unknown endpoint is listening on production port $PROD_PORT; refusing" >&2
  exit 2
fi
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "experiment port $PORT is already serving; refusing" >&2
  exit 2
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_pre.log"

say "start exact target-GDN INT8 candidate"
CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$NAME" \
  SERVED="$SERVED" IMG=sglang-xpu:mtp \
  CKPT="$MODEL_CONTAINER" API_KEY= \
  CONFIG_OVERLAY="$OUT/candidate_config.json" MEMFRAC=0.90 \
  TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3 METRICS=1 \
  THINKCAP=4096 KDIR="$KDIR" PUSHDIR="$PUSHDIR" \
  PUSH_AR_SO="$PUSH_AR_SO" SPEC_STEPS=10 SPEC_DRAFT=11 \
  REPLICATE_MTP_EMBED=1 PUSH_AR=1 PUSH_AR_MIN_NUMEL=0 \
  PUSH_AR_MAXB="$PUSH_AR_MAXB" DELAY_MLP_AR=0 \
  FUSED_MLP_AR_NORM=0 LMHEAD_INT8=0 \
  bash "$SHELF" start 2>&1 | tee "$OUT/start.log"

curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" \
  >"$OUT/models.json"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
  >"$OUT/server_info_before.json"
docker inspect "$NAME" >"$OUT/container_inspect.json"

docker logs "$NAME" >"$OUT/server_before_fixed.log" 2>&1
before_lines="$(wc -l <"$OUT/server_before_fixed.log")"
say "run fixed 640-token acceptance and coherence request"
python3 - "$PORT" "$SERVED" "$OUT/fixed_generation.json" <<'PY'
import json
import sys
import urllib.request

port, model, output = sys.argv[1:]
prompt = (
    "Write a continuous detailed technical essay of at least 1200 words about "
    "GPU memory hierarchy, tensor parallel inference, collective communication, "
    "and INT8 quantization. Use complete sentences and do not use a table."
)
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "top_p": 1,
    "seed": 1234,
    "max_tokens": 640,
    "ignore_eos": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
with open(output, "w", encoding="ascii") as handle:
    json.dump(result, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
print(
    "FIXED -> completion_tokens="
    f"{result.get('usage', {}).get('completion_tokens')} "
    f"finish={result.get('choices', [{}])[0].get('finish_reason')}"
)
PY
docker logs "$NAME" >"$OUT/server_after_fixed.log" 2>&1
tail -n "+$((before_lines + 1))" "$OUT/server_after_fixed.log" \
  >"$OUT/fixed_server_delta.log"

say "run same-process deterministic corpus twice"
for pass in 1 2; do
  python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --max-tokens 64 --out "$OUT/deterministic_$pass.json" \
    2>&1 | tee "$OUT/deterministic_$pass.log"
done
cmp "$OUT/deterministic_1.json" "$OUT/deterministic_2.json"

say "run 24-request mixed prefill/decode coherence gate"
python3 "$REPO/vllm/gate_concurrent_coherence.py" \
  "http://127.0.0.1:$PORT/v1" "$SERVED" 4 6 128 \
  2>&1 | tee "$OUT/mixed.log"

say "run 1600-token stability/coherence soak"
python3 "$REPO/sglang/soak_probe.py" "$PORT" "$SERVED" 1600 400 localhost \
  2>&1 | tee "$OUT/soak.log"

say "profile exactly five decode iterations for runtime route proof"
container_profile="/sgl_cache/$PROFILE_NAME"
curl -fsS -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"output_dir\":\"$container_profile\",\"num_steps\":5,\"activities\":[\"CPU\",\"XPU\"],\"profile_by_stage\":true,\"record_shapes\":true,\"with_stack\":false,\"profile_prefix\":\"gdn_int8\"}" \
  >"$OUT/start_profile_response.txt"
python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
  "http://127.0.0.1:$PORT/v1" "$SERVED" 1 32 2048 1 \
  2>&1 | tee "$OUT/profile_trigger.log"
found=0
for _ in $(seq 1 90); do
  if [ "$(find "$PROFILE_DIR" -maxdepth 1 -type f \
    -name '*DECODE.trace.json.gz' | wc -l)" = 2 ]; then
    found=1
    break
  fi
  sleep 1
done
[ "$found" = 1 ] || { echo "did not receive two decode traces" >&2; exit 1; }
find "$PROFILE_DIR" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' \
  -print | sort >"$OUT/trace_files.txt"
mapfile -t trace_files <"$OUT/trace_files.txt"
python3 "$REPO/sglang/parse_tp2_math_census.py" \
  "${trace_files[@]}" --steps 5 \
  2>&1 | tee "$OUT/math_census.log"

curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
  >"$OUT/server_info_after.json"
save_candidate
sha256sum "${artifacts[@]}" >"$OUT/artifacts_after.sha256"
cmp "$OUT/artifacts.sha256" "$OUT/artifacts_after.sha256"

say "gracefully stop candidate before final health and analysis"
stop_candidate
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_post.log"
ensure_endpoints_down

python3 "$ANALYZER" "$OUT"
say "mechanism gate complete; endpoints remain down"
