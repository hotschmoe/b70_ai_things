#!/usr/bin/env bash
# Position-balanced qualification for corrected target-GDN INT8 projections.
#
# Caller must hold the external dual-card lease for the whole campaign:
#   ./bin/gpu-run bash sglang/02_c4_gdn_int8_abba.sh
#
# A = current W8A8 SQ-GPTQ shelf checkpoint and its native config.
# B = target-GDN INT8 checkpoint plus the generated corrected config overlay.
# Endpoints stay down between arms and after every exit. Nothing is restored.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
PREPARE="$REPO/sglang/prepare_c4_gdn_int8_candidate.py"
ANALYZER="$REPO/sglang/analyze_c4_gdn_int8_abba.py"
SELF="$REPO/sglang/02_c4_gdn_int8_abba.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c4_gdn_int8_abba_$STAMP}"
PORT="${PORT:-31003}"
CTX="${CTX:-131072}"
MAXREQ="${MAXREQ:-4}"
MODEL_A_CONTAINER="/models/qwen3.6-27b/w8a8-sqgptq"
MODEL_B_CONTAINER="/models/qwen3.6-27b/w8a8-sqgptq-gdnint8"
MODEL_A_HOST="$REPO/models/files/qwen3.6-27b/w8a8-sqgptq"
MODEL_B_HOST="$REPO/models/files/qwen3.6-27b/w8a8-sqgptq-gdnint8"
SERVED_A="qwen36-27b-W8A8-sqgptq-base-mtp-c4-abba"
SERVED_B="qwen36-27b-W8A8-sqgptq-GDNRTN-mtp-c4-abba"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
PROD_NAME="${PROD_NAME:-qwen38_unsloth_ud_q4k_xl_tp2}"
PROD_PORT="${PROD_PORT:-18080}"
KDIR="${KDIR:-$ROOT/w8a8_kernel}"
PUSHDIR="${PUSHDIR:-$REPO/vllm/contrib/vllm_push_allreduce/prebuilt}"
PUSH_AR_SO="${PUSH_AR_SO:-/work/push_ar/libxpu_push_ar_graph.so}"
PUSH_AR_MAXB="${PUSH_AR_MAXB:-536870912}"
LOCK_BASE="${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}"
LEASE_CHECK_ONLY="${B70_C4_LEASE_CHECK_ONLY:-0}"
lease_proven=0
active_name=""
active_dir=""

case "$LEASE_CHECK_ONLY" in
  0|1) ;;
  *) echo "B70_C4_LEASE_CHECK_ONLY must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1

say() { echo "[c4-gdn-abba $(date -u +%H:%M:%S)] $*"; }

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

save_active() {
  if [ -n "$active_name" ] && docker inspect "$active_name" >/dev/null 2>&1; then
    docker logs "$active_name" >"$active_dir/server.log" 2>&1 || true
    docker inspect "$active_name" >"$active_dir/container_inspect.json" \
      2>/dev/null || true
  fi
}

stop_active() {
  if [ -n "$active_name" ] && docker inspect "$active_name" >/dev/null 2>&1; then
    docker stop -t 60 "$active_name" >/dev/null 2>&1 || true
    docker rm "$active_name" >/dev/null 2>&1 || true
  fi
  active_name=""
  active_dir=""
}

ensure_endpoints_down() {
  local failed=0
  if [ -n "$active_name" ] && docker ps --filter "name=^/${active_name}$" \
    --format '{{.Names}}' | rg -qx "$active_name"; then
    failed=1
  fi
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
    | rg -qx "$PROD_NAME"; then
    failed=1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" \
    >/dev/null 2>&1; then
    failed=1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PROD_PORT/health" \
    >/dev/null 2>&1; then
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
  save_active
  stop_active
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
    | rg -qx "$PROD_NAME"; then
    docker stop -t 60 "$PROD_NAME" >/dev/null 2>&1 || true
    docker rm "$PROD_NAME" >/dev/null 2>&1 || true
  fi
  "$REPO/bin/xpu-health" >"$OUT/health_final.log" 2>&1 || rc=1
  ensure_endpoints_down || rc=1
  cat "$OUT/health_final.log" || true
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
  "$SELF"
  "$SHELF"
  "$PREPARE"
  "$ANALYZER"
  "$REPO/sglang/configs/qwen36_w8a8_sqgptq_gdnrtn_quantization.json"
  "$REPO/sglang/patches/w8a8_shim.py"
  "$REPO/sglang/patches/woq_shim.py"
  "$REPO/sglang/patches/mtp_replicated_embedding.py"
  "$REPO/sglang/patches/push_ar_xpu.py"
  "$REPO/sglang/patches/xpu_delayed_mlp_ar.py"
  "$REPO/sglang/patches/xpu_fused_mlp_ar_norm.py"
  "$REPO/kernels/int8_gemm_w8a16.h"
  "$REPO/kernels/int8_gemm_w8a8.h"
  "$REPO/kernels/int8_quant_common.hpp"
  "$REPO/kernels/int8_gemm_kernel.patch"
  "$KDIR/_xpu_C.abi3.so"
  "$PUSHDIR/$(basename "$PUSH_AR_SO")"
  "$MODEL_A_HOST/config.json"
  "$MODEL_A_HOST/model.safetensors.index.json"
  "$MODEL_B_HOST/config.json"
  "$MODEL_B_HOST/model.safetensors.index.json"
  "$MODEL_B_HOST/GDN_INT8_NOTE.txt"
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

say "audit candidate checkpoint and materialize corrected config overlay"
python3 "$PREPARE" \
  --base-dir "$MODEL_A_HOST" \
  --candidate-dir "$MODEL_B_HOST" \
  --overlay-out "$OUT/candidate_config.json" \
  --report-out "$OUT/checkpoint_audit.json"
artifacts+=("$OUT/candidate_config.json" "$OUT/checkpoint_audit.json")
sha256sum "${artifacts[@]}" >"$OUT/artifacts.sha256"

verify_artifacts() {
  sha256sum --check "$OUT/artifacts.sha256"
}

IMAGE_ID="$(docker image inspect sglang-xpu:mtp --format '{{.Id}}')"
{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "claim=c4_target_gdn_int8_abba_qualification"
  echo "image=sglang-xpu:mtp"
  echo "image_id=$IMAGE_ID"
  echo "model_a=$MODEL_A_CONTAINER"
  echo "model_b=$MODEL_B_CONTAINER"
  echo "served_a=$SERVED_A"
  echo "served_b=$SERVED_B"
  echo "overlay=$OUT/candidate_config.json"
  echo "ctx=$CTX"
  echo "maxreq=$MAXREQ"
  echo "memfrac=0.90"
  echo "spec_steps=10"
  echo "spec_draft=11"
  echo "kdir=$KDIR"
  echo "pushdir=$PUSHDIR"
  echo "push_ar_so=$PUSH_AR_SO"
  echo "push_ar_min_numel=0"
  echo "replicate_mtp_embed=1"
  echo "delay_mlp_ar=0"
  echo "fused_mlp_ar_norm=0"
  echo "lmhead_int8=0"
  echo "ccl_topo_p2p_access=0"
  echo "order=01_A1:base,02_B1:gdnint8,03_B2:gdnint8,04_A2:base"
  echo "endpoint_policy=down_between_arms_and_after_campaign_no_restore"
} >"$OUT/manifest.txt"

say "snapshot and stop known production endpoint if present"
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' \
  >"$OUT/docker_ps_before.txt"
if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
  | rg -qx "$PROD_NAME"; then
  curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
    >"$OUT/production_models_before.json" || true
  docker inspect "$PROD_NAME" >"$OUT/production_inspect_before.json"
  docker stop -t 60 "$PROD_NAME" >"$OUT/production_stop.log"
  docker rm "$PROD_NAME" >>"$OUT/production_stop.log"
elif curl -fsS --max-time 3 "http://127.0.0.1:$PROD_PORT/health" \
  >/dev/null 2>&1; then
  echo "unknown endpoint is listening on production port $PROD_PORT; refusing" >&2
  exit 2
fi
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" \
  >/dev/null 2>&1; then
  echo "experiment port $PORT is already serving; refusing" >&2
  exit 2
fi
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_pre_campaign.log"

run_fixed_output() {
  local served="$1"
  local output="$2"
  python3 - "$PORT" "$served" "$output" <<'PY'
import json
import sys
import urllib.request

port, model, output = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": (
            "Write a detailed technical explanation of tensor-parallel LLM "
            "decode, including memory bandwidth, collectives, and INT8 matrix "
            "multiplication. Use complete prose and no table."
        ),
    }],
    "temperature": 0,
    "top_p": 1,
    "seed": 1234,
    "max_tokens": 512,
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
message = result["choices"][0]["message"]
stable = {
    "content": message.get("content"),
    "reasoning_content": message.get("reasoning_content"),
}
with open(output, "w", encoding="ascii") as handle:
    json.dump(stable, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
print(f"FIXED -> content_chars={len(message.get('content') or '')}")
PY
}

run_arm() {
  local label="$1"
  local variant="$2"
  local model served overlay
  if [ "$variant" = A ]; then
    model="$MODEL_A_CONTAINER"
    served="$SERVED_A"
    overlay=""
  else
    model="$MODEL_B_CONTAINER"
    served="$SERVED_B"
    overlay="$OUT/candidate_config.json"
  fi
  local slug name arm_dir
  slug="$(tr '[:upper:]' '[:lower:]' <<<"$label")"
  name="c4_gdn_abba_$slug"
  arm_dir="$OUT/$label"
  mkdir -p "$arm_dir"
  active_name="$name"
  active_dir="$arm_dir"

  say "ARM $label start variant=$variant model=$model served=$served"
  verify_artifacts 2>&1 | tee "$arm_dir/artifact_check_pre.log"
  "$REPO/bin/xpu-health" 2>&1 | tee "$arm_dir/health_pre.log"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" \
    SERVED="$served" IMG=sglang-xpu:mtp CKPT="$model" API_KEY= \
    CONFIG_OVERLAY="$overlay" MEMFRAC=0.90 TOOLCALL=1 \
    TOOLPARSER=qwen3_coder REASONPARSER=qwen3 METRICS=1 THINKCAP=4096 \
    KDIR="$KDIR" PUSHDIR="$PUSHDIR" PUSH_AR_SO="$PUSH_AR_SO" \
    SPEC_STEPS=10 SPEC_DRAFT=11 REPLICATE_MTP_EMBED=1 \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=0 PUSH_AR_MAXB="$PUSH_AR_MAXB" \
    DELAY_MLP_AR=0 FUSED_MLP_AR_NORM=0 LMHEAD_INT8=0 \
    bash "$SHELF" start 2>&1 | tee "$arm_dir/start.log"

  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" \
    >"$arm_dir/models.json"
  curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
    >"$arm_dir/server_info.json"
  docker inspect "$name" >"$arm_dir/container_inspect.json"

  run_fixed_output "$served" "$arm_dir/fixed_output.json" \
    2>&1 | tee "$arm_dir/fixed_output.log"
  python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
    --base "http://127.0.0.1:$PORT" --model "$served" \
    --max-tokens 128 --out "$arm_dir/deterministic.json" \
    2>&1 | tee "$arm_dir/deterministic.log"
  python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
    --base "http://127.0.0.1:$PORT" --model "$served" \
    --prompt-tokens 2048 --gen-tokens 128 --n 5 --ignore-eos \
    --label "$label" --out "$arm_dir/phase_p2048_g128.json" \
    2>&1 | tee "$arm_dir/phase_p2048_g128.log"
  bash "$REPO/sglang/perf_regime.sh" "$name" "$PORT" "$served" "$TOK" "$label" \
    2>&1 | tee "$arm_dir/perf_regime.log"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$served" 1 8 2048 5 \
    2>&1 | tee "$arm_dir/prefill_c1.log"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$served" 4 8 2048 5 \
    2>&1 | tee "$arm_dir/prefill_c4.log"
  python3 "$REPO/vllm/gate_concurrent_coherence.py" \
    "http://127.0.0.1:$PORT/v1" "$served" 4 6 256 \
    2>&1 | tee "$arm_dir/mixed.log"
  python3 "$REPO/sglang/soak_probe.py" "$PORT" "$served" 6400 800 localhost \
    2>&1 | tee "$arm_dir/soak6400.log"

  save_active
  if rg -i \
    'device_lost|out_of_resources|ur_result_error|enginedead|!!!!|segmentation fault|(^|[^a-z])nan([^a-z]|$)|missing key|unexpected key|size mismatch' \
    "$arm_dir/server.log"; then
    say "ARM $label fail: fatal marker"
    return 1
  fi
  say "ARM $label graceful stop"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" \
    SERVED="$served" CKPT="$model" CONFIG_OVERLAY="$overlay" \
    KDIR="$KDIR" PUSHDIR="$PUSHDIR" PUSH_AR_SO="$PUSH_AR_SO" \
    REPLICATE_MTP_EMBED=1 PUSH_AR=1 PUSH_AR_MIN_NUMEL=0 \
    DELAY_MLP_AR=0 FUSED_MLP_AR_NORM=0 LMHEAD_INT8=0 \
    bash "$SHELF" stop 2>&1 | tee "$arm_dir/stop.log"
  active_name=""
  active_dir=""
  "$REPO/bin/xpu-health" 2>&1 | tee "$arm_dir/health_post.log"
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" \
    >/dev/null 2>&1; then
    echo "ARM $label endpoint remained up after stop" >&2
    return 1
  fi
  say "ARM $label pass endpoint=down"
}

run_arm 01_A1 A
run_arm 02_B1 B
run_arm 03_B2 B
run_arm 04_A2 A

verify_artifacts 2>&1 | tee "$OUT/artifact_check_after.log"
sha256sum "${artifacts[@]}" >"$OUT/artifacts_after.sha256"
cmp "$OUT/artifacts.sha256" "$OUT/artifacts_after.sha256"
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_final.log"
ensure_endpoints_down
python3 "$ANALYZER" "$OUT"
say "A-B-B-A complete; endpoints remain down"
