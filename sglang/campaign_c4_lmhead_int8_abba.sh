#!/usr/bin/env bash
# Position-balanced C4 serving qualification for runtime INT8 lm_head.
#
# Caller must hold both cards for the entire campaign:
#   ./bin/gpu-run bash sglang/campaign_c4_lmhead_int8_abba.sh
#
# A = current W8A8 TP=2 baseline with BF16 lm_head.
# B = the identical stack with B70_W8A8_QUANT_LMHEAD=1.
# The endpoint stays down between arms and after the campaign. There is no
# production restore path in this script.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c4_lmhead_int8_abba_$STAMP}"
PORT="${PORT:-31003}"
CTX="${CTX:-131072}"
MAXREQ="${MAXREQ:-4}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp-c4-lmhead-int8-abba}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
PROD_NAME="${PROD_NAME:-qwen38_unsloth_ud_q4k_xl_tp2}"
PROD_PORT="${PROD_PORT:-18080}"
KDIR="${KDIR:-/mnt/vm_8tb/b70/w8a8_kernel}"
PUSHDIR="${PUSHDIR:-$REPO/vllm/contrib/vllm_push_allreduce/prebuilt}"
PUSH_AR_SO="${PUSH_AR_SO:-/work/push_ar/libxpu_push_ar_graph.so}"
PUSH_AR_MAXB="${PUSH_AR_MAXB:-536870912}"

KERNEL_SO="$KDIR/_xpu_C.abi3.so"
W8A16_SOURCE="$REPO/kernels/int8_gemm_w8a16.h"
W8A8_SOURCE="$REPO/kernels/int8_gemm_w8a8.h"
COMMON_SOURCE="$REPO/kernels/int8_quant_common.hpp"
KERNEL_PATCH="$REPO/kernels/int8_gemm_kernel.patch"
W8A8_SHIM="$REPO/sglang/patches/w8a8_shim.py"
PUSH_PATCH="$REPO/sglang/patches/push_ar_xpu.py"
PUSH_HOST_SO="$PUSHDIR/$(basename "$PUSH_AR_SO")"

for artifact in "$SHELF" "$KERNEL_SO" "$W8A16_SOURCE" "$W8A8_SOURCE" \
  "$COMMON_SOURCE" "$KERNEL_PATCH" "$W8A8_SHIM" "$PUSH_PATCH" "$PUSH_HOST_SO"; do
  [ -s "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 2; }
done
for op in int8_gemm_w8a16 int8_gemm_w8a8; do
  rg -a -q "$op" "$KERNEL_SO" || {
    echo "kernel SO is missing required op: $op" >&2
    exit 2
  }
done

mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1
active_name=""
active_dir=""

say() { echo "[c4-lmhead-abba $(date -u +%H:%M:%S)] $*"; }

save_active() {
  if [ -n "$active_name" ] && docker inspect "$active_name" >/dev/null 2>&1; then
    docker logs "$active_name" >"$active_dir/server.log" 2>&1 || true
    docker inspect "$active_name" >"$active_dir/container_inspect.json" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  save_active
  if [ -n "$active_name" ]; then
    docker stop -t 30 "$active_name" >/dev/null 2>&1 || true
    docker rm "$active_name" >/dev/null 2>&1 || true
    active_name=""
  fi
  if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
    | rg -qx "$PROD_NAME"; then
    docker stop -t 30 "$PROD_NAME" >/dev/null 2>&1 || true
    docker rm "$PROD_NAME" >/dev/null 2>&1 || true
  fi
  "$REPO/bin/xpu-health" >"$OUT/health_final.log" 2>&1 || rc=1
  cat "$OUT/health_final.log"
  say "exit rc=$rc artifacts=$OUT endpoint=down"
  exit "$rc"
}
trap cleanup EXIT INT TERM

sha256sum "$SHELF" "$W8A8_SHIM" "$KERNEL_SO" "$W8A16_SOURCE" \
  "$W8A8_SOURCE" "$COMMON_SOURCE" "$KERNEL_PATCH" "$PUSH_PATCH" \
  "$PUSH_HOST_SO" >"$OUT/artifacts.sha256"

verify_artifacts() {
  sha256sum --check "$OUT/artifacts.sha256"
}

{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "claim=lmhead_int8_serving_performance_and_coherence_qualification"
  echo "model=/models/qwen3.6-27b/w8a8-sqgptq"
  echo "served=$SERVED"
  echo "ctx=$CTX"
  echo "maxreq=$MAXREQ"
  echo "memfrac=0.90"
  echo "spec_steps=10"
  echo "spec_draft=11"
  echo "kdir=$KDIR"
  echo "pushdir=$PUSHDIR"
  echo "push_ar_so=$PUSH_AR_SO"
  echo "push_ar_min_numel=0"
  echo "push_ar_maxb=$PUSH_AR_MAXB"
  echo "replicate_mtp_embed=1"
  echo "delay_mlp_ar=0"
  echo "fused_mlp_ar_norm=0"
  echo "lmhead_compute=w8a16_only"
  echo "ccl_topo_p2p_access=0"
  echo "endpoint_policy=down_between_arms_and_after_campaign"
  echo "order=01_A1:lmhead0,02_B1:lmhead1,03_B2:lmhead1,04_A2:lmhead0"
  docker image inspect sglang-xpu:mtp --format 'image_id={{.Id}}' \
    2>/dev/null || true
} >"$OUT/manifest.txt"

say "snapshot and stop the daily-driver endpoint if present"
docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' >"$OUT/docker_ps_before.txt"
if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
  | rg -qx "$PROD_NAME"; then
  curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
    >"$OUT/production_models_before.json" || true
  docker inspect "$PROD_NAME" >"$OUT/production_inspect_before.json"
  docker stop -t 30 "$PROD_NAME" >"$OUT/production_stop.log"
  docker rm "$PROD_NAME" >>"$OUT/production_stop.log"
fi
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_pre_campaign.log"

validate_route_evidence() {
  local arm="$1"
  local enabled="$2"
  local log="$OUT/$arm/server.log"
  if [ "$enabled" = 0 ]; then
    if rg -q '\[lmhead-int8\]' "$log"; then
      echo "baseline unexpectedly installed lm_head INT8" >&2
      return 1
    fi
    return 0
  fi
  python3 - "$log" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="replace").read()
ready = re.findall(
    r"\[lmhead-int8\] ready role=(target|draft) rank=([01]) "
    r"N=124160 K=5120 storage=(replaced|aliased) w8a16_only=1 .*bf16_released=1",
    text,
)
shared = re.findall(
    r"\[lmhead-int8\] SHARED role=draft rank=([01]) "
    r"same_weight=1 same_scale=1 w8a16_only=1",
    text,
)
routes = [
    (role, int(rank), int(calls), int(rows))
    for role, rank, calls, rows in re.findall(
        r"\[lmhead-int8\] ROUTES role=(target|draft) rank=([01]) "
        r"calls=(\d+) latest_rows=(\d+) w8a16_only=1",
        text,
    )
]
expected_ready = {
    ("target", "0", "replaced"),
    ("target", "1", "replaced"),
    ("draft", "0", "aliased"),
    ("draft", "1", "aliased"),
}
expected_routes = {("target", 0), ("target", 1), ("draft", 0), ("draft", 1)}
ok = (
    len(ready) == 4
    and set(ready) == expected_ready
    and len(shared) == 2
    and set(shared) == {"0", "1"}
    and {(role, rank) for role, rank, calls, _rows in routes if calls == 1}
    == expected_routes
    and {(role, rank) for role, rank, calls, _rows in routes if calls >= 1000}
    == expected_routes
)
print(
    f"LMHEAD_ROUTE -> ready={ready} shared={shared} "
    f"routes={routes[-12:]} ok={ok}"
)
raise SystemExit(0 if ok else 1)
PY
}

run_arm() {
  local label="$1"
  local lmhead="$2"
  local slug
  slug="$(tr '[:upper:]' '[:lower:]' <<<"$label")"
  local name="c4_lmhead_abba_$slug"
  local arm_dir="$OUT/$label"
  mkdir -p "$arm_dir"
  active_name="$name"
  active_dir="$arm_dir"

  say "ARM $label start lmhead_int8=$lmhead"
  verify_artifacts 2>&1 | tee "$arm_dir/artifact_check_pre.log"
  "$REPO/bin/xpu-health" 2>&1 | tee "$arm_dir/health_pre.log"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" \
    SERVED="$SERVED" IMG=sglang-xpu:mtp \
    CKPT=/models/qwen3.6-27b/w8a8-sqgptq API_KEY= \
    CONFIG_OVERLAY= MEMFRAC=0.90 TOOLCALL=1 TOOLPARSER=qwen3_coder \
    REASONPARSER=qwen3 METRICS=1 THINKCAP=4096 \
    KDIR="$KDIR" PUSHDIR="$PUSHDIR" PUSH_AR_SO="$PUSH_AR_SO" \
    SPEC_STEPS=10 SPEC_DRAFT=11 REPLICATE_MTP_EMBED=1 \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=0 PUSH_AR_MAXB="$PUSH_AR_MAXB" \
    DELAY_MLP_AR=0 FUSED_MLP_AR_NORM=0 LMHEAD_INT8="$lmhead" \
    bash "$SHELF" start 2>&1 | tee "$arm_dir/start.log"

  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" \
    >"$arm_dir/models.json"
  curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
    >"$arm_dir/server_info.json"
  docker inspect "$name" >"$arm_dir/container_inspect.json"

  python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --max-tokens 128 --out "$arm_dir/deterministic.json" \
    2>&1 | tee "$arm_dir/deterministic.log"
  python3 "$REPO/vllm/cookbook_campaign/phase_bench.py" \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --prompt-tokens 2048 --gen-tokens 128 --n 5 --ignore-eos \
    --label "$label" --out "$arm_dir/phase_p2048_g128.json" \
    2>&1 | tee "$arm_dir/phase_p2048_g128.log"
  bash "$REPO/sglang/perf_regime.sh" "$name" "$PORT" "$SERVED" "$TOK" "$label" \
    2>&1 | tee "$arm_dir/perf_regime.log"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$SERVED" 1 8 2048 5 \
    2>&1 | tee "$arm_dir/prefill_c1.log"
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$SERVED" 4 8 2048 5 \
    2>&1 | tee "$arm_dir/prefill_c4.log"
  python3 "$REPO/vllm/nvfp4/bench_code.py" \
    "http://127.0.0.1:$PORT/v1" "$SERVED" 1 256 5 \
    2>&1 | tee "$arm_dir/code_c1.log"
  python3 "$REPO/vllm/nvfp4/bench_code.py" \
    "http://127.0.0.1:$PORT/v1" "$SERVED" 4 256 5 \
    2>&1 | tee "$arm_dir/code_c4.log"
  python3 "$REPO/vllm/gate_concurrent_coherence.py" \
    "http://127.0.0.1:$PORT/v1" "$SERVED" 4 6 256 \
    2>&1 | tee "$arm_dir/mixed.log"
  python3 "$REPO/sglang/soak_probe.py" "$PORT" "$SERVED" 6400 800 localhost \
    2>&1 | tee "$arm_dir/soak6400.log"

  save_active
  validate_route_evidence "$label" "$lmhead"
  if rg -i 'device_lost|out_of_resources|ur_result_error|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' \
    "$arm_dir/server.log"; then
    say "ARM $label fail: fatal marker"
    return 1
  fi
  say "ARM $label graceful stop"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" \
    SERVED="$SERVED" KDIR="$KDIR" PUSHDIR="$PUSHDIR" \
    PUSH_AR_SO="$PUSH_AR_SO" REPLICATE_MTP_EMBED=1 \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=0 DELAY_MLP_AR=0 FUSED_MLP_AR_NORM=0 \
    LMHEAD_INT8="$lmhead" bash "$SHELF" stop 2>&1 | tee "$arm_dir/stop.log"
  active_name=""
  active_dir=""
  "$REPO/bin/xpu-health" 2>&1 | tee "$arm_dir/health_post.log"
  say "ARM $label pass"
}

run_arm 01_A1 0
run_arm 02_B1 1
run_arm 03_B2 1
run_arm 04_A2 0

verify_artifacts 2>&1 | tee "$OUT/artifact_check_after.log"
sha256sum "$SHELF" "$W8A8_SHIM" "$KERNEL_SO" "$W8A16_SOURCE" \
  "$W8A8_SOURCE" "$COMMON_SOURCE" "$KERNEL_PATCH" "$PUSH_PATCH" \
  "$PUSH_HOST_SO" >"$OUT/artifacts_after.sha256"
cmp "$OUT/artifacts.sha256" "$OUT/artifacts_after.sha256"

if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
  | rg -qx "$PROD_NAME"; then
  echo "daily-driver container unexpectedly running" >&2
  exit 1
fi
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "experiment endpoint unexpectedly running" >&2
  exit 1
fi
echo "down" >"$OUT/endpoint_state_before_analysis.txt"

python3 "$REPO/sglang/analyze_c4_lmhead_int8_abba.py" "$OUT"
say "A-B-B-A complete; endpoint remains down"
