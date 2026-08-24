#!/usr/bin/env bash
# Position-balanced C3b serving qualification for the fused delayed-MLP
# all-reduce + residual + Gemma RMSNorm boundary.
#
# Caller must hold both cards for the entire campaign:
#   ./bin/gpu-run bash sglang/campaign_c3b_fused_boundary_abba.sh
#
# A = current promoted push AR + replicated MTP embedding, production push SO.
# B = the same stack plus delayed MLP AR and the FAST_MAX_ROWS=11 fused SO.
# The endpoint stays down between arms and after the campaign. There is no
# production restore path in this script.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c3b_fused_boundary_abba_$STAMP}"
PORT="${PORT:-31003}"
CTX="${CTX:-131072}"
MAXREQ="${MAXREQ:-4}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp-c3b-fused-abba}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
PROD_NAME="${PROD_NAME:-qwen38_unsloth_ud_q4k_xl_tp2}"
PROD_PORT="${PROD_PORT:-18080}"
BASE_PUSHDIR="${BASE_PUSHDIR:-$REPO/vllm/contrib/vllm_push_allreduce/prebuilt}"
BASE_PUSH_AR_SO="${BASE_PUSH_AR_SO:-/work/push_ar/libxpu_push_ar_graph.so}"
CAND_PUSHDIR="${CAND_PUSHDIR:-/mnt/vm_8tb/b70/fused_ar_rmsnorm}"
CAND_PUSH_AR_SO="${CAND_PUSH_AR_SO:-/work/push_ar/libxpu_push_ar_fused_rmsnorm.so}"
CAND_FAST_MAX_ROWS="${CAND_FAST_MAX_ROWS:-11}"
PUSH_AR_MIN_NUMEL="${PUSH_AR_MIN_NUMEL:-1048576}"
PUSH_AR_MAXB="${PUSH_AR_MAXB:-536870912}"
MECHANISM_TOKENS="${MECHANISM_TOKENS:-1600}"
MECHANISM_WINDOW="${MECHANISM_WINDOW:-400}"

BASE_HOST_SO="$BASE_PUSHDIR/$(basename "$BASE_PUSH_AR_SO")"
CAND_HOST_SO="$CAND_PUSHDIR/$(basename "$CAND_PUSH_AR_SO")"
KERNEL_SOURCE="$REPO/kernels/xpu_push_ar_fused_rmsnorm.cpp"
BASE_SOURCE="$REPO/vllm/contrib/vllm_push_allreduce/118_xpu_push_ar_graph.cpp"
DELAY_PATCH="$REPO/sglang/patches/xpu_delayed_mlp_ar.py"
FUSED_PATCH="$REPO/sglang/patches/xpu_fused_mlp_ar_norm.py"
PUSH_PATCH="$REPO/sglang/patches/push_ar_xpu.py"
WOQ_SHIM="$REPO/sglang/patches/woq_shim.py"

[ "$CAND_FAST_MAX_ROWS" = 11 ] || {
  echo "CAND_FAST_MAX_ROWS must be exactly 11 for this campaign" >&2
  exit 2
}
[ "$PUSH_AR_MIN_NUMEL" = 1048576 ] || {
  echo "PUSH_AR_MIN_NUMEL must remain at the promoted 1048576 baseline" >&2
  exit 2
}
for artifact in "$BASE_HOST_SO" "$CAND_HOST_SO" "$KERNEL_SOURCE" \
  "$BASE_SOURCE" "$DELAY_PATCH" "$FUSED_PATCH" "$PUSH_PATCH" "$WOQ_SHIM"; do
  [ -s "$artifact" ] || { echo "missing artifact: $artifact" >&2; exit 2; }
done
nm -D --defined-only "$CAND_HOST_SO" \
  | rg -q ' ar_allreduce_residual_gemma_rmsnorm_bf16$' || {
    echo "candidate SO does not export the fused boundary ABI" >&2
    exit 2
  }

mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1
active_name=""
active_dir=""

say() { echo "[c3b-fused-abba $(date -u +%H:%M:%S)] $*"; }

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

sha256sum "$BASE_SOURCE" "$BASE_HOST_SO" "$KERNEL_SOURCE" "$CAND_HOST_SO" \
  "$PUSH_PATCH" "$DELAY_PATCH" "$FUSED_PATCH" "$WOQ_SHIM" \
  >"$OUT/artifacts.sha256"

verify_artifacts() {
  sha256sum --check "$OUT/artifacts.sha256"
}

{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "claim=serving_performance_and_coherence_qualification"
  echo "model=/models/qwen3.6-27b/w8a8-sqgptq"
  echo "served=$SERVED"
  echo "ctx=$CTX"
  echo "maxreq=$MAXREQ"
  echo "replicate_mtp_embed=1"
  echo "push_ar_min_numel=$PUSH_AR_MIN_NUMEL"
  echo "push_ar_maxb=$PUSH_AR_MAXB"
  echo "base_pushdir=$BASE_PUSHDIR"
  echo "base_push_ar_so=$BASE_PUSH_AR_SO"
  echo "candidate_pushdir=$CAND_PUSHDIR"
  echo "candidate_push_ar_so=$CAND_PUSH_AR_SO"
  echo "candidate_fast_max_rows=$CAND_FAST_MAX_ROWS"
  echo "mechanism_tokens=$MECHANISM_TOKENS"
  echo "mechanism_window=$MECHANISM_WINDOW"
  echo "ccl_topo_p2p_access=0"
  echo "endpoint_policy=down_between_arms_and_after_campaign"
  echo "order=01_A1:baseline,02_B1:fused,03_B2:fused,04_A2:baseline"
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

validate_mechanism_snapshot() {
  local arm="$1"
  local enabled="$2"
  local log="$OUT/$arm/mechanism.log"
  if [ "$enabled" = 0 ]; then
    if rg -q '\[c3b-(fused|delayed-mlp)\]' "$log"; then
      echo "baseline unexpectedly installed C3b" >&2
      return 1
    fi
    return 0
  fi
  python3 - "$log" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="replace").read()
routes = [tuple(map(int, row)) for row in re.findall(
    r"\[c3b-delayed-mlp\] ROUTES rank=(\d+) eligible=(\d+) "
    r"consumed=(\d+) generic=(\d+)", text
)]
latest = {}
for rank, eligible, consumed, generic in routes:
    latest[rank] = (eligible, consumed, generic)
m11 = [int(value) for value in re.findall(
    r"\[c3b-fused\] calls=(\d+) rows=11 hidden=5120", text
)]
ok = (
    set(latest) == {0, 1}
    and all(e == c and g == 0 and c >= 4096 for e, c, g in latest.values())
    and len(m11) >= 2
    and max(m11, default=0) >= 4096
)
print(f"MECHANISM -> latest={latest} m11={m11[-4:]} ok={ok}")
raise SystemExit(0 if ok else 1)
PY
}

run_arm() {
  local label="$1"
  local fused="$2"
  local pushdir="$BASE_PUSHDIR"
  local push_so="$BASE_PUSH_AR_SO"
  local slug
  slug="$(tr '[:upper:]' '[:lower:]' <<<"$label")"
  local name="c3b_fused_abba_$slug"
  local arm_dir="$OUT/$label"
  if [ "$fused" = 1 ]; then
    pushdir="$CAND_PUSHDIR"
    push_so="$CAND_PUSH_AR_SO"
  fi
  mkdir -p "$arm_dir"
  active_name="$name"
  active_dir="$arm_dir"

  say "ARM $label start fused=$fused push_so=$push_so"
  verify_artifacts 2>&1 | tee "$arm_dir/artifact_check_pre.log"
  "$REPO/bin/xpu-health" 2>&1 | tee "$arm_dir/health_pre.log"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" \
    SERVED="$SERVED" IMG=sglang-xpu:mtp \
    CKPT=/models/qwen3.6-27b/w8a8-sqgptq API_KEY= \
    SPEC_STEPS=10 SPEC_DRAFT=11 REPLICATE_MTP_EMBED=1 \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL="$PUSH_AR_MIN_NUMEL" \
    PUSH_AR_MAXB="$PUSH_AR_MAXB" PUSHDIR="$pushdir" \
    PUSH_AR_SO="$push_so" DELAY_MLP_AR="$fused" \
    FUSED_MLP_AR_NORM="$fused" \
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

  # This c1-only warm/soak block must drive the candidate past the 4096-call
  # report before any c4 shape can legitimately take the generic fallback.
  python3 "$REPO/sglang/soak_probe.py" "$PORT" "$SERVED" \
    "$MECHANISM_TOKENS" "$MECHANISM_WINDOW" localhost \
    2>&1 | tee "$arm_dir/mechanism_soak.log"
  docker logs "$name" >"$arm_dir/mechanism.log" 2>&1
  validate_mechanism_snapshot "$label" "$fused"

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
  if rg -i 'device_lost|out_of_resources|ur_result_error|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' \
    "$arm_dir/server.log"; then
    say "ARM $label fail: fatal marker"
    return 1
  fi
  say "ARM $label graceful stop"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" \
    SERVED="$SERVED" PUSH_AR=1 PUSH_AR_MIN_NUMEL="$PUSH_AR_MIN_NUMEL" \
    PUSHDIR="$pushdir" PUSH_AR_SO="$push_so" DELAY_MLP_AR="$fused" \
    FUSED_MLP_AR_NORM="$fused" bash "$SHELF" stop \
    2>&1 | tee "$arm_dir/stop.log"
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
sha256sum "$BASE_SOURCE" "$BASE_HOST_SO" "$KERNEL_SOURCE" "$CAND_HOST_SO" \
  "$PUSH_PATCH" "$DELAY_PATCH" "$FUSED_PATCH" "$WOQ_SHIM" \
  >"$OUT/artifacts_after.sha256"
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

python3 "$REPO/sglang/analyze_c3b_fused_boundary_abba.py" "$OUT"
say "A-B-B-A complete; endpoint remains down"
