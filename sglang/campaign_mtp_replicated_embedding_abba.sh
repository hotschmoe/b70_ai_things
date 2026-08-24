#!/usr/bin/env bash
# Position-balanced C3a serving qualification. Caller must hold both cards:
#   ./bin/gpu-run bash sglang/campaign_mtp_replicated_embedding_abba.sh
# Set B70_RESTORE_PROD=0 to intentionally leave production stopped afterward.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
PROD_SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$REPO/results/logs/mtp_replicated_embedding_abba_$STAMP}"
PORT="${PORT:-31003}"
CTX="${CTX:-131072}"
MAXREQ="${MAXREQ:-4}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp-replicated-abba}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
PROD_NAME="${PROD_NAME:-qwen38_unsloth_ud_q4k_xl_tp2}"
PROD_ID="${PROD_ID:-hotschmoe-dd}"
PROD_PORT="${PROD_PORT:-18080}"
B70_RESTORE_PROD="${B70_RESTORE_PROD:-1}"

case "$B70_RESTORE_PROD" in
  0|1) ;;
  *) echo "B70_RESTORE_PROD must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1
active_name=""
active_dir=""
restore_prod=0
restoring=0
say() { echo "[c3a-abba $(date -u +%H:%M:%S)] $*"; }

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
  fi
  local health_rc=0
  "$REPO/bin/xpu-health" >"$OUT/health_before_restore.log" 2>&1 || health_rc=$?
  cat "$OUT/health_before_restore.log"
  if [ "$restore_prod" = 1 ] && [ "$restoring" = 0 ]; then
    if [ "$health_rc" = 0 ]; then
      restoring=1
      say "restoring exact Unsloth UD-Q4_K_XL production shelf"
      NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" \
        bash "$PROD_SHELF" start >"$OUT/production_restore.log" 2>&1 || rc=1
      cat "$OUT/production_restore.log"
      curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
        >"$OUT/production_models_after.json" || rc=1
    else
      say "cards unhealthy; refusing another TP=2 start"
      rc=1
    fi
  fi
  say "exit rc=$rc artifacts=$OUT"
  exit "$rc"
}
trap cleanup EXIT INT TERM

{
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "utc_start=$STAMP"
  echo "kernel=$(uname -r)"
  echo "image=sglang-xpu:mtp"
  echo "model=/models/qwen3.6-27b/w8a8-sqgptq"
  echo "served=$SERVED"
  echo "ctx=$CTX"
  echo "maxreq=$MAXREQ"
  echo "push_ar=1"
  echo "push_ar_min_numel=1048576"
  echo "p2paccess=0"
  echo "order=01_A1:0,02_B1:1,03_B2:1,04_A2:0"
} >"$OUT/manifest.txt"

say "snapshot current production"
if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' | rg -qx "$PROD_NAME"; then
  curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
    >"$OUT/production_models_before.json"
  python3 - "$OUT/production_models_before.json" "$PROD_ID" <<'PY'
import json
import sys
path, expected = sys.argv[1:]
actual = json.load(open(path))["data"][0]["id"]
print(f"PRODUCTION_SNAPSHOT -> id={actual}")
raise SystemExit(0 if actual == expected else 1)
PY
  restore_prod="$B70_RESTORE_PROD"
  NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" bash "$PROD_SHELF" stop
fi
"$REPO/bin/xpu-health" | tee "$OUT/health_pre_campaign.log"

run_arm() {
  local label="$1"
  local replicate="$2"
  local name="c3a_abba_$(tr '[:upper:]' '[:lower:]' <<<"$label")"
  local arm_dir="$OUT/$label"
  mkdir -p "$arm_dir"
  active_name="$name"
  active_dir="$arm_dir"
  say "ARM $label start replicate=$replicate"
  "$REPO/bin/xpu-health" | tee "$arm_dir/health_pre.log"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=1048576 REPLICATE_MTP_EMBED="$replicate" \
    bash "$SHELF" start 2>&1 | tee "$arm_dir/start.log"

  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$arm_dir/models.json"
  curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" >"$arm_dir/server_info.json"
  docker inspect "$name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sort >"$arm_dir/inspect_env.txt"
  rg -q "B70_XPU_REPLICATE_MTP_EMBED=$replicate" "$arm_dir/inspect_env.txt"
  if [ "$replicate" = 1 ]; then
    save_active
    [ "$(rg -c '\[mtp-replicated-embed\] target ENABLED' "$arm_dir/server.log")" = 2 ]
    [ "$(rg -c '\[mtp-replicated-embed\] draft SHARE OK' "$arm_dir/server.log")" = 2 ]
  fi

  python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --out "$arm_dir/deterministic.json" 2>&1 | tee "$arm_dir/deterministic.log"
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
    return 1
  fi
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=1048576 REPLICATE_MTP_EMBED="$replicate" \
    bash "$SHELF" stop 2>&1 | tee "$arm_dir/stop.log"
  active_name=""
  active_dir=""
  "$REPO/bin/xpu-health" | tee "$arm_dir/health_post.log"
  say "ARM $label pass"
}

run_arm 01_A1 0
run_arm 02_B1 1
run_arm 03_B2 1
run_arm 04_A2 0
python3 "$REPO/sglang/analyze_mtp_replicated_embedding_abba.py" "$OUT"
say "A-B-B-A complete"
