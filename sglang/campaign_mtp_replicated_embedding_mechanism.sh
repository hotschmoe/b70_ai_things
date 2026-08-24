#!/usr/bin/env bash
# C3a mechanism and deterministic-equivalence gate for native embedding replication.
# Caller must hold both cards for the entire block:
#   ./bin/gpu-run bash sglang/campaign_mtp_replicated_embedding_mechanism.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
PROD_SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-q4km/serve.sh"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/mtp_replicated_embedding_mechanism_$STAMP}"
PROFILE_ROOT="c3_mtp_replicated_embedding_$STAMP"
PORT="${PORT:-31003}"
CTX="${CTX:-131072}"
MAXREQ="${MAXREQ:-4}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp-replicated-mechanism}"
PROD_NAME="${PROD_NAME:-qwen38_stock_q4km_tp2}"
PROD_ID="${PROD_ID:-hotschmoe-dd}"
PROD_PORT="${PROD_PORT:-18080}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1

active_name=""
active_dir=""
restore_prod=0
restoring=0

say() { echo "[c3a $(date -u +%H:%M:%S)] $*"; }

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

  local health_rc=0
  "$REPO/bin/xpu-health" >"$OUT/health_before_restore.log" 2>&1 || health_rc=$?
  cat "$OUT/health_before_restore.log"
  if [ "$restore_prod" = 1 ] && [ "$restoring" = 0 ]; then
    if [ "$health_rc" = 0 ]; then
      restoring=1
      say "restoring exact stock Q4_K_M production shelf"
      NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" \
        bash "$PROD_SHELF" start >"$OUT/production_restore.log" 2>&1 || rc=1
      cat "$OUT/production_restore.log"
      if [ "$rc" = 0 ]; then
        curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
          >"$OUT/production_models_after.json" || rc=1
      fi
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
  echo "push_ar_graph=0"
  echo "ccl_topo_p2p_access=0"
  echo "order=baseline,replicated"
  docker image inspect sglang-xpu:mtp --format 'image_id={{.Id}}' 2>/dev/null || true
} >"$OUT/manifest.txt"

say "snapshot current production"
docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}' \
  >"$OUT/production_ps_before.txt"
if docker ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' | rg -qx "$PROD_NAME"; then
  curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
    >"$OUT/production_models_before.json"
  python3 - "$OUT/production_models_before.json" "$PROD_ID" <<'PY'
import json
import sys
path, expected = sys.argv[1:]
model_id = json.load(open(path))["data"][0]["id"]
print(f"PRODUCTION_SNAPSHOT -> id={model_id}")
raise SystemExit(0 if model_id == expected else 1)
PY
  docker inspect "$PROD_NAME" >"$OUT/production_inspect_before.json"
  restore_prod=1
  NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" bash "$PROD_SHELF" stop
fi
"$REPO/bin/xpu-health" | tee "$OUT/health_pre_campaign.log"

run_arm() {
  local arm="$1"
  local replicate="$2"
  local expected="$3"
  local name="c3a_mtp_embed_$arm"
  local arm_dir="$OUT/$arm"
  local host_profile="$ROOT/sgl_cache/$PROFILE_ROOT/$arm"
  local container_profile="/sgl_cache/$PROFILE_ROOT/$arm"
  mkdir -p "$arm_dir" "$host_profile"
  active_name="$name"
  active_dir="$arm_dir"

  say "ARM $arm start replicate=$replicate"
  "$REPO/bin/xpu-health" | tee "$arm_dir/health_pre.log"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=1048576 REPLICATE_MTP_EMBED="$replicate" \
    bash "$SHELF" start 2>&1 | tee "$arm_dir/start.log"

  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$arm_dir/models.json"
  curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" >"$arm_dir/server_info.json"
  docker inspect "$name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sort >"$arm_dir/inspect_env.txt"
  for device in 0 1; do
    xpu-smi stats -d "$device" >"$arm_dir/xpu_stats_$device.txt" 2>&1 || true
  done
  rg -q "B70_XPU_REPLICATE_MTP_EMBED=$replicate" "$arm_dir/inspect_env.txt"
  if [ "$replicate" = 1 ]; then
    save_active
    [ "$(rg -c '\[mtp-replicated-embed\] target ENABLED' "$arm_dir/server.log")" = 2 ]
    [ "$(rg -c '\[mtp-replicated-embed\] draft SHARE OK' "$arm_dir/server.log")" = 2 ]
  fi

  python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --out "$arm_dir/deterministic.json" 2>&1 | tee "$arm_dir/deterministic.log"

  say "ARM $arm profile five decode iterations"
  curl -fsS -X POST "http://127.0.0.1:$PORT/start_profile" \
    -H 'content-type: application/json' \
    -d "{\"output_dir\":\"$container_profile\",\"num_steps\":5,\"activities\":[\"CPU\",\"XPU\"],\"profile_by_stage\":true,\"record_shapes\":true,\"with_stack\":false,\"profile_prefix\":\"$arm\"}" \
    >"$arm_dir/start_profile_response.txt"
  local trigger_rc=0
  set +e
  python3 "$REPO/vllm/nvfp4/bench_prefill.py" \
    "http://127.0.0.1:$PORT/v1" "$SERVED" 1 32 2048 1 \
    2>&1 | tee "$arm_dir/profile_trigger.log"
  trigger_rc=${PIPESTATUS[0]}
  set -e
  echo "PROFILE_TRIGGER_RC -> $trigger_rc (trace census is authoritative)"

  local found=0
  for _ in $(seq 1 60); do
    if [ "$(find "$host_profile" -maxdepth 1 -type f -name '*DECODE.trace.json.gz' | wc -l)" = 2 ]; then
      found=1
      break
    fi
    sleep 1
  done
  [ "$found" = 1 ]
  python3 "$REPO/sglang/parse_tp2_collective_census.py" \
    "$host_profile" --arm "$expected" 2>&1 | tee "$arm_dir/collective_census.log"
  python3 "$REPO/scripts/112_parse_trace.py" "$host_profile" \
    >"$arm_dir/device_time.txt"

  save_active
  if rg -i 'device_lost|out_of_resources|ur_result_error|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' \
    "$arm_dir/server.log"; then
    say "ARM $arm fatal marker"
    return 1
  fi

  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=1048576 REPLICATE_MTP_EMBED="$replicate" \
    bash "$SHELF" stop 2>&1 | tee "$arm_dir/stop.log"
  active_name=""
  active_dir=""
  "$REPO/bin/xpu-health" | tee "$arm_dir/health_post.log"
  say "ARM $arm pass"
}

if [ "$SKIP_BASELINE" = 1 ]; then
  say "reusing preserved baseline artifacts in $OUT/baseline"
  [ -s "$OUT/baseline/deterministic.json" ]
  python3 "$REPO/sglang/parse_tp2_collective_census.py" \
    "$ROOT/sgl_cache/$PROFILE_ROOT/baseline" --arm baseline \
    2>&1 | tee "$OUT/baseline/collective_census.log"
else
  run_arm baseline 0 baseline
fi
run_arm replicated 1 replicated

cmp "$OUT/baseline/deterministic.json" "$OUT/replicated/deterministic.json"
echo "DETERMINISTIC_EQUIVALENCE -> PASS byte-identical 8/8"
say "mechanism pair complete"
