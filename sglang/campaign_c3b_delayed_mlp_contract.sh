#!/usr/bin/env bash
# Contract-only C3b A/B for SGLang's delayed MLP-AR marker route on XPU.
# No timing is collected and this script cannot authorize a performance promotion.
# Caller holds both cards for the entire stop/A/B/restore block:
#   ./bin/gpu-run bash sglang/campaign_c3b_delayed_mlp_contract.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
PROD_SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-q4km/serve.sh"
PATCH="$REPO/sglang/patches/xpu_delayed_mlp_ar.py"
WOQ_SHIM="$REPO/sglang/patches/woq_shim.py"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$REPO/results/logs/c3b_delayed_mlp_contract_$STAMP}"
PORT="${PORT:-31003}"
CTX="${CTX:-4096}"
MAXREQ="${MAXREQ:-1}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp-c3b-contract}"
PROD_NAME="${PROD_NAME:-qwen38_stock_q4km_tp2}"
PROD_ID="${PROD_ID:-hotschmoe-dd}"
PROD_PORT="${PROD_PORT:-18080}"
REAL_DOCKER="$(type -P docker)"

[ -f "$PATCH" ] || {
  echo "missing candidate patch: $PATCH" >&2
  exit 2
}
[ -f "$WOQ_SHIM" ] || {
  echo "missing install wiring: $WOQ_SHIM" >&2
  exit 2
}
PATCH_SHA256="$(sha256sum "$PATCH" | awk '{print $1}')"
WOQ_SHA256="$(sha256sum "$WOQ_SHIM" | awk '{print $1}')"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/campaign.log") 2>&1

active_name=""
active_dir=""
production_expected=0
production_restored=0
production_restore_attempted=0

say() { echo "[c3b-contract $(date -u +%H:%M:%S)] $*"; }

verify_sources() {
  [ "$(sha256sum "$PATCH" | awk '{print $1}')" = "$PATCH_SHA256" ]
  [ "$(sha256sum "$WOQ_SHIM" | awk '{print $1}')" = "$WOQ_SHA256" ]
}

# The shelf intentionally has no C3b knob.  Intercept only the named experiment
# container's docker-run and inject the candidate module and flag.  All shelf
# health, lifecycle, model, and production commands otherwise remain exact.
docker() {
  local action="${1:-}"
  shift || true
  if [ "$action" = run ]; then
    local args=" $* "
    if [[ "$args" == *" --name $C3B_CONTAINER "* ]]; then
      "$C3B_REAL_DOCKER" run \
        -v "$C3B_PATCH:/opt/venv/lib/python3.12/site-packages/xpu_delayed_mlp_ar.py:ro" \
        -e "B70_XPU_DELAY_MLP_AR=$C3B_ENABLE" "$@"
      return
    fi
  fi
  "$C3B_REAL_DOCKER" "$action" "$@"
}
export -f docker
export C3B_REAL_DOCKER="$REAL_DOCKER" C3B_PATCH="$PATCH"
export C3B_CONTAINER="__none__" C3B_ENABLE=0

save_active() {
  if [ -n "$active_name" ] && "$REAL_DOCKER" inspect "$active_name" >/dev/null 2>&1; then
    "$REAL_DOCKER" logs "$active_name" >"$active_dir/server.log" 2>&1 || true
    "$REAL_DOCKER" inspect "$active_name" >"$active_dir/container_inspect.json" 2>/dev/null || true
  fi
}

restore_production() {
  [ "$production_expected" = 1 ] || return 0
  [ "$production_restored" = 0 ] || return 0
  [ "$production_restore_attempted" = 0 ] || return 1
  production_restore_attempted=1
  say "restoring exact stock Q4_K_M production shelf"
  if ! NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" \
    CTX_SIZE=262144 BATCH=1024 UBATCH=256 LAB_DOORS=0 ENABLE_MTP=0 \
    bash "$PROD_SHELF" start >"$OUT/production_restore.log" 2>&1; then
    cat "$OUT/production_restore.log"
    "$REAL_DOCKER" logs "$PROD_NAME" >"$OUT/production_restore_server.log" 2>&1 \
      || true
    NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" \
      bash "$PROD_SHELF" stop >"$OUT/production_restore_stop.log" 2>&1 || true
    "$REPO/bin/xpu-health" >"$OUT/health_after_failed_restore.log" 2>&1 || true
    return 1
  fi
  cat "$OUT/production_restore.log"
  if ! curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
    >"$OUT/production_models_after.json" || ! python3 - \
    "$OUT/production_models_after.json" "$PROD_ID" <<'PY'
import json
import sys
path, expected = sys.argv[1:]
actual = json.load(open(path))["data"][0]["id"]
print(f"PRODUCTION_ID -> {actual}")
raise SystemExit(0 if actual == expected else 1)
PY
  then
    NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" \
      bash "$PROD_SHELF" stop >"$OUT/production_restore_stop.log" 2>&1 || true
    "$REPO/bin/xpu-health" >"$OUT/health_after_failed_restore.log" 2>&1 || true
    return 1
  fi
  if ! curl -fsS --max-time 15 -o /dev/null -w '%{http_code}\n' \
    "http://127.0.0.1:$PROD_PORT/health" \
    >"$OUT/production_health_after.txt" || ! "$REAL_DOCKER" inspect \
    "$PROD_NAME" >"$OUT/production_inspect_after.json"; then
    NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" \
      bash "$PROD_SHELF" stop >"$OUT/production_restore_stop.log" 2>&1 || true
    "$REPO/bin/xpu-health" >"$OUT/health_after_failed_restore.log" 2>&1 || true
    return 1
  fi
  production_restored=1
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  save_active
  if [ -n "$active_name" ]; then
    "$REAL_DOCKER" stop -t 30 "$active_name" >/dev/null 2>&1 || true
    "$REAL_DOCKER" rm "$active_name" >/dev/null 2>&1 || true
    active_name=""
  fi
  if [ "$production_expected" = 1 ] && [ "$production_restored" = 0 ] \
    && [ "$production_restore_attempted" = 0 ]; then
    local health_rc=0
    "$REPO/bin/xpu-health" >"$OUT/health_before_restore.log" 2>&1 || health_rc=$?
    cat "$OUT/health_before_restore.log"
    if [ "$health_rc" = 0 ]; then
      restore_production || rc=1
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
  echo "claim=contract_only_no_performance_promotion"
  echo "candidate_patch_sha256=$PATCH_SHA256"
  echo "woq_shim_sha256=$WOQ_SHA256"
  echo "model=/models/qwen3.6-27b/w8a8-sqgptq"
  echo "served=$SERVED"
  echo "ctx=$CTX"
  echo "maxreq=$MAXREQ"
  echo "replicate_mtp_embed=1"
  echo "push_ar_min_numel=1048576"
  echo "expected_target_edges=63"
  echo "order=baseline,candidate"
  "$REAL_DOCKER" image inspect sglang-xpu:mtp --format 'image_id={{.Id}}' \
    2>/dev/null || true
} >"$OUT/manifest.txt"

say "snapshot current production"
if "$REAL_DOCKER" ps --filter "name=^/${PROD_NAME}$" --format '{{.Names}}' \
  | rg -qx "$PROD_NAME"; then
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
  "$REAL_DOCKER" inspect "$PROD_NAME" >"$OUT/production_inspect_before.json"
  production_expected=1
  NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" bash "$PROD_SHELF" stop
fi
echo "$production_expected" >"$OUT/production_expected.txt"
"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_pre_campaign.log"

run_arm() {
  local arm="$1"
  local enable="$2"
  local name="c3b_contract_$arm"
  local arm_dir="$OUT/$arm"
  mkdir -p "$arm_dir"
  active_name="$name"
  active_dir="$arm_dir"
  export C3B_CONTAINER="$name" C3B_ENABLE="$enable"

  say "ARM $arm start delayed_mlp_ar=$enable"
  verify_sources
  "$REPO/bin/xpu-health" 2>&1 | tee "$arm_dir/health_pre.log"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    IMG=sglang-xpu:mtp CKPT=/models/qwen3.6-27b/w8a8-sqgptq API_KEY= \
    SPEC_STEPS=10 SPEC_DRAFT=11 REPLICATE_MTP_EMBED=1 \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL=1048576 \
    bash "$SHELF" start 2>&1 | tee "$arm_dir/start.log"

  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$arm_dir/models.json"
  "$REAL_DOCKER" inspect "$name" >"$arm_dir/container_inspect.json"
  python3 "$REPO/sglang/deterministic_equivalence_probe.py" \
    --base "http://127.0.0.1:$PORT" --model "$SERVED" \
    --max-tokens 128 --out "$arm_dir/deterministic.json" \
    2>&1 | tee "$arm_dir/deterministic.log"

  save_active
  if rg -i 'device_lost|out_of_resources|ur_result_error|enginedead|!!!!|(^|[^a-z])nan([^a-z]|$)' \
    "$arm_dir/server.log"; then
    say "ARM $arm fatal marker"
    return 1
  fi

  say "ARM $arm graceful stop"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    REPLICATE_MTP_EMBED=1 PUSH_AR=1 PUSH_AR_MIN_NUMEL=1048576 \
    bash "$SHELF" stop 2>&1 | tee "$arm_dir/stop.log"
  active_name=""
  active_dir=""
  "$REPO/bin/xpu-health" 2>&1 | tee "$arm_dir/health_post.log"
}

run_arm baseline 0
run_arm candidate 1
verify_sources

"$REPO/bin/xpu-health" 2>&1 | tee "$OUT/health_before_restore.log"
restore_production

python3 "$REPO/sglang/analyze_c3b_delayed_mlp_contract.py" "$OUT" \
  --served "$SERVED" --eligible 63 --production-id "$PROD_ID" \
  --production-name "$PROD_NAME"
say "contract-only A/B complete; no performance claim"
