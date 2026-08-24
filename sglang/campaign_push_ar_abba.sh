#!/usr/bin/env bash
# Position-balanced shelf-promotion campaign for eager decode push all-reduce.
#
# The caller must hold both cards for the full lifetime:
#   ./bin/gpu-run bash sglang/campaign_push_ar_abba.sh
#
# Sequence: A1=1M gate, B1=push-all, B2=push-all, A2=1M gate. The script
# snapshots/stops a running Unsloth UD-Q4_K_XL daily driver and restores it on exit.
# Set B70_RESTORE_PROD=0 to intentionally leave production stopped afterward.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELF="$REPO/rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh"
PROD_SHELF="$REPO/rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$REPO/results/logs/sglang_push_ar_abba_$STAMP}"
PORT="${PORT:-31003}"
SERVED="${SERVED:-qwen36-27b-w8a8-gptq-mtp-push-abba}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
CTX="${CTX:-8192}"
MAXREQ="${MAXREQ:-4}"
MAXB="${PUSH_AR_MAXB:-536870912}"
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

say() { echo "[campaign $(date -u +%H:%M:%S)] $*"; }

save_active() {
  if [ -n "$active_name" ] && docker inspect "$active_name" >/dev/null 2>&1; then
    mkdir -p "$active_dir"
    docker logs "$active_name" >"$active_dir/server.log" 2>&1 || true
    docker inspect "$active_name" >"$active_dir/container_inspect.json" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  save_active
  if [ -n "$active_name" ]; then
    say "cleanup experiment container $active_name"
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
      say "restoring exact Unsloth UD-Q4_K_XL production shelf"
      NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" \
        bash "$PROD_SHELF" start >"$OUT/production_restore.log" 2>&1 || rc=1
      cat "$OUT/production_restore.log"
      if [ "$rc" = 0 ]; then
        curl -fsS --max-time 15 "http://127.0.0.1:$PROD_PORT/v1/models" \
          >"$OUT/production_models_after.json" || rc=1
        if [ "$rc" = 0 ]; then
          python3 - "$OUT/production_models_after.json" "$PROD_ID" <<'PY' || rc=1
import json
import sys
path, expected = sys.argv[1:]
model_id = json.load(open(path))["data"][0]["id"]
print(f"PRODUCTION_ID -> {model_id}")
raise SystemExit(0 if model_id == expected else 1)
PY
        fi
      fi
    else
      say "cards unhealthy; refusing another TP=2 start, production not restored"
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
  docker image inspect sglang-xpu:mtp --format 'image_id={{.Id}}' 2>/dev/null || true
  echo "model=/models/qwen3.6-27b/w8a8-sqgptq"
  echo "served=$SERVED"
  echo "ctx=$CTX"
  echo "maxreq=$MAXREQ"
  echo "port=$PORT"
  echo "push_ar=1"
  echo "push_ar_graph=0"
  echo "push_ar_maxb=$MAXB"
  echo "ccl_topo_p2p_access=0"
  echo "order=01_A1:1048576,02_B1:0,03_B2:0,04_A2:1048576"
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
  restore_prod="$B70_RESTORE_PROD"
  say "stopping production inside held lease"
  NAME="$PROD_NAME" PORT="$PROD_PORT" SERVED="$PROD_ID" bash "$PROD_SHELF" stop
fi

"$REPO/bin/xpu-health" | tee "$OUT/health_pre_campaign.log"

run_arm() {
  local label="$1"
  local min_numel="$2"
  local slug
  slug="$(tr '[:upper:]' '[:lower:]' <<<"$label")"
  local name="sglang_push_abba_${slug}"
  local arm_dir="$OUT/$label"
  mkdir -p "$arm_dir"
  active_name="$name"
  active_dir="$arm_dir"

  say "ARM $label START min_numel=$min_numel"
  "$REPO/bin/xpu-health" | tee "$arm_dir/health_pre.log"

  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL="$min_numel" PUSH_AR_MAXB="$MAXB" \
    bash "$SHELF" start 2>&1 | tee "$arm_dir/start.log"

  curl -fsS --max-time 15 "http://127.0.0.1:$PORT/v1/models" >"$arm_dir/models.json"
  docker inspect "$name" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sort >"$arm_dir/inspect_env.txt"
  python3 - "$arm_dir/models.json" "$SERVED" "$CTX" "$arm_dir/inspect_env.txt" "$min_numel" <<'PY'
import json
import sys
models_path, expected_id, expected_ctx, env_path, expected_min = sys.argv[1:]
model = json.load(open(models_path))["data"][0]
env = set(open(env_path).read().splitlines())
required = {
    "B70_XPU_PUSH_AR=1",
    f"PUSH_AR_MIN_NUMEL={expected_min}",
    "PUSH_AR_GRAPH=0",
    "CCL_TOPO_P2P_ACCESS=0",
}
missing = sorted(required - env)
print(f"IDENTITY -> id={model['id']} max_model_len={model['max_model_len']}")
print(f"ENV -> missing={missing}")
ok = model["id"] == expected_id and int(model["max_model_len"]) >= int(expected_ctx) and not missing
raise SystemExit(0 if ok else 1)
PY

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
    say "ARM $label FAIL fatal marker"
    return 1
  fi
  rg -q '\[push-ar\] ENGAGED: sglang .* -> push collective' "$arm_dir/server.log"
  if [ "$min_numel" = 0 ]; then
    rg -q '\[push-ar-stats\].*below_min\(0\)=0' "$arm_dir/server.log"
  else
    python3 - "$arm_dir/server.log" <<'PY'
import re
import sys
text = open(sys.argv[1], errors="replace").read()
vals = [int(x) for x in re.findall(r"below_min\(1048576\)=(\d+)", text)]
print(f"PUSH_STATS -> below_min_latest={vals[-1] if vals else 'missing'}")
raise SystemExit(0 if vals and vals[-1] > 0 else 1)
PY
  fi

  say "ARM $label graceful stop"
  CTX="$CTX" RADIX=0 MAXREQ="$MAXREQ" PORT="$PORT" NAME="$name" SERVED="$SERVED" \
    PUSH_AR=1 PUSH_AR_MIN_NUMEL="$min_numel" PUSH_AR_MAXB="$MAXB" \
    bash "$SHELF" stop 2>&1 | tee "$arm_dir/stop.log"
  active_name=""
  active_dir=""
  "$REPO/bin/xpu-health" | tee "$arm_dir/health_post.log"
  say "ARM $label PASS"
}

run_arm 01_A1 1048576
run_arm 02_B1 0
run_arm 03_B2 0
run_arm 04_A2 1048576

python3 "$REPO/sglang/analyze_push_ar_abba.py" "$OUT"
say "campaign measurements complete"
