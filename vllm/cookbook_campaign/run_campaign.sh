#!/usr/bin/env bash
# End-to-end cookbook campaign for items 1-3 + 5 (methodology).
# Assumes DD is already stopped and card free.
#
# Env:
#   RESULTS_DIR  default results/cookbook_campaign/<ts>
#   CARD         default 0
#   ONLY         comma list: moe,dense,boundary,public  (default all available)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_DIR="${RESULTS_DIR:-$REPO/results/cookbook_campaign/$TS}"
mkdir -p "$RESULTS_DIR"
CARD="${CARD:-0}"
PORT="${PORT:-8000}"
ONLY="${ONLY:-moe,dense,boundary,public}"
PHASE_BENCH="$REPO/vllm/cookbook_campaign/phase_bench.py"
LAUNCH="$REPO/vllm/cookbook_campaign/launch.sh"
WAIT="$REPO/vllm/cookbook_campaign/wait_healthy.sh"
LOG="$RESULTS_DIR/campaign.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== campaign start $TS results=$RESULTS_DIR card=$CARD only=$ONLY ==="

have() { echo ",$ONLY," | grep -q ",$1,"; }

stop_cb() {
  docker ps -a --format '{{.Names}}' | grep -E '^b70_cb_' | while read -r n; do
    echo "stopping $n"; docker rm -f "$n" >/dev/null 2>&1 || true
  done
}

bench_cell() {
  local label=$1 model=$2 p=$3 g=$4 n=${5:-3}
  python3 "$PHASE_BENCH" \
    --base "http://127.0.0.1:$PORT" \
    --model "$model" \
    --prompt-tokens "$p" \
    --gen-tokens "$g" \
    --n "$n" \
    --label "$label" \
    --out "$RESULTS_DIR/${label}_p${p}_g${g}.json" || true
}

run_track_modes() {
  local track=$1
  shift
  local modes=("$@")
  local mode model
  for mode in "${modes[@]}"; do
    echo "=== $track $mode ==="
    stop_cb
    if ! bash "$LAUNCH" "$track" "$mode" on "$PORT" "$CARD"; then
      echo "LAUNCH FAILED $track $mode" | tee -a "$RESULTS_DIR/failures.txt"
      continue
    fi
    NAME="b70_cb_${track}_${mode}_on"
    if ! TIMEOUT=1500 bash "$WAIT" "$PORT" "$NAME"; then
      echo "HEALTH FAILED $track $mode" | tee -a "$RESULTS_DIR/failures.txt"
      docker logs "$NAME" >"$RESULTS_DIR/${track}_${mode}_boot.log" 2>&1 || true
      continue
    fi
    model=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
    echo "served model=$model"
    # short cells (item 1 / item 3 style)
    bench_cell "${track}_${mode}_short" "$model" 512 128 3
    bench_cell "${track}_${mode}_mid" "$model" 2048 128 3
    # save logs
    docker logs "$NAME" >"$RESULTS_DIR/${track}_${mode}_server.log" 2>&1 || true
  done
}

# --- item 1: MoE MTP reopen ---
if have moe; then
  if [ -d models/files/community/qwen36-35b-gptq-mtp-preserved ] \
     && [ "$(du -sm models/files/community/qwen36-35b-gptq-mtp-preserved | awk '{print $1}')" -ge 15000 ]; then
    run_track_modes moe35-gptq no-spec mtp2 mtp4
  elif [ -d models/files/qwen3.6-35b-a3b/int4-autoround ]; then
    echo "=== MoE GPTQ not ready; fallback autoround (quantized MTP experts) ==="
    run_track_modes moe35-autoround no-spec mtp2
  else
    echo "SKIP moe: no checkpoint" | tee -a "$RESULTS_DIR/failures.txt"
  fi
fi

# --- item 3 + dense part of 1: public/dense MTP ---
if have dense || have public; then
  if [ -d models/files/community/qwen36-27b-gptq-mtp-preserved ] \
     && [ "$(du -sm models/files/community/qwen36-27b-gptq-mtp-preserved | awk '{print $1}')" -ge 12000 ]; then
    run_track_modes dense27-gptq no-spec mtp4
  else
    echo "=== dense GPTQ not ready; fallback autoround + BF16 model-mtp.safetensors ==="
    run_track_modes dense27-autoround no-spec mtp4
  fi
fi

# --- item 2: exact 128k boundary with MTP4 ---
if have boundary; then
  track=""
  if [ -d models/files/community/qwen36-27b-gptq-mtp-preserved ] \
     && [ "$(du -sm models/files/community/qwen36-27b-gptq-mtp-preserved | awk '{print $1}')" -ge 12000 ]; then
    track=dense27-gptq
  else
    track=dense27-autoround
  fi
  echo "=== boundary exact-128k $track mtp4 ==="
  stop_cb
  MAXLEN=131072 bash "$LAUNCH" "$track" mtp4 on "$PORT" "$CARD" || true
  NAME="b70_cb_${track}_mtp4_on"
  if TIMEOUT=1500 bash "$WAIT" "$PORT" "$NAME"; then
    model=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
    # exact-ish long context: large prompt + short gen so total near max
    # Use p=130000 target with g=128 if memory allows; may OOM -- capture result either way
    python3 "$PHASE_BENCH" \
      --base "http://127.0.0.1:$PORT" \
      --model "$model" \
      --prompt-tokens 130000 \
      --gen-tokens 128 \
      --n 1 \
      --label "${track}_mtp4_boundary128k" \
      --out "$RESULTS_DIR/${track}_mtp4_boundary128k.json" || true
    docker logs "$NAME" >"$RESULTS_DIR/${track}_mtp4_boundary_server.log" 2>&1 || true
  else
    echo "BOUNDARY HEALTH FAILED" | tee -a "$RESULTS_DIR/failures.txt"
    docker logs "$NAME" >"$RESULTS_DIR/${track}_mtp4_boundary_boot.log" 2>&1 || true
  fi
fi

stop_cb
echo "=== campaign done; summarizing ==="
python3 - <<PY
import json, glob, os
rd="$RESULTS_DIR"
rows=[]
for p in sorted(glob.glob(rd+"/*.json")):
  try:
    d=json.load(open(p))
  except Exception:
    continue
  rows.append({
    "file": os.path.basename(p),
    "label": d.get("label"),
    "n_ok": d.get("n_ok"),
    "post_first": d.get("median_post_first_tok_s"),
    "prefill_proxy": d.get("median_prefill_proxy_tok_s"),
    "ttft": d.get("median_ttft_s"),
  })
print(json.dumps(rows, indent=2))
open(rd+"/summary_table.json","w").write(json.dumps(rows, indent=2)+"\n")
PY
echo "RESULTS=$RESULTS_DIR"
