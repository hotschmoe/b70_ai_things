#!/usr/bin/env bash
# X2: memory evidence only, eager serving; C7b and H1e go first.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
BASE=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT=$BASE/x2
cd "$REPO"
export B70_AGENT=X2 B70_GPU_LOCK_TIMEOUT=7200
mkdir -p "$OUT"
echo "X2 window controller $$ $(date -u +%FT%TZ)"
# Wait outside the GPU lease until the two priority windows have ended.
# The H1e window itself waits for C7b and exits with WINDOW_DONE.
while ! test -f "$OUT/priority-released"; do
  echo "PRIORITY_WAIT $(date -u +%FT%TZ)"
  sleep 30
done
if systemctl --user is-active --quiet hotschmoe-dd; then
  echo "DD_ACTIVE_ABORT"; exit 3
fi
tail -n 85 vllm/flashnext/JOURNAL.md > "$OUT/journal-before.txt"
cat /mnt/vm_8tb/b70/gpu.lock.{0,1}.owner
bin/gpu-run bin/xpu-health > "$OUT/health-before.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"
bin/gpu-run bin/xpu-collective-health > "$OUT/collective-before.txt" 2>&1
grep -q COLLECTIVE_HEALTH_OK "$OUT/collective-before.txt"

run() {
  local mode=$1 fix=$2 dtype=$3 batch=$4
  local max_len=2048
  if [[ "$mode" == capped-fp8 ]]; then max_len=200000; fi
  local name=vllm-fn-x2-$mode port=18122
  local evidence=$OUT/$mode
  mkdir -p "$evidence"
  echo "RUN $mode $(date -u +%FT%TZ)"
  local avail
  avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( avail < 8388608 )); then echo LOW_RAM_ABORT; return 4; fi
  local -a kvargs=()
  if [[ "$dtype" != auto ]]; then kvargs+=(--kv-cache-dtype "$dtype" --calculate-kv-scales); fi
  vllm/flashnext/harness/xpu-run.sh --card both --memory 100g --name "$name" --port "$port" \
    --env B70_GGUF=1 --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
    --env B70_QSA=0 --env B70_EP=0 --env B70_PREFILL=0 \
    --env B70_MEMORY_FIX="$fix" --env B70_MEMORY_REPORT="$evidence" \
    --env B70_GGUF_PATH=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
    --env B70_GGUF_SO="$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.c12.so" \
    --env PYTHONPATH="$BASE/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/plugins/b70_memory_xpu" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env SYCL_CACHE_PERSISTENT=0 \
    --env VLLM_ENGINE_READY_TIMEOUT_S=800 \
    -- vllm serve /mnt/vm_8tb/b70/build/backend-decision/hf-flashnext \
      --host 0.0.0.0 --port "$port" --served-model-name hotschmoe-dd qwen3.8-flash-next-iq3-xxs \
      --tensor-parallel-size 1 --data-parallel-size 2 --api-server-count 1 --enable-expert-parallel \
      --dtype bfloat16 --max-model-len "$max_len" --max-num-batched-tokens "$batch" \
      --gpu-memory-utilization 0.98 --max-num-seqs 2 --language-model-only --enforce-eager \
      --load-format b70_gguf --quantization b70_gguf --trust-remote-code "${kvargs[@]}" \
      > "$evidence/serve.log" 2>&1 &
  local serve_pid=$!
  cleanup_run() {
    docker stop -t 20 "$name" >/dev/null 2>&1 || true
    wait "$serve_pid" || true
  }
  trap cleanup_run EXIT INT TERM
  local ready=0
  for ((i=0; i<240; ++i)); do
    if ! kill -0 "$serve_pid" 2>/dev/null; then break; fi
    avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    echo "$(date -u +%FT%TZ) available_kb=$avail" >> "$evidence/host.log"
    if (( avail < 8388608 )); then echo LOW_RAM_ABORT; break; fi
    if curl -sf --max-time 2 "http://127.0.0.1:$port/v1/models" > "$evidence/models.json"; then ready=1; break; fi
    sleep 3
  done
  echo "READY=$ready $(date -u +%FT%TZ)"
  if (( ready )); then
    python3 vllm/flashnext/memory/greedy.py "$port" "$evidence/greedy.json" || true
  fi
  cleanup_run
  trap - EXIT INT TERM
  if (( ! ready )); then
    bin/gpu-run bin/xpu-health > "$evidence/health-after-failure.txt" 2>&1
    grep -q 'xpu-health: HEALTHY' "$evidence/health-after-failure.txt"
    bin/gpu-run bin/xpu-collective-health > "$evidence/collective-after-failure.txt" 2>&1
    grep -q COLLECTIVE_HEALTH_OK "$evidence/collective-after-failure.txt"
  fi
  echo "RUN_DONE $mode $(date -u +%FT%TZ)"
}
run current 0 auto 2048
run aliases 1 auto 2048
run capped-fp8 1 fp8 512
bin/gpu-run bin/xpu-health > "$OUT/health-after.txt" 2>&1
bin/gpu-run bin/xpu-collective-health > "$OUT/collective-after.txt" 2>&1
echo "WINDOW_DONE $(date -u +%FT%TZ)"
