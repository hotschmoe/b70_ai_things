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
bin/gpu-run bin/xpu-health >> "$OUT/health-before.txt" 2>&1
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt"
bin/gpu-run bin/xpu-collective-health >> "$OUT/collective-before.txt" 2>&1
grep -q COLLECTIVE_HEALTH_OK "$OUT/collective-before.txt"

run() {
  local mode=$1 fix=$2 dtype=$3 batch=$4
  last_mode=$mode
  local max_len=2048
  if [[ "$mode" == target-* || "$mode" == sidecar ]]; then max_len=200000; fi
  local name=vllm-fn-x2-$mode port=18122
  local evidence=$OUT/$mode
  mkdir -p "$evidence"
  echo "RUN $mode $(date -u +%FT%TZ)"
  local avail
  avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( avail < 8388608 )); then echo LOW_RAM_ABORT; return 4; fi
  local -a kvargs=()
  local -a parallelargs=(--data-parallel-size 2 --api-server-count 1 --enable-expert-parallel)
  local ep=0
  if [[ "$mode" == sidecar ]]; then parallelargs=(); ep=1; fi
  if [[ "$dtype" != auto ]]; then kvargs+=(--kv-cache-dtype "$dtype"); fi
  if [[ "$dtype" == fp8_per_token_head ]]; then kvargs+=(--attention-backend TRITON_ATTN); fi
  setsid vllm/flashnext/harness/xpu-run.sh --card both --memory 100g --name "$name" --port "$port" \
    --env B70_GGUF=1 --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
    --env B70_QSA=0 --env B70_EP="$ep" --env B70_PREFILL=0 \
    --env B70_EP_Q0=138 --env B70_EP_MAX_T="$batch" \
    --env B70_EP_PROFILE=/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin \
    --env B70_MEMORY_FIX="$fix" --env B70_MEMORY_REPORT="$evidence" \
    --env B70_EMBED_REF="$OUT/embedding-reference.npz" \
    --env B70_GGUF_PATH=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
    --env B70_GGUF_SO="$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.c12.so" \
    --env PYTHONPATH="$BASE/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu:$REPO/vllm/flashnext/plugins/b70_memory_xpu:$REPO/vllm/flashnext/plugins/b70_ep_xpu" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env SYCL_CACHE_PERSISTENT=0 \
    --env VLLM_ENGINE_READY_TIMEOUT_S=800 \
    -- vllm serve /mnt/vm_8tb/b70/build/backend-decision/hf-flashnext \
      --host 0.0.0.0 --port "$port" --served-model-name hotschmoe-dd qwen3.8-flash-next-iq3-xxs \
      --tensor-parallel-size 1 "${parallelargs[@]}" \
      --dtype bfloat16 --max-model-len "$max_len" --max-num-batched-tokens "$batch" \
      --gpu-memory-utilization 0.98 --max-num-seqs 2 --language-model-only --enforce-eager \
      --load-format b70_gguf --quantization b70_gguf --trust-remote-code "${kvargs[@]}" \
      > "$evidence/serve.log" 2>&1 &
  local serve_pid=$!
  cleanup_run() {
    docker stop -t 20 "$name" >/dev/null 2>&1 || true
    if kill -0 "$serve_pid" 2>/dev/null; then
      kill -- "-$serve_pid" 2>/dev/null || true
    fi
    wait "$serve_pid" || true
  }
  trap cleanup_run EXIT INT TERM
  local ready=0
  local started_at=0
  while kill -0 "$serve_pid" 2>/dev/null; do
    if ! kill -0 "$serve_pid" 2>/dev/null; then break; fi
    avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    echo "$(date -u +%FT%TZ) available_kb=$avail" >> "$evidence/host.log"
    if (( avail < 8388608 )); then echo LOW_RAM_ABORT; break; fi
    if (( ! started_at )) && grep -Fq 'gpu-run: [X2] acquired card(s)' "$evidence/serve.log"; then
      started_at=$(date +%s)
      echo "LEASE_ACQUIRED $mode $(date -u +%FT%TZ)"
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:$port/v1/models" > "$evidence/models.json"; then ready=1; break; fi
    if (( started_at && $(date +%s)-started_at > 1200 )); then echo STARTUP_TIMEOUT; break; fi
    sleep 3
  done
  echo "READY=$ready $(date -u +%FT%TZ)"
  if (( ready )); then
    local verify_rc=0
    python3 vllm/flashnext/memory/greedy.py "$port" "$evidence/greedy.json" || verify_rc=$?
    echo "GREEDY_RC=$verify_rc"
    if [[ "$fix" == 1 ]]; then
      local prefill_rc=0
      python3 vllm/flashnext/memory/prefill_check.py "$port" "$evidence/prefill.json" || prefill_rc=$?
      echo "PREFILL_RC=$prefill_rc"
    fi
  fi
  cleanup_run
  trap - EXIT INT TERM
  if (( ! ready )); then
    bin/gpu-run bin/xpu-health > "$evidence/health-after-failure.txt" 2>&1
    grep -q 'xpu-health: HEALTHY' "$evidence/health-after-failure.txt"
    bin/gpu-run bin/xpu-collective-health > "$evidence/collective-after-failure.txt" 2>&1
    grep -q COLLECTIVE_HEALTH_OK "$evidence/collective-after-failure.txt"
  else
    bin/gpu-run bin/xpu-health > "$evidence/health-after.txt" 2>&1
    grep -q 'xpu-health: HEALTHY' "$evidence/health-after.txt"
  fi
  echo "RUN_DONE $mode $(date -u +%FT%TZ)"
}
next_turn() {
  touch "$OUT/hold-next"
  echo "AWAIT_NEXT $1 $(date -u +%FT%TZ)"
  while test -f "$OUT/hold-next"; do sleep 3; done
}
started=0
active=0
for stage in current aliases target sidecar; do
  if [[ "$stage" == "${X2_START_ARM:-current}" ]]; then active=1; fi
  if (( ! active )); then continue; fi
  if (( started )); then next_turn "$stage"; fi
  started=1
  case "$stage" in
    current) run current 0 auto 2048 ;;
    aliases) run aliases 1 auto 2048 ;;
    target)
      dtype=${X2_TARGET_DTYPE:-$(cat "$OUT/target-dtype" 2>/dev/null || echo fp8)}
      run "target-$dtype" 1 "$dtype" 512 ;;
    sidecar) run sidecar 1 auto 2048 ;;
  esac
  if [[ "$stage" == "${X2_STOP_ARM:-none}" ]]; then break; fi
done
bin/gpu-run bin/xpu-collective-health > "$OUT/$last_mode/collective-after.txt" 2>&1
echo "WINDOW_DONE $(date -u +%FT%TZ)"
