#!/usr/bin/env bash
# 112 -- W8A8 27B TP=2 PER-STEP PROFILE (prefill + decode op decomposition).
# [!!! 2026-06-24 SUPERSEDED] The vLLM torch-profiler endpoint is ABSENT on :int8g: VLLM_TORCH_PROFILER_DIR is an
#   "unknown vLLM env" on this 0.23 build and POST /start_profile -> 404. So this in-situ Kineto approach is a dead
#   end on this image. Per-step decomposition was done instead via scripts/113 (component microbench at the exact
#   per-card TP=2 shapes) + scripts/allreduce_bench.py. Full diagnosis: 27b_w8a8_research.md. Kept for the record.
# Goal: disassemble each step of PP (prefill) and TG (decode) on TP=2 W8A8 Qwen3.6-27B into per-op XPU
# DEVICE time -> map cycles/time/bandwidth/compute per step. Method: serve EAGER (GRAPH=0) so every op is a
# distinct kernel the torch/XPU profiler can see (capture only removes launch overhead; per-op DEVICE time is
# identical eager vs captured). vLLM torch profiler via VLLM_TORCH_PROFILER_DIR + /start_profile,/stop_profile
# -> Kineto trace per rank; we join device kernels to parent CPU ops by correlation id and aggregate.
#
#   /mnt/vm_8tb/b70/gpu-run bash 112_w8a8_tp2_profile.sh
#
# MTP OFF here (clean body decomposition); collectives run eager oneCCL (exactly the per-layer allreduce we want
# to measure). Captured E2E + MTP verify characterized separately (recipe / scripts 111).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
PORT=18080; NAME=vllm_prof112
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
PROFDIR_HOST="$ROOT/tmp_ssd/prof112"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/results/prof112_${TS}"
mkdir -p "$OUT"
SUMM="$OUT/summary.txt"; : > "$SUMM"
log(){ echo "$@" | tee -a "$SUMM"; }

trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
rm -rf "$PROFDIR_HOST"; mkdir -p "$PROFDIR_HOST"

log "=== 112 W8A8 27B TP=2 per-step profile  ts=$TS ==="

# ---- serve EAGER TP=2, profiler ON (MTP off) -----------------------------------------------------
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p "${PORT}:${PORT}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_TORCH_PROFILER_DIR=/tmp_ssd/prof112 \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
  --dtype auto --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching --trust-remote-code \
  --distributed-executor-backend mp --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enforce-eager >/dev/null

log "--- waiting for /health (up to ~12 min) ---"
healthy=0
for i in $(seq 1 144); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then healthy=1; break; fi
  if docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited; then
    log "[!] EXITED EARLY"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$SUMM"; exit 1; fi
  sleep 5
done
[ "$healthy" = 1 ] || { log "[!] not healthy"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$SUMM"; exit 1; }
log "--- HEALTHY ---"

# coherence probe
resp=$(curl -s --max-time 60 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}")
log "coherence gen: $(printf '%s' "$resp" | grep -oE '"text":"[^"]*"' | head -1)"

# ---- build a long (~2000 token) prompt for the prefill window ------------------------------------
LONGP=$(printf 'The quick brown fox jumps over the lazy dog and then %.0s' $(seq 1 200))

prof_window(){ # $1 label  $2 prompt  $3 max_tokens
  local lab="$1" prompt="$2" mt="$3"
  log ">>> window $lab : max_tokens=$mt"
  curl -s --max-time 30 -X POST "http://localhost:$PORT/start_profile" >/dev/null 2>&1
  sleep 1
  local r; r=$(curl -s --max-time 300 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":$(printf '%s' "$prompt" | python_json_str),\"max_tokens\":$mt,\"temperature\":0,\"ignore_eos\":true}")
  local pt ct
  pt=$(printf '%s' "$r" | grep -oE '"prompt_tokens":[0-9]+' | head -1 | grep -oE '[0-9]+')
  ct=$(printf '%s' "$r" | grep -oE '"completion_tokens":[0-9]+' | head -1 | grep -oE '[0-9]+')
  log "    prompt_tokens=$pt completion_tokens=$ct"
  echo "$lab prompt_tokens=$pt completion_tokens=$ct" >> "$OUT/window_meta.txt"
  curl -s --max-time 60 -X POST "http://localhost:$PORT/stop_profile" >/dev/null 2>&1
  sleep 6   # let traces flush
}

# json-encode a string via the container python (host has none); read stdin -> json
python_json_str(){ docker exec -i "$NAME" python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.stdin.read()))'; }

# DECODE window: short prompt, many decode steps (per-step = total/completion_tokens)
prof_window decode "Count slowly: one two three four five six seven eight nine ten." 64
# PREFILL window: ~2000-token prompt, 1 decode step (prefill dominates)
prof_window prefill "$LONGP" 1

log "--- traces written ---"
ls -la "$PROFDIR_HOST" | tee -a "$SUMM"
cp -r "$PROFDIR_HOST"/* "$OUT/" 2>/dev/null || true

# ---- stop serve, free GPUs, then parse traces in a fresh python container ------------------------
docker rm -f "$NAME" >/dev/null 2>&1 || true
trap - EXIT

log "=== PARSE ==="
docker run --rm --entrypoint python3 -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$ROOT/scripts:/scripts:ro" \
  "$IMG" /scripts/112_parse_trace.py /tmp_ssd/prof112 "$OUT/window_meta.txt" 2>&1 | tee -a "$SUMM"

log "=== DONE -> $OUT ==="
