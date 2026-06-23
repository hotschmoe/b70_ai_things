#!/usr/bin/env bash
# 99 -- W8A8 27B TP=2 @ MAXLEN=262144 (model max): does a single 262K session FIT in VRAM, and how does PREFILL
# scale with prompt length? Serve MTP-OFF + EAGER (no capture memory -> max KV room; prefill is GEMM-bound so eager
# is representative). Read vLLM's reported KV capacity, then bench prefill TTFT at increasing input lengths.
# Hybrid math (config): 16 full-attn + 48 GDN layers, num_kv_heads=4, head_dim=256 -> KV = 64 KB/token (fp16);
# 262144 tok ~= 16 GiB total -> ~8 GiB/card (TP=2). Weights ~17.9/card -> should fit fp16; Half-KV if tight.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG=vllm-xpu-env:int8g
MODEL=/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
SERVED=qwen36-27b-w8a8-sqgptq-mtp
PORT=18080; NAME=vllm_mtp99
GDN_SO="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so"
GDN_LIB="$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
TS="$(date +%Y%m%d_%H%M%S)"
SUMM="$ROOT/results/mtp99_262k_${TS}.txt"; : > "$SUMM"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
MAXLEN="${MAXLEN:-262144}"; UTIL="${UTIL:-0.95}"
echo "=== 99 W8A8 TP=2 MAXLEN=$MAXLEN KV-capacity + prefill scaling (MTP-off, eager) ===" | tee -a "$SUMM"

serve() {  # $1 = kvdtype ("" = fp16)
  local kv="$1" KVARG=()
  [ -n "$kv" ] && KVARG=(--kv-cache-dtype "$kv")
  docker rm -f "$NAME" 2>/dev/null || true
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 32g -p ${PORT}:${PORT} \
    --pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556 \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
    -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
    -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e VLLM_LOGGING_LEVEL=INFO \
    -e OMP_NUM_THREADS=8 -e CCL_ENABLE_SYCL_KERNELS=0 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
    --entrypoint vllm "$IMG" serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" --dtype auto \
    --tensor-parallel-size 2 --max-model-len "$MAXLEN" --max-num-seqs 1 --gpu-memory-utilization "$UTIL" \
    --no-enable-prefix-caching --trust-remote-code --limit-mm-per-prompt '{"image":0,"video":0}' \
    --distributed-executor-backend mp --enforce-eager "${KVARG[@]}" >/dev/null
  local i; for i in $(seq 1 200); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && return 1; sleep 5; done; return 1
}
kv_report() {  # grep the engine log for the KV capacity / max-concurrency lines
  echo "    --- vLLM KV capacity report ---" | tee -a "$SUMM"
  docker logs "$NAME" 2>&1 | grep -iE "GPU KV cache size|KV cache size|Maximum concurrency|Available KV|kv_cache|blocks:" | grep -viE "OperatorEntry" | tail -8 | sed 's/^/      /' | tee -a "$SUMM"
}
prefill_at() {  # $1 = input_len  (MTP-off + --random is safe; the hang was MTP-on only)
  local L="$1"
  local raw ttft
  raw=$(timeout 420 docker exec -i "$NAME" vllm bench serve --backend vllm --model "$SERVED" --tokenizer "$MODEL" \
    --base-url "http://localhost:$PORT" --endpoint /v1/completions --dataset-name random \
    --random-input-len "$L" --random-output-len 8 --num-prompts 1 --max-concurrency 1 --ignore-eos 2>&1) || { echo "    prefill L=$L TIMEOUT(>420s)" | tee -a "$SUMM"; return; }
  ttft=$(echo "$raw" | grep -iE "Mean TTFT" | grep -oE '[0-9.]+' | head -1)
  echo "$raw" | grep -iE "Mean TTFT|Total Token throughput|Output token throughput" | sed 's/^/      /' | tee -a "$SUMM"
  awk -v L="$L" -v t="$ttft" 'BEGIN{ if(t>0) printf "    -> ctx %6d: TTFT %.0f ms, prefill ~= %.0f tok/s\n", L, t, L/(t/1000.0); else print "    -> ctx "L": no TTFT" }' | tee -a "$SUMM"
}

KV=""
echo ">>> serve @ MAXLEN=$MAXLEN fp16 KV, UTIL=$UTIL" | tee -a "$SUMM"
if ! serve ""; then
  echo "    fp16 KV OOM/fail at 262K -> retry Half-KV (fp8_e4m3)" | tee -a "$SUMM"
  docker logs "$NAME" 2>&1 | grep -iE "out of memory|memory|KV cache|less than|decrease" | tail -5 | sed 's/^/      /' | tee -a "$SUMM"
  KV="fp8_e4m3"
  if ! serve "$KV"; then echo "    Half-KV ALSO failed:" | tee -a "$SUMM"; docker logs "$NAME" 2>&1 | tail -12 | tee -a "$SUMM"; docker rm -f "$NAME" >/dev/null 2>&1; echo "=== 99 done ==="; exit 0; fi
fi
echo "    HEALTHY at MAXLEN=$MAXLEN (KV=${KV:-fp16})" | tee -a "$SUMM"
kv_report
echo ">>> prefill scaling (real KV reserved; MTP-off): ctx 2048 / 8192 / 32768 / 131072" | tee -a "$SUMM"
for L in 2048 8192 32768 131072; do prefill_at "$L"; done
docker rm -f "$NAME" >/dev/null 2>&1
echo "=== 99 done ===" | tee -a "$SUMM"
