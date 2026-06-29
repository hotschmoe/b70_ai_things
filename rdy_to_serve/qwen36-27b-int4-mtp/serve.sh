#!/usr/bin/env bash
# Qwen3.6-27B int4 (AutoRound W4A16, woqgemm) + NEXTN chain-MTP (num-steps=7) -- the LATENCY daily driver.
# Single card, GREEDY (deterministic), VISION retained. First config to STABLY beat the ~9.4 t/s sglang-XPU
# eager ceiling on the dual-B70 box: c1 ~15.3 t/s = 1.62x baseline, mean accept_len ~4.1-4.4. Correct under
# sustained mixed load (the agentic prefill+decode pattern that makes vLLM "!!!!").
#
# Self-contained: uses the BAKED sglang-xpu:mtp image (the 4 XPU MTP gates in mtp_tree_xpu.py + the
# spec-decode mamba memory_pool device fix + woqgemm int4 from :woq). NO runtime patch mounts.
#
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh start   # serve, wait healthy, coherence-gated gen probe, stay up
#   bash serve.sh stop                                      # stop + release the card
#   bash serve.sh logs                                      # follow server log
#   bash serve.sh gen                                       # one-shot chat probe vs the running server
#   bash serve.sh bench                                     # warm c1/c4 (pp/ttft/tg @ ctx2048) + soak vs running server
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh run      # start + bench + stop in one lease
#
# IMAGE: sglang-xpu:mtp (build: ../../images/sglang-xpu-mtp/). CKPT: Lorbus int4 + grafted BF16 MTP head.
#
# [!] GREEDY-ONLY: MTP verify runs greedily on XPU (gate 4) -> output is the target model's argmax (correct
#     greedy decoding) but temperature/top_p/top_k are IGNORED (exactly like the NPU/HIP spec path). For
#     sampling diversity use the non-MTP int4 woq DP=2 driver (../../sglang/serve_dp2.sh). Restoring sampling
#     under MTP = task #14 (a pure-torch chain rejection-sampler).
# [!] CONCURRENCY: --max-running-requests 4 fits one 32 GB card (the spec mamba intermediate-state cache
#     scales with it; 8 OOMs the KV at ctx 4096). Requests beyond 4 QUEUE and complete fine. MTP is a
#     single-stream / low-concurrency LATENCY lever; for high concurrency use the int4 woq DP=2 driver.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"

IMG="${IMG:-sglang-xpu:mtp}"
NAME="${NAME:-sglang_int4_mtp}"
CKPT="${CKPT:-/models/qwen3.6-27b/int4-autoround}"     # int4 AutoRound + grafted BF16 MTP head + vision
TOK="${TOK:-/models/qwen3.6-27b/bf16}"                 # bench tokenizer (base model)
SERVED="${SERVED:-qwen36-27b-int4-mtp-nextn}"          # id encodes scheme: int4 + mtp/nextn
PORT="${PORT:-30000}"
DEVICE="${DEVICE:-0}"                                   # single-card pin (ZE_AFFINITY_MASK)
CTX="${CTX:-4096}"
MEMFRAC="${MEMFRAC:-0.92}"
MAXREQ="${MAXREQ:-4}"                                   # spec mamba cache cap (>4 OOMs KV at ctx 4096)
SPEC_STEPS="${SPEC_STEPS:-7}"                           # chain depth-7 = near-peak 15.31 t/s (plateau at 9)
SPEC_DRAFT="${SPEC_DRAFT:-8}"                           # num-draft-tokens = steps+1
DENV="${DENV:-}"                                        # extra docker -e env, space-separated KEY=VAL

cmd="${1:-start}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

start() {
  mkdir -p "$ROOT"/{hf_cache,sgl_cache} 2>/dev/null || true
  local denv=(); for kv in $DENV; do denv+=(-e "$kv"); done
  say "=== sglang int4+MTP serve: $SERVED  IMG=$IMG  card=$DEVICE  steps=$SPEC_STEPS  ctx=$CTX  port=$PORT ==="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "${SHM:-16g}" -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK="$DEVICE" \
    -v "$REPO/models/files:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_MTP=1 -e B70_MTP_DEBUG="${DBG:-0}" "${denv[@]}" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
      --speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton --disable-cuda-graph \
      --max-running-requests $MAXREQ --skip-server-warmup \
      --tp 1 --context-length $CTX --mem-fraction-static $MEMFRAC \
      --host 0.0.0.0 --port $PORT" >/dev/null

  say "waiting for /health (skip-warmup -> fast; first gen JITs the spec path ~13s)..."
  local ok=0
  for i in $(seq 1 90); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED"; docker logs "$NAME" 2>&1|tail -40; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "NOT healthy; abort"; docker logs "$NAME" 2>&1|tail -30; return 1; }

  # COHERENCE GATE: a real generation must succeed (no spec-verify crash) and not be "!!!!".
  say "=== coherence gate (first gen JITs the spec path) ==="
  local g
  g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":96,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json
from collections import Counter
try:
 c=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('GATE FAIL: '+repr(sys.stdin.read()[:160])); sys.exit(1)
if not c: print('GATE FAIL: empty'); sys.exit(1)
if len(c)>=16:
 ch,n=Counter(c).most_common(1)[0]
 if n/len(c)>=0.6: print('GATE FAIL: GARBAGE '+repr(c[:120])); sys.exit(1)
print('GATE OK: '+repr(c[:140]))" || { say "coherence gate FAILED -- see: bash $0 logs"; return 1; }
  say "=== healthy + coherent; serving on :$PORT (model=$SERVED) ==="
}

case "$cmd" in
  start) start ;;
  run)   start && bash "$SCRIPT_DIR/serve.sh" bench; rc=$?; bash "$SCRIPT_DIR/serve.sh" stop; exit $rc ;;
  stop)  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  logs)  docker logs -f "$NAME" ;;
  status) docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'; curl -s "http://localhost:$PORT/health" && echo " <- /health" ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the ocean.\"}],\"max_tokens\":64,\"temperature\":0}" | python3 -m json.tool ;;
  bench) bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-NEXTN-mtp" ;;
  accept) docker logs "$NAME" 2>&1 | grep -oE "accept len: [0-9.]+" | tail -12
          docker logs "$NAME" 2>&1 | grep -oE "accept len: [0-9.]+" | awk -F': ' '{s+=$2;n++} END{if(n)printf "[mean accept len over %d batches] %.2f\n",n,s/n}' ;;
  *) echo "usage: $0 {start|run|stop|logs|status|gen|bench|accept}"; exit 2 ;;
esac
