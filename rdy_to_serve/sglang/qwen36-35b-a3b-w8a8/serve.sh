#!/usr/bin/env bash
# qwen36-35b-a3b-w8a8 -- Qwen3.6-35B-A3B Quark W8A8 INT8 MoE on sglang-XPU (2x B70, TP=2). The FIRST
# sglang serve of the int8 MoE (256 experts, top-8, shared expert + GDN hybrid attention). Route A:
# the 256 routed experts run TRUE int8 through sglang's in-tree Triton fused_moe (use_int8_w8a8); the
# dense linears (linear_attn.*, self_attn.*, shared_expert.*) load int8 + dequant->bf16 at load time.
#
#   VERIFIED (TP=2, eager, IN2048/OUT128 warm): TTFT 272ms (c1) / 637ms (c4) | decode c1 7.94 t/s,
#   c4 5.55/stream (agg 23.87) | single-stream soak 8.26 t/s STABLE (1.00x first/last, no degradation),
#   coherent throughout. KV ~1.04M tokens. TTFT is best-in-class for the 35B (int8-XMX prefill, 1.43x
#   over bf16 per the probe); decode is eager-slow (memory-bound -- graph capture / MTP are the levers).
#   vs vLLM Quark W8A8 (43.1 c1 / GRAPH=1): sglang trades raw decode for the production scheduler that
#   does NOT co-batch prefill+decode (the vLLM GDN "!!!!" risk). See ../../../research/w8a8/SGLANG_MOE_PLAN.md.
#
# WHAT MAKES IT WORK (the unblock chain, all mount-not-bake; ../../../research/w8a8/SGLANG_MOE_PLAN.md):
#   - sglang/patches/int8_actquant_xpu.py  -- XPU-safe per-token int8 act-quant (stock kernel's
#       tl.extra.cuda.libdevice.round does NOT link on triton-xpu; replaced by floor/ceil round).
#   - sglang/patches/quark_moe_int8.py     -- Int8MoEMethod (in-tree Triton int8 fused_moe) +
#       Int8DequantLinear (dense int8->bf16) + a QuarkConfig.get_quant_method monkeypatch; plus a
#       FusedMoE._load_per_channel_weight_scale unsqueeze (Quark stores 1-D [N] scales; sglang wants [N,1]).
#   - sglang/images/sglang-xpu-mtp/woq_shim.py -- the auto-imported hook; B70_QUARK_MOE_INT8=1 calls
#       quark_moe_int8.install() in EVERY process (the model builds in the TP workers).
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh start   # serve TP=2, coherence-gated, stay up
#   bash serve.sh bench                            # c1+c4 regime bench + soak (perf_regime.sh)
#   bash serve.sh stop                             # stop + release, health re-probe
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh run      # start + bench + stop in one lease
#
# [!] TP=2 = BOTH cards (int8 ~35GB -> ~17.9GB/card). GDN-safe: --mamba-ssm-dtype float32,
#     --skip-server-warmup, --disable-cuda-graph. Greedy on XPU. Re-verify: bash serve.sh smoke.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
IMG="${IMG:-sglang-xpu:mtp}"
NAME="${NAME:-sglang_w8a8_moe_35b}"
CKPT="${CKPT:-/models/qwen3.6-35b-a3b/quark-w8a8-int8}"
TOK="${TOK:-/models/qwen3.6-35b-a3b/quark-w8a8-int8}"
SERVED="${SERVED:-qwen36-35b-a3b-quark-w8a8-int8}"
PORT="${PORT:-30000}"; TP="${TP:-2}"; CTX="${CTX:-8192}"; MEMFRAC="${MEMFRAC:-0.88}"; MAXREQ="${MAXREQ:-8}"
SHIM="$REPO/sglang/images/sglang-xpu-mtp/woq_shim.py"
LOADER="$REPO/sglang/patches/quark_moe_int8.py"
ACTQ="$REPO/sglang/patches/int8_actquant_xpu.py"
SP=/opt/venv/lib/python3.12/site-packages
say(){ echo "[$(date +%H:%M:%S)] $*"; }

start(){
  say "pre-flight xpu-health"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; return 3; }
  for f in "$LOADER" "$ACTQ" "$SHIM"; do [ -f "$f" ] || { say "missing $f"; return 2; }; done
  docker rm -f "$NAME" >/dev/null 2>&1
  say "serve W8A8 int8 MoE (Route A, dequant dense) TP=$TP -> $SERVED on :$PORT (img=$IMG)"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 32g -p "${PORT}:${PORT}" \
    -v "$REPO/models/files:/models:ro" \
    -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$LOADER:$SP/quark_moe_int8.py:ro" -v "$ACTQ:$SP/int8_actquant_xpu.py:ro" -v "$SHIM:$SP/woq_shim.py:ro" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e TRITON_CACHE_DIR=/sgl_cache/triton -e B70_QUARK_MOE_INT8=1 -e B70_MOE_DEBUG="${B70_MOE_DEBUG:-0}" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec python -m sglang.launch_server --model-path '$CKPT' --tokenizer-path '$TOK' \
      --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
      --disable-cuda-graph --mamba-ssm-dtype float32 --disable-overlap-schedule \
      --page-size 64 --disable-radix-cache --skip-server-warmup \
      --tp $TP --context-length $CTX --mem-fraction-static $MEMFRAC --max-running-requests $MAXREQ \
      --host 0.0.0.0 --port $PORT" >/dev/null
  say "waiting for /health (load ~1min + first-forward int8 MoE Triton JIT; health pings race the JIT)..."
  for i in $(seq 1 200); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "EXITED"; docker logs "$NAME" 2>&1|tail -40; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null||echo 000)" = 200 ] && { say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null||echo 000)" = 200 ] || { say "NEVER HEALTHY"; docker logs "$NAME" 2>&1|tail -40; return 1; }
  local g; g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":64,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json
c=json.load(sys.stdin)['choices'][0]['message']['content'] or ''
print('COHERENCE OK:',repr(c[:160])) if c.strip() and (len(c)<16 or max(c.count(x) for x in set(c))/len(c)<0.6) else (print('GATE FAIL:',repr(c[:120])) or sys.exit(1))" \
    || { say "coherence gate FAILED -- see: docker logs $NAME"; return 1; }
  say "healthy + coherent; serving $SERVED on :$PORT"
}
stop(){ docker rm -f "$NAME" >/dev/null 2>&1; say "stopped $NAME"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || true; }
bench(){ bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "w8a8-moe-int8"; }
gen(){ curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"${2:-Why is the sky blue?}\"}],\"max_tokens\":128,\"temperature\":0}"; }

case "${1:-start}" in
  start) start ;;
  stop)  stop ;;
  bench) bench ;;
  gen)   gen "$@" ;;
  run)   start && bench; rc=$?; stop; exit $rc ;;
  smoke) start; rc=$?; stop; exit $rc ;;
  *) echo "usage: serve.sh {start|stop|bench|gen|run|smoke}"; exit 2 ;;
esac
