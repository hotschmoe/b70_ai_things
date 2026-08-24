#!/usr/bin/env bash
# Research serve for Ornith-1.5-35B-A3B W8A8 RTN + trained Shisa MTP.
# Not a shelf entry until TP=2 MTP, radix cache, coherence, and eval gates pass.
set -uo pipefail

REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-sglang-xpu:mtp-0515}"
NAME="${NAME:-sglang_ornith15_w8a8}"
CKPT="${CKPT:-/models/ornith-1.5-35b-a3b/w8a8-rtn-mtp-shisa}"
SERVED="${SERVED:-ornith-1.5-35b-a3b-W8A8-rtn-mtp-shisa}"
PORT="${PORT:-18080}"
TP="${TP:-2}"
CTX="${CTX:-8192}"
MEMFRAC="${MEMFRAC:-0.90}"
MAXREQ="${MAXREQ:-4}"
MTP="${MTP:-1}"
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"
RADIX="${RADIX:-0}"
SP=/opt/venv/lib/python3.12/site-packages
SHIM="$REPO/sglang/images/sglang-xpu-mtp-0515/woq_shim.py"
MTP_TREE="$REPO/sglang/images/sglang-xpu-mtp-0515/mtp_tree_xpu.py"
LOADER="$REPO/sglang/patches/quark_moe_int8.py"
ACTQ="$REPO/sglang/patches/int8_actquant_xpu.py"

say() { echo "[$(date +%H:%M:%S)] $*"; }

start() {
  say "pre-flight xpu-health"
  "$REPO/bin/xpu-health" 2>&1 | tail -2 || return 3
  for path in "$SHIM" "$MTP_TREE" "$LOADER" "$ACTQ"; do
    [ -f "$path" ] || { say "missing $path"; return 2; }
  done
  [ -f "$REPO/models/files/ornith-1.5-35b-a3b/w8a8-rtn-mtp-shisa/model.safetensors.index.json" ] || {
    say "missing quantized checkpoint"
    return 2
  }
  docker rm -f "$NAME" >/dev/null 2>&1 || true

  local spec_args=""
  [ "$MTP" = 1 ] && spec_args="--speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton"
  local radix_args="--disable-radix-cache --page-size 64"
  local radix_env=0
  if [ "$RADIX" = 1 ]; then
    radix_args="--mamba-radix-cache-strategy extra_buffer --enable-int8-mamba-checkpoint --enable-cache-report --page-size 128"
    radix_env=1
  fi

  say "serve TP=$TP ctx=$CTX MTP=$MTP steps=$SPEC_STEPS radix=$RADIX -> $SERVED"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 32g -p "$PORT:$PORT" \
    -v "$REPO/models/files:/models:ro" \
    -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$LOADER:$SP/quark_moe_int8.py:ro" \
    -v "$ACTQ:$SP/int8_actquant_xpu.py:ro" \
    -v "$SHIM:$SP/woq_shim.py:ro" \
    -v "$MTP_TREE:$SP/mtp_tree_xpu.py:ro" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache \
    -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor -e TRITON_CACHE_DIR=/sgl_cache/triton \
    -e B70_QUARK_MOE_INT8=1 -e B70_XPU_MTP="$MTP" \
    -e B70_XPU_MAMBA_EXTRA_BUFFER="$radix_env" -e CCL_TOPO_P2P_ACCESS=0 \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' \
      --trust-remote-code --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
      --disable-cuda-graph --mamba-ssm-dtype float32 --disable-overlap-schedule --skip-server-warmup \
      --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-metrics \
      $spec_args $radix_args --tp '$TP' --context-length '$CTX' --mem-fraction-static '$MEMFRAC' \
      --max-running-requests '$MAXREQ' --host 0.0.0.0 --port '$PORT'" >/dev/null

  for i in $(seq 1 180); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -Fxq "$NAME" || {
      say "server exited"
      docker logs "$NAME" 2>&1 | tail -80
      return 1
    }
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || true)" = 200 ]; then
      say "health 200 after $((i * 5))s"
      break
    fi
    sleep 5
  done
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || true)" = 200 ] || return 1
  curl -fsS "http://localhost:$PORT/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin); ids=[x['id'] for x in d['data']]; assert '$SERVED' in ids, ids; print('identity ->', ids)" || return 1
  curl -fsS --max-time 300 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Return only the integer sum of 19 and 23.\"}],\"max_tokens\":128,\"temperature\":0}" \
    | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; s=(m.get('content') or '')+(m.get('reasoning_content') or ''); assert '42' in s, repr(s); print('coherence ->', repr(s[-160:]))" || return 1
  say "healthy and coherent; endpoint left running"
}

stop() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  say "stopped $NAME"
  "$REPO/bin/xpu-health" 2>&1 | tail -2 || true
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  logs) docker logs -f "$NAME" ;;
  *) echo "usage: serve_ornith15_w8a8.sh {start|stop|logs}"; exit 2 ;;
esac
