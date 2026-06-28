#!/usr/bin/env bash
# serve_w4a8.sh -- serve Qwen3.6-27B-W4A8-sqgptq-prepacked on ONE Arc B70 (card 0, TP=1) with the
# freshly-built oneDNN int4 kernel hybrid (torch.ops._xpu_C.int4_gemm_w4a{8,16}) wired in via
# sglang/patches/w4a8_shim.py (HYBRID: decode M==1 -> w4a16 fp16-act; prefill M>1 -> w4a8 int8-act).
#
# This ckpt quantizes ONLY the MLP linears (GDN/attn stay bf16) and has NO vision -- EXPECTED.
# Goal: prove the kernel runs in the real serve path + a partial speedup, not a final model.
#
# Wiring (no image rebuild needed; runtime mounts over the baked shims):
#   - bind-mount the updated woq_shim.py + w4a8_shim.py over the baked site-packages copies
#   - bind-mount the built _xpu_C.abi3.so dir; B70_XPU_C_SO points the shim at it
#   - in-container: source oneAPI setvars, then PREPEND the oneAPI compiler lib to LD_LIBRARY_PATH
#     (required or the ctypes-loaded .so resolves but torch loses the XPU device -- W4A8_BUILD.md)
#   - B70_XPU_W4A8=1 makes woq_shim import+install w4a8_shim at startup; B70_W4A8_COMPILE=1
#     torch.compile-fuses the prefill per-token int8 act-quant.
#
#   GPU touch -> hold the lease:  ../bin/gpu-run --card 0 bash sglang/serve_w4a8.sh start
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"

IMG="${IMG:-sglang-xpu:woq}"
NAME="${NAME:-sglang_w4a8}"
CKPT="${CKPT:-/models/Qwen3.6-27B-W4A8-sqgptq-prepacked}"
TOK="${TOK:-$CKPT}"
SERVED="${SERVED:-qwen36-27b-w4a8-sqgptq}"
PORT="${PORT:-30000}"; DEVICE="${DEVICE:-0}"
CTX="${CTX:-4096}"; MEMFRAC="${MEMFRAC:-0.88}"
KERNEL_DIR="${KERNEL_DIR:-/mnt/vm_8tb/b70/w4a8_kernel}"
# Text-only config overlay: the ckpt config.json declares the multimodal Qwen3_5ForConditionalGeneration
# (vision_config) but the weights have NO vision + no preprocessor_config.json -> sglang's processor load
# crashes. We overlay a FLATTENED config (text_config lifted to top level, architectures=Qwen3_5ForCausalLM,
# quantization_config kept) so sglang picks the text-only loader (maps model.language_model.*->model.*,
# skips visual, is_multimodal=False -> no processor). Built by sglang/serve_w4a8.sh maketextcfg.
CONFIG_OVERLAY="${CONFIG_OVERLAY:-/mnt/vm_8tb/b70/w4a8_kernel/w4a8_textonly_config.json}"
SHIMS="${SHIMS:-$REPO/sglang/patches}"
SITE=/opt/venv/lib/python3.12/site-packages
W4A8_COMPILE="${W4A8_COMPILE:-1}"
W4A8_DEBUG="${W4A8_DEBUG:-0}"
LOG="${LOG:-$ROOT/w4a8_serve.log}"

cmd="${1:-start}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

start() {
  mkdir -p "$ROOT"/{hf_cache,sgl_cache} 2>/dev/null || true
  say "=== sglang W4A8 serve: $SERVED  IMG=$IMG  card=$DEVICE  ctx=$CTX  memfrac=$MEMFRAC  port=$PORT ==="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "${SHM:-16g}" -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK="$DEVICE" \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$KERNEL_DIR:/work/w4a8_kernel:ro" \
    -v "$CONFIG_OVERLAY:$CKPT/config.json:ro" \
    -v "$SHIMS/woq_shim.py:$SITE/woq_shim.py:ro" \
    -v "$SHIMS/w4a8_shim.py:$SITE/w4a8_shim.py:ro" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_W4A8=1 -e B70_XPU_C_SO=/work/w4a8_kernel/_xpu_C.abi3.so \
    -e B70_W4A8_COMPILE="$W4A8_COMPILE" -e B70_W4A8_DEBUG="$W4A8_DEBUG" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --dtype bfloat16 --attention-backend triton --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 \
      --disable-radix-cache --skip-server-warmup \
      --tp 1 --context-length $CTX --mem-fraction-static $MEMFRAC \
      --host 0.0.0.0 --port $PORT" >/dev/null
  say "container started; waiting for /health (logfile: $LOG)..."
  local ok=0
  for i in $(seq 1 180); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED"; docker logs "$NAME" > "$LOG" 2>&1; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  docker logs "$NAME" > "$LOG" 2>&1
  [ "$ok" = 1 ] || { say "NOT healthy; see $LOG"; return 1; }
  say "healthy; serving on :$PORT (model=$SERVED). logfile: $LOG"
}

maketextcfg() {
  local src="$ROOT/models/$(basename "$CKPT")/config.json"
  python3 - "$src" "$CONFIG_OVERLAY" <<'PY'
import json, sys
src, out = sys.argv[1], sys.argv[2]
c = json.load(open(src))
tc = dict(c["text_config"])           # lift text dims to top level
flat = dict(tc)
flat["architectures"] = ["Qwen3_5ForCausalLM"]
flat["quantization_config"] = c["quantization_config"]
flat["tie_word_embeddings"] = c.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))
flat.setdefault("torch_dtype", "bfloat16")
# Do NOT set layers_block_type: sglang's Qwen3_5TextConfig (registered for model_type qwen3_5_text by
# the w4a8 shim) exposes it as a DERIVED read-only property (from layer_types/full_attention_interval);
# putting it in the JSON breaks from_dict ("property has no setter"). Keep layer_types only.
flat.pop("layers_block_type", None)
json.dump(flat, open(out, "w"), indent=1)
print("wrote", out, "arch=", flat["architectures"], "model_type=", flat.get("model_type"),
      "layer_types_len=", len(flat.get("layer_types", [])))
PY
}

case "$cmd" in
  start) start ;;
  maketextcfg) maketextcfg ;;
  stop)  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  logs)  docker logs "$NAME" > "$LOG" 2>&1; echo "wrote $LOG" ;;
  status) docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'; curl -s "http://localhost:$PORT/health" && echo " <- /health" ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":96,\"temperature\":0}" | python3 -m json.tool ;;
  *) echo "usage: $0 {start|stop|logs|status|gen|maketextcfg}"; exit 2 ;;
esac
