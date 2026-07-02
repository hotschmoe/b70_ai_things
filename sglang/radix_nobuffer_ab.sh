#!/usr/bin/env bash
# radix_nobuffer_ab.sh -- EXPERIMENT (2026-07-02): can the W8A8 fused+MTP daily-driver config serve with
# sglang mamba PREFIX/RADIX caching on XPU via the no_buffer strategy? Prod runs --disable-radix-cache
# because RADIX=1 auto-picks extra_buffer (CUDA/MUSA/NPU-only -> crash). This forces the OTHER strategy:
#   --mamba-radix-cache-strategy no_buffer  +  --page-size 1   (no_buffer/MambaRadixCache v1 requires page_size=1).
#
# RUN 1 (intel_xpu attn) FAILED at scheduler init: MambaRadixCache asserts page_size==1, but the intel_xpu
# DECODE attention backend force-bumps page_size 1->128 (server_args.py:4835-4845: non-MLA XPU supports only
# [64,128]). MambaRadixCache v1 (page 1) and intel_xpu attn (page 64/128) are mutually exclusive.
# RUN 2 (this file): switch the main attention backend to TRITON (page_size=1-friendly, sanctioned XPU
# fallback) so page_size stays 1 and MambaRadixCache can build. Tradeoff: the 16/64 full-attn layers lose
# the XMX path -> also measure decode t/s. Everything else (TP=2, NEXTN MTP steps=10, fused int8 kernels,
# vision ckpt, skip-warmup, --disable-overlap-schedule for no_buffer) is KEPT as prod.
#
# A/B proof of caching: fire an ~8k-token prompt A twice (cold then warm), then a DIFFERENT prompt B twice.
# If radix works: A2 reports usage.prompt_tokens_details.cached_tokens ~= prompt_tokens and A2 wall-clock
# << A1; B1 is a fresh miss (~A1). Coherence gated on every reply. Own container/port, self-cleaning; does
# NOT touch the daily-driver container names. Run under the GPU lease (both cards, TP=2):
#   /mnt/vm_8tb/b70/gpu-run bash sglang/radix_nobuffer_ab.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
IMG=sglang-xpu:mtp
NAME=sglang_radix_ab
PORT=31000
CKPT=/models/qwen3.6-27b/w8a8-sqgptq
SERVED=qwen36-27b-w8a8-mtp
KDIR=$ROOT/w8a8_kernel
CTX=131072
say(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ say "cleanup: rm $NAME"; docker rm -f "$NAME" >/dev/null 2>&1; "$REPO/bin/xpu-health" 2>&1 | tail -2 || true; }
trap cleanup EXIT

say "pre-flight xpu-health"
"$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; exit 3; }
docker rm -f "$NAME" >/dev/null 2>&1

say "launch RADIX no_buffer + page_size=1 + TRITON attn (MTP+fused KEPT) TP=2 -> :$PORT"
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" -v "$KDIR:/work/kernel:ro" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -v "$REPO/sglang/patches/qwen3_coder_detector.py:/opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_W8A8_FUSED=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  -e SGLANG_MAX_THINK_TOKENS=4096 \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend triton --linear-attn-backend triton \
    --speculative-algorithm NEXTN --speculative-num-steps 10 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 11 --speculative-draft-attention-backend triton --disable-cuda-graph \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 1 \
    --mamba-radix-cache-strategy no_buffer \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-metrics \
    --tp 2 --context-length $CTX --mem-fraction-static 0.90 --max-running-requests 4 --skip-server-warmup \
    --host 0.0.0.0 --port $PORT" >/dev/null

say "waiting for /health (load + spec JIT ~3-8min)..."
ok=0
for i in $(seq 1 200); do
  docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "EXITED early -- logs:"; docker logs "$NAME" 2>&1 | tail -80; exit 1; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "NOT healthy after wait -- logs:"; docker logs "$NAME" 2>&1 | tail -80; exit 1; }

say "startup log (confirm no_buffer strategy chosen, radix on):"
docker logs "$NAME" 2>&1 | grep -iE 'radix|mamba|no_buffer|extra_buffer|page.?size|prefix|cache' | tail -25 || true

say "===== A/B prefix-cache test ====="
python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, time, urllib.request
port, served = sys.argv[1], sys.argv[2]
url = f"http://localhost:{port}/v1/chat/completions"

def block(word, reps):
    line = (word + " ") * 8
    return (line + "\n") * reps

# ~8k-token contexts. A and B diverge at the very first token -> B cannot reuse A's prefix.
ctxA = "CONTEXT-ALPHA unique marker 111.\n" + block("alpha beta gamma delta", 380)
ctxB = "CONTEXT-BRAVO unique marker 999.\n" + block("omega psi chi phi", 380)
Q = "\nIn one word, reply OK."
promptA = ctxA + Q
promptB = ctxB + Q

def fire(prompt, label):
    body = json.dumps({"model": served,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 4, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=420) as r:
        d = json.load(r)
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "") or (msg.get("reasoning_content") or "")
    u = d.get("usage", {}) or {}
    cached = ((u.get("prompt_tokens_details") or {}) or {}).get("cached_tokens")
    print(f"  {label:8s} t={dt:6.2f}s  prompt_tok={u.get('prompt_tokens')}  cached_tok={cached}  reply={content[:24]!r}")
    return dt, u.get("prompt_tokens"), cached

# small warmup to absorb spec/kernel JIT so it is NOT charged to the 'cold' A1 prefill
print("[warmup: absorb JIT]")
fire("Hello. Reply OK in one word.", "warmup")
print("[sequence]")
a1 = fire(promptA, "A1-cold")
a2 = fire(promptA, "A2-warm")
a3 = fire(promptA, "A3-warm")
b1 = fire(promptB, "B1-cold")
b2 = fire(promptB, "B2-warm")

print("\n[verdict]")
def spd(cold, warm):
    return f"{cold[0]/warm[0]:.1f}x faster" if warm[0] > 0 else "n/a"
print(f"  A warm speedup (A1->A2): {spd(a1,a2)}   A2 cached_tok={a2[2]} of {a2[1]}")
print(f"  A warm speedup (A1->A3): {spd(a1,a3)}")
print(f"  B cold ~= A cold?  A1={a1[0]:.2f}s B1={b1[0]:.2f}s  (B is a fresh prefix -> should miss)")
print(f"  B warm speedup (B1->B2): {spd(b1,b2)}   B2 cached_tok={b2[2]} of {b2[1]}")
hit = (a2[2] or 0) > 0.5 * (a2[1] or 1)
print(f"  RADIX PREFIX CACHE: {'WORKING (warm hit reuses prefix)' if hit else 'NO HIT (cached_tok ~0)'}")

# decode-throughput probe (triton attn vs prod intel_xpu ~25 t/s): short prompt, 128 new tokens
print("\n[decode throughput probe]")
def decode_probe(label, ntok=128):
    body = json.dumps({"model": served,
                       "messages": [{"role": "user", "content": "Count from 1 to 200 in words, one per line."}],
                       "max_tokens": ntok, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=420) as r:
        d = json.load(r)
    dt = time.time() - t0
    u = d.get("usage", {}) or {}
    ct = u.get("completion_tokens") or 0
    print(f"  {label}: {ct} tok in {dt:.2f}s -> {ct/dt:.2f} tok/s  (prod intel_xpu+MTP ~25 t/s)")
decode_probe("warm-decode")
PY

say "===== /metrics cache counters ====="
curl -s http://localhost:$PORT/metrics | grep -iE 'cache|prefix' | grep -ivE 'bucket|^#' | head -30 || true
say "done -- experiment container will be removed on exit"
