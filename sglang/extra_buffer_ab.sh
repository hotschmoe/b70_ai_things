#!/usr/bin/env bash
# extra_buffer_ab.sh -- EXPERIMENT (2026-07-02): prefix caching via the mamba extra_buffer strategy while
# KEEPING the intel_xpu XMX attention backend at page_size=128. This is the fix for the long-context DECODE
# COLLAPSE seen on the shipped no_buffer+page_size=1+triton config (29 t/s @8k -> 1-4 t/s @60k on the live DD).
# extra_buffer keeps XMX attn + page 128, so decode should stay fast at long context while still caching.
#
# The ONLY blocker was one assert (server_args._validate_mamba_extra_buffer: is_cuda()/musa()/npu()); our shim
# now drops it under B70_XPU_MAMBA_EXTRA_BUFFER=1 (mtp_tree_xpu.py DOMINO 5). With --page-size 128 the default
# mamba_track_interval=256 already satisfies the validator (256%128==0, 256>=11 draft tokens); chunk_size auto.
#
# Centerpiece measurement: DECODE t/s at 12k/30k/60k context (delta-timing: same cached prefix, 16 vs 144 out
# -> delta = pure decode). Compare vs the collapse. Own container/port; self-cleaning. Run under the lease:
#   /mnt/vm_8tb/b70/gpu-run bash sglang/extra_buffer_ab.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
IMG=sglang-xpu:mtp
NAME=sglang_extrabuf
PORT=31002
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

say "launch extra_buffer + intel_xpu XMX attn + page_size=128 (MTP+fused KEPT) TP=2 -> :$PORT"
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
  -v "$REPO/models/files:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" -v "$KDIR:/work/kernel:ro" \
  -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
  -v "$REPO/sglang/patches/mtp_tree_xpu.py:/opt/venv/lib/python3.12/site-packages/mtp_tree_xpu.py:ro" \
  -v "$REPO/sglang/patches/qwen3_coder_detector.py:/opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
  -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
  -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_W8A8_FUSED=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
  -e B70_XPU_MAMBA_EXTRA_BUFFER=1 -e SGLANG_MAX_THINK_TOKENS=4096 \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
    --device xpu --attention-backend intel_xpu --linear-attn-backend triton \
    --speculative-algorithm NEXTN --speculative-num-steps 10 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 11 --speculative-draft-attention-backend triton --disable-cuda-graph \
    --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 128 \
    --mamba-radix-cache-strategy extra_buffer --enable-cache-report \
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

say "startup: confirm extra_buffer + intel_xpu attn + un-gate shim"
docker logs "$NAME" 2>&1 | grep -iE "un-gated extra_buffer|extra_buffer un-gate|mamba_radix_cache_strategy=|Tree cache initialized|attention backend|Linear attention kernel" | head -15 || true

say "===== extra_buffer cache + LONG-CONTEXT DECODE test ====="
python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, time, urllib.request
port, served = sys.argv[1], sys.argv[2]
url = f"http://localhost:{port}/v1/chat/completions"
def block(w,r): return ((w+" ")*8+"\n")*r
def ctx(reps,tag): return f"CTX-{tag} marker.\n"+block("alpha beta gamma delta",reps)
def is_garbage(s):
    s=s.strip()
    if not s: return True
    if len(s)<4: return False
    return max(s.count(c) for c in set(s))/len(s) > 0.6
def fire(prompt,ntok):
    body=json.dumps({"model":served,"messages":[{"role":"user","content":prompt}],"max_tokens":ntok,"temperature":0}).encode()
    req=urllib.request.Request(url,data=body,headers={"content-type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=900) as r: d=json.load(r)
    dt=time.time()-t0; u=d.get("usage",{}) or {}
    m=d["choices"][0]["message"]; c=(m.get("content") or "") or (m.get("reasoning_content") or "")
    cached=((u.get("prompt_tokens_details") or {}) or {}).get("cached_tokens")
    return dt,u.get("prompt_tokens"),u.get("completion_tokens"),cached,c

print("[warmup: absorb JIT]"); fire("Hello. Reply OK.",4)

# cache A/B at ~12k
pA=ctx(360,"A")+"\nReply OK."
c1=fire(pA,4); w1=fire(pA,4)
print(f"[cache 12k] cold={c1[0]:.2f}s warm={w1[0]:.2f}s ({c1[0]/max(w1[0],0.01):.1f}x) warm_cached={w1[3]}/{w1[1]} coherent={'OK' if w1[4].strip() and not is_garbage(w1[4]) else 'FAIL'}")

# LONG-CONTEXT DECODE via delta-timing: same CACHED prefix, 16 vs 144 out -> delta 128 = pure decode.
print("[long-context decode -- extra_buffer + intel_xpu XMX @ page 128]")
print("  (compare vs shipped no_buffer+triton+page1: ~29 t/s @8k, 5.5 @26k, 1-4 @60k)")
for reps,tag in [(360,"~12k"),(900,"~30k"),(1800,"~60k")]:
    p=ctx(reps,tag)+"\nWrite a long detailed essay about the history of France."
    warm=fire(p,4)                      # cache the prefix
    a=fire(p,16); b=fire(p,144)         # delta = 128 decode tokens on the cached prefix
    dtps=128/(b[0]-a[0]) if b[0]-a[0]>0.05 else float('nan')
    print(f"  [{tag}] ptok={b[1]} cached={b[3]} a={a[0]:.2f}s(16out) b={b[0]:.2f}s(144out) -> DECODE ~{dtps:.1f} t/s  garbage={'no' if not is_garbage(b[4]) else 'CHECK'}")
PY

say "===== gen throughput from server logs (matches the DD-diagnosis metric) ====="
docker logs "$NAME" 2>&1 | grep -iE "Decode batch" | grep -oE "#full token: [0-9]+.*gen throughput \(token/s\): [0-9.]+" | tail -20 || true
say "===== /metrics cache counters ====="
curl -s http://localhost:$PORT/metrics | grep -iE 'cache_hit|cached_tokens|prefill_cache' | grep -ivE 'bucket|^#' | head -10 || true
say "done"
