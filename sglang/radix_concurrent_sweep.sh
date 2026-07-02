#!/usr/bin/env bash
# radix_concurrent_sweep.sh -- SHELF GATE for the prefix-cache config proven single-stream in
# sglang/radix_nobuffer_ab.sh (no_buffer + page_size=1 + triton attn, MTP+fused KEPT). The failure mode that
# matters on this box is CONCURRENT prefill+decode (vLLM's "!!!!"); single-stream coherence is not enough.
# This fires overlapping waves that mix long fresh prefills with in-flight decodes (max_running_requests=4 ->
# real queueing/mixing), checks EVERY reply for garbage, and confirms cache hits survive under concurrency.
# Adds --enable-cache-report so usage.cached_tokens populates. Own container/port; self-cleaning.
#   /mnt/vm_8tb/b70/gpu-run bash sglang/radix_concurrent_sweep.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
IMG=sglang-xpu:mtp
NAME=sglang_radix_sweep
PORT=31001
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

say "launch cache config (no_buffer+page1+triton attn, MTP+fused, cache-report) TP=2 -> :$PORT"
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
    --mamba-radix-cache-strategy no_buffer --enable-cache-report \
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

say "===== concurrent prefill+decode sweep ====="
python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, time, urllib.request, concurrent.futures as cf
port, served = sys.argv[1], sys.argv[2]
url = f"http://localhost:{port}/v1/chat/completions"

def block(word, reps):
    line = (word + " ") * 8
    return (line + "\n") * reps
SHARED = "SHARED-CONTEXT alpha.\n" + block("alpha beta gamma delta", 300)     # ~6.5k tok, reused across reqs
def uniq(tag):
    return f"UNIQUE-{tag} marker.\n" + block(f"{tag} lorem ipsum dolor sit", 300)

def is_garbage(s):
    s = s.strip()
    if not s: return True
    if len(s) < 4: return False
    return max(s.count(c) for c in set(s)) / len(s) > 0.6   # same heuristic as serve.sh coherence gate

def fire(kind, i, prompt, ntok=200):
    body = json.dumps({"model": served, "messages":[{"role":"user","content":prompt}],
                       "max_tokens": ntok, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type":"application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r: d = json.load(r)
    except Exception as e:
        return {"i":i,"kind":kind,"ok":False,"err":str(e)[:90],"dt":time.time()-t0}
    dt = time.time()-t0
    m = d["choices"][0]["message"]; c = (m.get("content") or "") or (m.get("reasoning_content") or "")
    u = d.get("usage",{}) or {}
    cached = ((u.get("prompt_tokens_details") or {}) or {}).get("cached_tokens")
    return {"i":i,"kind":kind,"ok":True,"dt":dt,"garbage":is_garbage(c),
            "ptok":u.get("prompt_tokens"),"ctok":u.get("completion_tokens"),"cached":cached,"head":c[:50]}

def wave(label, tasks):
    print(f"\n[{label}: {len(tasks)} concurrent reqs, server max_running=4 -> mixed prefill+decode]")
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        res = list(ex.map(lambda a: fire(*a), tasks))
    wall=time.time()-t0
    for r in sorted(res,key=lambda r:r["i"]):
        if r.get("ok"):
            print(f"  [{r['kind']:6s}{r['i']:2d}] {r['dt']:6.2f}s ptok={r['ptok']} ctok={r['ctok']} cached={r['cached']} garbage={r['garbage']} head={r['head']!r}")
        else:
            print(f"  [{r['kind']:6s}{r['i']:2d}] ERROR {r['err']}")
    ok=[r for r in res if r.get("ok")]; garb=[r for r in ok if r.get("garbage")]; errs=[r for r in res if not r.get("ok")]
    tot=sum(r.get("ctok") or 0 for r in ok)
    print(f"  -> wall={wall:.1f}s coherent={len(ok)-len(garb)}/{len(res)} garbage={len(garb)} errors={len(errs)} agg_decode={tot/wall:.1f} tok/s")
    return len(garb)==0 and len(errs)==0, res

# Wave 1: 8 share the big prefix (concurrent cold prefill race) + 4 unique fresh prefills. Coherence-under-load.
t1 = [("shared",i, SHARED+f"\nQ{i}: explain concept {i} in ~120 words.") for i in range(8)] \
   + [("unique",i, uniq(f"U{i}")+"\nSummarize the above in ~120 words.") for i in range(4)]
p1,_ = wave("WAVE 1 (cold, mixed)", t1)

# Wave 2: re-fire the shared prefix (now cached) concurrently + new unique -> cache hits UNDER concurrency.
t2 = [("shared",i, SHARED+f"\nQ{i}: explain concept {i} in ~120 words.") for i in range(8)] \
   + [("unique",i, uniq(f"V{i}")+"\nSummarize the above in ~120 words.") for i in range(4)]
p2,r2 = wave("WAVE 2 (warm shared)", t2)
cachedhits = sum(1 for r in r2 if r.get("ok") and r["kind"]=="shared" and (r.get("cached") or 0) > 1000)
print(f"\n[cache-under-load] wave2 shared reqs with cached>1000 tok: {cachedhits}/8")
print("\n[SWEEP VERDICT] " + ("PASS -- coherent under concurrent prefill+decode, cache hits survive" if (p1 and p2) else "FAIL -- garbage/errors under load (see above)"))
PY

say "===== /metrics cache counters ====="
curl -s http://localhost:$PORT/metrics | grep -iE 'cache_hit|cached_tokens|prefill_cache|uncached' | grep -ivE 'bucket|^#' | head -20 || true
say "done"
