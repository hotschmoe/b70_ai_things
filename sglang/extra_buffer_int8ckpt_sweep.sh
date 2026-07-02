#!/usr/bin/env bash
# extra_buffer_int8ckpt_sweep.sh -- EXPERIMENT (2026-07-02): add --enable-int8-mamba-checkpoint on top of the
# CHOSEN extra_buffer cache config (intel_xpu XMX + page 128). The int8 checkpoint pool stores radix-cached
# mamba states in int8 (separate pool) for ~2x cached-prefix CAPACITY at fixed memory. Constraints are clean
# (only rejects --enable-hierarchical-cache / --radix-cache-backend, neither used); quant/dequant is pure-torch.
# RISK to prove: does XPU int8-quant of mamba states stay COHERENT, and does it allocate without a CUDA path?
# Combined run: startup confirm + cache A/B + long-context decode (no regression?) + concurrent coherence gate.
#   /mnt/vm_8tb/b70/gpu-run bash sglang/extra_buffer_int8ckpt_sweep.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
IMG=sglang-xpu:mtp
NAME=sglang_ebi8
PORT=31004
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

say "launch extra_buffer + int8-mamba-checkpoint + intel_xpu XMX + page 128 (MTP+fused) TP=2 -> :$PORT"
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
    --mamba-radix-cache-strategy extra_buffer --enable-int8-mamba-checkpoint --enable-cache-report \
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

say "startup: confirm int8 checkpoint pool + extra_buffer + un-gate"
docker logs "$NAME" 2>&1 | grep -iE "int8.*checkpoint|checkpoint.*pool|un-gated extra_buffer|mamba_radix_cache_strategy=|enable_int8_mamba_checkpoint|Tree cache initialized" | head -15 || true

say "===== cache A/B + long-context decode (int8 checkpoint) ====="
python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, time, urllib.request
port, served = sys.argv[1], sys.argv[2]
url=f"http://localhost:{port}/v1/chat/completions"
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
print("[warmup]"); fire("Hello. Reply OK.",4)
pA=ctx(360,"A")+"\nReply OK."
c1=fire(pA,4); w1=fire(pA,4)
print(f"[cache 12k] cold={c1[0]:.2f}s warm={w1[0]:.2f}s ({c1[0]/max(w1[0],0.01):.1f}x) warm_cached={w1[3]}/{w1[1]} coherent={'OK' if w1[4].strip() and not is_garbage(w1[4]) else 'FAIL'}")
print("[long-context decode -- extra_buffer + int8 ckpt (expect ~ extra_buffer: 18.6@30k, 11.5@60k)]")
for reps,tag in [(900,"~30k"),(1800,"~60k")]:
    p=ctx(reps,tag)+"\nWrite a long detailed essay about the history of France."
    fire(p,4); a=fire(p,16); b=fire(p,144)
    dtps=128/(b[0]-a[0]) if b[0]-a[0]>0.05 else float('nan')
    print(f"  [{tag}] ptok={b[1]} cached={b[3]} -> DECODE ~{dtps:.1f} t/s  garbage={'no' if not is_garbage(b[4]) else 'CHECK'}")
PY

say "===== concurrent coherence gate (int8 checkpoint) ====="
python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, time, urllib.request, concurrent.futures as cf
port, served = sys.argv[1], sys.argv[2]
url=f"http://localhost:{port}/v1/chat/completions"
def block(w,r): return ((w+" ")*8+"\n")*r
SHARED="SHARED-CONTEXT alpha.\n"+block("alpha beta gamma delta",300)
def uniq(t): return f"UNIQUE-{t}.\n"+block(f"{t} lorem ipsum dolor sit",300)
def is_garbage(s):
    s=s.strip()
    if not s: return True
    if len(s)<4: return False
    return max(s.count(c) for c in set(s))/len(s) > 0.6
def fire(kind,i,prompt,ntok=200):
    body=json.dumps({"model":served,"messages":[{"role":"user","content":prompt}],"max_tokens":ntok,"temperature":0}).encode()
    req=urllib.request.Request(url,data=body,headers={"content-type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=600) as r: d=json.load(r)
    except Exception as e:
        return {"i":i,"ok":False,"err":str(e)[:80]}
    dt=time.time()-t0; m=d["choices"][0]["message"]; c=(m.get("content") or "") or (m.get("reasoning_content") or "")
    u=d.get("usage",{}) or {}
    return {"i":i,"ok":True,"dt":dt,"g":is_garbage(c),"ctok":u.get("completion_tokens"),"cached":((u.get("prompt_tokens_details") or {}) or {}).get("cached_tokens")}
def wave(label,tasks):
    print(f"[{label}: {len(tasks)} concurrent, max_running=4]")
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        res=list(ex.map(lambda a: fire(*a), tasks))
    wall=time.time()-t0
    ok=[r for r in res if r.get("ok")]; g=[r for r in ok if r.get("g")]; e=[r for r in res if not r.get("ok")]
    tot=sum(r.get("ctok") or 0 for r in ok)
    print(f"  coherent={len(ok)-len(g)}/{len(res)} garbage={len(g)} errors={len(e)} agg_decode={tot/wall:.1f} t/s wall={wall:.1f}s")
    return len(g)==0 and len(e)==0
t1=[("s",i,SHARED+f"\nQ{i}: explain concept {i} in ~120 words.") for i in range(8)]+[("u",i,uniq(f"U{i}")+"\nSummarize in ~120 words.") for i in range(4)]
p1=wave("WAVE1 cold",t1)
t2=[("s",i,SHARED+f"\nQ{i}: explain concept {i} in ~120 words.") for i in range(8)]+[("u",i,uniq(f"V{i}")+"\nSummarize in ~120 words.") for i in range(4)]
p2=wave("WAVE2 warm",t2)
print("[INT8-CKPT VERDICT] "+("PASS -- coherent + caching with int8 mamba checkpoint" if (p1 and p2) else "FAIL"))
PY

say "===== /metrics cache counters ====="
curl -s http://localhost:$PORT/metrics | grep -iE 'cache_hit|cached_tokens|prefill_cache' | grep -ivE 'bucket|^#' | head -8 || true
say "done"
