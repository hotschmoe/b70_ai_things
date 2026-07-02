#!/usr/bin/env bash
# extra_buffer_sweep.sh -- SHELF GATE for the extra_buffer cache config (intel_xpu XMX attn + page_size=128 +
# extra_buffer strategy, un-gated on XPU via B70_XPU_MAMBA_EXTRA_BUFFER=1). Single-stream proved (extra_buffer_ab.sh):
# caching + NO long-context decode collapse (22.7/18.6/11.5 t/s @12k/30k/60k vs triton+page1's 29/5.5/1-4).
# This is the concurrent prefill+decode coherence gate (the "!!!!" failure mode) + a clean steady-state decode
# throughput read over 200-token generations. Own container/port; self-cleaning. Run under the lease:
#   /mnt/vm_8tb/b70/gpu-run bash sglang/extra_buffer_sweep.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
IMG=sglang-xpu:mtp
NAME=sglang_eb_sweep
PORT=31003
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

say "launch extra_buffer + intel_xpu XMX + page_size=128 (MTP+fused) TP=2 -> :$PORT"
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

say "===== concurrent prefill+decode sweep (extra_buffer) ====="
python3 - "$PORT" "$SERVED" <<'PY'
import sys, json, time, urllib.request, concurrent.futures as cf
port, served = sys.argv[1], sys.argv[2]
url = f"http://localhost:{port}/v1/chat/completions"
def block(w,r): return ((w+" ")*8+"\n")*r
SHARED = "SHARED-CONTEXT alpha.\n" + block("alpha beta gamma delta", 300)     # ~6.5k reused
def uniq(tag): return f"UNIQUE-{tag} marker.\n" + block(f"{tag} lorem ipsum dolor sit", 300)
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
        return {"i":i,"kind":kind,"ok":False,"err":str(e)[:90],"dt":time.time()-t0}
    dt=time.time()-t0; m=d["choices"][0]["message"]; c=(m.get("content") or "") or (m.get("reasoning_content") or "")
    u=d.get("usage",{}) or {}; cached=((u.get("prompt_tokens_details") or {}) or {}).get("cached_tokens")
    return {"i":i,"kind":kind,"ok":True,"dt":dt,"garbage":is_garbage(c),"ptok":u.get("prompt_tokens"),"ctok":u.get("completion_tokens"),"cached":cached,"head":c[:44]}
def wave(label,tasks):
    print(f"\n[{label}: {len(tasks)} concurrent, max_running=4 -> mixed prefill+decode]")
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        res=list(ex.map(lambda a: fire(*a), tasks))
    wall=time.time()-t0
    for r in sorted(res,key=lambda r:r["i"]):
        if r.get("ok"): print(f"  [{r['kind']:6s}{r['i']:2d}] {r['dt']:6.2f}s ptok={r['ptok']} ctok={r['ctok']} cached={r['cached']} garbage={r['garbage']} head={r['head']!r}")
        else: print(f"  [{r['kind']:6s}{r['i']:2d}] ERROR {r['err']}")
    ok=[r for r in res if r.get("ok")]; garb=[r for r in ok if r.get("garbage")]; errs=[r for r in res if not r.get("ok")]
    tot=sum(r.get("ctok") or 0 for r in ok)
    print(f"  -> wall={wall:.1f}s coherent={len(ok)-len(garb)}/{len(res)} garbage={len(garb)} errors={len(errs)} agg_decode={tot/wall:.1f} tok/s")
    return len(garb)==0 and len(errs)==0
t1=[("shared",i, SHARED+f"\nQ{i}: explain concept {i} in ~120 words.") for i in range(8)] + [("unique",i, uniq(f"U{i}")+"\nSummarize in ~120 words.") for i in range(4)]
p1=wave("WAVE 1 (cold, mixed)", t1)
t2=[("shared",i, SHARED+f"\nQ{i}: explain concept {i} in ~120 words.") for i in range(8)] + [("unique",i, uniq(f"V{i}")+"\nSummarize in ~120 words.") for i in range(4)]
p2=wave("WAVE 2 (warm shared)", t2)
print("\n[SWEEP VERDICT] "+("PASS -- coherent under concurrent prefill+decode (extra_buffer)" if (p1 and p2) else "FAIL -- garbage/errors under load"))
PY

say "===== steady-state decode t/s from logs (200-tok gens) ====="
docker logs "$NAME" 2>&1 | grep -iE "Decode batch" | grep -oE "#full token: [0-9]+.*gen throughput \(token/s\): [0-9.]+" | tail -20 || true
say "===== /metrics cache counters ====="
curl -s http://localhost:$PORT/metrics | grep -iE 'cache_hit|cached_tokens|prefill_cache' | grep -ivE 'bucket|^#' | head -10 || true
say "done"
