#!/usr/bin/env bash
# Single-stream COHERENT-generation bench (real text -> meaningful spec-decode acceptance).
# Measures TTFT + decode t/s over a few coherent prompts; reads vLLM /metrics spec-decode
# acceptance counters. Args: [container] [served_model] [label]
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
# Accept env (NAME/MODEL/LABEL) with positional fallback, so runremote.sh (env-only transport) can drive it.
NAME="${NAME:-${1:-vllm_qwen3}}"; MODEL="${MODEL:-${2:-qwen3-14b}}"; LABEL="${LABEL:-${3:-qwen3-14b}}"; PORT=18080
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT="$ROOT/results/specbench_${LABEL}_${STAMP}.txt"
curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || { echo "not healthy"; exit 1; }

docker exec -i -e PORT="$PORT" -e MODEL="$MODEL" "$NAME" python - <<'PY' 2>&1 | tee "$OUT"
import os,time,json,urllib.request
PORT=os.environ["PORT"]; MODEL=os.environ["MODEL"]
U=f"http://localhost:{PORT}/v1/chat/completions"
def run(msg,n,label):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":msg}],
        "max_tokens":n,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}).encode()
    r=urllib.request.Request(U,data=body,headers={"Content-Type":"application/json"})
    t0=time.time();ttft=None;k=0
    try:
        with urllib.request.urlopen(r) as resp:
            for raw in resp:
                s=raw.decode("utf-8","ignore").strip()
                if not s.startswith("data:"):continue
                d=s[5:].strip()
                if d=="[DONE]":break
                o=json.loads(d);ch=o.get("choices") or []
                if ch and ch[0].get("delta",{}).get("content"):
                    if ttft is None:ttft=time.time()-t0
                    k+=1
    except Exception as e:
        print(f"[{label}] ERR {e}");return None
    dt=time.time()-t0; dec=(k-1)/(dt-ttft) if (ttft and k>1 and dt>ttft) else float("nan")
    print(f"[{label}] gen={k} TTFT={ (ttft*1000 if ttft else -1):.0f}ms decode={dec:.2f} tok/s total={dt:.2f}s")
    return dec
prompts=["Write a detailed 200-word explanation of how a GPU executes a matrix multiplication.",
         "Explain step by step how HTTP works when you load a web page.",
         "Write a Python function that merges two sorted lists, with comments."]
print("== warmup =="); run(prompts[0],16,"warm")
decs=[]
for i,p in enumerate(prompts):
    d=run(p,200,f"gen{i}");
    if d==d: decs.append(d)
if decs: print(f"MEAN coherent decode: {sum(decs)/len(decs):.2f} tok/s over {len(decs)} runs")
PY

echo "=== spec-decode acceptance metrics (if any) ===" | tee -a "$OUT"
curl -s "http://localhost:${PORT}/metrics" 2>/dev/null | grep -iE 'spec_decode|draft|accept|num_accepted|num_draft' | grep -v '#' | head -20 | tee -a "$OUT"
echo "=== saved $OUT ==="
