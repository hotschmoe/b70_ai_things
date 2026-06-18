#!/usr/bin/env bash
# Minimal generation test via /v1/completions (no chat template needed). Tries Qwen
# chat format. Prints the actual text + decode t/s. NAME, MODEL(served name), PORT.
set -uo pipefail
NAME="${NAME:-vllm_qwen3}"; MODEL="${MODEL:-qwen36-int4}"; PORT="${PORT:-18080}"
docker exec -i -e MODEL="$MODEL" -e PORT="$PORT" "$NAME" python - <<'PY'
import os,time,json,urllib.request
M=os.environ["MODEL"]; P=os.environ["PORT"]
def comp(prompt,n,stream=True):
    body=json.dumps({"model":M,"prompt":prompt,"max_tokens":n,"temperature":0,"stream":stream,
                     "stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(f"http://localhost:{P}/v1/completions",data=body,
        headers={"Content-Type":"application/json"})
    t0=time.time();ttft=None;k=0;txt=""
    with urllib.request.urlopen(req) as r:
        for raw in r:
            s=raw.decode("utf-8","ignore").strip()
            if not s.startswith("data:"):continue
            d=s[5:].strip()
            if d=="[DONE]":break
            o=json.loads(d);ch=o.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:ttft=time.time()-t0
                k+=1; txt+=ch[0]["text"]
    dt=time.time()-t0; dec=(k-1)/(dt-ttft) if (ttft and k>1 and dt>ttft) else float("nan")
    return txt,ttft,dec,k
p="<|im_start|>user\nIn one sentence, what is an Intel Arc GPU?<|im_end|>\n<|im_start|>assistant\n"
print("=== generation ===")
txt,ttft,dec,k=comp(p,80)
print("OUTPUT:",repr(txt[:300]))
print(f"gen={k} TTFT={(ttft*1000 if ttft else -1):.0f}ms decode={dec:.2f} tok/s")
print("=== 3x coherent decode (200 tok) ===")
ps=["<|im_start|>user\nWrite a 200-word explanation of how a GPU does matrix multiply.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain step by step how HTTPS works.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a python quicksort with comments.<|im_end|>\n<|im_start|>assistant\n"]
ds=[]
for i,pp in enumerate(ps):
    _,tt,dd,kk=comp(pp,200); print(f"[gen{i}] tok={kk} decode={dd:.2f} t/s")
    if dd==dd: ds.append(dd)
if ds: print(f"MEAN decode: {sum(ds)/len(ds):.2f} tok/s")
PY
