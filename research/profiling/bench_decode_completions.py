#!/usr/bin/env python3
"""Usage-based decode bench over /v1/completions (no chat template -> immune to the
THINK_BUDGET/thinking_token_budget chat-400 on serves without a reasoning parser).
Same spirit as vllm/nvfp4/bench_code.py: real coding prompts, non-streaming, decode
t/s = completion_tokens / wall. Usage: bench_decode_completions.py <base/v1> <model> [conc=1] [out=256] [reps=3]
"""
import sys, os, json, time, threading, urllib.request
BASE=sys.argv[1].rstrip("/"); MODEL=sys.argv[2]
NC=int(sys.argv[3]) if len(sys.argv)>3 else 1
OUT=int(sys.argv[4]) if len(sys.argv)>4 else 256
REPS=int(sys.argv[5]) if len(sys.argv)>5 else 3
KEY=os.environ.get("API_KEY","")
URL=BASE+"/completions"
PROMPTS=[
 "Write a complete Python implementation of an LRU cache with O(1) get/put using a doubly linked list and a dict, with type hints and a usage example:\n\n",
 "Write a thread-safe bounded blocking queue in Python using threading.Condition, with a producer/consumer demo:\n\n",
 "Implement Dijkstra's shortest path in Python with a binary heap and a Graph class with add_edge, plus a worked example:\n\n",
]
def one(i,res):
    body=json.dumps({"model":MODEL,"prompt":PROMPTS[i%len(PROMPTS)],"max_tokens":OUT,"temperature":0.0,"ignore_eos":True}).encode()
    hdr={"content-type":"application/json"}
    if KEY: hdr["Authorization"]="Bearer "+KEY
    t0=time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL,data=body,headers=hdr),timeout=600) as r:
            o=json.load(r)
    except Exception as e:
        res[i]={"err":str(e)[:80]}; return
    w=time.time()-t0; u=o.get("usage") or {}; ct=u.get("completion_tokens")
    res[i]={"wall":w,"ct":ct,"tps":(ct/w if ct and w else float("nan"))}
def run(N):
    res=[None]*N; ts=[threading.Thread(target=one,args=(i,res)) for i in range(N)]
    for t in ts: t.start()
    for t in ts: t.join()
    ok=[r for r in res if r and r.get("tps")==r.get("tps") and r.get("tps")]
    errs=[r["err"] for r in res if r and r.get("err")]
    if errs: print("   errs:",errs[:2])
    if not ok: return None
    return {"tps":sum(r["tps"] for r in ok)/len(ok),"agg":sum(r["tps"] for r in ok),"ct":sum(r["ct"] for r in ok)//len(ok)}
if __name__=="__main__":
    print(f"model={MODEL} conc={NC} out={OUT} reps={REPS} (completions, usage-based decode)")
    _=run(1)
    rows=[r for r in (run(NC) for _ in range(REPS)) if r]
    if rows:
        best=max(rows,key=lambda x:x["tps"])
        print(f"c{NC}: decode avg={sum(x['tps'] for x in rows)/len(rows):.1f} best={best['tps']:.1f} t/s | agg={sum(x['agg'] for x in rows)/len(rows):.1f} | out~{rows[0]['ct']}")
    else:
        print(f"c{NC}: ALL FAILED")
