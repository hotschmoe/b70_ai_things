#!/usr/bin/env python3
import json, glob, os, sys, statistics
d = sys.argv[1]
def f(x, n=1): return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
for n in (1, 2, 4, 8, 16):
    p = f"{d}/conc_{n}.json"
    if not os.path.exists(p): continue
    for c in json.load(open(p))["cells"]:
        r = c["request"]; reps = c["reps"]; tt = sorted(t for rep in reps for t in rep["ttft_s"])
        pct = lambda q: tt[min(len(tt) - 1, int(round(q * (len(tt) - 1))))]
        print(f"conc n={n} {c['prompt']:4s} T={c['temperature']}: agg {f(c['aggregate_tps'])} per-stream {f(c['per_stream_tps'])} "
              f"ttft p50 {f(pct(.5),2)} p95 {f(pct(.95),2)} max {f(c['ttft_s_max'],2)} chunk-gap p50/p95 ms {f(r['round_gap_p50_ms'])}/{f(r['round_gap_p95_ms'])} "
              f"tok/chunk {f(r['tokens_per_chunk'],2)} alone eq/uneq {c['alone']['equal']}/{c['alone']['unequal']} serial eq/uneq {c['serial']['equal']}/{c['serial']['unequal']}")
for p in sorted(glob.glob(f"{d}/cache_*.json")):
    j = json.load(open(p)); 
    ph = j.get("phases") or j.get("results") or j
    print(os.path.basename(p), json.dumps(j)[:0])
    def walk(o, path=""):
        if isinstance(o, dict):
            if "prompt_tokens" in o and "ttft_s" in o:
                print(f"  {o.get('phase','?'):18s} prompt {o['prompt_tokens']} cached {o.get('cached_tokens')} ttft {f(o['ttft_s'],3)} s -> prefill {f(o['prompt_tokens']/o['ttft_s'] if o['ttft_s'] else None,0)} tok/s  cache_state {o.get('cache_state')}")
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(j); print("  qualified:", j.get("qualified"))
