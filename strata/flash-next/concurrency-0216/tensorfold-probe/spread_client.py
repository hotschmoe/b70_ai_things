#!/usr/bin/env python3
"""N concurrent greedy streams spread round-robin over several base URLs. Per-stream decode tok/s, TTFT, aggregate.
usage: spread_client.py MODEL_PREFIX TOKENS OUT.json URL [URL ...] [--n N] ; model id = MODEL_PREFIX + '-r<i>' if --suffix
Stdlib only."""
import json, sys, threading, time, urllib.request, statistics, argparse
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=16)
ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--prompts", default="/mnt/vm_8tb/b70/build/tensorfold/main/tools/bench_prompts.json")
ap.add_argument("--model", action="append", required=True, help="served id per URL, in order"); ap.add_argument("url", nargs="+")
a = ap.parse_args()
prompts = json.load(open(a.prompts)); prompts = [p if isinstance(p, str) else p.get("prompt") or p.get("text") for p in prompts]
res = [None] * a.n
def one(i):
    u = a.url[i % len(a.url)]; m = a.model[i % len(a.model)]
    body = {"model": m, "max_tokens": a.tokens, "temperature": 0, "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(u + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; ntok = 0; pieces = 0; stamps = []
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("usage"): ntok = d["usage"].get("completion_tokens", ntok)
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
                    now = time.perf_counter(); pieces += 1; stamps.append(now)
                    if first is None: first = now
                    last = now
        res[i] = dict(i=i, url=u, ttft=first - t0 if first else None, tokens=ntok, pieces=pieces,
                      tps=((ntok - 1) / (last - first)) if first and last and last > first and ntok > 1 else None,
                      first=first, last=last, t0=t0, itl_p50=pct([b - a for a, b in zip(stamps, stamps[1:])], 0.5) if len(stamps) > 2 else None,
                      itl_p95=pct([b - a for a, b in zip(stamps, stamps[1:])], 0.95) if len(stamps) > 2 else None, tok_per_chunk=(ntok / pieces) if pieces else None)
    except Exception as e:
        res[i] = dict(i=i, url=u, error=repr(e))
def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
ths = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
T0 = time.perf_counter()
for t in ths: t.start()
for t in ths: t.join()
ok = [r for r in res if r and not r.get("error") and r.get("tps")]
tt = [r["ttft"] for r in ok]; tp = [r["tps"] for r in ok]
firsts = min(r["first"] for r in ok); lasts = max(r["last"] for r in ok)
agg = sum(r["tokens"] - 1 for r in ok) / (lasts - firsts)
summ = dict(n=a.n, ok=len(ok), errors=[r for r in res if r and r.get("error")], wall_s=time.perf_counter() - T0,
            per_stream_tps_median=statistics.median(tp), per_stream_tps_min=min(tp), per_stream_tps_max=max(tp),
            ttft_p50=pct(tt, 0.5), ttft_p95=pct(tt, 0.95), aggregate_tps=agg,
            itl_p50_median_over_streams=statistics.median([r['itl_p50'] for r in ok if r.get('itl_p50')]),
            itl_p95_median_over_streams=statistics.median([r['itl_p95'] for r in ok if r.get('itl_p95')]),
            itl_p95_worst_stream=max([r['itl_p95'] for r in ok if r.get('itl_p95')]),
            tokens_per_chunk_median=statistics.median([r['tok_per_chunk'] for r in ok if r.get('tok_per_chunk')]))
json.dump(dict(summary=summ, streams=[{k: v for k, v in r.items() if k not in ("first", "last", "t0")} for r in res]), open(a.out, "w"), indent=1)
print(json.dumps(summ, indent=1))
for r in res: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("first", "last", "t0")})
