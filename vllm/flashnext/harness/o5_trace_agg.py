#!/usr/bin/env python3
"""O5: per-kernel totals over the last traced segment, divided by N steps.

  python3 o5_trace_agg.py trace.json NSTEPS [--top 50] [--seg-gap-s 1.0]
"""
import argparse, collections, json, re


def short(n):
    n = re.sub(r"\[SIMD.*$", "", n).strip()
    n = re.sub(r"<.*", "<>", n) if len(n) > 120 else n
    return n[:120]


ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("nsteps", type=float)
ap.add_argument("--top", type=int, default=50); ap.add_argument("--after", default="")
ap.add_argument("--skip", type=float, default=0.0); ap.add_argument("--seg-gap-s", type=float, default=1.0)
a = ap.parse_args()
d = json.load(open(a.trace))
ev = d["traceEvents"] if isinstance(d, dict) else d
pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
gpu = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "gpu_op":
        gpu[e["pid"]].append((e["ts"], e["ts"] + e["dur"], short(e["name"])))
allv = sorted(x for v in gpu.values() for x in v)
cut, end = allv[0][0], None
for s, t, _ in allv:
    if end is not None and s - end > a.seg_gap_s * 1e6:
        cut = s
    end = t if end is None else max(end, t)
if a.after:
    # start after the last kernel whose name contains --after (e.g. prefill-only kernels), plus --skip ms
    last = max((t for v in gpu.values() for s0, t, n in v if s0 >= cut and a.after in n), default=cut)
    cut = last + a.skip * 1e3
for p in sorted(gpu, key=lambda p: pname.get(p, "")):
    v = [x for x in gpu[p] if x[0] >= cut]
    if not v:
        continue
    agg = collections.defaultdict(lambda: [0, 0.0])
    for s, t, n in v:
        agg[n][0] += 1; agg[n][1] += t - s
    tot = sum(x[1] for x in agg.values())
    print(f"\nDEVICE {pname.get(p, p)} kernels/step={len(v)/a.nsteps:.0f} sum/step={tot/a.nsteps/1e3:.2f} ms")
    for n, (c, du) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(f"  {du/a.nsteps/1e3:8.3f} ms  n/step={c/a.nsteps:7.1f}  mean_us={du/c:8.2f}  {n}")
