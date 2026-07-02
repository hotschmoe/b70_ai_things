#!/usr/bin/env python3
# analyze_trace.py -- decompose a torch-profiler (kineto) chrome trace from the sglang decode loop.
# Answers: how much of an MTP iteration is GPU-busy vs idle (launch/python-bound), which ops dominate,
# and how big the inter-kernel gaps are. Works on .json or .json.gz (pt.trace.json).
#   usage: analyze_trace.py <trace.json[.gz]> [top_n]
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)

events = data["traceEvents"] if isinstance(data, dict) else data

# device-side kernel events: kineto marks GPU/XPU kernels with cat in {kernel, gpu_op, Kernel,
# gpu_memcpy, ...}; XPU (PTI) traces use cat "kernel"/"gpu_op" on a device pid track.
kern = []
cpu_ops = defaultdict(lambda: [0, 0.0])   # name -> [count, total_us]
cats = defaultdict(int)
for e in events:
    if not isinstance(e, dict) or e.get("ph") != "X":
        continue
    cat = (e.get("cat") or "").lower()
    cats[cat] += 1
    dur = e.get("dur", 0)
    name = e.get("name", "?")
    if "kernel" in cat or "gpu" in cat:
        kern.append((e["ts"], dur, name))
    elif cat in ("cpu_op", "operator"):
        c = cpu_ops[name]
        c[0] += 1
        c[1] += dur

print("== event categories ==")
for c, n in sorted(cats.items(), key=lambda x: -x[1])[:12]:
    print(f"  {c or '(none)':24s} {n}")

if kern:
    kern.sort()
    t0, t1 = kern[0][0], max(ts + d for ts, d, _ in kern)
    span = t1 - t0
    busy = sum(d for _, d, _ in kern)
    # merge-overlap busy (multi-queue): sweep
    ivs = sorted((ts, ts + d) for ts, d, _ in kern)
    merged = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s > cur_e:
            merged += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    merged += cur_e - cur_s
    print(f"\n== device track ==\nkernels={len(kern)} span={span/1e3:.1f}ms busy(sum)={busy/1e3:.1f}ms "
          f"busy(merged)={merged/1e3:.1f}ms idle={100*(1-merged/span):.1f}%")
    # gap histogram
    gaps = defaultdict(int)
    prev_end = None
    big_gaps = 0.0
    for ts, d, _ in kern:
        if prev_end is not None and ts > prev_end:
            g = ts - prev_end
            if g >= 1000:
                gaps[">=1ms"] += 1
                big_gaps += g
            elif g >= 100:
                gaps["100us-1ms"] += 1
                big_gaps += g
            elif g >= 20:
                gaps["20-100us"] += 1
                big_gaps += g
        prev_end = max(prev_end or 0, ts + d)
    print(f"gap buckets: {dict(gaps)}  total-gap>=20us={big_gaps/1e3:.1f}ms")
    # top kernels
    agg = defaultdict(lambda: [0, 0.0])
    for _, d, name in kern:
        a = agg[name[:110]]
        a[0] += 1
        a[1] += d
    print(f"\n== top {top_n} device kernels by total time ==")
    for name, (n, tot) in sorted(agg.items(), key=lambda x: -x[1][1])[:top_n]:
        print(f"  {tot/1e3:9.2f}ms  n={n:6d}  avg={tot/n:7.1f}us  {name}")
else:
    print("\nNO device kernel events found -- profiler may be CPU-only on this build")

if cpu_ops:
    print(f"\n== top {top_n} CPU ops by total time ==")
    for name, (n, tot) in sorted(cpu_ops.items(), key=lambda x: -x[1][1])[:top_n]:
        print(f"  {tot/1e3:9.2f}ms  n={n:6d}  avg={tot/n:7.1f}us  {name[:110]}")
