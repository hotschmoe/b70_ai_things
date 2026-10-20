#!/usr/bin/env python3
"""O4: summarize a unitrace Chrome trace of the traced C1 segment.

Takes the GPU events after the last long idle gap on every device (the
script pauses unitrace after load and resumes it only around one short
C1 request). Splits card-0 activity into steps at idle gaps, then ranks
kernels per device by summed duration per decode step.

  python3 o4_trace.py trace.json [--gap-ms 0.4] [--top 60]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics


def short(name: str) -> str:
    name = re.sub(r"\[SIMD.*$", "", name).strip()
    if len(name) > 150:
        name = name[:150]
    return name


def union(iv):
    tot = 0.0
    end = None
    for a, b in sorted(iv):
        if end is None or a > end:
            tot += b - a
            end = b
        elif b > end:
            tot += b - end
            end = b
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-ms", type=float, default=0.4)
    ap.add_argument("--seg-gap-s", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--dump-step", type=int, default=-1)
    a = ap.parse_args()
    d = json.load(open(a.trace))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    gpu = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "gpu_op":
            gpu[e["pid"]].append((e["ts"], e["ts"] + e["dur"], short(e["name"])))
    # Last segment: after the last gap > seg_gap on the union of devices.
    allv = sorted(x for v in gpu.values() for x in v)
    cut = allv[0][0]
    end = None
    for s, t, _ in allv:
        if end is not None and s - end > a.seg_gap_s * 1e6:
            cut = s
        end = t if end is None else max(end, t)
    print(f"segment starts at {cut:.0f} us; devices {len(gpu)}")
    names = sorted(gpu, key=lambda p: pname.get(p, ""))
    for p in names:
        v = sorted(x for x in gpu[p] if x[0] >= cut)
        if not v:
            continue
        span = v[-1][1] - v[0][0]
        busy = union([(s, t) for s, t, _ in v])
        print(f"\nDEVICE {pname.get(p, p)} kernels={len(v)} span_ms={span/1e3:.2f} busy_ms={busy/1e3:.2f}")
        # Steps at idle gaps.
        steps = []
        cur = [v[0]]
        for prev, x in zip(v, v[1:]):
            if x[0] - max(y[1] for y in cur[-4:]) > a.gap_ms * 1e3:
                steps.append(cur)
                cur = []
            cur.append(x)
        steps.append(cur)
        lens = [(s[-1][1] - s[0][0]) / 1e3 for s in steps]
        cnts = [len(s) for s in steps]
        print("steps(ms,kernels): " + " ".join(f"{l:.1f}/{c}" for l, c in zip(lens, cnts)))
        gaps = [(steps[i + 1][0][0] - steps[i][-1][1]) / 1e3 for i in range(len(steps) - 1)]
        if gaps:
            print("idle gaps ms: " + " ".join(f"{g:.2f}" for g in gaps))
        # Decode steps: the modal kernel count.
        mode = statistics.mode(cnts)
        dec = [s for s in steps if len(s) == mode]
        print(f"decode-like steps: {len(dec)} with {mode} kernels; span p50 "
              f"{statistics.median([(s[-1][1]-s[0][0])/1e3 for s in dec]):.2f} ms; busy p50 "
              f"{statistics.median([union([(x[0],x[1]) for x in s])/1e3 for s in dec]):.2f} ms")
        agg = collections.defaultdict(lambda: [0, 0.0])
        for s in dec:
            for x in s:
                agg[x[2]][0] += 1
                agg[x[2]][1] += x[1] - x[0]
        n = len(dec)
        tot = sum(v2[1] for v2 in agg.values()) / n
        print(f"per-step kernel sum {tot/1e3:.2f} ms")
        for k, (c, du) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
            print(f"  {du/n/1e3:8.3f} ms  n/step={c/n:6.1f}  mean_us={du/c:8.2f}  {k}")
        if a.dump_step >= 0 and a.dump_step < len(dec):
            s = dec[a.dump_step]
            t0 = s[0][0]
            for x in s:
                print(f"    +{(x[0]-t0)/1e3:8.3f} {(x[1]-x[0]):8.1f}us {x[2][:110]}")


if __name__ == "__main__":
    main()
