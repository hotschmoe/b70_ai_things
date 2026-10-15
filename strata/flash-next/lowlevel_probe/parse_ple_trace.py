#!/usr/bin/env python3
"""parse_ple_trace.py -- summarize STRATA_PLE_TRACE=1 stderr lines from one Strata serve log (CPU only, ASCII).

Line format (sycl/src/ngram/ple_reader.cpp PleReader::collect, threaded path only):
  ple trace: <jobs> jobs  issue->submit <us>  submit->last <us>  last->return <us>  collect wait <us> us
All values are steady_clock microsecond DIFFERENCES within one ticket. No absolute timestamp is logged, so these
lines cannot be joined exactly to GPU (unitrace) events; only the log order is available.
Printed only when 0 < jobs < 200: all-cache-hit gathers (jobs 0) and big prefill gathers (jobs >= 200) are absent.
Usage: parse_ple_trace.py LOG [--json OUT] [--window START_LINE:END_LINE]
"""
import json, re, sys

PAT = re.compile(r"ple trace:\s+(\d+) jobs\s+issue->submit\s+(-?[\d.]+)\s+submit->last\s+(-?[\d.]+)\s+"
                 r"last->return\s+(-?[\d.]+)\s+collect wait\s+(-?[\d.]+) us")
FIELDS = ["issue_to_submit_us", "submit_to_last_us", "last_to_return_us", "collect_wait_us"]


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith("-"):
        print(__doc__)
        return 2
    path, out, win = a[0], None, None
    if "--json" in a:
        out = a[a.index("--json") + 1]
    if "--window" in a:
        s, e = a[a.index("--window") + 1].split(":")
        win = (int(s), int(e))
    rows, other = [], {"strata_batch_summary": [], "ple_other": []}
    with open(path, "r", errors="replace") as f:
        for n, line in enumerate(f, 1):
            if win and not (win[0] <= n <= win[1]):
                continue
            m = PAT.search(line)
            if m:
                j = int(m.group(1))
                rows.append({"line": n, "jobs": j, **{k: float(m.group(i + 2)) for i, k in enumerate(FIELDS)}})
            elif "strata batch:" in line and "windows" in line:
                other["strata_batch_summary"].append(line.strip()[:400])
            elif "ple" in line.lower() and ("hit" in line.lower() or "miss" in line.lower()):
                other["ple_other"].append(line.strip()[:300])
    summ = {"log": path, "n_traced_gathers": len(rows),
            "note": "differences only; no absolute anchor; jobs==0 (all cache hits) and jobs>=200 not logged; "
                    "collect_wait is the caller's blocking time (the stall); last_to_return includes caller work "
                    "between completion and collect when the read finished early (slack, not stall)"}
    if rows:
        jobs = [r["jobs"] for r in rows]
        summ["jobs"] = {"sum": sum(jobs), "min": min(jobs), "p50": pct(jobs, 50), "max": max(jobs)}
        for k in FIELDS:
            v = [r[k] for r in rows]
            summ[k] = {"sum": round(sum(v), 1), "p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99), "max": max(v)}
        waits = [r["collect_wait_us"] for r in rows]
        summ["gathers_blocking_gt_10us"] = sum(1 for w in waits if w > 10)
        summ["gathers_blocking_gt_100us"] = sum(1 for w in waits if w > 100)
        summ["anomalies"] = {"negative_values": sum(1 for r in rows for k in FIELDS if r[k] < 0)}
    summ.update(other)
    txt = json.dumps(summ, indent=1)
    print(txt)
    if out:
        with open(out, "w") as f:
            f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
