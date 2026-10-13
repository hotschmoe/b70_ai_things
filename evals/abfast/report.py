#!/usr/bin/env python3
"""abfast report: per-arm summary, and paired table for two arms.

  report.py RUN_DIR_A [RUN_DIR_B]
"""
import json, os, statistics as st, sys


def load(d):
    meta = json.load(open(f"{d}/meta.json"))
    res = {}
    td = f"{d}/tasks"
    for t in sorted(os.listdir(td)):
        p = f"{td}/{t}/result.json"
        if os.path.exists(p):
            res[t] = json.load(open(p))
    return meta, res


def fmt(x):
    return "-" if x is None else f"{x:.0f}"


def summary(label, meta, res):
    rs = list(res.values())
    n = len(rs)
    passed = [r for r in rs if r["pass"]]
    walls = [r["agent_wall_s"] for r in rs if r.get("agent_wall_s") is not None]
    pw = [r["agent_wall_s"] for r in passed if r.get("agent_wall_s") is not None]
    ends = {}
    for r in rs:
        ends[r["end_reason"]] = ends.get(r["end_reason"], 0) + 1
    tot = lambda k: sum(r.get(k) or 0 for r in rs)
    print(f"== {label}: model={meta['model']} thinking={meta['thinking']} parallel={meta['parallel']} timeout={meta['timeout_s']}s")
    print(f"   tasks={n} passed={len(passed)} ({100*len(passed)/max(n,1):.0f}%)  end_reasons={ends}")
    print(f"   agent wall all:    median={fmt(st.median(walls) if walls else None)}s mean={fmt(st.mean(walls) if walls else None)}s")
    print(f"   agent wall passed: median={fmt(st.median(pw) if pw else None)}s mean={fmt(st.mean(pw) if pw else None)}s")
    print(f"   turns={tot('turns')} tool_calls={tot('tool_calls')} prompt_tokens={tot('prompt_tokens')} "
          f"(cached {tot('cache_read_tokens')}) completion_tokens={tot('completion_tokens')}")
    print(f"   run wall={meta.get('total_wall_s')}s end_health={meta.get('end_health')}")
    bad = [r["id"] for r in rs if r["end_reason"] in ("endpoint_error", "harness_error")]
    if bad:
        print(f"   NOTE endpoint/harness failures (exclude from paired score): {bad}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ma, ra = load(sys.argv[1])
    summary(ma["arm"], ma, ra)
    if len(sys.argv) < 3:
        return
    mb, rb = load(sys.argv[2])
    summary(mb["arm"], mb, rb)
    ids = sorted(set(ra) & set(rb))
    print(f"\n== paired ({ma['arm']} vs {mb['arm']}), {len(ids)} common tasks")
    print(f"{'task':36} {'A':>5} {'t_A':>6} {'B':>5} {'t_B':>6}  result")
    w = l = t = 0
    for i in ids:
        a, b = ra[i], rb[i]
        pa, pb = a["pass"], b["pass"]
        if pa and not pb: r = "A wins"; w += 1
        elif pb and not pa: r = "B wins"; l += 1
        else: r = "tie"; t += 1
        f = lambda x: f"{x.get('hidden_passed')}/{x.get('hidden_total')}" if x.get("hidden_total") else "-"
        print(f"{i:36} {'PASS' if pa else 'fail':>5} {fmt(a.get('agent_wall_s')):>6} {'PASS' if pb else 'fail':>5} {fmt(b.get('agent_wall_s')):>6}  {r}"
              f"  [{a['end_reason']}|{b['end_reason']}]")
    print(f"A wins={w} B wins={l} ties={t}")
    both = [i for i in ids if ra[i]["pass"] and rb[i]["pass"]]
    if both:
        ta = [ra[i]["agent_wall_s"] for i in both]; tb = [rb[i]["agent_wall_s"] for i in both]
        print(f"both passed ({len(both)}): median wall A={fmt(st.median(ta))}s B={fmt(st.median(tb))}s")


if __name__ == "__main__":
    main()
