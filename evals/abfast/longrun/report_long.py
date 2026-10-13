#!/usr/bin/env python3
"""Longrun report: per-arm mean milestone fraction, paired per-task table, progress curves, hygiene metrics.

  report.py RUN_A [RUN_B]      (dispatches here when meta tier == longrun)
"""
import json, os, statistics as st, sys


def load(d):
    meta = json.load(open(f"{d}/meta.json")); res = {}; curves = {}
    for t in sorted(os.listdir(f"{d}/tasks")):
        p = f"{d}/tasks/{t}/result.json"
        if os.path.exists(p):
            res[t] = json.load(open(p))
            c = f"{d}/tasks/{t}/curve.json"
            if os.path.exists(c):
                curves[t] = json.load(open(c))
    return meta, res, curves


def frac(r):
    return r.get("final_fraction") if r.get("final_fraction") is not None else 0.0


def at(curve, t):
    v = 0.0
    for p in curve:
        if p["t_s"] <= t and p.get("fraction") is not None:
            v = p["fraction"]
    return v


def summary(label, meta, res, curves):
    rs = list(res.values())
    ok = [r for r in rs if r["end_reason"] not in ("harness_error",)]
    fr = [frac(r) for r in rs]
    tf = [r.get("tests_fraction") or 0.0 for r in rs]
    ends = {}
    for r in rs:
        ends[r["end_reason"]] = ends.get(r["end_reason"], 0) + 1
    print(f"== {label}: model={meta['model']} budget={meta['timeout_s']}s tools={set(meta.get('tools', {}).values())} idle_rounds={meta.get('idle_rounds')}")
    print(f"   tasks={len(rs)}  mean milestone fraction={st.mean(fr) if fr else 0:.3f}  mean hidden-test fraction={st.mean(tf) if tf else 0:.3f}  end_reasons={ends}")
    tot = lambda k: sum(r.get(k) or 0 for r in rs)
    print(f"   turns={tot('turns')} tool_calls={tot('tool_calls')} tool_errors={tot('tool_errors')} unknown_tool={tot('unknown_tool_calls')} "
          f"bad_args={tot('bad_arg_calls')} compactions={tot('compactions')} (fail {tot('compaction_failures')}, overflow {tot('overflow_compactions')}) "
          f"completion_tokens={tot('completion_tokens')} prompt_tokens={tot('prompt_tokens')}")
    bad = [r["id"] for r in rs if r["end_reason"] in ("endpoint_error", "harness_error")]
    if bad:
        print(f"   NOTE endpoint/harness failures (exclude from paired score): {bad}")
    print(f"   run wall={meta.get('total_wall_s')}s end_health={meta.get('end_health')}")


def main(argv):
    ma, ra, ca = load(argv[0])
    summary(ma["arm"], ma, ra, ca)
    runs = [(ma, ra, ca)]
    if len(argv) > 1:
        mb, rb, cb = load(argv[1])
        summary(mb["arm"], mb, rb, cb)
        runs.append((mb, rb, cb))
    ids = sorted(set.intersection(*[set(r) for _, r, _ in runs]))
    names = [m["arm"] for m, _, _ in runs]
    print(f"\n== per task milestone fraction (and hidden tests passed), {len(ids)} common tasks")
    hdr = f"{'task':16}" + "".join(f" | {n[:14]:>14} {'tests':>9} {'wall_h':>6} {'end':>14} {'stop1_h':>7}" for n in names)
    print(hdr)
    for i in ids:
        line = f"{i:16}"
        for _, r, _ in runs:
            x = r[i]
            line += f" | {frac(x):>14.3f} {str(x.get('hidden_passed'))+'/'+str(x.get('hidden_total')):>9} " \
                    f"{(x.get('agent_wall_s') or 0)/3600:>6.2f} {x['end_reason']:>14} {((x.get('first_stop_s') or 0)/3600):>7.2f}"
        print(line)
    if len(runs) == 2:
        d = [frac(runs[0][1][i]) - frac(runs[1][1][i]) for i in ids]
        w = sum(1 for x in d if x > 1e-9); l = sum(1 for x in d if x < -1e-9)
        print(f"paired mean difference ({names[0]} - {names[1]}) = {st.mean(d):+.3f}; {names[0]} better on {w}, {names[1]} better on {l}, equal {len(d)-w-l}")
    print("\n== hygiene per task (turns, tool calls, tool errors, unknown/bad-arg calls, compactions, longest identical-call run, rounds, last file change h)")
    for i in ids:
        for (m, r, _) in runs:
            x = r[i]
            print(f"{i:16} {m['arm'][:12]:12} turns={x.get('turns')} calls={x.get('tool_calls')} err={x.get('tool_errors')} "
                  f"unk/bad={x.get('unknown_tool_calls')}/{x.get('bad_arg_calls')} compact={x.get('compactions')} "
                  f"repeat={x.get('max_identical_tool_call_run')} rounds={len(x.get('rounds') or [])} last_change={((x.get('last_change_s') or 0)/3600):.2f}h "
                  f"out_tok={x.get('completion_tokens')}")
    horizon = max([m["timeout_s"] for m, _, _ in runs])
    steps = [t * 3600 for t in range(1, int(horizon // 3600) + 1)] or [horizon]
    if any(c for _, _, c in runs):
        print("\n== milestone fraction vs time (mean over tasks; snapshots scored after the run; carry-forward between snapshots)")
        print(f"{'hour':>6}" + "".join(f" {n[:14]:>14}" for n in names))
        for t in [1800] + steps:
            row = f"{t/3600:>6.1f}"
            for m, r, c in runs:
                v = [at(c[i], t) for i in ids if i in c]
                row += f" {st.mean(v) if v else float('nan'):>14.3f}"
            print(row)
        print("\n== per task curve (fraction at 0.5h,1h,2h,3h,4h,...)")
        for i in ids:
            for m, r, c in runs:
                if i in c:
                    pts = " ".join(f"{at(c[i], t):.2f}" for t in [1800] + steps)
                    print(f"{i:16} {m['arm'][:12]:12} {pts}")


if __name__ == "__main__":
    main(sys.argv[1:])
