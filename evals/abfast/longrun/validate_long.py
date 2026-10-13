#!/usr/bin/env python3
"""No-model validation of longrun tasks: stub (initial agent state), partial reference, full reference; twice.

  validate_long.py [--tasks id,id] [--runs 2]
Expect: stub ~0, partial strictly between, full == 100 percent of milestones, identical results across runs.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_common as C, verify as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=""); ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tasks-file", default=C.TASKS_FILE)
    a = ap.parse_args()
    tasks = C.load_tasks(a.tasks_file)
    if a.tasks:
        tasks = [t for t in tasks if t["id"] in a.tasks.split(",")]
    ok_all = True
    for t in tasks:
        td = f"{C.LDATA}/{t['id']}"
        rows = {}
        for which, d in (("stub", "agent"), ("partial", "ref_partial"), ("full", "ref_sub")):
            if not os.path.isdir(f"{td}/{d}") or not os.listdir(f"{td}/{d}"):
                rows[which] = None; continue
            sigs = []
            for i in range(a.runs):
                r = V.score(t, C.tar_dir(f"{td}/{d}"), tag=f"v{which}{i}")
                sigs.append((r["milestones_passed"], r["milestones_total"], r["tests_passed"], r["tests_total"],
                             tuple((m["id"], m["passed"], m["total"]) for m in r["milestones"])))
            rows[which] = {"ms": f"{sigs[0][0]}/{sigs[0][1]}", "tests": f"{sigs[0][2]}/{sigs[0][3]}",
                           "deterministic": all(s == sigs[0] for s in sigs), "built": r["built"], "check_s": r.get("check_s")}
        s, p, f = rows["stub"], rows["partial"], rows["full"]
        good = bool(f and f["ms"].split("/")[0] == f["ms"].split("/")[1] and f["ms"][0] != "0")
        good = good and all(x and x["deterministic"] for x in (s, p, f) if x)
        if p and f:
            pm = int(p["ms"].split("/")[0]); fm = int(f["ms"].split("/")[0])
            good = good and 0 < pm < fm
        if s:
            good = good and int(s["ms"].split("/")[0]) <= t.get("stub_may_pass_milestones", 1)
        ok_all &= good
        print(f"{t['id']:22} {'OK ' if good else 'BAD'} stub={s and s['ms']} partial={p and p['ms']} full={f and f['ms']} "
              f"tests(full)={f and f['tests']} det={[x and x['deterministic'] for x in (s,p,f)]}", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
