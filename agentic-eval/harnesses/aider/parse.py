#!/usr/bin/env python3
"""parse.py -- aider polyglot benchmark native output -> parsed.json (HARNESS_CONTRACT schema).

Reads the per-exercise `.aider.results.json` files the benchmark writes under the run dir
(<run>/<lang>/exercises/practice/<exercise>/.aider.results.json) and emits to stdout:

  {
    "score": <pass_rate_2 as 0..1>,
    "score_name": "pass_rate_2",
    "n_tasks": <int>,
    "per_task": [{"task_id": "<lang>/<exercise>", "passed": <bool within 2 tries>}, ...],
    "extra": {"pass_rate_1", "percent_cases_well_formed", "syntax_errors", "timeouts", ...}
  }

task_id is "<lang>/<exercise>" (NOT the bare exercise name): 58 of the 100 exercise names recur
across languages, so the bare leaf would collide. The lang-qualified relative path is unique and
stable across configs, so lib/stats.py can pair tasks.

aider per-exercise results fields used:
  tests_outcomes        : list[bool], one per try; LAST element True => passed within tries.
                          pass_rate_1 = first try passed; pass_rate_2 = passed within 2 tries.
  num_malformed_responses: >0 => this case was NOT well-formed (diff-format breakage).
  syntax_errors          : count of SyntaxError lines seen across tries.
  test_timeouts          : count of timed-out test runs.

Usage:
  parse.py <native_run_dir> [--selected selected.txt]
  --selected : optional file of "<lang>/exercises/practice/<exercise>" paths that were REQUESTED.
               Requested-but-missing exercises (e.g. a crash that produced no results json) are
               counted as n_tasks members with passed=false, so n_tasks == the requested subset
               size and a missing result is a fail (conservative, matches the upstream stat which
               divides by completed_tests but we want config-stable denominators).
"""
import argparse
import json
import os
import sys


def find_results(run_dir):
    """Yield (lang, exercise, results_dict) for every .aider.results.json under run_dir."""
    for root, _dirs, files in os.walk(run_dir):
        if ".aider.results.json" not in files:
            continue
        parts = root.split(os.sep)
        # .../<lang>/exercises/practice/<exercise>
        if len(parts) >= 4 and parts[-2] == "practice" and parts[-3] == "exercises":
            lang = parts[-4]
            exercise = parts[-1]
        else:
            # fall back: leaf is exercise, no lang context
            lang = "?"
            exercise = parts[-1]
        fpath = os.path.join(root, ".aider.results.json")
        try:
            with open(fpath) as f:
                res = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"parse.py: skip unreadable {fpath}: {e}", file=sys.stderr)
            continue
        yield lang, exercise, res


def task_id_from_path(rel_path):
    """'<lang>/exercises/practice/<exercise>' -> '<lang>/<exercise>'."""
    parts = rel_path.strip().split("/")
    if len(parts) >= 4 and parts[1] == "exercises" and parts[2] == "practice":
        return f"{parts[0]}/{parts[-1]}"
    return rel_path.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--selected", default=None,
                    help="file of requested <lang>/exercises/practice/<ex> paths (missing => fail)")
    args = ap.parse_args()

    by_task = {}            # task_id -> dict(passed, passed1, malformed, syntax, timeouts)
    for lang, exercise, res in find_results(args.run_dir):
        tid = f"{lang}/{exercise}"
        outcomes = res.get("tests_outcomes", []) or []
        passed2 = bool(outcomes[-1]) if outcomes else False
        passed1 = bool(outcomes[0]) if outcomes else False
        malformed = int(res.get("num_malformed_responses", 0) or 0)
        by_task[tid] = {
            "passed": passed2,
            "passed1": passed1,
            "malformed": malformed > 0,
            "syntax": int(res.get("syntax_errors", 0) or 0),
            "timeouts": int(res.get("test_timeouts", 0) or 0),
            "completed": True,
        }

    # Reconcile against the REQUESTED subset so denominators are config-stable.
    requested_ids = []
    if args.selected and os.path.exists(args.selected):
        with open(args.selected) as f:
            for line in f:
                line = line.strip()
                if line:
                    requested_ids.append(task_id_from_path(line))
    # Union: every requested task is in the table; if it has no results json it's a missing fail.
    all_ids = list(dict.fromkeys(requested_ids + sorted(by_task)))
    for tid in all_ids:
        if tid not in by_task:
            by_task[tid] = {"passed": False, "passed1": False, "malformed": False,
                            "syntax": 0, "timeouts": 0, "completed": False}

    # Order: requested order first (stable pairing), then any extras sorted.
    ordered = requested_ids + [t for t in sorted(by_task) if t not in requested_ids]
    ordered = list(dict.fromkeys(ordered))

    n_tasks = len(ordered)
    if n_tasks == 0:
        out = {"score": None, "score_name": "pass_rate_2", "n_tasks": 0, "per_task": [],
               "extra": {"error": "no exercises found", "run_dir": args.run_dir}}
        print(json.dumps(out, indent=2))
        return 0

    n_pass2 = sum(1 for t in ordered if by_task[t]["passed"])
    n_pass1 = sum(1 for t in ordered if by_task[t]["passed1"])
    n_completed = sum(1 for t in ordered if by_task[t]["completed"])
    # percent_cases_well_formed: of COMPLETED cases, fraction with no malformed responses
    # (upstream divides by completed_tests). Blind to missing cases by design.
    n_malformed_cases = sum(1 for t in ordered if by_task[t]["completed"] and by_task[t]["malformed"])
    well_formed = (1.0 - n_malformed_cases / n_completed) if n_completed else None
    syntax_errors = sum(by_task[t]["syntax"] for t in ordered)
    timeouts = sum(by_task[t]["timeouts"] for t in ordered)

    per_task = [{"task_id": t, "passed": bool(by_task[t]["passed"])} for t in ordered]

    out = {
        "score": round(n_pass2 / n_tasks, 6),
        "score_name": "pass_rate_2",
        "n_tasks": n_tasks,
        "per_task": per_task,
        "extra": {
            "pass_rate_1": round(n_pass1 / n_tasks, 6),
            "percent_cases_well_formed": (round(well_formed, 6) if well_formed is not None else None),
            "syntax_errors": syntax_errors,
            "timeouts": timeouts,
            "n_completed": n_completed,
            "n_missing": n_tasks - n_completed,
            "n_passed_2tries": n_pass2,
            "n_passed_1try": n_pass1,
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
