#!/usr/bin/env python3
"""agentic-eval/harnesses/swe/parse.py

Turn mini-swe-agent predictions + the official swebench evaluation report into the
canonical parsed.json (schema in docs/HARNESS_CONTRACT.md). Stdlib only.

    parse.py <run_dir>

<run_dir> must contain:
  preds.json                       mini-swe-agent predictions
                                   {instance_id: {model_name_or_path, instance_id, model_patch}, ...}
  report/<model>.<run_id>.json     swebench grader report (resolved_ids / submitted_ids / error_ids ...)
                                   (we glob report/*.json and take the swebench-shaped one)

Optionally:
  <instance_id>/<instance_id>.traj.json    per-instance mini trajectory (for turn counts / exit status)

Output (stdout):
  {"score": <resolved fraction>, "score_name": "resolved", "n_tasks": N,
   "per_task": [{"task_id": instance_id, "passed": resolved?}, ...],
   "extra": {"n_submitted":, "n_errored":, "n_empty_patch":, "avg_turns":, ...}}

score = resolved / n_tasks, where n_tasks is the number of instances mini SUBMITTED
(attempted) -- i.e. the slice it was asked to run. A task mini submitted but the
grader could not resolve (failed tests, empty patch, harness error) counts as a
fail, which is the correct SWE-bench accounting. task_id == instance_id is stable
across configs because mini runs the same dataset slice in dataset order.
"""
import glob
import json
import os
import sys


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _find_report(run_dir):
    """Locate the swebench grader report json under <run_dir>/report (or run_dir)."""
    cands = sorted(glob.glob(os.path.join(run_dir, "report", "*.json")))
    cands += sorted(glob.glob(os.path.join(run_dir, "*.json")))
    for p in cands:
        if os.path.basename(p) == "preds.json":
            continue
        try:
            d = _load_json(p)
        except Exception:
            continue
        # swebench report has these keys (reporting.make_run_report).
        if isinstance(d, dict) and "resolved_ids" in d and "submitted_ids" in d:
            return p, d
    return None, None


def _avg_turns(run_dir, instance_ids):
    """Mean assistant turns across available per-instance trajectories, or None."""
    turns = []
    for iid in instance_ids:
        traj = os.path.join(run_dir, iid, f"{iid}.traj.json")
        if not os.path.exists(traj):
            continue
        try:
            d = _load_json(traj)
        except Exception:
            continue
        msgs = d.get("messages") or d.get("trajectory") or []
        if isinstance(msgs, list):
            n = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")
            if n:
                turns.append(n)
    if not turns:
        return None
    return round(sum(turns) / len(turns), 2)


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: parse.py <run_dir>\n")
        sys.exit(2)
    run_dir = sys.argv[1]

    preds_path = os.path.join(run_dir, "preds.json")
    preds = _load_json(preds_path) if os.path.exists(preds_path) else {}

    rpath, report = _find_report(run_dir)
    if report is None:
        sys.stderr.write(f"parse.py: no swebench report json found under {run_dir}\n")
        # Degrade gracefully: emit zero-score over whatever mini submitted so the
        # campaign still records a (failed-grade) result rather than crashing.
        report = {"resolved_ids": [], "submitted_ids": list(preds.keys()),
                  "error_ids": [], "empty_patch_ids": []}

    resolved = set(report.get("resolved_ids", []))
    # Submitted = what we asked mini to attempt. Prefer mini's preds (the true slice);
    # fall back to the grader's submitted_ids.
    submitted = list(preds.keys()) or list(report.get("submitted_ids", []))
    submitted = sorted(submitted)
    errored = set(report.get("error_ids", []))
    empty = set(report.get("empty_patch_ids", []))

    per_task = [{"task_id": iid, "passed": iid in resolved} for iid in submitted]
    n_tasks = len(per_task)
    n_resolved = sum(1 for t in per_task if t["passed"])
    score = (n_resolved / n_tasks) if n_tasks else 0.0

    extra = {
        "n_submitted": n_tasks,
        "n_resolved": n_resolved,
        "n_errored": sum(1 for iid in submitted if iid in errored),
        "n_empty_patch": sum(1 for iid in submitted if iid in empty),
        "report_file": os.path.relpath(rpath, run_dir) if rpath else None,
        "dataset_total_instances": report.get("total_instances"),
    }
    avg_turns = _avg_turns(run_dir, submitted)
    if avg_turns is not None:
        extra["avg_turns"] = avg_turns

    out = {
        "score": round(score, 6),
        "score_name": "resolved",
        "n_tasks": n_tasks,
        "per_task": per_task,
        "extra": extra,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
