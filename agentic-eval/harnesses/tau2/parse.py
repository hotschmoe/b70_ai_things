#!/usr/bin/env python3
"""parse.py -- tau2-bench results.json -> agentic-eval parsed.json (stdout).

Usage:
    python parse.py <results-json-or-its-dir>

Reads a tau2 run's monolithic results.json (text runs use "json" format):
    {
      "info": {...},
      "tasks": [{"id": "0", ...}, ...],
      "simulations": [
        {"task_id": "0", "trial": 0, "reward_info": {"reward": 1.0}, ...},
        ...
      ]
    }

Emits the HARNESS_CONTRACT schema:
    {"score": <pass^1 in 0..1>, "score_name": "pass^1", "n_tasks": <int>,
     "per_task": [{"task_id": "<str>", "passed": <bool>}, ...],
     "extra": {"avg_reward": <float>, "pass^2": <float|None>, "num_trials": <int>}}

pass^k semantics (upstream tau2 metrics/agent_metrics.py):
  is_successful(reward) == (1 - 1e-6) <= reward <= (1 + 1e-6)   (i.e. reward == 1)
  per task with n trials and c successes: pass^k = C(c,k)/C(n,k)
  campaign pass^k = mean over tasks of that per-task value.
For the greedy temp=0 single-trial regime (num_trials=1) this reduces to the
fraction of tasks that passed, and per_task.passed is just reward==1 for that
one trial. task_id is the tau2 task id (stable across configs) so lib/stats.py
can pair runs.
"""
import json
import math
import os
import sys


def _load_results(path):
    """Accept either the results.json file or a directory containing it."""
    if os.path.isdir(path):
        cand = os.path.join(path, "results.json")
        if os.path.exists(cand):
            path = cand
        else:
            # tau2 'dir' format keeps results.json next to a simulations/ subdir;
            # but our text runs are monolithic. Fall back to any results.json found.
            for root, _dirs, files in os.walk(path):
                if "results.json" in files:
                    path = os.path.join(root, "results.json")
                    break
    with open(path) as f:
        return json.load(f)


def _is_successful(reward):
    if reward is None:
        return False
    return (1 - 1e-6) <= float(reward) <= (1 + 1e-6)


def _pass_hat_k(n_trials, success_count, k):
    """C(c,k)/C(n,k); upstream tau2 pass^k. Requires n_trials >= k."""
    if n_trials < k:
        return None
    denom = math.comb(n_trials, k)
    if denom == 0:
        return None
    return math.comb(success_count, k) / denom


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: parse.py <results.json|dir>\n")
        sys.exit(2)
    data = _load_results(sys.argv[1])

    sims = data.get("simulations", []) or []
    # Group successes by task_id. reward lives at reward_info.reward.
    by_task = {}  # task_id(str) -> {"n": int, "succ": int, "rewards": [floats]}
    for sim in sims:
        tid = str(sim.get("task_id"))
        ri = sim.get("reward_info") or {}
        reward = ri.get("reward")
        slot = by_task.setdefault(tid, {"n": 0, "succ": 0, "rewards": []})
        slot["n"] += 1
        if reward is not None:
            slot["rewards"].append(float(reward))
        if _is_successful(reward):
            slot["succ"] += 1

    # Stable task ordering: numeric id if possible, else lexical.
    def _key(t):
        try:
            return (0, int(t))
        except (ValueError, TypeError):
            return (1, str(t))

    task_ids = sorted(by_task.keys(), key=_key)
    n_tasks = len(task_ids)

    # Per-task pass^1: success_count / n_trials. For num_trials=1 this is 0/1.
    per_task = []
    p1_vals = []
    p2_vals = []
    all_rewards = []
    min_trials = min((by_task[t]["n"] for t in task_ids), default=0)
    for t in task_ids:
        slot = by_task[t]
        n, c = slot["n"], slot["succ"]
        all_rewards.extend(slot["rewards"])
        p1 = _pass_hat_k(n, c, 1)
        if p1 is not None:
            p1_vals.append(p1)
        # passed: for single-trial this is reward==1; for multi-trial we report
        # "passed all trials" so the paired McNemar test stays a clean 0/1 per task.
        per_task.append({"task_id": t, "passed": bool(c == n and n > 0)})
        if n >= 2:
            p2 = _pass_hat_k(n, c, 2)
            if p2 is not None:
                p2_vals.append(p2)

    score = (sum(p1_vals) / len(p1_vals)) if p1_vals else None
    avg_reward = (sum(all_rewards) / len(all_rewards)) if all_rewards else None
    pass2 = (sum(p2_vals) / len(p2_vals)) if (p2_vals and min_trials >= 2) else None

    out = {
        "score": round(score, 6) if score is not None else None,
        "score_name": "pass^1",
        "n_tasks": n_tasks,
        "per_task": per_task,
        "extra": {
            "avg_reward": round(avg_reward, 6) if avg_reward is not None else None,
            "pass^2": round(pass2, 6) if pass2 is not None else None,
            "num_trials": min_trials,
            "n_simulations": len(sims),
        },
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
