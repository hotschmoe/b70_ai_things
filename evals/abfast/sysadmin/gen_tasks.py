#!/usr/bin/env python3
"""Merge sysadmin/tasks/*/task.json into ../tasks_sysadmin.json (keeps existing "validation" entries)."""
import glob, json, os
H = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(H, "..", "tasks_sysadmin.json")
old = {}
if os.path.exists(out):
    old = {t["id"]: t for t in json.load(open(out))["tasks"]}
tasks = []
for p in sorted(glob.glob(f"{H}/tasks/*/task.json")):
    t = json.load(open(p))
    d = os.path.dirname(p)
    t.update(sys=True, repo="sysadmin", image="abfast-sys:1", cwd="/root",
             has_lazy_check=os.path.exists(f"{d}/lazy.sh"))
    if t["id"] in old and "validation" in old[t["id"]]:
        t["validation"] = old[t["id"]]["validation"]
    tasks.append(t)
order = {"easy": 0, "medium": 1, "hard": 2}
tasks.sort(key=lambda t: (order[t["difficulty"]], t["id"]))
json.dump({"image": "abfast-sys:1", "note": "sysadmin tier: broken-machine repair tasks, Pi runs as root in a bridge-network container; "
           "see evals/abfast/sysadmin/sysrun.py", "tasks": tasks}, open(out, "w"), indent=1)
open(out, "a").write("\n")
print(len(tasks), "tasks")
