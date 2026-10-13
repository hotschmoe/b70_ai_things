#!/usr/bin/env python3
"""Merge tasks/<id>/task.json (+ INSTRUCTION.md as 'instruction') into ../tasks_longrun.json."""
import glob, json, os
H = os.path.dirname(os.path.abspath(__file__))
out = []
for tj in sorted(glob.glob(f"{H}/tasks/*/task.json")):
    t = json.load(open(tj))
    t["instruction"] = open(os.path.join(os.path.dirname(tj), "INSTRUCTION.md")).read().strip()
    out.append(t)
json.dump({"tier": "longrun", "tasks": out}, open(f"{H}/../tasks_longrun.json", "w"), indent=1)
open(f"{H}/../tasks_longrun.json", "a").write("\n")
print("assembled", [t["id"] for t in out])
