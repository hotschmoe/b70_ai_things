#!/usr/bin/env python3
"""CPU-only host evidence summary; budget expiry alone is not a stall."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
args = parser.parse_args()
end_path = args.out / "d1-client-end.json"
end = json.loads(end_path.read_text()) if end_path.exists() else None
rows = []
for path in (args.out / "d1-liveness").glob("phases-*.jsonl"):
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if end is None or row["wall_ns"] <= end["wall_ns"]:
            rows.append(row)
phases = [r for r in rows if r["event"] in ("enter", "return", "exception")]
latest = max(phases, key=lambda r:r["wall_ns"]) if phases else None
gap = (end["wall_ns"] - latest["wall_ns"]) / 1e9 if end and latest else None
stall = [r for r in rows if r["event"] in ("stall", "long_active_phase")]
pending = {}
for row in sorted(phases, key=lambda r:r['wall_ns']):
    if type(row.get('token')) is not int:
        continue
    key = (row.get('pid'),row.get('tid'),row['token'])
    if row['event'] == 'enter':
        pending[key] = row
    else:
        pending.pop(key, None)
recent = [r for r in phases if end and r['wall_ns'] >= end['wall_ns'] - 10_000_000_000]
host_progress = dict(phase_events=len(recent), execute_returns=sum(r['event']=='return' and r.get('phase')=='execute' for r in recent),
                     sample_returns=sum(r['event']=='return' and r.get('phase')=='sample' for r in recent))
verdict = "insufficient host evidence"
if end and latest:
    if end["client_rc"] == 0:
        verdict = "finite request sequence completed; no stability qualification"
    elif gap < 8:
        verdict = "client body ended while host phases were advancing; timeout alone is not a liveness failure"
    elif latest["event"] == "enter":
        verdict = "host call has no return for >=8s; inspect native/Python stack, duration alone does not prove deadlock"
    else:
        verdict = "no phase advance for >=8s after last host return; inspect engine/API/scheduler stack"
report = dict(classification=verdict, end=end, last_phase=latest, gap_seconds=gap,
              unreturned_host_calls=list(pending.values()), last10s_host_progress=host_progress,
              watchdog_witnesses=stall, actual_instantiate_calls=[r for r in phases if r.get("phase")=="instantiate"],
              mixed_T21=[r for r in phases if r.get("phase")=="execute" and r["event"]=="enter"
                         and r.get("total")==21 and sorted(r.get("scheduled", []))==[1,20]],
              limits="Duration alone does not prove lost liveness: coarse 32K prefill may normally exceed8s; the120s budget can expire during normal progress or instrumentation overhead. Compare host counters with client/scheduler/generation progress and stack. Host returns and zero error words do not establish device completion or transport progress. Native post-health, engine teardown and source checks remain required.")
(args.out / "d1-localization.json").write_text(json.dumps(report, indent=2, ensure_ascii=True)+"\n")
print(json.dumps(dict(classification=verdict, gap_seconds=gap, watchdog_witnesses=len(stall))))
