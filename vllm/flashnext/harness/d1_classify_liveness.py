#!/usr/bin/env python3
"""CPU-only host evidence summary; budget expiry alone is not a stall."""
import argparse
import json
import os
from pathlib import Path
import re
import signal

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--output-name", default="d1-localization.json")
args = parser.parse_args()
if Path(args.output_name).name != args.output_name:
    parser.error("output-name must be a filename")
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
component_path = args.out / 'd1-client-status.env'
components = dict((key, int(value)) for key,value in
                  re.findall(r'^([A-Z_]+_RC)=([0-9]+)$', component_path.read_text(), re.M)) if component_path.exists() else {}
window_path = args.out / 'window.log'
window_text = window_path.read_text(errors='replace') if window_path.exists() else ''
serve_path = args.out / 'serve.log'
serve_text = serve_path.read_text(errors='replace') if serve_path.exists() else ''
fatal_engine = bool(re.search(r'ENGINE_ERROR_RC:[1-9]|EngineDeadError|EngineCore encountered an issue|engine core exited unexpectedly',
                              window_text + '\n' + serve_text, re.I))
exit_witness = None
proc_path = args.out / 'd1-liveness/proc/proc.jsonl'
if proc_path.exists():
    for line in proc_path.read_text().splitlines():
        try:
            sample = json.loads(line)
            for thread in sample['threads']:
                if thread['tid'] != sample['pid'] or thread['stat'].startswith('READ_ERROR'):
                    continue
                raw = thread['stat']
                fields = raw[raw.rfind(')')+2:].split()
                code = int(fields[49])
                if code:
                    exit_witness = dict(wall_ns=sample['wall_ns'], host_pid=sample['pid'],
                                        identity=thread.get('identity'), stat_field52_exit_code=code,
                                        waitstatus_exitcode=os.waitstatus_to_exitcode(code), raw_stat=raw)
                    if os.WIFSIGNALED(code):
                        exit_witness['signal'] = signal.Signals(os.WTERMSIG(code)).name
        except (ValueError, KeyError, IndexError):
            continue
if fatal_engine or (exit_witness and exit_witness.get('signal')):
    verdict = 'engine failure; ' + (exit_witness.get('signal', 'native exit signal unavailable') if exit_witness else 'native exit signal unavailable')
elif end and end['client_rc'] in (124,137):
    verdict = 'client body budget expired; duration alone does not prove lost liveness'
elif any(code for code in components.values()):
    verdict = 'individual client/check failures; shell body0 does not imply request success'
elif end and latest:
    if end["client_rc"] == 0:
        verdict = "finite client body returned0; inspect individual check status, no stability qualification"
    elif gap < 8:
        verdict = "client body ended while host phases were advancing; timeout alone is not a liveness failure"
    elif latest["event"] == "enter":
        verdict = "host call has no return for >=8s; inspect native/Python stack, duration alone does not prove deadlock"
    else:
        verdict = "no phase advance for >=8s after last host return; inspect engine/API/scheduler stack"
report = dict(classification=verdict, end=end, last_phase=latest, gap_seconds=gap,
              individual_client_rcs=components, fatal_engine=fatal_engine, engine_exit_witness=exit_witness,
              unreturned_host_calls=list(pending.values()), last10s_host_progress=host_progress,
              watchdog_witnesses=stall, actual_instantiate_calls=[r for r in phases if r.get("phase")=="instantiate"],
              mixed_T21=[r for r in phases if r.get("phase")=="execute" and r["event"]=="enter"
                         and r.get("total")==21 and sorted(r.get("scheduled", []))==[1,20]],
              limits="Duration alone does not prove lost liveness: coarse 32K prefill may normally exceed8s; the120s budget can expire during normal progress or instrumentation overhead. Compare host counters with client/scheduler/generation progress and stack. Host returns and zero error words do not establish device completion or transport progress. Native post-health, engine teardown and source checks remain required.")
(args.out / args.output_name).write_text(json.dumps(report, indent=2, ensure_ascii=True)+"\n")
print(json.dumps(dict(classification=verdict, gap_seconds=gap, watchdog_witnesses=len(stall))))
