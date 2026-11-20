#!/usr/bin/env python3
"""Extract diagnostic head-to-head timing distributions with one scan per trace."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/mnt/vm_8tb/b70/results/conc-0216')
TASKS = {
    'b1': ('b-trace-v3/strata-dd-0216.453.json', 'summary.json'),
    'b2': ('b-multi2-trace/strata-dd-0216.453.json', 'summary-closed.json'),
    'b3': ('b-multi2-trace/strata-dd-0216.453.json', 'summary-closed.json'),
    'b7': ('b7-trace/strata-dd-0216.453.json', 'summary.json'),
}


def percentile(values, p):
    values = sorted(values)
    q = (len(values)-1)*p
    i = int(q)
    return values[i] + (values[min(i+1,len(values)-1)]-values[i])*(q-i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenarios', nargs='*', choices=sorted(TASKS), default=sorted(TASKS))
    args = parser.parse_args()
    selected = args.scenarios or sorted(TASKS)
    by_trace = defaultdict(list)
    data = {}
    for scenario in selected:
        trace, summary = TASKS[scenario]
        d = json.loads((ROOT/'ledger'/scenario/summary).read_text())
        by_trace[RESULTS/trace].append((scenario,d['start_us'],d['end_us']))
        data[scenario] = []
    for trace, windows in by_trace.items():
        with trace.open() as f:
            for line in f:
                if 'KregQ5K' not in line or ('{31040;' not in line and '{62080;' not in line):
                    continue
                try: event = json.loads(line.rstrip().rstrip(','))
                except json.JSONDecodeError: continue
                if event.get('cat') != 'gpu_op' or event.get('ph') != 'X' or event.get('pid') != 4294952799:
                    continue
                name = event['name']; match = re.search(r'KregQ5K, dpct_kernel_scalar<(\d+)>', name)
                if not match: continue
                for scenario, start, end in windows:
                    if start <= event['ts'] <= end:
                        data[scenario].append(dict(start_us=event['ts'],end_us=event['ts']+event['dur'],
                                                   rows=int(match.group(1)),kernel_us=event['dur']))
    for scenario, events in data.items():
        events.sort(key=lambda e:e['end_us'])
        if len(events) < 2: raise RuntimeError('Too few full-head events for '+scenario)
        for i,event in enumerate(events):
            event['previous_head_gap_ms'] = '' if not i else (event['end_us']-events[i-1]['end_us'])/1000
        gaps = [e['previous_head_gap_ms'] for e in events[1:]]
        summary = dict(scenario=scenario,head_events=len(events),rows=dict(Counter(str(e['rows']) for e in events)),
                       head_to_head_ms_p50=percentile(gaps,.5),head_to_head_ms_p95=percentile(gaps,.95),
                       note='Profiled full-vocabulary head-to-head intervals, including commit, drafts, waits, admission and capture. Not unprofiled per-window latency or a causal breakdown.')
        out=ROOT/'ledger'/scenario
        with (out/'head-intervals.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(events[0]));w.writeheader();w.writerows(events)
        (out/'head-interval-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary))


if __name__ == '__main__': main()
