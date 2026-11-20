#!/usr/bin/env python3
"""Stream an unclosed unitrace JSON and label all resulting data incomplete."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import statistics

ROOT = Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/concurrency-0216/ledger')
RESULTS = Path('/mnt/vm_8tb/b70/results/conc-0216')
TRACE = RESULTS / 'b-multi-trace/strata-dd-0216.453.json'


def pct(v, p):
    v.sort()
    x = (len(v) - 1) * p
    lo = int(x)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (x - lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', type=Path, default=TRACE)
    ap.add_argument('--markers', type=Path, default=RESULTS / 'b-multi-markers.json')
    ap.add_argument('--tag', default='partial')
    args = ap.parse_args()
    marks = json.loads(args.markers.read_text())
    for m in marks:
        m['trace_us_estimate'] = ((m['real_before_ns'] + m['real_after_ns']) / 2000
                                   + (m['raw_ns'] - m['boot_ns']) / 1000)
    def t(label):
        return next(x['trace_us_estimate'] for x in marks if x['label'] == label)
    bounds = {
        'b2': (max(t('b2_first_0'), t('b2_first_1')), t('b2_end')),
        'b4': (t('b4_begin'), t('b4_end')),
        'b3': (max(t('b3_first_' + str(i)) for i in range(4)), t('b3_end')),
        'b6': (t('b6_begin'), t('b6_first_0')),
    }
    groups = defaultdict(list)
    metadata = {}
    thread_names = {}
    invalid = 0
    lines = 0
    selected = defaultdict(int)
    ends = defaultdict(lambda: [float('inf'), float('-inf')])
    with args.trace.open('rb') as check:
        check.seek(max(0, args.trace.stat().st_size - 32))
        closed = re.sub(rb'\s+', b'', check.read()).endswith(b']}')
    with args.trace.open('rb', buffering=1024*1024) as f:
        for line in f:
            lines += 1
            if not line.startswith(b'{"ph"'):
                continue
            if b'"ph": "M"' in line:
                try:
                    e = json.loads(line.rstrip(b',\r\n'))
                    if e.get('name') == 'process_name': metadata[e['pid']] = e['args']['name']
                    if e.get('name') == 'thread_name': thread_names[(e['pid'], e['tid'])] = e['args']['name']
                except (ValueError, KeyError): invalid += 1
                continue
            if b'"ph": "X"' not in line or (b'"cat": "gpu_op"' not in line and b'"cat": "cpu_op"' not in line):
                continue
            try:
                e = json.loads(line.rstrip(b',\r\n'))
                start, duration = e['ts'], e['dur']
                for name, (lo, hi) in bounds.items():
                    if start >= lo and start + duration <= hi:
                        key = (name, e['cat'], e['pid'], e['tid'], e['name'])
                        groups[key].append(duration)
                        selected[name] += 1
                        ends[name][0] = min(ends[name][0], start)
                        ends[name][1] = max(ends[name][1], start + duration)
                        break
            except (ValueError, KeyError, TypeError): invalid += 1
    for name, (lo, hi) in bounds.items():
        rows = []
        for (scenario, kind, pid, tid, op), durations in groups.items():
            if scenario != name:
                continue
            total = sum(durations)
            rows.append({'kind': kind, 'process': metadata.get(pid, str(pid)),
                         'engine_or_thread': thread_names.get((pid, tid), str(tid)),
                         'operation': op, 'count': len(durations), 'total_ms': total / 1000,
                         'mean_us': total / len(durations), 'median_us': statistics.median(durations),
                         'p95_us': pct(durations, .95)})
        rows.sort(key=lambda r: (r['kind'], -r['total_ms']))
        out = ROOT / name
        out.mkdir(parents=True, exist_ok=True)
        if rows:
            with (out / ('operations-' + args.tag + '.csv')).open('w', newline='', encoding='ascii') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary = {'scenario': name, 'trace': str(args.trace), 'start_us': lo, 'end_us': hi,
                   'selected_events': selected[name], 'first_last_selected_us': ends[name],
                   'source_closed': closed, 'complete': False, 'invalid_event_lines': invalid,
                   'total_lines': lines, 'metadata': metadata,
                   'warning': ('Trace closed normally, but event coverage and causal completeness remain unverified. ' if closed else
                               'Unclosed trace: counts and durations may be missing events. ') +
                              'Whole client interval, overlap not removed; no wall-time or bandwidth inference.',
                   'operations': rows}
        (out / ('summary-' + args.tag + '.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=True) + '\n', encoding='ascii')
    print(json.dumps({'bounds': bounds, 'selected_events': selected, 'invalid_event_lines': invalid,
                      'lines': lines, 'metadata': metadata, 'source_closed': closed}, ensure_ascii=True))


if __name__ == '__main__':
    main()
