#!/usr/bin/env python3
"""Inventory real collective shapes and completed CPU calls in paired rank traces."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    records = []
    for path in sorted(args.root.rglob('*.pt.trace.json*')):
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as f:
            events = json.load(f)['traceEvents']
        calls = []
        for event in events:
            name = event.get('name', '').lower()
            if (event.get('ph') == 'X' and event.get('cat') == 'cpu_op'
                    and any(word in name for word in ('all_reduce', 'allreduce', 'all_gather', 'allgather'))):
                calls.append(event)
        shapes = Counter((e['name'], json.dumps(e.get('args', {}).get('Input Dims'), sort_keys=True)) for e in calls)
        records.append({'trace': str(path), 'completed_collective_cpu_calls': len(calls),
                        'calls_with_duration': sum('dur' in e for e in calls),
                        'shapes': [{'name': name, 'input_dims': json.loads(shape), 'count': count}
                                   for (name, shape), count in sorted(shapes.items())],
                        'driver_calls': dict(Counter(e['name'] for e in events if e.get('ph') == 'X'
                            and e.get('name') in ['zeCommandQueueExecuteCommandLists', 'zeFenceReset',
                                                 'zeEventHostSynchronize', 'zeFenceHostSynchronize']))})
    workers = [r for r in records if r['completed_collective_cpu_calls']]
    report = {'traces': records, 'paired_collective_traces': len(workers),
              'paired_shapes_and_counts_match': len(workers) == 2 and workers[0]['shapes'] == workers[1]['shapes'],
              'scope': 'Completed CPU collective entry/return events; captured graph internals may be opaque. Require response and post-health evidence separately.'}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
