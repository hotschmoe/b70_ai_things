#!/usr/bin/env python3
"""Extract explicit unitrace copy records; do not infer physical bus traffic."""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / 'ledger'
PATTERN = re.compile(r'zeCommandListAppendMemoryCopy\(([^)]+)\)\[(\d+)\]')


def main():
    for scenario in ('b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'):
        directory = ROOT / scenario
        source = directory / ('summary-closed.json' if scenario in ('b2', 'b3', 'b4', 'b6') else 'summary.json')
        payload = json.loads(source.read_text())
        rows = []
        for item in payload['operations']:
            if item['kind'] not in ('gpu', 'gpu_op'):
                continue
            m = PATTERN.search(item['operation'])
            if not m:
                continue
            n = int(item['count']); size = int(m.group(2))
            rows.append({'trace_direction': m.group(1), 'bytes_per_call': size, 'count': n,
                         'logical_bytes': n * size, 'total_ms': item['total_ms'],
                         'mean_us': item['mean_us'], 'median_us': item['median_us'], 'p95_us': item['p95_us'],
                         'device': item['process'], 'engine': item['engine_or_thread'], 'operation': item['operation']})
        rows.sort(key=lambda r: -r['total_ms'])
        if rows:
            with (directory / 'explicit-transfers.csv').open('w', newline='', encoding='ascii') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(scenario, len(rows), sum(x['logical_bytes'] for x in rows))


if __name__ == '__main__':
    main()
