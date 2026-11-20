#!/usr/bin/env python3
"""Summarize diagnostic router IDs by row count and layer, without timing claims."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def pct(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    groups = defaultdict(list)
    layer_groups = defaultdict(list)
    malformed = 0
    with args.input.open() as f:
        for line in f:
            row = json.loads(line)
            n = row['rows']
            mult = row['multiplicities']
            if not (1 <= n <= 8 and row['entries'] == n * 10 and len(mult) == row['unique']
                    and sum(mult) == row['entries'] and all(1 <= x <= n for x in mult)):
                malformed += 1
                continue
            groups[n].append(row)
            layer_groups[(n, row['layer'])].append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for n, rows in sorted(groups.items()):
        unique = [r['unique'] for r in rows]
        mults = Counter(x for r in rows for x in r['multiplicities'])
        chunks = [sum((x + 3) // 4 for x in r['multiplicities']) for r in rows]
        unique_bytes = [r['unique'] * r['weight_bytes_per_expert'] for r in rows]
        summary.append(dict(rows=n, layer_windows=len(rows), entries_per_window=n*10,
            unique_p50=pct(unique,.5), unique_p95=pct(unique,.95),
            unique_expert_logical_bytes_p50=pct(unique_bytes,.5),
            unique_expert_logical_bytes_p95=pct(unique_bytes,.95),
            weight_groups_of_up_to_4_p50=pct(chunks,.5),
            weight_groups_of_up_to_4_p95=pct(chunks,.95),
            expert_multiplicity_counts=dict(sorted(mults.items()))))
    per_layer = []
    for (n, layer), rows in sorted(layer_groups.items()):
        unique = [r['unique'] for r in rows]
        per_layer.append(dict(rows=n, layer=layer, layer_windows=len(rows),
                              gu_type=rows[0]['gu_type'], down_type=rows[0]['down_type'],
                              weight_bytes_per_expert=rows[0]['weight_bytes_per_expert'],
                              unique_p50=pct(unique,.5), unique_p95=pct(unique,.95),
                              unique_weight_bytes_p50=pct([r['unique'] * r['weight_bytes_per_expert'] for r in rows],.5),
                              unique_weight_bytes_p95=pct([r['unique'] * r['weight_bytes_per_expert'] for r in rows],.95)))
    (args.out/'summary.json').write_text(json.dumps(dict(summary=summary, malformed=malformed,
        note='Profiled route-ID snapshots; group-of-four counts are idealized lower bounds, not measured weight traffic.'),
        indent=2,ensure_ascii=True)+'\n')
    with (args.out/'per-layer.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rows','layer','layer_windows','gu_type','down_type','weight_bytes_per_expert','unique_p50','unique_p95','unique_weight_bytes_p50','unique_weight_bytes_p95'])
        w.writeheader(); w.writerows(per_layer)
    print(json.dumps(dict(summary=summary,malformed=malformed),ensure_ascii=True))
    if malformed or not summary:
        raise RuntimeError('Missing or malformed route audit records')


if __name__ == '__main__':
    main()
