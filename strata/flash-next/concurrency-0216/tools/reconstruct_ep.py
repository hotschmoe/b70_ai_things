#!/usr/bin/env python3
"""Bounded C1 per-layer EP wait extraction from a closed UnitTrace JSON trace.

Device clocks have not been independently calibrated to prove cross-card event
causality. Per-layer labels are emitted only if card-0 order is validated
against all 48 waits preceding each full-vocabulary head. Card-1 grouping is
ordinal, not a causal timestamp join.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def pct(values, p):
    values = sorted(values)
    q = (len(values)-1)*p
    i = int(q)
    return values[i] + (values[min(i+1, len(values)-1)]-values[i])*(q-i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    summary = json.loads(a.summary.read_text())
    low, high = summary['start_us'], summary['end_us']
    waits = {0: [], 1: []}
    heads = []
    pidmap = {4294952799: 0, 4294952798: 1}
    with a.trace.open() as f:
        for line in f:
            if 'ep_wait(' not in line and not ('KregQ5K, 4, 4, 128' in line and '62080' in line):
                continue
            try:
                event = json.loads(line.rstrip().rstrip(','))
            except json.JSONDecodeError:
                continue
            if event.get('ph') != 'X' or event.get('cat') != 'gpu_op':
                continue
            card = pidmap.get(event.get('pid'))
            ts = event.get('ts', -1)
            if card is None or not low <= ts <= high:
                continue
            if 'ep_wait(' in event['name']:
                waits[card].append((ts, event['dur']))
            elif card == 0:
                heads.append((ts, event['dur']))
    for rows in waits.values(): rows.sort()
    heads.sort()
    n = len(heads)
    if not n or any(len(waits[c]) != n*48 for c in (0,1)):
        raise RuntimeError('Wait/head cardinality does not support 48-layer windows: '
                           + str({0:len(waits[0]),1:len(waits[1]),'heads':n}))
    for j,(ts,_) in enumerate(heads):
        group = waits[0][j*48:(j+1)*48]
        if group[-1][0] >= ts or (j and group[0][0] <= heads[j-1][0]):
            raise RuntimeError('Card-0 head boundaries do not validate ordinal 48-wait grouping at window '+str(j))
    out = []
    for card in (0,1):
        for layer in range(48):
            durations = [waits[card][j*48+layer][1] for j in range(n)]
            out.append(dict(card=card,layer=layer,windows=n,mean_us=statistics.mean(durations),
                            p50_us=pct(durations,.5),p95_us=pct(durations,.95),
                            total_ms=sum(durations)/1000,
                            direction='wait_for_peer_return' if card==0 else 'wait_for_card0_inbox',
                            grouping='card0_head_validated' if card==0 else 'ordinal_only_not_causally_joined'))
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'ep-wait-by-layer.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    (a.out/'ep-wait-method.json').write_text(json.dumps(dict(windows=n,waits_by_card={str(k):len(v) for k,v in waits.items()},
        note='C1 only; GPU spin service, profiled; card-1 ordinal mapping and cross-card causal timing remain unvalidated.'),indent=2)+'\n')
    print(json.dumps({'windows':n,'card0_total_ms':sum(x[1] for x in waits[0])/1000,
                      'card1_total_ms':sum(x[1] for x in waits[1])/1000}))


if __name__ == '__main__':
    main()
