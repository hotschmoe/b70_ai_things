#!/usr/bin/env python3
"""Summarize completed matched Stage C arms; preserve per-stream latency."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/mnt/vm_8tb/b70/results/conc-0216'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('arms', nargs='+')
    args = ap.parse_args()
    rows = []
    for arm in args.arms:
        c1 = args.root / ('c-' + arm + '-c1')
        c2 = args.root / ('c-' + arm + '-c2')
        c4 = args.root / ('c-' + arm + '-c4')
        runs1 = [json.loads(p.read_text()) for p in sorted(c1.glob('coding-*.json'))]
        runs2 = json.loads((c2 / 'summary.json').read_text())['windows']
        runs4 = [json.loads(p.read_text()) for p in sorted(c4.glob('measured-*.json'))]
        assert len(runs1) == 2 and len(runs2) == 2 and len(runs4) == 2
        for repeat, run in enumerate(runs1):
            rows.append(dict(arm=arm, scenario='C1', repeat=repeat, stream=0,
                             decode_tps=run['client_decode_tps'], ttft_s=run['ttft_s'],
                             output_sha256=run['output_sha256'],
                             completion_tokens=run['usage']['completion_tokens']))
        for repeat, run in enumerate(runs2):
            for i, tps in enumerate(run['client_decode_tps']):
                gap = run['content_event_gap_ms'][i]
                rows.append(dict(arm=arm, scenario='C2', repeat=repeat, stream=i,
                                 decode_tps=tps, aggregate_tps=run['aggregate_output_tps'],
                                 ttft_s=run['ttft_s'][i], event_gap_p50_ms=gap['p50'],
                                 event_gap_p95_ms=gap['p95'], output_sha256=run['hashes'][i]))
        for repeat, run in enumerate(runs4):
            for i, stream in enumerate(run['streams']):
                ts = [e['t'] for e in stream['events']]
                gaps = [1000 * (b - a) for a, b in zip(ts, ts[1:])]
                rows.append(dict(arm=arm, scenario='C4', repeat=repeat, stream=i,
                                 decode_tps=stream['decode_tps'], aggregate_tps=run['tg_aggregate_decode_window_tps'],
                                 ttft_s=stream['ttft_s'], event_gap_p50_ms=percentile(gaps, .5),
                                 event_gap_p95_ms=percentile(gaps, .95),
                                 completion_tokens=stream['usage']['completion_tokens'],
                                 output_sha256=hashlib.sha256(stream['text'].encode()).hexdigest()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['arm', 'scenario', 'repeat', 'stream', 'decode_tps', 'aggregate_tps', 'ttft_s',
              'event_gap_p50_ms', 'event_gap_p95_ms', 'completion_tokens', 'output_sha256']
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for arm in args.arms:
        for scenario in ['C1', 'C2', 'C4']:
            group = [r for r in rows if r['arm'] == arm and r['scenario'] == scenario]
            agg = [r['aggregate_tps'] for r in group if r.get('stream') == 0 and 'aggregate_tps' in r]
            print(json.dumps(dict(arm=arm, scenario=scenario,
                median_stream_tps=statistics.median(r['decode_tps'] for r in group),
                aggregate_runs=agg, ttft_p50=percentile([r['ttft_s'] for r in group], .5),
                ttft_p95=percentile([r['ttft_s'] for r in group], .95),
                median_event_gap_p50_ms=statistics.median([r['event_gap_p50_ms'] for r in group if r.get('event_gap_p50_ms') is not None]) if scenario != 'C1' else None,
                median_event_gap_p95_ms=statistics.median([r['event_gap_p95_ms'] for r in group if r.get('event_gap_p95_ms') is not None]) if scenario != 'C1' else None)))


if __name__ == '__main__':
    main()
