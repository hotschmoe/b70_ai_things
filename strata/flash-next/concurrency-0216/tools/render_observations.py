#!/usr/bin/env python3
"""Render recorded GPU service on each card, not an additive latency budget."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def label(name):
    name = name.replace('strata::kernels::', '').replace('(anonymous namespace)::', '')
    name = name.replace('strata::prefill::', 'prefill::').replace('dpct_kernel_name<', '')
    shape = re.search(r'\[SIMD(?:16|32) \{[^}]+\} \{[^}]+\}\]$', name)
    suffix = shape.group(0) if shape else ''
    head = name.split('(', 1)[0]
    head = re.sub(r'\s+', ' ', head)
    return (head[:100] + ('...' if len(head) > 100 else '') + (' ' + suffix if suffix else '')).replace('|', '/')


def main():
    out = ['# Ranked recorded operation service', '',
           'These are per-card operation totals in scenario-bounded profiled intervals, not proven critical-path',
           'costs. They can overlap other work, include edge windows and profiler overhead, and omit work',
           'not recorded by unitrace. Exact names, shapes, p50/p95 and device/thread attribution are in the',
           'per-scenario CSV/JSON. B5 may omit graphs captured during paused preparation.',
           'Host API duration includes waits and profiler overhead; it is not CPU time or a separate',
           'additive cost, and can overlap device execution.', '']
    for s in ('b1','b2','b3','b4','b5','b6','b7'):
        path = ROOT/'ledger'/s/('summary-closed.json' if s in ('b2','b3','b4','b6') else 'summary.json')
        data = json.loads(path.read_text())
        out += ['## ' + s.upper(), '']
        for card, device in ((0, '0:b:0:0 #0'), (1, '0:44:0:0 #1')):
            rows = [r for r in data['operations'] if r['kind'] in ('gpu', 'gpu_op') and device in r['process']]
            rows.sort(key=lambda r: -r['total_ms'])
            out += ['### Card ' + str(card), '', '| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |',
                    '| ---: | --- | ---: | ---: | ---: |']
            for i, r in enumerate(rows[:10], 1):
                out.append('| %d | %s | %d | %.2f | %.2f / %.2f |' %
                           (i, label(r['operation']), r['count'], r['total_ms'], r['median_us'], r['p95_us']))
            if not rows:
                out.append('| - | No recorded GPU events for this device in the selected interval. | - | - | - |')
            out.append('')
        host = [r for r in data['operations'] if r['kind'] in ('cpu_op', 'host')]
        host.sort(key=lambda r: -r['total_ms'])
        out += ['### Recorded host Level Zero calls', '', '| Rank | API | Calls | Summed duration ms | p50 / p95 us |',
                '| ---: | --- | ---: | ---: | ---: |']
        for i, r in enumerate(host[:10], 1):
            out.append('| %d | %s | %d | %.2f | %.2f / %.2f |' %
                       (i, label(r['operation']), r['count'], r['total_ms'], r['median_us'], r['p95_us']))
        if not host:
            out.append('| - | No host API events in the selected interval. | - | - | - |')
        out.append('')
    (ROOT/'OBSERVED-OPERATIONS.md').write_text('\n'.join(out)+'\n',encoding='ascii')


if __name__ == '__main__':
    main()
