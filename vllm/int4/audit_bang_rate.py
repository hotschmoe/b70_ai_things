#!/usr/bin/env python3
"""Count observed raw bang requests by workload without claiming a long-run rate."""
import argparse
import json
from pathlib import Path
import re


def bang(value):
    if isinstance(value, str):
        return '!' * 32 in re.sub(r'\\u0021', '!', value, flags=re.I)
    if isinstance(value, dict):
        return any(bang(v) for v in value.values())
    if isinstance(value, list):
        return any(bang(v) for v in value)
    return False


def audit(root):
    groups = {}
    incidents = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        counts = []
        errors = 0
        recovered = 0
        if (directory / 'results.json').exists():
            sessions = json.loads((directory / 'results.json').read_text())
            if not isinstance(sessions, list) or not all('rows' in s for s in sessions):
                continue
            for session in sessions:
                attempts = session.get('attempts')
                if attempts is None:
                    attempts = [{'phase': r['phase'], 'turn': r['turn'],
                                 'bang': bang(r.get('response')), 'error': r.get('error')}
                                for r in session['rows']]
                for attempt in attempts:
                    same = [r for r in session['rows']
                            if (r['phase'], r['turn']) == (attempt['phase'], attempt['turn'])]
                    last = [a for a in attempts if (a['phase'], a['turn']) ==
                            (attempt['phase'], attempt['turn'])][-1]
                    flagged = attempt['bang'] or (attempt is last and any(bang(r.get('response')) for r in same))
                    counts.append(flagged)
                    errors += bool(attempt.get('error'))
                    if flagged:
                        incidents.append({'workload': directory.name, 'session': session['session'],
                                          **attempt, 'observed_bang': True})
                for r in session['rows']:
                    if r['passed'] and any(a['bang'] and (a['phase'], a['turn']) == (r['phase'], r['turn']) for a in attempts):
                        recovered += 1
        elif (directory / 'responses.jsonl').exists():
            for line in (directory / 'responses.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row.get('client_cancelled'):
                    continue
                counts.append(bang([row.get('text'), row.get('reasoning')]))
                errors += bool(row.get('error'))
        elif directory.name == '10-code' and (directory / 'summary.json').exists():
            summary = json.loads((directory / 'summary.json').read_text())
            for line in Path(summary['raw_samples']).read_text().splitlines():
                row = json.loads(line)
                counts.append(bang(row.get('solution', row.get('completion', ''))))
        if counts:
            groups[directory.name] = {'requests': len(counts), 'bang_requests': sum(counts),
                'observed_fraction': sum(counts) / len(counts), 'request_errors': errors,
                'correct_turns_after_bang_retry': recovered}
    total = sum(g['requests'] for g in groups.values())
    failures = sum(g['bang_requests'] for g in groups.values())
    return {'root': str(root), 'groups': groups, 'requests': total, 'bang_requests': failures,
        'observed_fraction': failures / total if total else None, 'incidents': incidents,
        'target_fraction': .05,
        'scope': 'Observed test mix only. Injected Pi failures and intentional cancellations excluded. Not a representative long-run rate or independence assumption; inspect each workload and temporal clusters.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
