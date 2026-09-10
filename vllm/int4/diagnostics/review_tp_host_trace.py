#!/usr/bin/env python3
"""Check rank-matched startup boundaries; never infer device completion."""
import argparse
import collections
import json
from pathlib import Path


def review(files):
    issues = []
    scopes = {}
    calls = {}
    profiles = collections.defaultdict(list)
    identities = []
    detailed = collections.defaultdict(collections.Counter)
    for path in files:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            event = row['event']
            pid = row['pid']
            if event in ('installed', 'route_source'):
                identities.append(row)
            if event == 'truncated' or event.endswith('_exception'):
                issues.append(event)
            if event.startswith('dummy_'):
                key = (pid, row['id'])
                if event == 'dummy_enter':
                    if key in scopes:
                        issues.append('duplicate scope entry')
                    scopes[key] = row
                elif event == 'dummy_return':
                    if key not in scopes:
                        issues.append('scope return without entry')
                    else:
                        del scopes[key]
                    for name, count in row['counts'].items():
                        if name.endswith('_enter') and count != row['counts'].get(name[:-6] + '_return', 0):
                            issues.append('unbalanced scope collective counts')
                    if row['phase'] == 'profile':
                        profiles[row['rank']].append(row)
            if event in ('collective_enter', 'collective_return'):
                key = (pid, row['scope'], row['call'])
                detailed[(pid, row['scope'])][row['op'] + ('_enter' if event == 'collective_enter' else '_return')] += 1
                if row['communicator'].get('world_size') != 2 or row['communicator'].get('rank_in_group') != row['rank']:
                    issues.append('invalid TP communicator rank/world size')
                if event == 'collective_enter':
                    if key in calls:
                        issues.append('duplicate collective entry')
                    calls[key] = row
                else:
                    previous = calls.pop(key, None)
                    if previous is None:
                        issues.append('collective return without entry')
                    elif any(previous[k] != row[k] for k in ('rank', 'op', 'communicator')):
                        issues.append('collective identity mismatch')
    if scopes or calls:
        issues.append('unreturned host boundaries')
    if set(profiles) != {0, 1}:
        issues.append('missing TP2 profile ranks')
    for rank_profiles in profiles.values():
        for row in rank_profiles:
            expected = {k: v for k, v in row['counts'].items() if k.endswith(('_enter', '_return'))}
            if dict(detailed[(row['pid'], row['id'])]) != expected:
                issues.append('profile detailed boundaries differ from aggregate counts')
    matched = []
    left, right = profiles[0], profiles[1]
    if len(left) != len(right):
        issues.append('profile count differs between ranks')
    for first, second in zip(left, right):
        # Tensor device strings legitimately differ by local device ordinal.
        def normalized_shapes(row):
            result = {}
            for encoded, count in row['shape_counts'].items():
                value = json.loads(encoded)
                if isinstance(value.get('input'), dict):
                    value['input'].pop('device', None)
                result[json.dumps(value, sort_keys=True)] = count
            return result
        equal = (first['arguments'] == second['arguments'] and
                 first['counts'] == second['counts'] and
                 normalized_shapes(first) == normalized_shapes(second))
        if not equal:
            issues.append('profile arguments/counts/shapes differ between ranks')
        if not first['counts'].get('all_reduce_enter'):
            issues.append('profile has no observed all-reduce')
        matched.append({'matched': equal, 'arguments': first['arguments'],
                        'ranks': [{'rank': row['rank'], 'counts': row['counts'],
                                   'shape_counts': row['shape_counts']} for row in (first, second)]})
    return {'passed': not issues, 'issues': issues, 'profiles': matched,
            'source_identities': identities,
            'scope': 'Python host boundaries only; not device completion or replay operation counts'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = review(sorted(args.directory.glob('host-*.jsonl')))
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps({'passed': result['passed'], 'issues': result['issues']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
