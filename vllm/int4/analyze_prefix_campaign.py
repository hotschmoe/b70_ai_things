#!/usr/bin/env python3
"""Audit actual prefix hits and latency; correctness flags alone are insufficient."""
import argparse
import json
from pathlib import Path
import re


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def cached(response):
    return response['usage']['prompt_tokens_details']['cached_tokens']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    r = args.root
    reuse = rows(r / '03-reuse/responses.jsonl')
    assert len(reuse) == 7 and [row['task'] for row in reuse] == [0, 0, 1, 2, 3, 0, 0]
    warm = []
    for index in (1, 6):
        row = reuse[index]
        fraction = cached(row) / row['usage']['prompt_tokens']
        ratio = row['ttft_s'] / reuse[0]['ttft_s']
        warm.append({'index': index, 'cached_tokens': cached(row),
                     'prompt_tokens': row['usage']['prompt_tokens'],
                     'cache_fraction': fraction, 'ttft_s': row['ttft_s'],
                     'ttft_vs_initial_cold': ratio,
                     'passed': fraction >= .9 and ratio <= .25 and row['passed']})
    groups = {}
    for name in ('04-tools', '09-evict-tools'):
        data = json.loads((r / name / 'results.json').read_text())
        followups = [row for session in data for row in session['rows']
                     if row['phase'] != 'tool' or row['turn'] != 0]
        fractions = [cached(row['response']) / row['response']['usage']['prompt_tokens']
                     for row in followups if row.get('response')]
        groups[name] = {'sessions_passed': all(s['passed'] for s in data),
                        'missing_responses': sum(not row.get('response') for row in followups),
                        'followup_cached_fractions': fractions,
                        'followups_with_cache': sum(f > 0 for f in fractions)}
    guides = rows(r / '02-guides/responses.jsonl')
    base_guides = rows(args.baseline / '03-decode/responses.jsonl')
    pool = int(re.search(r'GPU KV cache size: ([\d,]+) tokens',
                        (r / 'server.log').read_text()).group(1).replace(',', ''))
    report = {'root': str(r), 'baseline': str(args.baseline), 'gpu_pool_tokens': pool,
              'cold_ttft_s': reuse[0]['ttft_s'], 'warm': warm,
              'revisit_after_four_histories': {'cached_tokens': cached(reuse[5]),
                  'ttft_s': reuse[5]['ttft_s'], 'correct': reuse[5]['passed'],
                  'distinct_prompt_tokens': sum(reuse[i]['usage']['prompt_tokens']
                                                for i in (0, 2, 3, 4))},
              'tools': groups,
              'guides_exact_against_prefix_off': [x['text_sha256'] for x in guides]
                  == [x['text_sha256'] for x in base_guides],
              'workloads_passed': (r / 'WORKLOADS_PASSED').exists(),
              'health_passed': (r / 'exit.rc').read_text().strip() == '0',
              'promotion_qualified': False,
              'remaining_review': 'Paired coding/raw output review and fresh serving lifecycle.'}
    report['reuse_gate_passed'] = all(w['passed'] for w in warm) and all(x['passed'] for x in reuse)
    report['growing_history_reuse_passed'] = (groups['04-tools']['sessions_passed']
        and all(f >= .8 for f in groups['04-tools']['followup_cached_fractions']))
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
