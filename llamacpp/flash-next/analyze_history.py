#!/usr/bin/env python3
"""Offline comparison of native history captures; no network or GPU access."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def digest(value):
    raw = value.encode('utf-8') if isinstance(value, str) else json.dumps(value, sort_keys=True).encode('ascii')
    return hashlib.sha256(raw).hexdigest()


def load_case(root, name):
    request = json.loads((root / (name + '.request.json')).read_text())['payload']
    events = [json.loads(line)['payload'] for line in (root / (name + '.events.jsonl')).read_text().splitlines()]
    progress = [e['prompt_progress'] for e in events if 'prompt_progress' in e]
    steps = []
    for event in events:
        if event.get('stop') is not False or 'prompt_progress' in event:
            continue
        tokens = event.get('tokens', [])
        probs = event.get('completion_probabilities', [])
        if len(tokens) != 1 or len(probs) != 1 or probs[0]['id'] != tokens[0]:
            raise ValueError(name + ': cannot align one token and probability record per event')
        steps.append({'id': tokens[0], 'token': probs[0]['token'], 'logprob': probs[0].get('logprob'),
                      'top': probs[0].get('top_logprobs', []), 'timings': event.get('timings')})
    finals = [e for e in events if e.get('stop') is True]
    if len(finals) != 1:
        raise ValueError(name + ': exactly one terminal event required')
    return {'name': name, 'prompt': request['prompt'], 'prompt_sha256': digest(request['prompt']),
            'request_except_prompt': {k: v for k, v in request.items() if k != 'prompt'},
            'progress': progress, 'progress_processed': [x['processed'] for x in progress],
            'tokens': [s['id'] for s in steps], 'steps': steps,
            'final_timings': finals[0].get('timings'), 'final_prompt_equal': finals[0].get('prompt') == request['prompt'],
            'input_token_ids': request['prompt'] if isinstance(request['prompt'], list) else None}


def compare(a, b):
    same_prompt = a['prompt'] == b['prompt']
    common = 0
    for x, y in zip(a['tokens'], b['tokens']):
        if x != y:
            break
        common += 1
    rows = []
    if same_prompt:
        # Include logits for the first divergent output token: its input prefix
        # is still identical. Later autoregressive comparisons are not matched.
        count = min(len(a['steps']), len(b['steps']), common + 1)
        for i in range(count):
            x, y = a['steps'][i], b['steps'][i]
            xt, yt = {p['id']: p for p in x['top']}, {p['id']: p for p in y['top']}
            shared = sorted(xt.keys() & yt.keys())
            deltas = [{'id': t, 'token': xt[t]['token'], 'a_logprob': xt[t]['logprob'],
                       'b_logprob': yt[t]['logprob'], 'b_minus_a': yt[t]['logprob'] - xt[t]['logprob']} for t in shared]
            def margin(top):
                return top[0]['logprob'] - top[1]['logprob'] if len(top) >= 2 else None
            rows.append({'position': i, 'a_token': x['id'], 'b_token': y['id'],
                         'a_selected_logprob': x['logprob'], 'b_selected_logprob': y['logprob'],
                         'a_top_margin': margin(x['top']), 'b_top_margin': margin(y['top']),
                         'a_top': x['top'], 'b_top': y['top'], 'shared_top_deltas': deltas,
                         'max_abs_shared_logprob_delta': max((abs(d['b_minus_a']) for d in deltas), default=None)})
    changed = [r for r in rows if r['a_top'] != r['b_top']]
    return {'a': a['name'], 'b': b['name'], 'prompt_exact_equal': same_prompt,
            'request_parameters_equal': a['request_except_prompt'] == b['request_except_prompt'],
            'progress_processed_equal': a['progress_processed'] == b['progress_processed'],
            'output_token_ids_equal': a['tokens'] == b['tokens'], 'common_output_prefix_tokens': common,
            'first_output_divergence_position': None if a['tokens'] == b['tokens'] else common,
            'first_top_probability_difference_position': changed[0]['position'] if changed else None,
            'matched_prefix_steps': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Existing profile-requests directory')
    parser.add_argument('--output', required=True, type=Path, help='New JSON artifact; refuses overwrite')
    args = parser.parse_args()
    names = ['prose_first', 'prose_immediate_repeat', 'prose_after_json']
    cases = [load_case(args.input, name) for name in names]
    report = {'scope': 'offline_history_diagnostic', 'promoted': False,
              'limitations': ['Top-five probabilities cannot reconstruct full logits or full-distribution distances',
                              'Prompt progress is not proof of every internal kernel batch shape',
                              'String prompts do not independently prove input token IDs; no tokenizer endpoint called',
                              'No speed or correctness pass verdict'],
              'cases': cases, 'pairs': [compare(a, b) for a, b in itertools.combinations(cases, 2)]}
    with args.output.open('x', encoding='ascii') as out:
        out.write(json.dumps(report, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps({'artifact': str(args.output), 'pairs': [{k: v for k, v in r.items() if k != 'matched_prefix_steps'} for r in report['pairs']]}, ensure_ascii=True))


if __name__ == '__main__':
    main()
