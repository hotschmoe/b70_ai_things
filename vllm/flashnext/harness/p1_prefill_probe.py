#!/usr/bin/env python3
"""Matched P1 HTTP prefill probes; the server owns the GPU lease."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from bench_latency import identity, make_prompt, post_json, stream_completion


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--alias', required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=('0', '1'), required=True)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    assert args.samples > 0
    model = 'hotschmoe-dd'
    report = {'arm': args.arm, 'rows': [], 'ok': False}

    def save():
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n')

    try:
        report['identity'] = identity(args.base_url, model, args.alias)
        assert report['identity']['first_ok'] and report['identity']['alias_ok']
        if args.fixtures.exists():
            fixtures = json.loads(args.fixtures.read_text())
        else:
            assert args.arm == '0', 'treatment requires frozen control fixtures'
            fixtures = []
            for target in (8192, 12288, 32768):
                prompt, _ = make_prompt(args.base_url, model, target, f'p1-fixed-{target}')
                prompt += '\nIn summary, the main lesson is'
                tokenized = post_json(args.base_url + '/tokenize',
                                      {'model': model, 'prompt': prompt,
                                       'add_special_tokens': True}, 60)
                fixtures.append({'target': target, 'prompt': prompt,
                                 'prompt_sha256': digest(prompt),
                                 'token_ids': tokenized['tokens']})
            args.fixtures.parent.mkdir(parents=True, exist_ok=True)
            with args.fixtures.open('x') as output:
                output.write(json.dumps(fixtures, indent=2, ensure_ascii=True) + '\n')
        report['fixtures_sha256'] = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
        assert [row['target'] for row in fixtures] == [8192, 12288, 32768]
        save()
        for fixture in fixtures:
            assert digest(fixture['prompt']) == fixture['prompt_sha256']
            tokenized = post_json(args.base_url + '/tokenize',
                                  {'model': model, 'prompt': fixture['prompt'],
                                   'add_special_tokens': True}, 60)
            assert tokenized['tokens'] == fixture['token_ids'], 'fixture tokenization changed'
            row = {'target': fixture['target'], 'prompt_sha256': fixture['prompt_sha256'],
                   'prompt_tokens': len(fixture['token_ids']), 'requests': []}
            report['rows'].append(row)
            for repeat in range(args.samples + 1):
                result = stream_completion(args.base_url, model, fixture['prompt'], 16, 120)
                result['warmup'] = repeat == 0
                row['requests'].append(result)
                save()
                assert result['prompt_tokens'] == row['prompt_tokens'], 'usage/fixture count mismatch'
                assert result['ttft_ms'] is not None and result['n_chunks'] > 0 and result['text'], \
                    'no observed content event; request wall is not TTFT'
                assert isinstance(result['completion_tokens'], int) and result['completion_tokens'] > 0
                print('P1_PREFILL', args.arm, fixture['target'], repeat,
                      'tokens', result['prompt_tokens'], 'content_ttft_ms', result['ttft_ms'],
                      'wall_ms', result['wall_ms'], flush=True)
            measured = row['requests'][1:]
            row['measured_median_ttft_ms'] = statistics.median(r['ttft_ms'] for r in measured)
            row['measured_input_tok_s'] = row['prompt_tokens'] * 1000 / row['measured_median_ttft_ms']
            save()
        # Long chat coherence is separate from completion timing and is not
        # included in the prefill medians. Use the same frozen 32K context.
        report['long_math'] = []
        for a, b in ((17, 19), (23, 29)):
            body = post_json(args.base_url + '/v1/chat/completions',
                             {'model': model, 'temperature': 0, 'max_tokens': 16,
                              'chat_template_kwargs': {'enable_thinking': False},
                              'messages': [{'role': 'user', 'content': fixtures[-1]['prompt'] +
                                            f'\nReturn only the integer: {a} * {b}.'}]}, 120)
            text = body['choices'][0]['message']['content'].strip()
            report['long_math'].append({'expected': str(a * b), 'text': text,
                                        'passed': text == str(a * b), 'usage': body.get('usage')})
            save()
        assert all(row['passed'] for row in report['long_math']), 'long coherence failed'
        report['ok'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()
    print('P1_PREFILL_DONE', args.arm, report['ok'], flush=True)


if __name__ == '__main__':
    main()
