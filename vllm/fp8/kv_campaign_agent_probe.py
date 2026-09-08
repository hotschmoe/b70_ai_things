#!/usr/bin/env python3
"""Repeated tool-call/history correctness gate, no external tool execution."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from kv_campaign_probe import request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18125')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--records', type=int, default=180)
    p.add_argument('--timeout', type=int, default=180)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / 'config.json').write_text(json.dumps(config, default=str, indent=2) + '\n')
    assert args.model in [m['id'] for m in json.loads(request(args.base, '/v1/models'))['data']]
    (args.out / 'metrics-before.txt').write_text(request(args.base, '/metrics'))
    started = time.monotonic()
    def session(index):
        rows = []
        def record(row):
            rows.append(row)
            with (args.out / f'session-{index}.jsonl').open('a') as f:
                f.write(json.dumps(row, ensure_ascii=True) + '\n')
        def invoke(payload, phase, turn):
            began = time.monotonic()
            response = None
            try:
                response = json.loads(request(args.base, '/v1/chat/completions', payload, timeout=args.timeout))
                if not isinstance(response['choices'][0]['message'], dict):
                    raise ValueError('missing response message')
                return response, time.monotonic() - began
            except Exception as exc:
                record({'phase': phase, 'turn': turn, 'passed': False,
                        'elapsed_s': time.monotonic() - began, 'error': ascii(exc),
                        'response': response})
                return None, None
        history = [{'role': 'system', 'content': 'You are testing a warehouse tool. Always use lookup_stock to look up stock. After receiving its result, report only its count as an integer. Background records:\n' + ('The warehouse stores parts and maintains an inventory ledger.\n' * args.records)}]
        for turn in range(4):
            sku = f'part-{index}-{turn}'
            count = 730 + index * 10 + turn
            history.append({'role': 'user', 'content': 'Look up stock for SKU ' + sku + '.'})
            payload = {'model': args.model, 'messages': history, 'temperature': 0, 'seed': 42, 'max_tokens': 512,
                       'tools': [{'type': 'function', 'function': {'name': 'lookup_stock', 'description': 'Get stock count for a SKU', 'parameters': {'type': 'object', 'properties': {'sku': {'type': 'string'}}, 'required': ['sku']}}}],
                       'tool_choice': 'auto', 'chat_template_kwargs': {'enable_thinking': False}, 'cache_salt': 'agent-v1-' + str(index)}
            response, elapsed = invoke(payload, 'tool', turn)
            if response is None:
                break
            message = response['choices'][0]['message']
            calls = message.get('tool_calls') or []
            try:
                good = len(calls) == 1 and bool(calls[0].get('id')) and calls[0]['function']['name'] == 'lookup_stock' and json.loads(calls[0]['function']['arguments']) == {'sku': sku}
            except (KeyError, TypeError, ValueError):
                good = False
            record({'phase': 'tool', 'turn': turn, 'passed': good, 'response': response, 'elapsed_s': elapsed})
            if not good:
                break
            history.append({'role': 'assistant', 'content': message.get('content'), 'tool_calls': calls})
            history.append({'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': json.dumps({'sku': sku, 'count': count})})
            payload['messages'] = history
            payload['tool_choice'] = 'none'
            answer, elapsed = invoke(payload, 'answer', turn)
            if answer is None:
                break
            msg = answer['choices'][0]['message']
            good = isinstance(msg.get('content'), str) and msg['content'].strip() == str(count)
            record({'phase': 'answer', 'turn': turn, 'passed': good, 'response': answer, 'elapsed_s': elapsed})
            history.append({'role': 'assistant', 'content': msg.get('content')})
            if not good:
                break
        return {'session': index, 'rows': rows, 'passed': len(rows) == 8 and all(r['passed'] for r in rows)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(session, range(4)))
    (args.out / 'results.json').write_text(json.dumps(results, ensure_ascii=True, indent=2) + '\n')
    passed = all(r['passed'] for r in results)
    try:
        (args.out / 'metrics-after.txt').write_text(request(args.base, '/metrics'))
    except Exception as exc:
        passed = False
        (args.out / 'metrics-error.txt').write_text(ascii(exc) + '\n')
    summary = {'passed': passed, 'checks': sum(len(r['rows']) for r in results),
               'elapsed_s': time.monotonic() - started}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
