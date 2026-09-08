#!/usr/bin/env python3
"""Repeated tool-call/history correctness gate, no external tool execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
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
    (args.out / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    assert args.model in [m['id'] for m in json.loads(request(args.base, '/v1/models'))['data']]
    def session(index):
        rows = []
        history = [{'role': 'system', 'content': 'You are testing a warehouse tool. Always use lookup_stock to look up stock. After receiving its result, report only its count as an integer. Background records:\n' + ('The warehouse stores parts and maintains an inventory ledger.\n' * args.records)}]
        for turn in range(4):
            sku = f'part-{index}-{turn}'
            count = 730 + index * 10 + turn
            history.append({'role': 'user', 'content': 'Look up stock for SKU ' + sku + '.'})
            payload = {'model': args.model, 'messages': history, 'temperature': 0, 'seed': 42, 'max_tokens': 512,
                       'tools': [{'type': 'function', 'function': {'name': 'lookup_stock', 'description': 'Get stock count for a SKU', 'parameters': {'type': 'object', 'properties': {'sku': {'type': 'string'}}, 'required': ['sku']}}}],
                       'tool_choice': 'auto', 'chat_template_kwargs': {'enable_thinking': False}, 'cache_salt': 'agent-v1-' + str(index)}
            response = json.loads(request(args.base, '/v1/chat/completions', payload, timeout=args.timeout))
            message = response['choices'][0]['message']
            calls = message.get('tool_calls') or []
            good = len(calls) == 1 and calls[0]['function']['name'] == 'lookup_stock' and json.loads(calls[0]['function']['arguments']) == {'sku': sku}
            rows.append({'phase': 'tool', 'turn': turn, 'passed': good, 'response': response})
            if not good:
                break
            history.append({'role': 'assistant', 'content': message.get('content'), 'tool_calls': calls})
            history.append({'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': json.dumps({'sku': sku, 'count': count})})
            payload['messages'] = history
            payload['tool_choice'] = 'none'
            answer = json.loads(request(args.base, '/v1/chat/completions', payload, timeout=args.timeout))
            msg = answer['choices'][0]['message']
            good = (msg.get('content') or '').strip() == str(count)
            rows.append({'phase': 'answer', 'turn': turn, 'passed': good, 'response': answer})
            history.append({'role': 'assistant', 'content': msg.get('content')})
            if not good:
                break
        return {'session': index, 'rows': rows, 'passed': len(rows) == 8 and all(r['passed'] for r in rows)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(session, range(4)))
    (args.out / 'results.json').write_text(json.dumps(results, ensure_ascii=True, indent=2) + '\n')
    passed = all(r['passed'] for r in results)
    print(json.dumps({'passed': passed, 'checks': sum(len(r['rows']) for r in results)}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
