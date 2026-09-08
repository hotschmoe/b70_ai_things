#!/usr/bin/env python3
"""Repeated tool-call/history correctness gate, no external tool execution."""
import argparse
from datetime import datetime, timezone
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
from kv_campaign_probe import request


class BangLoopError(RuntimeError):
    pass


def contains_bang(value):
    if isinstance(value, str):
        return '!' * 32 in re.sub(r'\\u0021', '!', value, flags=re.I)
    if isinstance(value, dict):
        return any(contains_bang(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_bang(v) for v in value)
    return False


def stream_request(base, payload, timeout, trace, headers=None, bang_limit=None):
    """Retain tool deltas and partial text if a long request times out."""
    body = dict(payload, stream=True, stream_options={'include_usage': True})
    req = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **(headers or {})})
    message = {'role': 'assistant', 'content': None}
    response = {'choices': [{'index': 0, 'message': message, 'finish_reason': None}]}
    calls = {}
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as stream, trace.open('w') as log:
            for line in stream:
                if time.monotonic() - started > timeout:
                    raise TimeoutError('total request deadline exceeded')
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                log.write(json.dumps({'elapsed_s': time.monotonic() - started, 'event': event}) + '\n')
                log.flush()
                for key in ('id', 'model', 'created', 'usage'):
                    if event.get(key) is not None:
                        response[key] = event[key]
                for choice in event.get('choices', []):
                    assert choice['index'] == 0
                    delta = choice.get('delta', {})
                    if delta.get('content'):
                        message['content'] = (message['content'] or '') + delta['content']
                    for key in ('reasoning', 'reasoning_content'):
                        if delta.get(key):
                            message[key] = message.get(key, '') + delta[key]
                    for item in delta.get('tool_calls', []):
                        call = calls.setdefault(item['index'], {'id': '', 'type': 'function',
                                                    'function': {'name': '', 'arguments': ''}})
                        if item.get('id'):
                            call['id'] = item['id']
                        for key in ('name', 'arguments'):
                            call['function'][key] += item.get('function', {}).get(key) or ''
                        message['tool_calls'] = [calls[i] for i in sorted(calls)]
                    if bang_limit:
                        fields = [message.get(k) or '' for k in ('content', 'reasoning', 'reasoning_content')]
                        fields += [c['function']['arguments'] for c in calls.values()]
                        if any('!' * bang_limit in re.sub(r'\\u0021', '!', value, flags=re.I)
                               for value in fields):
                            raise BangLoopError('32 consecutive bangs; cancel and isolate retry cache')
                    if choice.get('finish_reason'):
                        response['choices'][0]['finish_reason'] = choice['finish_reason']
            if not response['choices'][0]['finish_reason'] or 'usage' not in response:
                raise RuntimeError('incomplete streaming response')
        return response
    except Exception as exc:
        exc.partial_response = response
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18125')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--records', type=int, default=180)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--shared-cache', action='store_true', help='omit cache_salt, matching ordinary clients')
    p.add_argument('--stream', action='store_true', help='retain incremental tool/text output including partial failures')
    p.add_argument('--bang-retries', type=int, default=0,
                   help='Diagnostic retry emulation, not the Pi extension: cancel at32 bangs, rotate salt')
    p.add_argument('--salt', default='', help='Independent workload namespace')
    p.add_argument('--logprobs', action='store_true', help='Retain top5 token logprobs in raw SSE for failure diagnosis')
    p.add_argument('--session-order', default='0,1,2,3',
                   help='Logical session IDs in submission order; singleton isolates concurrency')
    p.add_argument('--turns', type=int, default=4, choices=range(1, 5))
    argv = sys.argv[1:]
    # A namespace may begin with a single dash. Preserve it as literal data,
    # including jobs already queued with separate --salt/value arguments.
    if '--salt' in argv:
        i = argv.index('--salt')
        if i + 1 < len(argv) and argv[i + 1].startswith('-') and not argv[i + 1].startswith('--'):
            argv[i:i + 2] = ['--salt=' + argv[i + 1]]
    args = p.parse_args(argv)
    if args.bang_retries < 0 or args.bang_retries > 3:
        p.error('--bang-retries must be between0 and3')
    if args.bang_retries and not args.stream:
        p.error('bang recovery requires --stream')
    session_order = [int(v) for v in args.session_order.split(',')]
    if not session_order or len(session_order) > 4 or len(set(session_order)) != len(session_order):
        p.error('--session-order requires one to four unique integer IDs')
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / 'config.json').write_text(json.dumps(config, default=str, indent=2) + '\n')
    assert args.model in [m['id'] for m in json.loads(request(args.base, '/v1/models'))['data']]
    (args.out / 'metrics-before.txt').write_text(request(args.base, '/metrics'))
    started = time.monotonic()
    def session(index):
        rows = []
        salt = 'agent-v1-' + str(index) + args.salt
        isolation_active = not args.shared_cache
        attempts = []
        def record(row):
            row['recorded_at'] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
            with (args.out / f'session-{index}.jsonl').open('a') as f:
                f.write(json.dumps(row, ensure_ascii=True) + '\n')
        def invoke(payload, phase, turn):
            nonlocal salt, isolation_active
            began = time.monotonic()
            response = None
            try:
                for attempt in range(args.bang_retries + 1):
                    attempt_started = datetime.now(timezone.utc).isoformat()
                    try:
                        if isolation_active:
                            payload['cache_salt'] = salt
                        else:
                            payload.pop('cache_salt', None)
                        with (args.out / f'session-{index}-requests.jsonl').open('a') as f:
                            f.write(json.dumps({'phase': phase, 'turn': turn, 'attempt': attempt,
                                'started_at': attempt_started, 'payload': payload}, ensure_ascii=True) + '\n')
                        if args.stream:
                            suffix = f'-retry{attempt}' if attempt else ''
                            response = stream_request(args.base, payload, args.timeout,
                                args.out / f'session-{index}-{turn}-{phase}{suffix}-sse.jsonl',
                                bang_limit=32 if args.bang_retries else None)
                        else:
                            response = json.loads(request(args.base, '/v1/chat/completions', payload, timeout=args.timeout))
                        attempts.append({'phase': phase, 'turn': turn, 'attempt': attempt,
                                         'bang': contains_bang(response['choices'][0]['message']),
                                         'started_at': attempt_started})
                        break
                    except BangLoopError as exc:
                        row = {'phase': phase, 'turn': turn, 'attempt': attempt, 'bang': True,
                               'started_at': attempt_started,
                               'detected_at': datetime.now(timezone.utc).isoformat(),
                               'partial_response': exc.partial_response, 'cache_salt': payload.get('cache_salt')}
                        attempts.append(row)
                        with (args.out / f'session-{index}-bangs.jsonl').open('a') as f:
                            f.write(json.dumps(row) + '\n')
                        if attempt == args.bang_retries:
                            raise
                        salt += f'-recovery-{turn}-{phase}-{attempt}'
                        isolation_active = True
                if not isinstance(response['choices'][0]['message'], dict):
                    raise ValueError('missing response message')
                return response, time.monotonic() - began
            except Exception as exc:
                if not isinstance(exc, BangLoopError):
                    attempts.append({'phase': phase, 'turn': turn, 'attempt': attempt,
                                     'bang': False, 'error': ascii(exc), 'started_at': attempt_started})
                response = getattr(exc, 'partial_response', response)
                record({'phase': phase, 'turn': turn, 'passed': False,
                        'elapsed_s': time.monotonic() - began, 'error': ascii(exc),
                        'response': response})
                return None, None
        history = [{'role': 'system', 'content': 'You are testing a warehouse tool. Always use lookup_stock to look up stock. After receiving its result, report only its count as an integer. Background records:\n' + ('The warehouse stores parts and maintains an inventory ledger.\n' * args.records)}]
        for turn in range(args.turns):
            sku = f'part-{index}-{turn}'
            count = 730 + index * 10 + turn
            history.append({'role': 'user', 'content': 'Look up stock for SKU ' + sku + '.'})
            payload = {'model': args.model, 'messages': history, 'temperature': 0, 'seed': 42, 'max_tokens': 512,
                       'tools': [{'type': 'function', 'function': {'name': 'lookup_stock', 'description': 'Get stock count for a SKU', 'parameters': {'type': 'object', 'properties': {'sku': {'type': 'string'}}, 'required': ['sku']}}}],
                       'tool_choice': 'auto', 'chat_template_kwargs': {'enable_thinking': False}, 'cache_salt': 'agent-v1-' + str(index)}
            if args.logprobs:
                payload.update(logprobs=True, top_logprobs=5)
            if args.shared_cache:
                payload.pop('cache_salt')
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
        return {'session': index, 'rows': rows, 'attempts': attempts,
                'passed': len(rows) == args.turns * 2 and all(r['passed'] for r in rows)}
    with ThreadPoolExecutor(max_workers=len(session_order)) as pool:
        results = list(pool.map(session, session_order))
    (args.out / 'results.json').write_text(json.dumps(results, ensure_ascii=True, indent=2) + '\n')
    passed = all(r['passed'] for r in results)
    try:
        (args.out / 'metrics-after.txt').write_text(request(args.base, '/metrics'))
    except Exception as exc:
        passed = False
        (args.out / 'metrics-error.txt').write_text(ascii(exc) + '\n')
    summary = {'passed': passed, 'checks': sum(len(r['rows']) for r in results),
               'attempts': sum(len(r['attempts']) for r in results),
               'bang_attempts': sum(a['bang'] for r in results for a in r['attempts']),
               'recovery_scope': 'Diagnostic cancellation/salt rotation; not actual Pi/OMP extension execution',
               'elapsed_s': time.monotonic() - started}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
