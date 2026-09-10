#!/usr/bin/env python3
"""Reconstruct bounded Pi histories for inference only; never execute tools.

This is a controlled reconstruction, not an exact wire replay: the supplied
bundle omits Pi's system prompt, tool schemas and forwarded sampling payload.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fp8'))
from kv_campaign_agent_probe import stream_request, contains_bang, BangLoopError


TOOLS = [dict(type='function', function=dict(name=name, description=description,
    parameters=dict(type='object', properties=properties, required=required)))
    for name, description, properties, required in [
        ('bash', 'Run a shell command.', {'command': {'type': 'string'}, 'timeout': {'type': 'number'}}, ['command']),
        ('read', 'Read a file.', {'path': {'type': 'string'}, 'offset': {'type': 'number'}, 'limit': {'type': 'number'}}, ['path']),
        ('write', 'Write a file.', {'path': {'type': 'string'}, 'content': {'type': 'string'}}, ['path', 'content']),
        ('edit', 'Replace exact text in a file.', {'path': {'type': 'string'}, 'oldText': {'type': 'string'}, 'newText': {'type': 'string'}}, ['path', 'oldText', 'newText'])]]


def text_blocks(content):
    if isinstance(content, str):
        return content
    return '\n'.join(block.get('text', '') for block in content if block.get('type') == 'text')


def reconstruct(session):
    history = [{'role': 'system', 'content': 'You are a coding assistant. Continue the user task using the available tools.'}]
    prior_payloads = []
    for line in session.read_text().splitlines():
        record = json.loads(line)
        message = record.get('message', {})
        role = message.get('role')
        if role == 'assistant':
            if contains_bang(message.get('content')):
                return prior_payloads[-1:], history, message['timestamp']
            if message.get('stopReason') not in ('toolUse', 'stop'):
                continue
            prior_payloads.append(json.loads(json.dumps(history)))
            converted = {'role': 'assistant', 'content': text_blocks(message.get('content', [])) or None}
            calls = [dict(id=b['id'], type='function', function=dict(name=b['name'],
                     arguments=json.dumps(b['arguments'], ensure_ascii=True)))
                     for b in message.get('content', []) if b.get('type') == 'toolCall']
            if calls:
                converted['tool_calls'] = calls
            history.append(converted)
        elif role == 'toolResult':
            history.append(dict(role='tool', tool_call_id=message['toolCallId'],
                                content=text_blocks(message.get('content', []))))
        elif role == 'user':
            history.append(dict(role='user', content=text_blocks(message.get('content', []))))
    raise ValueError('No first bang found in session')


def prepare(bundle, output, agents):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    cases = []
    for agent in agents:
        session, = [p for p in (bundle / agent).glob('*.jsonl') if p.name != 'rpc.jsonl']
        warm, history, stamp = reconstruct(session)
        seed = next(a['seed'] for a in manifest['agents'] if a['name'] == agent)
        for phase, messages in [('warm', m) for m in warm] + [('target', history)]:
            payload = dict(messages=messages, tools=TOOLS, temperature=.7, seed=seed,
                           max_tokens=8192, chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='medium'))
            path = output / (agent + '-' + phase + '.json')
            raw = json.dumps(payload, ensure_ascii=True, indent=2) + '\n'
            path.write_text(raw)
            cases.append(dict(agent=agent, phase=phase, payload=str(path), sha256=hashlib.sha256(raw.encode()).hexdigest(),
                              original_failed_timestamp_ms=stamp, messages=len(messages)))
    result = dict(cases=cases, limitations=[
        'Reconstructed system prompt and minimal tool schemas; not exact Pi payload.',
        'Past thinking omitted; recorded tool results (including errors) retained without execution.',
        'Manifest per-agent seed used for control; original forwarded seed unverified.',
        'First bang response and all subsequent history excluded; earlier non-bang degeneration may remain.'])
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.json').write_text(json.dumps(dict(
        args=vars(args), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest()), default=str, indent=2) + '\n')
    identity = json.load(urllib.request.urlopen(args.base + '/v1/models', timeout=10))
    if args.model not in [m['id'] for m in identity['data']]:
        raise ValueError('served identity mismatch')
    (args.out / 'models.json').write_text(json.dumps(identity, indent=2) + '\n')
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for case in manifest['cases']:
        repeats = args.repeats if case['phase'] == 'target' else 1
        for repeat in range(repeats):
            payload_bytes = Path(case['payload']).read_bytes()
            if hashlib.sha256(payload_bytes).hexdigest() != case['sha256']:
                raise ValueError('payload hash mismatch')
            payload = json.loads(payload_bytes)
            payload.update(model=args.model, cache_salt='pi-isolation-' + case['agent'])
            if args.cold_each:
                payload['cache_salt'] += '-' + case['phase'] + '-' + str(repeat)
            name = case['agent'] + '-' + case['phase'] + '-' + str(repeat)
            (args.out / (name + '-request.json')).write_text(json.dumps(payload, ensure_ascii=True) + '\n')
            started = time.monotonic()
            row = dict(case=case, repeat=repeat, started_at=datetime.now(timezone.utc).isoformat())
            try:
                response = stream_request(args.base, payload, args.timeout, args.out / (name + '.sse.jsonl'), bang_limit=32)
                row.update(response=response, bang=contains_bang(response), error=None)
            except Exception as exc:
                row.update(response=getattr(exc, 'partial_response', None), bang=isinstance(exc, BangLoopError), error=ascii(exc))
            row['elapsed_s'] = time.monotonic() - started
            rows.append(row)
            (args.out / 'results.json').write_text(json.dumps(rows, ensure_ascii=True, indent=2) + '\n')
            print(json.dumps({k: v for k, v in row.items() if k not in ('case', 'response')}, ensure_ascii=True), flush=True)
            # Corruption is an experimental result, not an instruction to retry
            # the same poisoned state or proceed to unbounded generation.
            if row['error'] or row['bang']:
                (args.out / 'OUTCOME.json').write_text(json.dumps(dict(bang=row['bang'], requests=len(rows), error=row['error'])) + '\n')
                return 1
    (args.out / 'OUTCOME.json').write_text(json.dumps(dict(bang=False, requests=len(rows), error=None)) + '\n')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--agents', nargs='+', default=['agent_3', 'agent_2', 'agent_1'])
    p = sub.add_parser('run')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--base', default='http://127.0.0.1:18125')
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--cold-each', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.bundle, args.out, args.agents), indent=2))
    else:
        raise SystemExit(run(args))
