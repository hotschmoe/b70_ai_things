#!/usr/bin/env python3
"""Observe raw stop/token metadata; only adds return_token_ids to a frozen request."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def diagnostic_request(original):
    assert original.get('stream') is True
    assert original.get('n', 1) == 1
    assert not original.get('stop') and not original.get('stop_token_ids')
    assert not original.get('ignore_eos', False)
    assert 'return_token_ids' not in original
    result = dict(original)
    result['return_token_ids'] = True
    assert {k: v for k, v in result.items() if k != 'return_token_ids'} == original
    return result


def parse_sse(raw):
    tokens = []; finishes = []; ids = set(); prompt = None; usage = None
    content = []; reasoning = []; errors = []; done = False; token_chunks = 0
    frames = raw.decode('utf-8').replace('\r\n', '\n').split('\n\n')
    for frame in frames:
        data = '\n'.join(line[5:].lstrip() for line in frame.splitlines() if line.startswith('data:'))
        if not data: continue
        if data.strip() == '[DONE]': done = True; continue
        event = json.loads(data)
        if 'error' in event: errors.append(event['error'])
        if event.get('id'): ids.add(event['id'])
        if event.get('prompt_token_ids') is not None: prompt = event['prompt_token_ids']
        if event.get('usage') is not None: usage = event['usage']
        for choice in event.get('choices', []):
            assert choice.get('index', 0) == 0
            values = choice.get('token_ids')
            if values is not None:
                assert isinstance(values, list) and all(type(v) is int for v in values)
                tokens.extend(values); token_chunks += 1
            delta = choice.get('delta', {})
            content.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
            if choice.get('finish_reason') is not None:
                finishes.append(dict(finish_reason=choice['finish_reason'], stop_reason=choice.get('stop_reason'), terminal_token_ids=values))
    return dict(response_ids=sorted(ids), done=done, errors=errors, finishes=finishes,
                token_ids=tokens, token_count=len(tokens), last_token_ids=tokens[-16:],
                token_metadata_chunks=token_chunks, prompt_token_ids=prompt,
                usage=usage, content=''.join(content), reasoning=''.join(reasoning),
                observation_complete=done and len(finishes)==1 and not errors,
                caveat='Diagnostic completion is not semantic success. Missing EOS in visible text is expected; missing token metadata cannot establish absence of sampled EOS.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    original_bytes = a.request.read_bytes(); original = json.loads(original_bytes)
    payload = diagnostic_request(original)
    if not a.run:
        print(json.dumps(dict(status='PREPARED_NOT_RUN',request_sha256=hashlib.sha256(original_bytes).hexdigest(),delta={'return_token_ids':True},unchanged_fields=sorted(original),base=a.base)))
        return 0
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'request.json').write_text(json.dumps(payload,ensure_ascii=True)+'\n')
    (a.out/'config.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),request_source=str(a.request),request_sha256=hashlib.sha256(original_bytes).hexdigest(),base=a.base,started=time.time(),delta={'return_token_ids':True}),indent=2)+'\n')
    request = urllib.request.Request(a.base.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Accept':'text/event-stream'})
    try:
        with urllib.request.urlopen(request,timeout=a.timeout) as response, (a.out/'response.sse').open('wb') as f:
            (a.out/'response-headers.json').write_text(json.dumps(dict(response.headers),indent=2)+'\n')
            start = time.monotonic(); size = 0
            for line in response:
                f.write(line); f.flush(); size += len(line)
                if time.monotonic()-start > a.timeout or size > 32*1024*1024:
                    raise RuntimeError('diagnostic read bound exceeded')
        parsed = parse_sse((a.out/'response.sse').read_bytes())
        (a.out/'result.json').write_text(json.dumps(parsed,ensure_ascii=True,indent=2)+'\n')
        return 0 if parsed['observation_complete'] else 1
    except Exception as exc:
        (a.out/'error.json').write_text(json.dumps({'error':ascii(exc)})+'\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
