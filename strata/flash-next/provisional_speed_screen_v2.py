#!/usr/bin/env python3
"""Bounded client-only diagnostic screen; owned GPU lifecycle is external."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request

MAX_LINE = 1 << 20
MAX_BYTES = 16 << 20
MAX_EVENTS = 4096
OBSERVER_PREFIXES = ('STRATA_FIDELITY_DIAG', 'STRATA_PREFIX30', 'STRATA_PLE_INPUT33',
    'STRATA_QSA3', 'STRATA_BATCH_FIDELITY', 'STRATA_CRITICAL_PATH_TRACE',
    'STRATA_FULL_CACHE', 'STRATA_MEMORY', 'STRATA_MIRROR_OWNER_TRACE',
    'STRATA_SLOT_OWNER_TRACE', 'SYCL_UR_TRACE', 'UR_LOG', 'UR_TRACE', 'STRATA_HOST_TRACE')


def observer_off(env):
    require(not any(any(k.startswith(x) for x in OBSERVER_PREFIXES) and str(v) not in ('0', '')
                    for k, v in env.items()), 'Heavy diagnostic environment forbidden')


def repeat_binding(requests):
    keys = ('output_text', 'reasoning_text', 'finish_reason', 'usage')
    require(len(requests) == 4, 'Four request repeat corpus required')
    for i, j in ((0, 3), (1, 2)):
        require(all(requests[i][k] == requests[j][k] for k in keys), 'Deterministic repeat coherence failed')


def request_binding(value, body, expected):
    require(value['usage']['completion_tokens'] <= body['max_tokens'], 'Actual output exceeds request budget')
    require(value['sse_model'] in expected, 'SSE served model mismatch')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prompts():
    task = ('Explain how water moves through evaporation, condensation, precipitation, '
            'and collection. Give a detailed numbered explanation with practical examples '
            'and a final paragraph about conserving water. Write approximately 250 words.')
    notes = ('Observation: warm sunlight heats a lake; moist air rises and cools; '
             'clouds form; rain returns water to soil and streams. ')
    return {'short': task, 'prefill_about512': notes * 18 + '\n' + task}


def parse_stream(stream, started_ns, clock=time.perf_counter_ns, on_line=None):
    """Read SSE lines; timestamps describe client receipt, never device/token clocks."""
    raw = []
    events = []
    data = []
    size = 0
    done = False
    model = None
    previous = -1
    while True:
        line = stream.readline(MAX_LINE + 1)
        if not line:
            break
        at = clock() - started_ns
        require(type(at) is int and at >= 0 and at >= previous, 'Finite ordered client timestamps required')
        previous = at
        require(len(line) <= MAX_LINE, 'SSE line bound exceeded')
        size += len(line)
        require(size <= MAX_BYTES, 'SSE byte bound exceeded')
        require(line.endswith(b'\n'), 'Truncated SSE line')
        raw.append({'arrival_ns': at, 'line_hex': line.hex()})
        if on_line is not None:
            on_line(raw[-1])
        text = line.decode('utf-8').rstrip('\r\n')
        if not text:
            if data:
                payload = '\n'.join(data)
                data = []
                require(not done, 'SSE event after DONE')
                if payload == '[DONE]':
                    done = True
                else:
                    value = json.loads(payload)
                    require(isinstance(value, dict) and 'error' not in value, 'SSE error/nonobject')
                    require(isinstance(value.get('model'), str), 'Actual SSE model required')
                    if model is None: model = value['model']
                    require(value['model'] == model, 'SSE model drift')
                    require(type(value.get('choices')) is list and len(value['choices']) == 1 and
                            type(value['choices'][0].get('index')) is int and value['choices'][0]['index'] == 0,
                            'Exact single typed SSE choice lane required')
                    events.append({'arrival_ns': at, 'value': value})
                    require(len(events) <= MAX_EVENTS, 'SSE event bound exceeded')
        elif text.startswith('data:'):
            data.append(text[5:].lstrip(' '))
        elif text.startswith(':') or text.startswith(('event:', 'id:', 'retry:')):
            pass
        else:
            raise ValueError('Unknown SSE field')
    require(done and not data, 'Missing DONE or incomplete SSE event')
    usage = [e for e in events if e['value'].get('usage') is not None]
    require(len(usage) == 1, 'Exactly one actual usage event required')
    require(usage[0] is events[-1], 'Usage must be the final terminal SSE event')
    u = usage[0]['value']['usage']
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        require(type(u.get(key)) is int and u[key] >= 0, 'Typed actual token usage required')
    require(u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens'], 'Usage total mismatch')
    cached = u.get('prompt_tokens_details', {}).get('cached_tokens')
    require(type(cached) is int and cached == 0, 'Fresh cacheOFF usage required')
    finished = [(e, c['finish_reason']) for e in events for c in e['value'].get('choices', [])
                if c.get('finish_reason') is not None]
    require(len(finished) == 1 and finished[0][1] in ('stop', 'length'), 'One natural stop/length finish required')
    require(finished[0][0] is usage[0], 'Actual terminal finish and usage must share final event')
    require(not any(usage[0]['value']['choices'][0].get('delta', {}).get(k)
                    for k in ('content', 'reasoning_content')), 'Terminal usage contains late text')
    content = [(e['arrival_ns'], c.get('delta', {})) for e in events
               for c in e['value'].get('choices', [])
               if any(c.get('delta', {}).get(k) for k in ('content', 'reasoning_content'))]
    times = [x[0] for x in content]
    elapsed = usage[0]['arrival_ns'] / 1e9
    timing = usage[0]['value'].get('timings')
    if timing is not None:
        require(type(timing) is dict, 'Native timing object required')
        require(all(v is None or (type(v) in (int, float) and math.isfinite(v) and v >= 0)
                    for v in timing.values()), 'Finite nonnegative native timings required')
    first = times[0] / 1e9 if times else None
    gaps = [(b-a)/1e9 for a, b in zip(times, times[1:])]
    return {'raw_lines': raw, 'events': events, 'sse_model': model, 'usage': u, 'finish_reason': finished[0][1],
            'output_text': ''.join(d.get('content', '') for _, d in content),
            'reasoning_text': ''.join(d.get('reasoning_content', '') for _, d in content),
            'client_first_text_s': first, 'client_usage_arrival_s': elapsed,
            'client_text_chunk_gaps_s': gaps, 'text_chunk_count': len(times),
            'completion_tokens_at_least128': u['completion_tokens'] >= 128,
            'prompt_tokens_within384_768': 384 <= u['prompt_tokens'] <= 768,
            'completion_tokens_per_client_wall_s': u['completion_tokens']/elapsed if elapsed else None,
            'native_reported_timings': usage[0]['value'].get('timings'),
            'native_timing_frontend_count_matches_usage': timing.get('predicted_n') == u['completion_tokens'] if timing else None,
            'native_rate_numerator_independently_observed': False,
            'true_per_token_gaps_observed': False, 'device_prefill_rate_observed': False}


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    require(sys.flags.optimize == 0 and not os.environ.get('PYTHONOPTIMIZE'), 'Optimized Python unsupported')
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    a = p.parse_args()
    require(0 < a.timeout <= 1800, 'Bounded request timeout required')
    a.output.mkdir(parents=True, exist_ok=False)
    directory = a.prepared.resolve()
    files = ('prepared.json', 'server-config.json', 'artifact-identity.json')
    binding = {f: sha(directory/f) for f in files}
    m = json.loads((directory/'prepared.json').read_text())
    cfg = json.loads((directory/'server-config.json').read_text())
    args = cfg['args']
    require(args[args.index('--prompt-cache')+1] == '0', 'Declared prompt cacheOFF required')
    env = cfg.get('env', {})
    observer_off(env)
    base = 'http://127.0.0.1:' + str(m['port'])
    expected = ['hotschmoe-dd', m['alias']]
    models = get_json(base+'/v1/models')
    require([x['id'] for x in models['data']] == expected, 'Actual served registry aliases/order mismatch')
    require(all(x['meta']['artifact_identity']['artifact_identity_sha256'] == m['artifact_manifest_sha256']
                for x in models['data']), 'Actual artifact identity mismatch')
    report = {'schema': 1, 'scope': 'provisional_client_speed_screen', 'passed': False,
              'prepared': str(directory), 'source_sha256': sha(__file__), 'binding': binding,
              'models_before': models, 'config': cfg, 'prepared_metadata': m,
              'registry_sha256': sha(Path(__file__).resolve().parents[2]/'evals/configs/models.yaml'),
              'started_epoch': time.time(), 'requests': [], 'errors': [],
              'correctness_qualified': False, 'shelf_qualified': False,
              'health_teardown_qualified': False, 'gpu_lifecycle_owned_by_client': False}
    try:
        corpus = prompts()
        for index, name in enumerate(('short', 'prefill_about512', 'prefill_about512', 'short')):
            body = {'model': expected[0], 'messages': [{'role': 'user', 'content': corpus[name]}],
                    'temperature': 0, 'seed': 1, 'max_tokens': 192, 'stream': True,
                    'stream_options': {'include_usage': True}, 'strata_fresh': True,
                    'chat_template_kwargs': {'enable_thinking': False}}
            packet = json.dumps(body, sort_keys=True).encode()
            req = urllib.request.Request(base+'/v1/chat/completions', data=packet,
                                         headers={'Content-Type': 'application/json'})
            started_epoch = time.time()
            started_ns = time.perf_counter_ns()
            with (a.output/('request-'+str(index)+'.raw.jsonl')).open('x') as raw:
                def retain(row):
                    raw.write(json.dumps(row)+'\n')
                    raw.flush()
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    require(r.headers.get_content_type() == 'text/event-stream', 'SSE response required')
                    value = parse_stream(r, started_ns, on_line=retain)
            value.update(index=index, prompt_name=name, request=body, started_epoch=started_epoch,
                         finished_epoch=time.time(), request_sha256=hashlib.sha256(packet).hexdigest())
            (a.output/('request-'+str(index)+'.json')).write_text(json.dumps(value, indent=2, ensure_ascii=True)+'\n')
            report['requests'].append({k: v for k, v in value.items() if k not in ('events', 'raw_lines')})
            request_binding(value, body, expected)
        repeat_binding(report['requests'])
        report['repeat_coherence_passed'] = True
        report['models_after'] = get_json(base+'/v1/models')
        require(report['models_after'] == models, 'Live model metadata changed')
        require({f: sha(directory/f) for f in files} == binding, 'Prepared/config binding changed')
        report['passed'] = True
    except BaseException as e:
        report['errors'].append(type(e).__name__+': '+str(e))
        raise
    finally:
        report['finished_epoch'] = time.time()
        report['artifacts'] = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
                               for p in sorted(a.output.iterdir()) if p.is_file()]
        (a.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=True)+'\n')


if __name__ == '__main__':
    main()
