#!/usr/bin/env python3
"""CPU-side gates for gpu_window.sh (no GPU use; talks HTTP to the window server on 127.0.0.1:18150 only).

  identity URL                 model ids, context, capabilities vs the env file (WIN_EXPECT_CAPS)
  compare RESULTS REF [MIN]    serve-bench rates vs a reference results.json (MIN ratio, default 0.95)
  logscan LOG [START_REGEX]    fatal markers in a server log since the last launch line
Each prints one GATE line and exits 0 on PASS, 1 on FAIL. Never prints prompt or response text.
"""
import json, os, re, sys, urllib.request

def gate(name, ok, detail):
    print(f"GATE {name} {'PASS' if ok else 'FAIL'} {detail}")
    sys.exit(0 if ok else 1)

def identity(url):
    with urllib.request.urlopen(url.rstrip('/') + '/v1/models', timeout=10) as r:
        models = json.load(r)['data']
    ids = [m['id'] for m in models]
    first = models[0]
    want_name = os.environ.get('TF_NAME', 'hotschmoe-dd')
    aliases = os.environ.get('TF_ALIASES', '').split()
    listed = set(ids) | {a for m in models for a in (m.get('aliases') or [])}
    ctx = int(os.environ.get('TF_CONTEXT', '200000'))
    caps = first.get('capabilities', {})
    bad = []
    if first['id'] != want_name: bad.append(f'first id {first["id"]} != {want_name}')
    missing = [a for a in aliases if a not in listed]
    if missing: bad.append(f'aliases missing {missing}')
    if first.get('context_length') != ctx: bad.append(f'context_length {first.get("context_length")} != {ctx}')
    for kv in filter(None, os.environ.get('WIN_EXPECT_CAPS', '').split(',')):
        k, v = kv.split('=')
        if str(caps.get(k)).lower() != v.lower(): bad.append(f'cap {k}={caps.get(k)} want {v}')
    gate('identity', not bad, json.dumps({'ids': ids, 'context_length': first.get('context_length'),
         'reasoning': first.get('reasoning'), 'caps': {k: v for k, v in caps.items() if v is True}, 'bad': bad}))

def rates(path):
    r = json.load(open(path))
    d = {f"C{m['concurrency']}": m['aggregate_tokens_s'] for m in r['measurements']
         if m['kind'] == 'decode' and m['config'] == 'policy'}
    d.update({f"PF{m['prompt_tokens'] // 1024}k": m['tokens_s'] for m in r['measurements'] if m['kind'] == 'prefill'})
    return r, d

def compare(results, ref, minimum='0.95'):
    r, got = rates(results)
    _, want = rates(ref)
    ratio = {k: round(got[k] / want[k], 3) for k in got if k in want}
    low = {k: v for k, v in ratio.items() if v < float(minimum)}
    gate('bench-vs-int-rf2', r.get('oracle_equal') in (True, 'recorded') and not low,
         json.dumps({'got': {k: round(v, 1) for k, v in got.items()}, 'ratio': ratio, 'min': float(minimum),
                     'oracle_equal': r.get('oracle_equal'), 'below': low}))

FATAL = re.compile(r'DEVICE_LOST|WEDGED|HUNG|panic|Segmentation fault|SIGSEGV|SIGABRT|error\.[A-Z][A-Za-z]+|OutOfMemory|reached unreachable')
IGNORE = re.compile(r'spills \d+ bytes of registers|libtfgrammar')

def logscan(log, start=r'\[dd_launch_tf\] variant='):
    lines = open(log, errors='replace').read().splitlines()
    idx = max([i for i, l in enumerate(lines) if re.search(start, l)] or [0])
    hits = [l[:160] for l in lines[idx:] if FATAL.search(l) and not IGNORE.search(l)]
    gate('server-log', not hits, json.dumps({'lines': len(lines) - idx, 'fatal': hits[:5]}))

if __name__ == '__main__':
    cmd, *rest = sys.argv[1:] or ['help']
    {'identity': identity, 'compare': compare, 'logscan': logscan}.get(cmd, lambda *_: sys.exit(__doc__))(*rest)
