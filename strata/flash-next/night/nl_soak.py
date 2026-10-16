#!/usr/bin/env python3
"""Night-lab mixed-traffic soak: random C1/C2/C3 coding, short bursts, a long-context turn; coherence probes.
usage: nl_soak.py <minutes> <out.json>"""
import json, random, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nl

minutes, out = float(sys.argv[1]), sys.argv[2]
rng = random.Random(1016)
t_end = time.time() + minutes * 60
log, errors = [], []
docs = sorted((nl.REPO / 'docs').glob('2026*.md'))
corpus = ''.join(ch if ord(ch) < 128 else ' ' for p in docs for ch in p.read_text(errors='replace'))


def one(kind, i):
    try:
        if kind == 'coh':
            r = nl.send([{"role": "user", "content": f"[soak {i}] Return only the integer: 41 * 43."}], 16)
            ok = r['text'].strip() == '1763'
            return dict(kind=kind, ok=ok, text=r['text'][:40])
        if kind == 'long':
            off = rng.randrange(0, len(corpus) - 100000)
            r = nl.send([{"role": "user", "content": f"[soak long {i}] " + corpus[off:off + 80000] + "\n\nSummarize the key numbers above."}], 128)
        elif kind == 'short':
            r = nl.send([{"role": "user", "content": f"[soak short {i}] Name three uses of a hash map in one line each."}], 64)
        else:
            r = nl.send([{"role": "user", "content": f"[soak {kind} {i}] " + rng.choice([nl.CODING, nl.CODING2, nl.PROSE])}], rng.choice([128, 256, 512]))
        return dict(kind=kind, ok=True, tps=(r['timings'] or {}).get('predicted_per_second'), n=r['usage']['completion_tokens'])
    except Exception as e:
        return dict(kind=kind, ok=False, error=repr(e)[:300])


i = 0
while time.time() < t_end:
    shape = rng.choice(['c1', 'c2', 'c2', 'c3', 'short4', 'long', 'coh'])
    i += 1
    if shape == 'c1': jobs = ['code']
    elif shape == 'c2': jobs = ['code', 'code']
    elif shape == 'c3': jobs = ['code', 'code', 'short']
    elif shape == 'short4': jobs = ['short'] * 4
    elif shape == 'long': jobs = ['long', 'code']
    else: jobs = ['coh', 'coh']
    with ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda k: one(k, i), jobs))
    log.append(dict(i=i, shape=shape, res=res, t=time.time()))
    bad = [r for r in res if not r['ok']]
    errors += bad
    print(i, shape, ['ok' if r['ok'] else 'FAIL' for r in res], flush=True)
    if any('engine stopped' in (r.get('error') or '') for r in bad):
        print('ENGINE DIED', bad, flush=True)
        break
Path(out).write_text(json.dumps(dict(rounds=i, errors=errors, log=log), indent=1))
print(json.dumps(dict(rounds=i, errors=len(errors))))
