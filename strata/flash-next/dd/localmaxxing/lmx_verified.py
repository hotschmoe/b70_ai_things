#!/usr/bin/env python3
"""LocalMaxxing verified-run measurement for the live Strata endpoint (lmx v0.1.54 has no --canonical-prompt).

Same method as `lmx speed-test run --mode remote`: streaming chat completion, a leading
"[LocalMaxxing cache-bust nonce: ...]" line per request, 1 untimed warmup, 3 timed iterations, median reported,
TTFT = first streamed token, decode rate = (completion_tokens - 1) / (last token - first token).
Adds what a Verified run needs and the CLI does not capture here: promptSha256/promptSample (canonical text, nonce
line excluded), outputSha256/outputSample, engineTimingsRaw (Strata's own per-request `timings`), and the MTP draft
counts from those timings (draft_n / draft_n_accepted) of the median iteration.

usage: lmx_verified.py <canonical-id> <template-run.json> <out-payload.json> [max_tokens]
"""
import hashlib, json, secrets, statistics, subprocess, sys, time, urllib.request

import os
BASE = os.environ.get('LMX_BASE', 'http://127.0.0.1:18124') + '/v1/chat/completions'
cid, template, out = sys.argv[1], sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 512
ctx = json.loads(subprocess.run([f'{__import__("os").path.expanduser("~")}/.local/bin/lmx', 'context'],
                                capture_output=True, text=True, check=True).stdout)
canon = next(p for p in ctx['canonicalPrompts'] if p['id'] == cid)
text = canon['text']
assert hashlib.sha256(text.encode()).hexdigest() == canon['sha256'], 'canonical text hash mismatch'


def one():
    nonce = f"[LocalMaxxing cache-bust nonce: {secrets.token_hex(8)}]\n"
    body = dict(model='hotschmoe-dd', messages=[{'role': 'user', 'content': nonce + text}],
                max_tokens=max_tokens, stream=True)
    req = urllib.request.Request(BASE, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.perf_counter(); t_first = t_last = None; parts = []; usage = timings = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            ev = json.loads(line[6:])
            for ch in ev.get('choices') or []:
                d = ch.get('delta') or {}
                piece = (d.get('reasoning_content') or '') + (d.get('content') or '')
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now; t_last = now; parts.append(piece)
            usage = ev.get('usage') or usage; timings = ev.get('timings') or timings
    n = usage['completion_tokens']
    return dict(ttft_ms=(t_first - t0) * 1000, tok_s_out=(n - 1) / (t_last - t_first), output_tokens=n,
                prompt_tokens=usage['prompt_tokens'], output=''.join(parts), timings=timings)


one()                                                             # untimed warmup
runs = [one() for _ in range(3)]
med = sorted(runs, key=lambda r: r['tok_s_out'])[1]
p = json.load(open(template))
keep = ('backend', 'batchSize', 'benchmarkMode', 'engineBuild', 'engineCommit', 'engineName', 'engineRepository',
        'engineVersion', 'hardware', 'hfId', 'modelRevision', 'quantization', 'metricSource', 'timingSource')
pay = {k: p[k] for k in keep if k in p}
t = med['timings']
spec = bool(t.get('draft_n'))
pay.update(
    engineFlags={**{k: v for k, v in p['engineFlags'].items() if k not in ('promptSource', 'promptFile')},
                 'maxTokens': max_tokens, 'promptSource': f'canonical:{cid}', 'iterations': 3, 'warmup': 1,
                 **({'specDraftTokens': t['draft_n'], 'specAcceptedTokens': t['draft_n_accepted'],
                     'specAcceptanceRate': round(t['draft_n_accepted'] / t['draft_n'], 4),
                     'specMeanAcceptedLength': round(t['predicted_n'] / max(1, t['predicted_n'] - t['draft_n_accepted']), 3)}
                    if spec else {})},
    tokSOut=round(med['tok_s_out'], 1), tokSOutSource='inter_token', ttftMs=round(med['ttft_ms'], 2),
    ttftSource='stream_first_token', outputTokens=med['output_tokens'], promptTokens=med['prompt_tokens'],
    prefillTokens=0, prompt=text, promptSha256=canon['sha256'], promptSample=text[:2000],
    outputSha256=hashlib.sha256(med['output'].encode()).hexdigest(), outputSample=med['output'][:4000],
    engineTimingsRaw=t,
    samples=[dict(iteration=i + 1, tokSOut=round(r['tok_s_out'], 1), ttftMs=round(r['ttft_ms'], 2),
                  outputTokens=r['output_tokens'], promptTokens=r['prompt_tokens'],
                  draftN=r['timings'].get('draft_n'), draftAccepted=r['timings'].get('draft_n_accepted'),
                  enginePredictedPerSecond=r['timings']['predicted_per_second']) for i, r in enumerate(runs)])
if not spec:
    for k in ('specDecoding', 'specMethod', 'specNumTokens', 'mtpEnabled'):
        pay['engineFlags'].pop(k, None)
json.dump(pay, open(out, 'w'), indent=1)
print(json.dumps({k: pay[k] for k in ('tokSOut', 'ttftMs', 'outputTokens')} | {'samples': pay['samples']}))
