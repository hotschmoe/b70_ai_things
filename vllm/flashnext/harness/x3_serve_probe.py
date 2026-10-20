"""Matched GGUF prefill A/B and coherence on the 35B proxy."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench_latency import identity, coherence, stream_completion, make_prompt, summarize_itl

ROOT = Path('/mnt/vm_8tb/b70/build/vllm-flashnext/x3')
BASE = 'http://127.0.0.1:18133'
MODEL = 'hotschmoe-dd'
report = {'identity': identity(BASE, MODEL, 'qwen3.6-35b-a3b-ud-iq3-xxs')}
assert report['identity']['first_ok'] and report['identity']['alias_ok'], report
print('IDENTITY', report, flush=True)
prompts = {n: make_prompt(BASE, MODEL, n, 'x3-matched-' + str(n))[0] for n in (1024,8192,12288)}
for mode, sizes in (('0', (1024,8192)), ('1', (1024,8192,12288,12288))):
    (ROOT/'prefill-mode').write_text(mode)
    report[mode] = {'coherence': coherence(BASE, MODEL, 300), 'prefill': []}
    assert report[mode]['coherence']['ok'], report[mode]['coherence']
    # Warm one vendor-path step before reporting new-path TTFT.
    if mode == '1':
        stream_completion(BASE, MODEL, prompts[1024], 1, 300)
    for n in sizes:
        row = stream_completion(BASE, MODEL, prompts[n], 1, 300)
        summary = summarize_itl(row)
        report[mode]['prefill'].append({'target': n, 'measured': summary})
        print('RESULT', mode, n, json.dumps(summary), flush=True)
        (ROOT/'serve-ab.json').write_text(json.dumps(report, indent=2))
# Deterministic token IDs/text under both prefill paths on a nontrivial prompt.
texts = []
for mode in ('0', '1'):
    (ROOT/'prefill-mode').write_text(mode)
    row = stream_completion(BASE, MODEL, prompts[1024], 32, 300)
    texts.append(row['text'])
report['matched_long_prompt_texts'] = texts
report['matched_long_prompt_equal'] = texts[0] == texts[1]
with ThreadPoolExecutor(2) as pool:
    rows=list(pool.map(lambda _:stream_completion(BASE,MODEL,'The capital of France is',16,300),range(2)))
report['concurrent_texts']=[row['text'] for row in rows]
assert all('paris' in text.lower() for text in report['concurrent_texts'])
assert report['concurrent_texts'][0]==report['concurrent_texts'][1]
(ROOT/'serve-ab.json').write_text(json.dumps(report, indent=2))
print('GREEDY_LONG_EQUAL', report['matched_long_prompt_equal'], flush=True)
