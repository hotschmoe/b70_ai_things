#!/usr/bin/env python3
"""CPU-only fixture preparation; run with campaign tokenizers installed."""
import hashlib
import json
from pathlib import Path
import sys
from tokenizers import Tokenizer

repo = Path('/mnt/vm_8tb/github/b70_ai_things')
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
if (out / 'fixture.json').exists():
    raise FileExistsError('Refuse to replace an existing comparison fixture')
tokenizer = Tokenizer.from_file('/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext/tokenizer.json')
sources = ['evals/abfast/run_arm.py', 'evals/abfast/report.py',
           'evals/abfast/sysadmin/sysrun.py', 'evals/abfast/longrun/run_long.py',
           'strata/flash-next/night/nl.py', 'vllm/flashnext/harness/bench_latency.py']
corpus = '\n'.join('\nFILE: ' + name + '\n' + (repo / name).read_text() for name in sources)
tokens = tokenizer.encode(corpus, add_special_tokens=False).ids
assert len(tokens) > 20000, len(tokens)
context = tokenizer.decode(tokens[:19700], skip_special_tokens=False)
instruction = ('\n\nTASK: Review this code bundle for concurrency, cancellation, resource ownership, '
               'measurement validity, and failure handling. Give a detailed technical review with concrete '
               'examples and proposed Python fixes. Explain at least ten separate findings, with code and '
               'tests. Write a long answer of at least 2000 words. Start directly with your first finding.\n')
batches = []
for batch in range(3):
    prompts = [f'Comparison batch {batch}, independent stream {stream}. Review request {batch * 4 + stream:04d}.\n' +
               context + instruction for stream in range(4)]
    batches.append(dict(name='warmup' if batch == 0 else f'measured-{batch}', warmup=batch == 0,
                        prompts=prompts,
                        content_token_counts=[len(tokenizer.encode(p, add_special_tokens=False).ids) for p in prompts]))
fixture = dict(schema=1, description='Same four cold coding-review prompts per batch on all recipes; greedy, thinking off, output512',
               sources={name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in sources},
               batches=batches)
(out / 'fixture.json').write_text(json.dumps(fixture, indent=2, ensure_ascii=True) + '\n', encoding='ascii')
print(json.dumps({'fixture': str(out / 'fixture.json'), 'body_tokens': [b['content_token_counts'] for b in batches]}))
