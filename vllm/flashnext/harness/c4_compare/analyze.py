#!/usr/bin/env python3
"""Offline steady-overlap TG; require exact text token/usage count agreement."""
import bisect
import json
from pathlib import Path
import statistics
import sys
from tokenizers import Tokenizer

root = Path(sys.argv[1])
tokenizers = {
    'flashnext': '/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext/tokenizer.json',
    'strata': '/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext/tokenizer.json',
    'qwen27b': '/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212/tokenizer.json',
}
results = []
for path in sorted(root.glob('*/bench/summary.json')):
    armroot = path.parent.parent
    if armroot.is_symlink():
        continue
    metadata = json.loads((armroot / 'metadata.json').read_text())
    arm = metadata['arm']
    tokenizer = Tokenizer.from_file(tokenizers[arm])
    rows = []
    for batchpath in sorted(path.parent.glob('measured-*.json')):
        batch = json.loads(batchpath.read_text())
        streams = batch['streams']
        begin = max(r['first'] for r in streams)
        end = min(r['last'] for r in streams)
        checks, count = [], 0
        for stream in streams:
            encoding = tokenizer.encode(stream['text'], add_special_tokens=False)
            offsets = encoding.offsets
            # Assign a token to the SSE event by which its complete text arrived.
            char_ends, times, length = [], [], 0
            for event in stream['events']:
                length += len(event['text'])
                char_ends.append(length)
                times.append(event['t'])
            checked = len(encoding.ids) == stream['usage']['completion_tokens']
            checks.append(dict(encoded_tokens=len(encoding.ids), usage_tokens=stream['usage']['completion_tokens'], exact=checked))
            count += sum(begin < times[min(bisect.bisect_left(char_ends, last), len(times)-1)] <= end
                         for first, last in offsets)
        valid = end > begin and all(c['exact'] for c in checks)
        rows.append(dict(batch=batch['batch'], steady_window_s=end - begin,
                         tokens_in_steady_window=count, token_usage_checks=checks,
                         steady_tg_valid=valid,
                         steady_tg_aggregate_tps=count / (end - begin) if valid else None))
    summary = json.loads(path.read_text())
    summary.update(arm=arm, tag=metadata['tag'], steady=rows,
                   steady_tg_aggregate_tps=statistics.median(r['steady_tg_aggregate_tps'] for r in rows)
                       if all(r['steady_tg_valid'] for r in rows) else None,
                   steady_definition='Retokenized output tokens arriving after all four first content events and through earliest final content; require exact reencoded output count == server usage for every stream')
    results.append(summary)
(root / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n', encoding='ascii')
print(json.dumps(results, indent=2))
