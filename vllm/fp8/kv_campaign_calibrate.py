#!/usr/bin/env python3
"""Collect a reproducible synthetic calibration corpus, or merge TP scales."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from kv_campaign_probe import stream, request, long_prompt


TOPICS = [
    'Python iterator and generator error handling', 'Rust ownership and lifetimes',
    'Go goroutines with cancellation and bounded channels', 'TypeScript async APIs and retries',
    'C++ vector invalidation and RAII', 'SQL joins, nulls and transaction isolation',
    'JSON tool-call arguments and schema validation', 'shell scripts with careful quoting',
    'unit tests for parsing and serialization', 'filesystem traversal and symbolic links',
    'HTTP streaming and partial responses', 'database migration rollback procedures',
    'Unicode normalization and string slicing', 'sorting and searching invariants',
    'matrix multiplication and numerical conditioning', 'probability and combinatorics',
    'geometric proofs and coordinate transformations', 'differentiation and integration',
    'distributed consensus and leader election', 'cache eviction and backpressure',
    'network timeout diagnosis using logs', 'debugging concurrent queue races',
    'poetry with internal rhyme', 'a historical essay on ocean navigation',
    'explain photosynthesis to a beginner', 'compare English and Spanish grammar',
    'translate an everyday dialogue into French', 'write a German technical summary',
    'write a Japanese greeting and explain it in English', 'a structured project handoff',
    'an agent workflow with tool results and corrections', 'a configuration file review',
]


def merge(files, out, headroom):
    if not math.isfinite(headroom) or headroom < 1:
        raise ValueError('headroom must be finite and >= 1')
    ranks = {}; hashes = {}
    for path in files:
        raw = path.read_bytes(); data = json.loads(raw)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for layer, rec in data.items():
            rank = rec['rank']
            if rank not in (0, 1) or rec['n'] < 1 or rec['tokens'] < 1:
                raise ValueError('invalid rank or empty observation')
            if layer in ranks.setdefault(rank, {}):
                raise ValueError('duplicate layer/rank observation')
            if any(not math.isfinite(rec[k]) or rec[k] < 0 for k in ('q_amax', 'k_amax', 'v_amax')):
                raise ValueError('invalid amax')
            ranks[rank][layer] = rec
    if set(ranks) != {0, 1} or set(ranks[0]) != set(ranks[1]):
        raise ValueError('missing or mismatched TP layer coverage')
    if len(ranks[0]) != 17:
        raise ValueError('expected 16 full-attention target layers plus 1 MTP layer')
    layers = {name: {k + '_scale': max(max(ranks[r][name][k + '_amax'] for r in (0, 1)) / 448 * headroom, 1e-6) for k in ('q', 'k', 'v')} for name in ranks[0]}
    artifact = {'schema': 'b70.qwen38-official-fp8-kv-scales.v1', 'weights': 'qwen3.8-27b/fp8-official', 'headroom': headroom, 'sources': hashes, 'observations': ranks, 'layers': layers}
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
    return artifact


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18125')
    p.add_argument('--model')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--merge', type=Path, nargs='+')
    p.add_argument('--headroom', type=float, default=1.)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--long', action='store_true')
    args = p.parse_args()
    if args.merge:
        artifact = merge(args.merge, args.out, args.headroom)
        print('Merged %d layers on both ranks' % len(artifact['layers']))
        return
    assert args.model in [m['id'] for m in json.loads(request(args.base, '/v1/models'))['data']]
    args.out.mkdir(parents=True, exist_ok=False)
    corpus = []
    for i in range(args.samples):
        topic = TOPICS[i % len(TOPICS)]
        style = ['Give a concrete worked example.', 'Find a subtle bug and correct it.', 'Explain the tradeoffs.', 'Return structured JSON with examples.', 'Write a concise implementation.', 'Write tests and explain edge cases.', 'Give a step-by-step derivation.', 'Simulate a user/assistant discussion.'][i // len(TOPICS) % 8]
        corpus.append({'id': i, 'prompt': f'Case {i}: {topic}. {style} Use the numbers {i+11} and {i+29} when an example needs numbers.', 'limit': 48})
    if args.long:
        for tokens in [32000, 96000, 150000, 180000]:
            prompt, _ = long_prompt(tokens, 77)
            corpus.append({'id': 'long' + str(tokens), 'prompt': prompt, 'limit': 32})
        for topic in TOPICS[:2]:
            corpus.append({'id': 'continuation-' + topic, 'prompt': 'Write a comprehensive tutorial with code and many examples about ' + topic, 'limit': 4096})
    serialized = json.dumps(corpus, ensure_ascii=True, indent=2)
    (args.out / 'corpus.json').write_text(serialized + '\n')
    (args.out / 'corpus.sha256').write_text(hashlib.sha256(serialized.encode()).hexdigest() + '\n')
    for item in corpus:
        row = stream(args.base, args.model, item['prompt'], item['limit'], salt='calibration-v1', timeout=900, thinking=isinstance(item['id'], int) and item['id'] % 4 == 0)
        row['id'] = item['id']
        with (args.out / 'responses.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=True) + '\n')
        if row['error'] or not row['usage'].get('completion_tokens'):
            raise RuntimeError('calibration request failed: ' + ascii(row))
        print('Collected ' + str(item['id']), flush=True)


if __name__ == '__main__':
    main()
