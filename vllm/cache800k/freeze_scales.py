#!/usr/bin/env python3
"""Freeze INT4 target KV scales only after a healthy complete collection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fp8'))
from kv_campaign_calibrate import merge


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--preservation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mtp', type=int, default=0)
    args = p.parse_args()
    if args.out.exists():
        raise RuntimeError('refusing to overwrite frozen scales')
    root = args.calibration
    assert (root / 'exit.rc').read_text().strip() == '0'
    assert (root / 'WORKLOADS_PASSED').exists()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['args']['mtp'] == args.mtp
    assert manifest['args']['hook'] == 'record' and manifest['args']['eager']
    assert manifest['args']['prefix_off'] and manifest['args']['kv_dtype'] == 'auto'
    corpus = json.loads((root / '02-calibrate/corpus.json').read_text())
    rows = [json.loads(x) for x in (root / '02-calibrate/responses.jsonl').read_text().splitlines()]
    assert len(rows) == len(corpus) and len(rows) >= 262
    assert all(not r['error'] and r['usage'].get('completion_tokens', 0) > 0 for r in rows)
    source = json.loads((args.preservation / 'source-hashes.json').read_text())
    ops = next(v for k, v in source.items() if k.endswith('/_xpu_ops.py'))
    assert ops in (root / 'server.log').read_text()
    weights = json.loads((args.preservation / 'model-files.json').read_text())
    config_sha = weights['config.json']['sha256']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact = merge(sorted(root.glob('kv-record-*.json')), args.out, 1.1,
                     17 if args.mtp else 16,
                     'qwen3.8-27b/int4-autoround-gptq-relabel-r212')
    artifact['model_config_sha256'] = config_sha
    artifact['provenance'] = dict(image=manifest['image'], xpu_ops_sha256=ops,
        model_files=weights, source_sha256=manifest['source_sha256'],
        calibration_manifest_sha256=hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256((root / '02-calibrate/corpus.json').read_bytes()).hexdigest(),
        responses_sha256=hashlib.sha256((root / '02-calibrate/responses.jsonl').read_bytes()).hexdigest(),
        requests=len(rows), prompt_tokens=sum(r['usage'].get('prompt_tokens', 0) for r in rows),
        completion_tokens=sum(r['usage'].get('completion_tokens', 0) for r in rows),
        note='Synthetic text calibration. Periodic activation snapshots; not exhaustive final-step counts.')
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(path=str(args.out), sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                          layers=len(artifact['layers']), requests=len(rows)), indent=2))


if __name__ == '__main__':
    main()
