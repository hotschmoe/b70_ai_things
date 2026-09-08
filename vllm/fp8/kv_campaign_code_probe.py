#!/usr/bin/env python3
"""Reuse the repository's sandboxed EvalPlus grader with campaign identity."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:18125/v1')
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=32)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'evals/orchestrator'))
    import common
    import tier1_code
    identity = common.check_endpoint(args.base)
    assert identity['ok'] and args.model in identity['models'], identity
    registry = common.load_models_config(repo / 'evals/configs/models.yaml')
    assert args.model in [m['served_model_id'] for m in registry['models']]
    args.out.mkdir(parents=True, exist_ok=False)
    sampling = {'temperature': 0., 'top_p': 1., 'seed': 42, 'max_tokens': 2048}
    ctx = SimpleNamespace(out_dir=args.out, endpoint=args.base, model_id=args.model, sampling=sampling)
    common.write_json(args.out / 'config.json', {'model': args.model, 'identity': identity,
                      'sampling': sampling, 'limit': args.limit, 'dataset': 'humaneval',
                      'thinking': False, 'git': common.get_git_sha()})
    result = tier1_code.run(ctx, limit=args.limit)
    common.write_json(args.out / 'summary.json', result)
    print(json.dumps(result), flush=True)
    return int(bool(result.get('error') or result.get('skipped')))


if __name__ == '__main__':
    raise SystemExit(main())
