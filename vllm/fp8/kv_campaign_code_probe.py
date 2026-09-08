#!/usr/bin/env python3
"""Reuse the repository's sandboxed EvalPlus grader with campaign identity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
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
    from evalplus.data import get_human_eval_plus
    dataset = (json.dumps(get_human_eval_plus(), sort_keys=True, ensure_ascii=True) + '\n').encode()
    dataset_sha = hashlib.sha256(dataset).hexdigest()
    assert dataset_sha == 'b52c70fb955ba9c1139174f56d58b0c2804558acf7513553d4edb08ab5df818f', dataset_sha
    grader_image = subprocess.check_output(['docker', 'image', 'inspect', 'evalplus-sandbox:0.3.1',
                                           '--format', '{{.Id}}'], text=True).strip()
    assert grader_image == 'sha256:c0522083adc7557aab54abff248a4a4a9f7c32b4d9d66ddf9af05200ae5a3339', grader_image
    identity = common.check_endpoint(args.base)
    assert identity['ok'] and args.model in identity['models'], identity
    registry = common.load_models_config(repo / 'evals/configs/models.yaml')
    assert args.model in [m['served_model_id'] for m in registry['models']]
    args.out.mkdir(parents=True, exist_ok=False)
    sampling = {'temperature': 0., 'top_p': 1., 'seed': 42, 'max_tokens': 2048}
    ctx = SimpleNamespace(out_dir=args.out, endpoint=args.base, model_id=args.model, sampling=sampling)
    common.write_json(args.out / 'config.json', {'model': args.model, 'identity': identity,
                      'sampling': sampling, 'limit': args.limit, 'dataset': 'humaneval',
                      'dataset_sha256': dataset_sha, 'grader_image': grader_image,
                      'source_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in [Path(__file__).resolve(), Path(common.__file__), Path(tier1_code.__file__)]},
                      'thinking': False, 'git': common.get_git_sha()})
    result = tier1_code.run(ctx, limit=args.limit)
    common.write_json(args.out / 'summary.json', result)
    print(json.dumps(result), flush=True)
    return int(bool(result.get('error') or result.get('skipped')))


if __name__ == '__main__':
    raise SystemExit(main())
