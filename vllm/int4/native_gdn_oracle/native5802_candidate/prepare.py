#!/usr/bin/env python3
"""Freeze an exact native5802 candidate without changing numerical expectations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

BASE_SHA = '22ee62b3495d46cc8fa72bb4c291d8e36abd317814ee23e0f13fd5073bf125e0'
BASE_ROOT = Path('/mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/native-gdn-tp1-plan-card0')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('IMAGE', 'NATIVE_SHA'):
            node.value = ast.Constant(value='IDENTITY_ONLY')
    return ast.dump(tree, include_attributes=False)


def repin(source, old_image, old_native, image, native):
    assert source.count(old_image) == source.count(old_native) == 1
    candidate = source.replace(old_image, image).replace(old_native, native)
    assert normalized(source) == normalized(candidate), 'numerical oracle changed'
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--native-sha256', required=True)
    parser.add_argument('--pair-pass', type=Path, required=True)
    parser.add_argument('--pair-image-inspect', type=Path, required=True)
    parser.add_argument('--build-evidence', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', args.image)
    assert re.fullmatch(r'[0-9a-f]{64}', args.native_sha256)
    assert args.root.is_absolute() and args.output.is_absolute()
    assert not args.root.exists() and not args.output.exists()
    assert args.pair_pass.is_absolute() and args.pair_image_inspect.is_absolute()
    assert args.pair_pass.parent in args.pair_image_inspect.parents
    base = BASE_ROOT / 'inputs/oracle.py'
    assert sha(base) == BASE_SHA
    plan = json.loads((BASE_ROOT / 'plan.json').read_text())
    old_image, old_native, old_output = plan['image'], plan['native_sha256'], plan['output']
    assert args.image != old_image and args.native_sha256 != old_native
    candidate = repin(base.read_text(), old_image, old_native, args.image, args.native_sha256)
    evidence_sha = sha(args.build_evidence)
    inputs = args.root / 'inputs'
    inputs.mkdir(parents=True)
    oracle = inputs / 'oracle.py'
    oracle.write_text(candidate, encoding='ascii')
    here = Path(__file__).resolve().parent
    original_lifecycle = here.parent / 'lifecycle.py'
    assert sha(original_lifecycle) == plan['frozen_files'][str(original_lifecycle)]
    plan.update(image=args.image, native_sha256=args.native_sha256,
                output=str(args.output), pair_preflight_pass=str(args.pair_pass),
                pair_preflight_image_inspect=str(args.pair_image_inspect),
                oracle_sha256=sha(oracle), original_lifecycle=str(original_lifecycle),
                contract='unchanged original rolling6; all 12 checks; 128 steps',
                status='PREPARED_NOT_RUN: candidate pair health and explicit allocation required')
    plan['oracle_command'] = [arg.replace(old_image, args.image).replace(str(BASE_ROOT / 'inputs'), str(inputs)).replace(old_output, str(args.output)).replace('b70-native-gdn-tp1-oracle-card0', 'b70-native5802-gdn-oracle-card0') for arg in plan['oracle_command']]
    files = {p: d for p, d in plan['frozen_files'].items() if p != str(base)}
    for path in [original_lifecycle, here / 'lifecycle.py', Path(__file__).resolve(), oracle, args.build_evidence]:
        files[str(path)] = sha(path)
    plan['frozen_files'] = files
    plan['launch'] = ['bin/gpu-run', '--card', '0', 'python3', str(here / 'lifecycle.py'), str(args.root / 'plan.json'), '--run']
    (args.root / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (args.root / 'identity-delta.json').write_text(json.dumps(dict(original_oracle_sha256=BASE_SHA, candidate_oracle_sha256=sha(oracle), original_image=old_image, candidate_image=args.image, original_native_sha256=old_native, candidate_native_sha256=args.native_sha256, numerical_ast_identical=True, build_evidence_sha256=evidence_sha), indent=2)+'\n')
    print(args.root / 'plan.json')


if __name__ == '__main__':
    main()
