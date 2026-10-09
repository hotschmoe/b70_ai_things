#!/usr/bin/env python3
"""Actual pinned Python-runtime import paths; no device or model payload mount."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:002f80b6cf11b107d2427fe41a3a87958535e3be2a2cc3fd457c82688ae4acc2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT / 'strata/flash-next/combined-numerical-batch-engine-build-plan-v2.json'
    plan = json.loads(plan_path.read_text())
    source = out / 'source'
    subprocess.run(['git', 'clone', '--local', '--no-hardlinks', '--no-checkout', plan['source_root'], str(source)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', plan['source_revision']], check=True, capture_output=True)
    for row in plan['patches']:
        patch = ROOT / row['path']
        assert sha(patch) == row['sha256']
        subprocess.run(['git', '-C', str(source), 'apply', '--check', str(patch)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(source), 'apply', str(patch)], check=True, capture_output=True)
    for name, want in plan['expected_patched_source_sha256'].items():
        assert sha(source / name) == want, name
    settings = {'vocab_size': 248320, 'special_ids': {'tokenizer.ggml.eos_token_id': 248046}}
    (out / 'tokenizer').mkdir()
    (out / 'tokenizer/tokenizer.json').write_text(json.dumps(settings) + '\n', encoding='ascii')
    (out / 'artifact-identity.json').write_text(json.dumps({'tokenizer_files': {'tokenizer.json': sha(out / 'tokenizer/tokenizer.json')}}) + '\n', encoding='ascii')
    (out / 'config.json').write_text(json.dumps({'tokenizer': '/work/tokenizer', 'artifact_identity_manifest': '/work/artifact-identity.json'}) + '\n', encoding='ascii')
    previous = Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T144515Z-k4gb2xtx/source')
    calls = [
        ('old_package_negative', previous, ['-c', 'from serve import server'], False),
        ('package_identity', source, ['-c', 'from serve import server; from serve.batch_request_identity import RequestField; assert server.RequestField is RequestField; print("CANONICAL_PACKAGE_IMPORT_PASS")'], True),
        ('module_help', source, ['-m', 'serve.server', '--help'], True),
        ('script_help', source, ['/src/serve/server.py', '--help'], True),
        ('actual_trace_help', source, ['/controller/c1_api_trace.py', '--config', '/work/config.json', '--help'], True),
    ]
    observations = []
    for label, tree, command, passing in calls:
        argv = ['docker', 'run', '--rm', '--network', 'none', '--user', str(os.getuid()) + ':' + str(os.getgid()), '--workdir', '/src', '--entrypoint', '/opt/b70-c1-python/bin/python',
                '-e', 'B70_C1_TRACE=/work/unwritten-trace.jsonl', '-v', str(tree) + ':/src:ro', '-v', str(ROOT / 'strata/flash-next') + ':/controller:ro', '-v', str(out) + ':/work:ro', IMAGE, *command]
        result = subprocess.run(argv, capture_output=True, timeout=60)
        (out / (label + '.stdout')).write_bytes(result.stdout)
        (out / (label + '.stderr')).write_bytes(result.stderr)
        accepted = result.returncode == 0 if passing else result.returncode != 0 and b"No module named 'batch_request_identity'" in result.stderr
        observations.append({'case': label, 'command': argv, 'return_code': result.returncode, 'passed': accepted, 'expected_pass': passing,
                             'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(), 'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()})
        assert accepted, label + ': ' + result.stderr.decode('ascii', errors='backslashreplace')
    assert not (out / 'unwritten-trace.jsonl').exists()
    receipt = {'CONFIG': 'Actual package/script/trace imports in pinned Python runtime; synthetic settings only, no GPU devices or model mount',
               'COMMAND': observations, 'RESULT': 'All four corrected entry modes pass; old package import reproduces actual failure',
               'VERDICT': 'API import compatibility only; no model math, GPU lifecycle, serving or shelf qualification',
               'passed': True, 'plan_sha256': sha(plan_path), 'test_sha256': sha(Path(__file__)), 'source_files': len(plan['expected_patched_source_sha256']),
               'runtime_image': IMAGE, 'gpu_executed': False, 'model_payload_read': False}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='ascii')
    print('PASS old failure negative and4 real package/script/trace imports; noGPU/model payload')


if __name__ == '__main__':
    main()
