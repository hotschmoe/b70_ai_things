#!/usr/bin/env python3
"""ROOT-only source snapshot producer. No compilation, Docker, or model access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

EXCLUDED_DIRS = {'.git', 'node_modules', '.venv', '__pycache__'}
ROOT_PAYLOAD_DIR = 'models'
EXCLUDED_SUFFIXES = {'.gguf', '.safetensors', '.pt', '.pth', '.o', '.a', '.so', '.pyc'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def manifest(source):
    source = Path(source).resolve()
    records = []
    for root, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('build')
                         and not (Path(root) == source and d == ROOT_PAYLOAD_DIR))
        for d in dirs:
            if (Path(root) / d).is_symlink():
                raise ValueError('source directory symlink refused')
        for name in sorted(files):
            p = Path(root) / name
            if p.suffix in EXCLUDED_SUFFIXES:
                continue
            if p.is_symlink() or not p.is_file():
                raise ValueError('source file symlink/nonregular refused')
            records.append({'path': str(p.relative_to(source)), 'size': p.stat().st_size,
                            'mode': p.stat().st_mode & 0o777, 'sha256': sha(p)})
    records.sort(key=lambda x: x['path'])
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'files': records, 'file_count': len(records), 'total_bytes': sum(x['size'] for x in records),
            'manifest_sha256': digest}

def source_completeness(binding, plan):
    records = {item['path']: item for item in binding['files']}
    for category in ('source_evidence', 'essential_sources'):
        for path, expected in plan.get(category, {}).items():
            if path not in records or records[path]['sha256'] != expected:
                raise ValueError('required source missing or changed: ' + path)
    return True

def check_snapshot(source, plan):
    binding = manifest(source)
    source_completeness(binding, plan)
    if binding['manifest_sha256'] != plan['current_worktree_manifest_sha256']:
        raise ValueError('snapshot corpus changed')
    return binding

def git_metadata(source):
    # These commands run only when ROOT explicitly invokes this producer.
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'GIT_OPTIONAL_LOCKS': '0', 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': '/dev/null'}
    result = {}
    for key, args in [('head', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all']),
                      ('tracked', ['ls-files', '--stage']), ('dirty_patch', ['diff', '--no-ext-diff', '--no-textconv', '--binary', 'HEAD', '--']),
                      ('submodules', ['submodule', 'status', '--recursive'])]:
        result[key] = subprocess.check_output(['git', '-c', 'core.fsmonitor=false', '-C', str(source), *args], env=env).decode('utf-8')
    if any(line.startswith('160000 ') for line in result['tracked'].splitlines()):
        raise ValueError('gitlinks require a separate pinned source plan')
    return result

def prepare(source, output, plan):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output.exists():
        raise ValueError('output must be new and outside source')
    binding = manifest(source)
    source_completeness(binding, plan)
    if binding['manifest_sha256'] != plan['current_worktree_manifest_sha256']:
        raise ValueError('source corpus changed; prepare a new immutable plan')
    before = git_metadata(source)
    if before['head'].strip() != plan['expected_head']:
        raise ValueError('HEAD differs')
    output.mkdir(parents=True)
    snapshot = output / 'source'
    snapshot.mkdir()
    for item in binding['files']:
        src, dst = source / item['path'], snapshot / item['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if manifest(snapshot) != binding or manifest(source) != binding or git_metadata(source) != before:
        raise ValueError('source changed during snapshot; retain failed output')
    check_snapshot(snapshot, plan)
    (output / 'dirty.patch').write_text(before.pop('dirty_patch'), encoding='utf-8')
    receipt = {'schema': 1, 'source': str(source), 'snapshot': str(snapshot), 'source_binding': binding,
               'git': before, 'dirty_patch_sha256': sha(output / 'dirty.patch'),
               'producer_sha256': sha(__file__), 'plan': plan, 'snapshot_verified': True, 'source_completeness_verified': True,
               'actual_build': False, 'actual_inference': False, 'quality_qualified': False}
    (output / 'source-receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source')
    p.add_argument('--output')
    p.add_argument('--check-snapshot')
    p.add_argument('--plan', required=True)
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    if sha(__file__) != plan['producer_sha256']:
        raise ValueError('producer binding changed')
    if a.check_snapshot:
        if a.source or a.output:
            raise ValueError('check-only mode cannot prepare output')
        check_snapshot(a.check_snapshot, plan)
        print(json.dumps({'source_completeness_verified': True, 'actual_build': False}))
        return
    if not a.source or not a.output:
        raise ValueError('prepare requires source and new output')
    receipt = prepare(a.source, a.output, plan)
    print(json.dumps({'snapshot_verified': receipt['snapshot_verified'], 'actual_build': False}))

if __name__ == '__main__':
    main()
