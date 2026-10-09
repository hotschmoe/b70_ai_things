#!/usr/bin/env python3
"""CPU-only dependency and native GPU library identity probe."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def gpu_libraries():
    roots = [Path('/usr/lib/x86_64-linux-gnu'), Path('/opt/intel/oneapi')]
    prefixes = ('libze_', 'libsycl', 'libur_', 'libigc', 'libigdfcl', 'libigdgmm')
    files = {}
    for root in roots:
        for path in root.rglob('*.so*'):
            if path.name.startswith(prefixes) and path.is_file():
                files[str(path)] = sha(path)
    if not any('libze_intel_gpu' in p for p in files) or not any('libsycl' in p for p in files):
        raise RuntimeError('GPU library census is incomplete')
    return dict(sorted(files.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu-libraries', action='store_true')
    ap.add_argument('--receipt', type=Path)
    ap.add_argument('--before', type=Path)
    a = ap.parse_args()
    libraries = gpu_libraries()
    if a.gpu_libraries:
        print(json.dumps(libraries, sort_keys=True))
        return
    before = json.loads(a.before.read_text())
    if libraries != before:
        raise RuntimeError('Installing isolated Python dependencies changed GPU libraries')
    deps = {name: version(name) for name in ['regex', 'jinja2', 'markupsafe']}
    if deps != {'regex': '2026.9.10', 'jinja2': '3.1.6', 'markupsafe': '3.0.3'}:
        raise RuntimeError('Python dependency pins differ')
    if platform.python_version() != '3.12.3':
        raise RuntimeError('Runtime Python changed from the base image')
    a.receipt.write_text(json.dumps({'schema': 1, 'passed': True, 'python_versions':
        {'python': platform.python_version(), **deps}, 'gpu_libraries_unchanged': True,
        'gpu_libraries': libraries}, indent=2) + '\n')


if __name__ == '__main__':
    main()
