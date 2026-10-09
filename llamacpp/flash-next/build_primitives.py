#!/usr/bin/env python3
"""Build a test-only production-shape overlay; backend/model source stays pristine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/vm_8tb/github/llama.cpp-flashnext')
BUILD = Path('/mnt/vm_8tb/b70/build/flashnext-llamacpp')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    plan = json.loads((REPO / 'llamacpp/flash-next/primitive-extra-plan.json').read_text())
    original = SOURCE / 'tests/test-backend-ops.cpp'
    patch = REPO / plan['patch']
    assert sha(original) == plan['source_file_sha256']
    assert sha(patch) == plan['patch_sha256']
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == plan['source_revision']
    directory = BUILD / ('primitive-overlay-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    (directory / 'tests').mkdir(parents=True, exist_ok=False)
    overlay = directory / 'tests/test-backend-ops.cpp'
    shutil.copy2(original, overlay)
    subprocess.run(['patch', '--batch', '-d', str(directory), '-p1', '-i', str(patch)], check=True)
    server_before = sha(BUILD / 'bin/llama-server')
    if not (BUILD / 'bin/test-backend-ops-upstream').exists():
        shutil.copy2(BUILD / 'bin/test-backend-ops', BUILD / 'bin/test-backend-ops-upstream')
    with (directory / 'build.log').open('w') as log:
        result = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--user', str(os.getuid()) + ':' + str(os.getgid()),
                                 '-v', str(SOURCE) + ':/src:ro', '-v', str(BUILD) + ':/build',
                                 '-v', str(overlay) + ':/src/tests/test-backend-ops.cpp:ro', args.image,
                                 'cmake --build /build --target test-backend-ops -j 4'], stdout=log, stderr=subprocess.STDOUT)
    receipt = {'image': args.image, 'source_revision': plan['source_revision'], 'patch_sha256': sha(patch),
               'test_source_sha256': sha(overlay), 'build_rc': result.returncode,
               'server_sha256_before': server_before, 'server_sha256_after': sha(BUILD / 'bin/llama-server'),
               'test_binary_sha256': sha(BUILD / 'bin/test-backend-ops')}
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(directory)
    return result.returncode or int(receipt['server_sha256_before'] != receipt['server_sha256_after'])


if __name__ == '__main__':
    raise SystemExit(main())
