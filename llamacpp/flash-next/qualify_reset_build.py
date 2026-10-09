#!/usr/bin/env python3
"""Add test-only fixtures to a finished isolated reset build, preserving serving artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-receipt', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    original = json.loads(args.build_receipt.read_text())
    assert original['build_rc'] == 0 and original['baseline_unchanged'] and original['external_source_unchanged']
    assert original['image'] == IMAGE and original['devices_exposed'] is False and not original['cmake_mismatches']
    plan = json.loads(args.plan.read_text())
    assert plan['source_revision'] == original['source_revision']
    build = Path(original['build'])
    source = Path(original['source_copy'])
    out = args.build_receipt.parent
    receipt_path = out / 'qualification-build-receipt.json'
    assert not receipt_path.exists()
    result = dict(original)
    result.update(build_rc=1, status='test augmentation pending',
                  original_build_receipt_path=str(args.build_receipt.resolve()),
                  original_build_receipt_sha256=sha(args.build_receipt),
                  preserved_original_binary_sha256=original['binary_sha256'],
                  plan_path=str(args.plan.resolve()), plan_sha256=sha(args.plan), patches=plan['patches'],
                  qualifier_sha256=sha(Path(__file__)), augmentation_commands=[])
    (out / 'qualification-controller.py').write_bytes(Path(__file__).read_bytes())
    try:
        for path, expected in original['binary_sha256'].items():
            assert sha(Path(path)) == expected
        previous = {spec['path']: spec['sha256'] for spec in original['patches']}
        current = {spec['path']: spec['sha256'] for spec in plan['patches']}
        assert all(current.get(path) == expected for path, expected in previous.items())
        additions = [spec for spec in plan['patches'] if spec['path'] not in previous]
        assert len(additions) == 1
        patch = REPO / additions[0]['path']
        assert sha(patch) == additions[0]['sha256']
        assert re.findall(r'^\+\+\+ b/(.+)$', patch.read_text(), re.M) == ['tests/test-backend-ops.cpp']
        for argv in [['git', '-C', str(source), 'apply', '--check', str(patch)],
                     ['git', '-C', str(source), 'apply', str(patch)]]:
            result['augmentation_commands'].append(argv)
            subprocess.run(argv, check=True)
        script = 'set -eo pipefail\ncmake --build /build --target test-backend-ops -j 4'
        command = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
                   '-v', str(source) + ':/src:ro', '-v', str(build) + ':/build', IMAGE, script]
        result['augmentation_commands'].append(command)
        with (out / 'qualification-build.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        for path, expected in original['binary_sha256'].items():
            assert sha(Path(path)) == expected, 'Serving artifact changed: ' + path
        assert (build / 'bin/test-backend-ops').is_file()
        result['binary_sha256'] = dict(original['binary_sha256'])
        result['binary_sha256'][str(build / 'bin/test-backend-ops')] = sha(build / 'bin/test-backend-ops')
        assert sha(build / 'CMakeCache.txt') == original['cmake_cache_sha256']
        result['patched_source_sha256'] = {rel: sha(source / rel) for rel in plan['overlay_files']}
        result['source_copy_status'] = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
        result['build_rc'] = 0
        result['status'] = 'built; same serving artifacts preserved; same-build GPU primitives/reset observation remain unqualified'
    except Exception as error:
        result['error'] = str(error)
        result['status'] = 'test augmentation failed'
    result['external_source_unchanged'] = not subprocess.check_output(['git', '-C', original['source_tree'], 'status', '--porcelain'])
    if not result['external_source_unchanged']:
        result['build_rc'] = 1
    receipt_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['build_rc'] == 0, receipt=str(receipt_path))), flush=True)
    return result['build_rc']


if __name__ == '__main__':
    sys.exit(main())
