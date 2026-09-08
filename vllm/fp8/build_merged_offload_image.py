#!/usr/bin/env python3
"""Backport three merged scheduler fixes without changing the native stack."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from build_kv_offload_image import BASE, BASE_TAG

TARGET = 'vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py'
PURELIB = '/opt/venv/lib/python3.12/site-packages/'
BEFORE = '616e7fd4cb0d09064cbc4d5735f607b37964c6be3b81e26de00d5913e0a9a3e3'
AFTER = '0d2fd9a20e02e2b1560d757658ded738ae6c5d7cc292bf75d20c49636f6338b0'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--native-manifest', type=Path, required=True)
    p.add_argument('--tag', default='b70/vllm-r187:offload-merged-20260908')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'cat', BASE, PURELIB + TARGET])
    assert hashlib.sha256(data).hexdigest() == BEFORE, 'unmatched scheduler source'
    dest = args.out / TARGET
    dest.parent.mkdir(parents=True)
    dest.write_bytes(data)
    patch = Path(__file__).resolve().parent / 'patches/offload-merged-52771-52807-54288.patch'
    shutil.copy2(patch, args.out / patch.name)
    subprocess.run(['git', 'apply', '--unsafe-paths', str(patch)], cwd=args.out, check=True)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == AFTER
    subprocess.run(['docker', 'tag', BASE, BASE_TAG], check=True)
    (args.out / 'Dockerfile').write_text('\n'.join([
        'FROM ' + BASE_TAG,
        'COPY ' + TARGET + ' ' + PURELIB + TARGET,
        'LABEL b70.offload-merged-prs="52771,52807,54288"', '',
    ]))
    subprocess.run(['docker', 'build', '-t', args.tag, str(args.out)], check=True)
    assert subprocess.check_output(['docker', 'image', 'inspect', BASE_TAG, '--format', '{{.Id}}'], text=True).strip() == BASE
    image = subprocess.check_output(['docker', 'image', 'inspect', args.tag, '--format', '{{.Id}}'], text=True).strip()
    expected = args.native_manifest.read_text()
    paths = [line.split(maxsplit=1)[1] for line in expected.splitlines() if line.strip()]
    actual = subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'sha256sum', image, *paths], text=True)
    (args.out / 'native-sha256.txt').write_text(actual)
    assert actual == expected, 'native bytes changed'
    manifest = dict(base=BASE, image=image, tag=args.tag, scheduler_before=BEFORE,
                    scheduler_after=AFTER, patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                    merged_prs=[52771, 52807, 54288], native_bytes_equal=True, qualified=False)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
