#!/usr/bin/env python3
"""Package only the guarded Python offload repair over the frozen native image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASE = 'sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152'
BASE_TAG = 'b70/kv-native-base:f46780e1a72c'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tag', default='b70/vllm-r187:kv-offload-20260908')
    p.add_argument('--native-manifest', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent / 'kv_hooks'
    hashes = {}
    for name in ('sitecustomize.py', 'kv_offload_group_fix.py'):
        shutil.copy2(source / name, args.out / name)
        hashes[name] = hashlib.sha256((args.out / name).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    # BuildKit treats a bare image ID in FROM as a registry name. Give the
    # already installed immutable image a local build alias instead.
    subprocess.run(['docker', 'tag', BASE, BASE_TAG], check=True)
    dockerfile = '\n'.join([
        'FROM ' + BASE_TAG,
        'COPY sitecustomize.py kv_offload_group_fix.py /opt/b70-kv/hooks/',
        'ENV PYTHONPATH=/opt/b70-kv/hooks B70_OFFLOAD_GROUP_FIX=1',
        'LABEL b70.kv-offload-hooks-sha256="' + fingerprint + '"',
        '',
    ])
    (args.out / 'Dockerfile').write_text(dockerfile)
    subprocess.run(['docker', 'build', '-t', args.tag, str(args.out)], check=True)
    if subprocess.check_output(['docker', 'image', 'inspect', BASE_TAG,
                                '--format', '{{.Id}}'], text=True).strip() != BASE:
        raise RuntimeError('native build alias changed during build')
    image = subprocess.check_output(['docker', 'image', 'inspect', args.tag,
                                    '--format', '{{.Id}}'], text=True).strip()
    expected = args.native_manifest.read_text()
    paths = [line.split(maxsplit=1)[1] for line in expected.splitlines() if line.strip()]
    actual = subprocess.check_output(['docker', 'run', '--rm', '--entrypoint',
                                     'sha256sum', image, *paths], text=True)
    (args.out / 'native-sha256.txt').write_text(actual)
    if actual != expected:
        raise RuntimeError('derived image changed native runtime bytes')
    manifest = {'base': BASE, 'image': image, 'tag': args.tag,
                'hook_sha256': hashes, 'hook_fingerprint': fingerprint,
                'native_bytes_equal': True, 'qualified': False}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
