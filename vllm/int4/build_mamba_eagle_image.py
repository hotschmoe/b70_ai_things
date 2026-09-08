#!/usr/bin/env python3
"""Apply a Python-only PR 48375 backport to the exact installed R276 package."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from prepare_replica import IMAGE

PURELIB = '/opt/venv/lib/python3.12/site-packages/'
TARGET = 'vllm/v1/core/single_type_kv_cache_manager.py'
BEFORE = '1a0dedb76ed07c64fa780e10a89ff718c37975804bf70594a9577c6ecebd5787'
NATIVE = ['vllm/_xpu_ops.py', 'vllm/model_executor/layers/layernorm.py',
          'vllm_xpu_kernels/_xpu_C.abi3.so', 'torch/lib/libc10_xpu.so',
          'torch/lib/libtorch_xpu.so', 'torch/lib/libtorch-xpu-ops-sycltla.so']


def run(*args):
    return subprocess.check_output(list(args), text=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = run('docker', 'run', '--rm', '--entrypoint', 'cat', IMAGE, PURELIB + TARGET)
    assert hashlib.sha256(original.encode()).hexdigest() == BEFORE
    dest = args.out / TARGET
    dest.parent.mkdir(parents=True)
    dest.write_text(original)
    patch = Path(__file__).resolve().parent / 'patches/r276-mamba-eagle-drop.patch'
    subprocess.run(['git', 'apply', '--unsafe-paths', str(patch)], cwd=args.out, check=True)
    base_tag = 'b70/r276-prefix-base:20260908'
    tag = 'b70/r276-prefix-mamba-eagle:20260908'
    subprocess.run(['docker', 'tag', IMAGE, base_tag], check=True)
    (args.out / 'Dockerfile').write_text('FROM ' + base_tag + '\nCOPY ' + TARGET + ' ' + PURELIB + TARGET + '\n')
    subprocess.run(['docker', 'build', '-t', tag, str(args.out)], check=True)
    assert run('docker', 'image', 'inspect', base_tag, '--format', '{{.Id}}').strip() == IMAGE
    image = run('docker', 'image', 'inspect', tag, '--format', '{{.Id}}').strip()
    manifests = []
    for label, candidate in [('base', IMAGE), ('candidate', image)]:
        native = run('docker', 'run', '--rm', '--entrypoint', 'sha256sum', candidate,
                     *[PURELIB + f for f in NATIVE])
        (args.out / (label + '-native-sha256.txt')).write_text(native)
        manifests.append(native)
    assert manifests[0] == manifests[1], 'native stack mismatch'
    data = dict(base=IMAGE, image=image, target=TARGET, before=BEFORE,
                after=hashlib.sha256(dest.read_bytes()).hexdigest(),
                patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                upstream_pr='https://github.com/vllm-project/vllm/pull/48375',
                upstream_head='9e04ba754a6aecd6383074227e7fca101c1faf50',
                native_bytes_equal=True, qualified=False)
    (args.out / 'manifest.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
