#!/usr/bin/env python3
"""Compile native HC model TUs in an independent overlay, with no DRM devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/vm_8tb/github/strata')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch', type=Path, action='append', required=True)
    p.add_argument('--ggml-headers', type=Path, default=Path('/mnt/vm_8tb/b70/research/strata-ggml-3cf03257'))
    p.add_argument('--leased', action='store_true')
    args = p.parse_args()
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == 'fb58e0dbc8399662c0e47c76578c6e878b14f6cf'
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'])
    out = Path(tempfile.mkdtemp(prefix='strata-native-hc-tus-', dir='/mnt/vm_8tb/b70/build'))
    result = dict(image=IMAGE, devices_exposed=False, scope='SYCL object compilation only; no GPU/model execution',
                  source_revision='fb58e0dbc8399662c0e47c76578c6e878b14f6cf', commands=[], objects=[], passed=False)
    result['patches'] = [dict(path=str(p.resolve()), sha256=sha(p)) for p in args.patch]
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    touched = set()
    try:
        # Migrated prefill includes these source headers by relative path.
        shutil.copytree(SOURCE / 'src/prefill', out / 'src/prefill')
        result['relative_header_source_hashes'] = {
            str(path.relative_to(out)): sha(path) for path in (out / 'src/prefill').rglob('*') if path.is_file()}
        ggml = args.ggml_headers.resolve()
        ggml_receipt = json.loads((ggml / 'source-receipt.json').read_text())
        assert ggml_receipt['revision'] == '3cf03257f219afbe7334045ff7c6a06ac68c627d'
        for rel, expected in ggml_receipt['files'].items():
            assert sha(ggml / rel) == expected
        result['ggml_headers'] = ggml_receipt
        for patch in args.patch:
            for rel in re.findall(r'^\+\+\+ b/(.+)$', patch.read_text(), re.M):
                path = out / rel
                touched.add(rel)
                if not path.exists() and (SOURCE / rel).exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((SOURCE / rel).read_bytes())
            subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=out, check=True)
            subprocess.run(['git', 'apply', str(patch.resolve())], cwd=out, check=True)
        result['source_hashes'] = {rel: sha(out / rel) for rel in sorted(touched)}
        tus = sorted(rel for rel in touched if rel.endswith('.cpp'))
        for built in [0, 1]:
            for i, rel in enumerate(tus):
                obj = f'tu-{built}-{i}.o'
                compiler = ['/opt/intel/oneapi/compiler/2026.1/bin/icpx', '-fsycl', '-std=c++20', '-O2',
                            '-DSTRATA_NATIVE_EXPERTS=1', '-I/work/sycl/include', '-I/work/include',
                            '-I/src/sycl/include', '-I/src/include', '-I/src/third_party/ggml',
                            '-I/src/third_party/ggml/include', '-I/src/third_party/ggml/src',
                            '-I/ggml/ggml/include',
                            '-c', '/work/' + rel, '-o', '/work/' + obj]
                if built:
                    compiler.insert(4, '-DSTRATA_SYCL_Q8_HC_BUILT=1')
                command = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
                           '-v', str(SOURCE) + ':/src:ro', '-v', str(out) + ':/work',
                           '-v', str(ggml) + ':/ggml:ro', IMAGE,
                           'exec ' + shlex.join(compiler)]
                result['commands'].append(command)
                with (out / f'tu-{built}-{i}.log').open('w') as log:
                    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                if done.returncode or not (out / obj).is_file():
                    raise RuntimeError(f'{rel} built={built} compile failed, log tu-{built}-{i}.log')
                result['objects'].append(dict(source=rel, native_built=built, sha256=sha(out / obj)))
                (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        result['passed'] = True
    except Exception as error:
        result['error'] = str(error)
    result['external_source_unchanged'] = not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'])
    result['passed'] = result['passed'] and result['external_source_unchanged']
    (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], receipt=str(out / 'receipt.json'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
