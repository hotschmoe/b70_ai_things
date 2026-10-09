#!/usr/bin/env python3
"""Build the default-off cache byte diagnostic in an independent source/build pair."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/vm_8tb/github/llama.cpp-flashnext')
BASELINE = Path('/mnt/vm_8tb/b70/build/flashnext-llamacpp')
BUILD_ROOT = BASELINE.parent
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def read_cache(path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^([^/#][^:]*):([^=]+)=(.*)$', line)
        if match:
            result[match[1]] = (match[2], match[3])
    return result


def baseline_artifacts():
    paths = [BASELINE / 'bin/llama-server', BASELINE / 'CMakeCache.txt']
    paths += sorted((BASELINE / 'bin').glob('lib*.so*'))
    return {str(path): sha(path) for path in paths if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default=IMAGE, help='Exact previously pinned build image ID')
    parser.add_argument('--jobs', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--leased', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.leased:
        os.execv(str(REPO / 'bin/gpu-run'),
                 ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        lock = os.environ.get('B70_GPU_LOCK', '/mnt/vm_8tb/b70/gpu.lock') + '.' + str(card)
        if not os.path.samefile('/proc/self/fd/' + str(fd), lock):
            raise RuntimeError('Missing inherited whole-host GPU lease')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    plan_path = REPO / 'llamacpp/flash-next/cache-byte-invariant-plan.json'
    plan = json.loads(plan_path.read_text())
    patch = REPO / plan['patch']
    if sha(patch) != plan['patch_sha256']:
        raise RuntimeError('Diagnostic patch hash mismatch')
    revision = output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'])
    if revision != plan['source_revision'] or output(['git', '-C', str(SOURCE), 'status', '--porcelain']):
        raise RuntimeError('Source must be clean at the reviewed revision')
    for name, expected in plan['source_file_sha256'].items():
        if sha(SOURCE / name) != expected:
            raise RuntimeError('Source hash mismatch: ' + name)
    image = json.loads(output(['docker', 'image', 'inspect', args.image]))
    if image[0]['Id'] != IMAGE:
        raise RuntimeError('Build image differs from the pinned baseline image')

    cache = read_cache(BASELINE / 'CMakeCache.txt')
    required = {'CMAKE_BUILD_TYPE': 'Release', 'GGML_SYCL': 'ON', 'GGML_SYCL_DNN': 'ON',
                'GGML_SYCL_F16': 'OFF', 'LLAMA_OPENSSL': 'ON', 'LLAMA_BUILD_TESTS': 'ON',
                'BUILD_SHARED_LIBS': 'ON', 'CMAKE_GENERATOR': 'Ninja',
                'CMAKE_C_COMPILER': '/opt/intel/oneapi/compiler/2026.1/bin/icx',
                'CMAKE_CXX_COMPILER': '/opt/intel/oneapi/compiler/2026.1/bin/icpx',
                'DNNL_DIR': '/opt/intel/oneapi/dnnl/2026.0/lib/cmake/dnnl'}
    for key, expected in required.items():
        if cache.get(key, ('', None))[1] != expected:
            raise RuntimeError('Baseline configuration mismatch: ' + key)
    flags = {key: value for key, (kind, value) in cache.items()
             if kind in ('BOOL', 'STRING') and key.startswith(('GGML_', 'LLAMA_'))}
    for key in required:
        if key != 'CMAKE_GENERATOR':
            flags[key] = cache[key][1]
    for key in ('CMAKE_C_FLAGS', 'CMAKE_CXX_FLAGS', 'CMAKE_C_FLAGS_RELEASE', 'CMAKE_CXX_FLAGS_RELEASE'):
        flags[key] = cache[key][1]

    directory = Path(tempfile.mkdtemp(prefix='flashnext-cache-diagnostic-' +
                         time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-', dir=BUILD_ROOT))
    source_copy = directory / 'source'
    build = directory / 'build'
    build.mkdir()
    before = baseline_artifacts()
    receipt = {'schema': 1, 'source_revision': revision, 'source_tree': str(SOURCE),
               'source_copy': str(source_copy), 'build': str(build), 'image': IMAGE,
               'plan_sha256': sha(plan_path), 'patch_sha256': sha(patch),
               'builder_sha256': sha(Path(__file__)), 'cmake_flags': flags,
               'baseline_artifacts_before': before, 'devices_exposed': False,
               'runtime_required_env': {'GGML_SYCL_ENABLE_OPT': '0', 'LLAMA_MOE_VERIFY_BYTES': '1'},
               'status': 'preparing', 'build_rc': None}
    (directory / 'image-inspect.json').write_text(json.dumps(image, indent=2) + '\n')
    (directory / 'baseline-CMakeCache.txt').write_bytes((BASELINE / 'CMakeCache.txt').read_bytes())
    print('CONFIG diagnostic=' + str(directory), flush=True)
    rc = 1
    try:
        with (directory / 'build.log').open('w') as log:
            def run(argv):
                receipt.setdefault('commands', []).append(argv)
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
            # Independent git objects preserve build identity without writing to the baseline repository.
            run(['git', 'clone', '--local', '--no-hardlinks', '--no-checkout', str(SOURCE), str(source_copy)])
            run(['git', '-C', str(source_copy), 'checkout', '--detach', revision])
            run(['git', '-C', str(source_copy), 'apply', '--check', str(patch)])
            run(['git', '-C', str(source_copy), 'apply', str(patch)])
            receipt['patched_source_sha256'] = {name: sha(source_copy / name) for name in plan['overlay_files']}
            receipt['source_copy_status'] = output(['git', '-C', str(source_copy), 'status', '--porcelain'])
            configure = ['cmake', '-S', '/src', '-B', '/build', '-G', 'Ninja']
            configure += ['-D' + key + '=' + value for key, value in sorted(flags.items())]
            script = '\n'.join([
                'set -eo pipefail',
                'if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh; fi',
                'set -u', 'export LC_ALL=C',
                'icpx --version > /build/compiler-version.txt',
                'dpkg-query -W > /build/packages.txt',
                shlex.join(configure),
                'cmake --build /build --target llama-server -j ' + str(args.jobs),
            ])
            receipt['container_script'] = script
            run(['docker', 'run', '--rm', '--network', 'none', '--user', str(os.getuid()) + ':' + str(os.getgid()),
                 '-v', str(source_copy) + ':/src:ro', '-v', str(build) + ':/build', IMAGE, script])
        actual = read_cache(build / 'CMakeCache.txt')
        mismatches = {key: {'expected': value, 'actual': actual.get(key)}
                      for key, value in flags.items() if actual.get(key, ('', None))[1] != value}
        receipt['cmake_mismatches'] = mismatches
        if mismatches:
            raise RuntimeError('Diagnostic configuration differs from baseline flags')
        artifacts = [build / 'bin/llama-server', *sorted((build / 'bin').glob('lib*.so*'))]
        receipt['binary_sha256'] = {str(path): sha(path) for path in artifacts if path.is_file()}
        receipt['cmake_cache_sha256'] = sha(build / 'CMakeCache.txt')
        receipt['status'] = 'built; GPU compatibility and byte invariants unqualified'
        rc = 0
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['error'] = str(error)
        print('RESULT build failed: ' + str(error), file=sys.stderr)
    finally:
        receipt['baseline_artifacts_after'] = baseline_artifacts()
        receipt['baseline_unchanged'] = receipt['baseline_artifacts_after'] == before
        receipt['external_source_unchanged'] = (
            output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD']) == revision and
            not output(['git', '-C', str(SOURCE), 'status', '--porcelain']))
        if not receipt['baseline_unchanged'] or not receipt['external_source_unchanged']:
            rc = 1
            receipt['status'] = 'failed isolation check'
        receipt['build_rc'] = rc
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('RESULT receipt=' + str(directory / 'receipt.json'))
    print('VERDICT ' + receipt['status'])
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
