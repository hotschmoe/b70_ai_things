#!/usr/bin/env python3
"""Run ported upstream scheduler regressions without exposing GPU devices.

Tests are vLLM Apache-2.0 source from the frozen image, plus regression hunks
from merged PRs 52771, 52807 and 54288. Only test API/fixture adaptations are
included: older partial_tail_offloads, existing block-hash setter, and a
mocked hybrid capability for CPU scheduler tests. No scheduler method is mocked.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from build_kv_offload_image import BASE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--select', default='')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = 'tests/v1/kv_connector/unit/'
    names = [prefix + 'offloading_connector', prefix + 'utils.py',
             'tests/__init__.py', 'tests/v1/__init__.py',
             'tests/v1/kv_connector/__init__.py', prefix + '__init__.py']
    raw = subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'tar',
                                   BASE, '-C', '/workspace/vllm', '-cf', '-', *names])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            if member.isfile() and path.suffix == '.py':
                dest = args.out / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.extractfile(member).read())
    port = Path(__file__).resolve().parent / 'patches/offload-merged-regression-port.json'
    for name, record in json.loads(port.read_text()).items():
        target = args.out / name
        before = target.read_bytes()
        assert hashlib.sha256(before).hexdigest() == record['before_sha256'], name
        lines = before.decode().splitlines(True)
        for change in reversed(record['changes']):
            lines[change['start']:change['end']] = change['text'].splitlines(True)
        after = ''.join(lines).encode()
        assert hashlib.sha256(after).hexdigest() == record['after_sha256'], name
        target.write_bytes(after)
    command = ['docker', 'run', '--rm', '--entrypoint', '/opt/venv/bin/python',
               '-w', '/testroot', '-e', 'PYTHONPATH=/testroot',
               '-v', str(args.out.resolve()) + ':/testroot', args.image,
               '-m', 'pytest', '-q', prefix + 'offloading_connector/test_scheduler.py']
    if args.select:
        command += ['-k', args.select]
    (args.out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (args.out / 'pytest.log').open('w') as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    (args.out / 'exit.rc').write_text(str(rc) + '\n')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
