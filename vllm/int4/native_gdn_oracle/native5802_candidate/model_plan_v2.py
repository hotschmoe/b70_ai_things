#!/usr/bin/env python3
"""Prepare native-only model controls, or validate prerequisites and exec server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

RAW = Path('/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910')


def validate_inherited_lease(server):
    assert server.count('--leased') == 1
    assert server[server.index('--card')+1] == '0'
    lease = Path(os.environ.get('B70_GPU_LOCK', '/mnt/vm_8tb/b70/gpu.lock') + '.0')
    assert Path('/proc/self/fd/8').resolve(strict=True) == lease.resolve()
    assert os.fstat(8).st_ino == lease.stat().st_ino


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image')
    p.add_argument('--native-sha256')
    p.add_argument('--root', type=Path)
    p.add_argument('--pair-pass', type=Path)
    p.add_argument('--pair-image-inspect', type=Path)
    p.add_argument('--oracle-outcome', type=Path)
    p.add_argument('--oracle-result', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if a.plan:
        plan = json.loads(a.plan.read_text())
        if not a.run:
            print(json.dumps(plan, indent=2)); return
        validate_inherited_lease(plan['server'])
        assert os.environ.get('B70_XPU_GDN_PREFIX_CONV_COPY', '0') == '0'
        for file, digest in plan['frozen_files'].items():
            assert sha(Path(file)) == digest, file
        assert Path(plan['pair_pass']).is_file()
        assert Path(plan['pair_pass']).parent in Path(plan['pair_image_inspect']).parents
        assert json.loads(Path(plan['pair_image_inspect']).read_text())[0]['Id'] == plan['image']
        outcome = json.loads(Path(plan['oracle_outcome']).read_text())
        result = json.loads(Path(plan['oracle_result']).read_text())
        assert outcome['passed'] is True and outcome['numeric_passed'] is True and outcome['image'] == plan['image']
        assert result['passed'] is True and result['image'] == plan['image'] and result['native_sha256'] == plan['native_sha256']
        assert result['requested_steps'] == result['completed_steps'] == 128
        assert len(result['rows']) == 128 and all(len(r['checks']) == 12 and all(r['checks'].values()) for r in result['rows'])
        assert '--xpu-gdn-prefix-conv-copy' not in plan['server']
        os.execv(sys.executable, [sys.executable, *plan['server'][1:]])
    assert not a.run
    assert a.image and re.fullmatch(r'sha256:[0-9a-f]{64}', a.image)
    assert a.native_sha256 and re.fullmatch(r'[0-9a-f]{64}', a.native_sha256)
    assert all(x and x.is_absolute() for x in [a.root, a.pair_pass, a.pair_image_inspect, a.oracle_outcome, a.oracle_result])
    assert a.pair_pass.parent in a.pair_image_inspect.parents
    assert not a.root.exists()
    original = RAW/'tp1-card0-mtp3-ctx100k.plan.json'
    plan = json.loads(original.read_text()); server = plan['server']
    assert server[server.index('--mtp')+1] == '3'
    assert server[server.index('--card')+1] == '0'
    assert not any('adapter' in v or 'prefix-conv-copy' in v for v in server)
    a.root.mkdir(); config = a.root/'config'; out = a.root/'run'
    oldconfig = Path(server[server.index('--preservation')+1])
    shutil.copytree(oldconfig, config)
    for f in oldconfig.rglob('*'):
        if f.is_file(): assert f.read_bytes() == (config/f.relative_to(oldconfig)).read_bytes()
    changes = {'--image':a.image, '--preservation':str(config), '--out':str(out), '--port':'18161', '--name':'b70-native5802-tp1-card0-mtp3', '--scales':str(config/'fresh-scales.json')}
    for flag, value in changes.items(): server[server.index(flag)+1] = value
    assert '--leased' not in server
    server.append('--leased')
    alias = server[server.index('--served-alias')+1]
    assert alias in Path('evals/configs/models.yaml').read_text()
    boundary = json.loads((RAW/'tp1-card0-mtp3-conv-adapter1-crossover-ctx100k.jobs.json').read_text())[0]
    serial = json.loads((RAW/'card1-serial-session0.job.json').read_text())
    ample = json.loads((RAW/'conv-adapter1-extended-jobs.prepared.json').read_text())['jobs'][0]
    jobs = [boundary, serial, ample]
    frozen = [Path(__file__).resolve(), original, Path(server[1])]
    for job, name, subdir in zip(jobs, ['01-boundary10', '02-serial-session0', '03-ample4'], ['boundary10', 'serial-session0-control', 'ample4']):
        job['name'] = name; cmd = job['command']
        for flag, value in {'--base':'http://127.0.0.1:18161', '--alias':alias, '--out':str(out/subdir)}.items(): cmd[cmd.index(flag)+1] = value
        source = Path(cmd[1]); frozen.extend(source.parent.glob('*.py'))
        flag = '--manifest' if '--manifest' in cmd else '--corpus'
        manifest = Path(cmd[cmd.index(flag)+1]); frozen.append(manifest)
        # Freeze the full read-only corpus payload directory as well as metadata.
        frozen.extend(manifest.parent.glob('*.json'))
        if flag == '--manifest': frozen.append(Path(json.loads(manifest.read_text())['payload']))
        (a.root/(name+'.job.json')).write_text(json.dumps(job, indent=2)+'\n')
    frozen.extend(f for f in config.rglob('*') if f.is_file())
    # probe_generation imports the common frozen probe modules from this tree.
    for name in ['client-source', 'serial-session0-client-source']:
        frozen.extend((RAW/name).rglob('*.py'))
    plan.update(image=a.image, native_sha256=a.native_sha256, out=str(out), jobs=jobs,
                pair_pass=str(a.pair_pass), pair_image_inspect=str(a.pair_image_inspect),
                oracle_outcome=str(a.oracle_outcome), oracle_result=str(a.oracle_result),
                frozen_files={str(f):sha(f) for f in frozen},
                notes='PREPARED_NOT_RUN. Native-only candidate; adapter absent. Byte-identical original MTP3 FP8/FULL/prefixON100K config. Initial boundary10, exact serial2, ample4 only; parent queues/reviews jobs after READY and owns STOP/strict posthealth. No automatic extended qualification.')
    plan['launch'] = ['bin/gpu-run', '--card', '0', 'python3', str(Path(__file__).resolve()), '--plan', str(a.root/'plan.json'), '--run']
    (a.root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(a.root/'plan.json')


if __name__ == '__main__':
    main()
