#!/usr/bin/env python3
"""CPU-only new native TP2 100K plan from the preserved interrupted recipe."""
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[4]
OLD = Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/mrv1-tp2-fp8-clean100k')
RAW = Path('/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910')
ROOT = RAW/'native5802-tp2-100k-plan'
IMAGE = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
NATIVE = '6717e3ec7fe3e7cdc26cd66b84aabdd076f211e5bed62b8711b7377fca8668d9'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not ROOT.exists()
    ROOT.mkdir()
    for name in ['config', 'source']:
        shutil.copytree(OLD/name, ROOT/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['fresh-scales.json', 'early-history-manifest.json']:
        shutil.copyfile(OLD/name, ROOT/name)
    assert sha(ROOT/'fresh-scales.json') == 'be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062'
    plan = json.loads((OLD/'plan.json').read_text())
    plan = json.loads(json.dumps(plan).replace(str(OLD), str(ROOT)).replace('mrv1-tp2-fp8-clean100k-', 'native5802-tp2-100k-'))
    server = plan['server']
    server[server.index('--image')+1] = IMAGE
    server[server.index('--name')+1] = 'b70-native5802-tp2-100k'
    alias = server[server.index('--served-alias')+1]
    assert alias in (REPO/'evals/configs/models.yaml').read_text()
    assert '--leased' not in server  # Server acquires both leases itself.
    assert '--health-p2p-check' not in server and server[server.index('--p2p')+1] == '0'
    assert not any('PREFIX_CONV_COPY' in s or 'prefix-conv-copy' in s for s in server)
    cfg = json.loads((ROOT/'config/Config.json').read_text())
    assert cfg == json.loads((OLD/'config/Config.json').read_text())
    assert cfg['Cmd'][cfg['Cmd'].index('--max-model-len')+1] == '100000'
    assert 'CCL_TOPO_P2P_ACCESS=0' in cfg['Env']
    assert not any('B70_XPU_GDN_PREFIX_CONV_COPY=' in e for e in cfg['Env'])
    initial = [json.loads((RAW/'tp1-card0-mtp3-conv-adapter1-crossover-ctx100k.jobs.json').read_text())[0],
               json.loads((RAW/'card1-serial-session0.job.json').read_text()),
               json.loads((RAW/'card0-concurrent-v1.job.json').read_text())]
    for job, name in zip(initial, ['01-boundary10', '01-serial2', '01-concurrent4client']):
        job['name'] = name; cmd = job['command']
        for flag, value in {'--base':'http://127.0.0.1:18125', '--out':str(ROOT/'run'/name), '--alias':alias}.items():
            if flag in cmd: cmd[cmd.index(flag)+1] = value
            else: cmd.extend([flag, value])
    startup = json.loads(json.dumps(plan['jobs'][-1]))
    startup['name'] = '00-startup-host-trace-review'
    startup['command'][-1] = str(ROOT/'run/startup-host-trace-before-workloads.json')
    plan['jobs'] = [startup] + initial + plan['jobs']
    # Keep the original review and all complete qualification rounds. The
    # interrupted old marker cannot satisfy this new candidate's completion.
    plan['notes'] = 'PREPARED_NOT_RUN. Native-only image update; exact old 100K config. Initial boundary10/serial2/four-client controls precede complete tiny24/early3/three deterministic and three sampled rounds with individual quality gates and startup trace review. No old partial passes inherited.'
    (ROOT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    files = {str(p):sha(p) for p in ROOT.rglob('*') if p.is_file()}
    for p in [Path(__file__).resolve(), REPO/'vllm/cache800k/run_arm.py', Path(server[1]), REPO/'vllm/int4/diagnostics/xpu_health_strict.sh', RAW/'native-contract-provenance/image-receipt.json']:
        files[str(p)] = sha(p)
    for folder in ['client-source', 'serial-session0-client-source', 'concurrent-v1-client-source']:
        for p in (RAW/folder).rglob('*.py'): files[str(p)] = sha(p)
    for folder in ['corpus-adapter-pair10', 'corpus-concurrent-v1']:
        for p in (RAW/folder).glob('*.json'): files[str(p)] = sha(p)
    files[str(RAW/'serial-session0-manifest.json')] = sha(RAW/'serial-session0-manifest.json')
    for name, digest in json.loads((OLD/'manifest.json').read_text())['external'].items():
        assert sha(Path(name)) == digest, name
        files[name] = digest
    files[str(Path(__file__).with_name('tp2_launch.py'))] = sha(Path(__file__).with_name('tp2_launch.py'))
    prerequisites = dict(image=IMAGE, native_sha256=NATIVE,
        pair_pass='/mnt/vm_8tb/b70/results/bang_isolation_20260910/native5802-conv-contract-preflight/PASS',
        pair_inspect='/mnt/vm_8tb/b70/results/bang_isolation_20260910/native5802-conv-contract-preflight/native5802-conv-contract/image-inspect.json',
        numeric_outcome=str(RAW/'native5802-oracle-card0/OUTCOME.json'),
        numeric_tp2_local_outcome=str(RAW/'native5802-oracle-tp2-local-card1/OUTCOME.json'),
        pending=['parent review of completed native TP1 matrix and local TP2 numerical outcome', 'both GPUs allocated and previous workloads fully cleaned/healthy', 'actual profile/capacity and paired collective audit on this TP2 startup', 'all new full100K jobs and strict teardown/posthealth pass before separate200K successor'],
        launch=['python3',str(REPO/'vllm/cache800k/run_arm.py'),str(ROOT/'plan.json')],
        lease_rule='Do not wrap run_arm in gpu-run; server owns both leases.',
        files=files)
    (ROOT/'prerequisites.json').write_text(json.dumps(prerequisites, indent=2)+'\n')
    print(ROOT/'plan.json')


if __name__ == '__main__':
    main()
