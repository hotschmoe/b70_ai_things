"""Root-owned isolated NoDri expert-pair operator build, original SDK read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / 'iq-expert-pair-operator-fixture-source-plan-v1.json'
PLAN_SHA = '65b76970b1e0042fc50fd33bdd3f8b296a1ed90a1f494251b2518ae00ec4e65e'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.fstat(9).st_ino == Path('/mnt/vm_8tb/b70/gpu.lock.1').stat().st_ino,
            'Root card1 lease required even for NoDri compilation')
    require(sha(PLAN) == PLAN_SHA, 'Frozen fixture plan changed')
    plan = json.loads(PLAN.read_bytes())
    sdk = Path(plan['actual52_SDK'])
    require(sha(sdk / 'receipt.json') == plan['actual52_receipt_sha256'], 'SDK receipt changed')
    receipt = json.loads((sdk / 'receipt.json').read_bytes())
    require(receipt['build_rc'] == 0 and receipt['devices_exposed'] is False,
            'Completed original NoDri SDK required')
    for name, want in plan['files'].items():
        require(sha(ROOT / name) == want, 'Frozen fixture input changed: ' + name)
    for name, want in receipt['patched_source_sha256'].items():
        require(sha(sdk / 'source' / name) == want, 'Actual SDK source changed: ' + name)
    require(sha(plan['production_Ninja_source']) == plan['production_Ninja_sha256'], 'Ninja changed')
    output = args.output
    require(output.is_absolute() and output.resolve() == output and not output.exists(), 'New canonical output required')
    output.mkdir()
    candidate = output / 'candidate'
    subprocess.run(['python3', str(HERE / 'prepare_expert_pair_reuse_v1.py'),
                    '--output', str(candidate)], check=True, stdout=subprocess.DEVNULL)
    require(sha(candidate / 'iq_kernels.dp.cpp') == plan['fresh_iq_source_sha256'], 'Candidate source changed')
    build = sdk / 'build'
    originals = [build / 'libstrata_kernels.a', build / 'libstrata_core.a',
                 build / 'CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o',
                 build / 'compiler-version.txt', build / 'packages.txt']
    before = {str(p): sha(p) for p in originals}
    includes = {str(p.relative_to(sdk / 'source')): sha(p)
                for directory in ('include', 'sycl/include', 'third_party/ggml')
                for p in sorted((sdk / 'source' / directory).rglob('*')) if p.is_file()}
    commands = [plan['fresh_iq_object_compile_argv'], plan['fixture_compile_argv'], plan['link_argv']]
    script = '\n'.join([
        'set -Eeo pipefail', 'test ! -e /dev/dri',
        'if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh; fi',
        'icpx --version > /out/compiler-version.txt', 'dpkg-query -W > /out/packages.txt',
        'cmp /original-build/compiler-version.txt /out/compiler-version.txt',
        'cmp /original-build/packages.txt /out/packages.txt',
        'ar t /original-build/libstrata_kernels.a > /out/original-members.txt',
        'cp /original-build/libstrata_kernels.a /out/libstrata_kernels.without_iq.a',
        'ar d /out/libstrata_kernels.without_iq.a iq_kernels.dp.cpp.o',
        'ar t /out/libstrata_kernels.without_iq.a > /out/filtered-members.txt',
        *[shlex.join(command) for command in commands]])
    (output / 'compile.sh').write_text(script + '\n', encoding='ascii')
    command = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '2', '--memory', '4g',
               '--user', str(os.getuid()) + ':' + str(os.getgid()),
               '-v', str(sdk / 'source') + ':/src:ro', '-v', str(build) + ':/original-build:ro',
               '-v', str(candidate) + ':/candidate:ro', '-v', str(output) + ':/out',
               '-v', str(ROOT / 'kernels/iq_expert_pair_reuse_fixture_v1.cpp') + ':/fixture.cpp:ro',
               receipt['image'], 'bash', '/out/compile.sh']
    report = dict(schema=1, started_epoch=time.time(), fixture_plan_sha256=PLAN_SHA,
                  compiler_image=receipt['image'], devices_exposed=False,
                  original_inputs_sha256=before, include_inputs_sha256=includes,
                  candidate_sha256=plan['fresh_iq_source_sha256'], command=command,
                  fresh_compile_commands=commands, passed=False)
    try:
        with (output / 'compile.log').open('wb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report['return_code'] = result.returncode
        require(result.returncode == 0, 'NoDri compile/link failed; preserve compile.log')
        old = (output / 'original-members.txt').read_text().splitlines()
        new = (output / 'filtered-members.txt').read_text().splitlines()
        require(old.count('iq_kernels.dp.cpp.o') == 1 and new == [x for x in old if x != 'iq_kernels.dp.cpp.o'],
                'Exact single original IQ archive member removal required')
        report['new_artifacts_sha256'] = {name: sha(output / name) for name in
                                         ('iq_kernels.pair.o', 'fixture.o', 'libstrata_kernels.without_iq.a', 'iq-pair-fixture')}
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
    finally:
        report['original_inputs_unchanged'] = all(sha(p) == want for p, want in before.items())
        report['include_inputs_unchanged'] = all(sha(sdk / 'source' / p) == want for p, want in includes.items())
        report['passed'] &= report['original_inputs_unchanged'] and report['include_inputs_unchanged']
        report['finished_epoch'] = time.time()
        (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n', encoding='ascii')
    print(json.dumps({k: v for k, v in report.items() if k in ('passed', 'error', 'return_code', 'finished_epoch')}, ensure_ascii=True))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
