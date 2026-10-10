#!/usr/bin/env python3
"""Fresh leased leaf link against pinned live source35; no GPU execution."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from prepare_hc_composition35_leaf_v1 import prepare, sha, require
from c1_serve_controller_combined_v13 import leased

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = 'd55e6fb79581cf8dadbba561dc89dbaf28cb48fc77617c483ee0ad5c84559ffd'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--leased', action='store_true')
    args = parser.parse_args()
    require(sha(args.plan) == PLAN_SHA, 'Frozen composition compile plan changed before lease')
    admission_plan = json.loads(args.plan.read_text())
    require(prepare(Path(admission_plan['engine_root'])) == admission_plan, 'Current composition inputs changed before lease')
    if not args.leased:
        os.execv(str(ROOT/'bin/gpu-run'), ['gpu-run', '--card', '0', sys.executable,
                 __file__, *sys.argv[1:], '--leased'])
    leased([0])
    require(sha(args.plan) == PLAN_SHA, 'Frozen leaf compile plan changed')
    plan = json.loads(args.plan.read_text())
    require(prepare(Path(plan['engine_root'])) == plan, 'Current leaf inputs changed')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'plan.snapshot.json').write_bytes(args.plan.read_bytes())
    (out/'compile-controller.snapshot.py').write_bytes(Path(__file__).read_bytes())
    name = 'b70-hc35comp1-compile-'+str(os.getpid())
    receipt = {'schema': 1, 'started_epoch': time.time(), 'passed': False,
               'plan_sha256': PLAN_SHA, 'plan': plan, 'container': name,
               'controller_sha256': sha(__file__), 'errors': [],
               'actual_GPU_run': False, 'model_math_qualified': False,
               'archive_association_limit': plan['archive_association_limit']}
    def save():
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='ascii')
    command = ['docker', 'run', '--name', name, '--network', 'none', '--user', '1000:1000',
               '--entrypoint', '/bin/bash',
               '--label', 'b70.hc35comp1.compile='+PLAN_SHA,
               '-v', plan['engine_root']+':/sdk:ro',
               '-v', str(Path(plan['leaf_source']).parent)+':/leaf:ro',
               '-v', str(out)+':/out', plan['image'], '-c',
               'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1\nicpx --version\nexec '+
               shlex.join(plan['compile_argv_inside_pinned_image'])]
    receipt['command'] = command
    save()
    try:
        with (out/'compile.log').open('wb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=900, check=False)
        receipt['compile_return_code'] = result.returncode
        require(result.returncode == 0, 'Leaf compiler/link failed; preserve compile.log')
        binary = out/'hc_composition_arithmetic35_gpu_v1'
        with binary.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'Fresh leaf output is not ELF')
        receipt['binary'] = str(binary)
        receipt['binary_sha256'] = sha(binary)
        receipt['compile_log_sha256'] = sha(out/'compile.log')
        require(prepare(Path(plan['engine_root'])) == plan, 'Source/SDK changed during compile')
        receipt['post_source_unchanged'] = True
    except Exception as error:
        receipt['errors'].append(str(error))
    finally:
        try:
            observed = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
            require(observed['Name'] == '/'+name and observed['Config']['Image'] == plan['image']
                    and observed['Config']['Labels'].get('b70.hc35comp1.compile') == PLAN_SHA,
                    'Compile container ownership differs')
            if observed['State']['Running']:
                subprocess.run(['docker', 'stop', '-t', '15', name],
                               stdout=subprocess.DEVNULL, timeout=45, check=True)
                observed = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
            require(not observed['State']['Running'], 'Compile container remains live')
            receipt['container_terminal'] = observed['State']
            subprocess.run(['docker', 'rm', name], stdout=subprocess.DEVNULL, check=True)
            receipt['container_removed'] = True
            if observed['State']['ExitCode'] != 0:
                receipt['errors'].append('Compile container exit '+str(observed['State']['ExitCode']))
        except Exception as error:
            receipt['errors'].append('Owned cleanup: '+str(error))
        receipt['finished_epoch'] = time.time()
        receipt['passed'] = not receipt['errors']
        save()
    print(json.dumps({'passed': receipt['passed'], 'receipt': str(out/'receipt.json'),
                      'errors': receipt['errors'], 'actual_GPU_run': False}))
    return 0 if receipt['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
