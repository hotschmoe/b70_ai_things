#!/usr/bin/env python3
"""Build pinned full Strata SYCL native-HC source in fresh trees, without devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name('native-hc-engine-build-plan.json')
SOURCE = Path('/mnt/vm_8tb/github/strata')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN, help='New immutable plan generation for later PLE/mirror patches')
    parser.add_argument('--jobs', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--ggml-source', type=Path, help='Optional complete clean dependency checkout at exact pin; 22 headers alone are insufficient')
    parser.add_argument('--leased', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.leased:
        os.execv(str(REPO/'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        if not os.path.samefile('/proc/self/fd/'+str(fd), '/mnt/vm_8tb/b70/gpu.lock.'+str(card)):
            raise RuntimeError('Inherited pair lease required for parent compiler workflow')
    raw_plan = args.plan.read_bytes()
    plan = json.loads(raw_plan)
    if output(['git','-C',str(SOURCE),'rev-parse','HEAD']) != plan['source_revision'] or output(['git','-C',str(SOURCE),'status','--porcelain']):
        raise RuntimeError('Strata external source must be clean at the reviewed pin')
    for name, expected in plan['source_file_sha256'].items():
        if sha(SOURCE/name) != expected:
            raise RuntimeError('Source identity mismatch: '+name)
    for patch in plan['patches']:
        if sha(REPO/patch['path']) != patch['sha256']:
            raise RuntimeError('Patch identity mismatch: '+patch['path'])
    if output(['docker','image','inspect',IMAGE,'--format','{{.Id}}']) != IMAGE:
        raise RuntimeError('Exact pinned compiler/runtime image required')
    directory = Path(tempfile.mkdtemp(prefix='strata-native-hc-engine-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-',dir='/mnt/vm_8tb/b70/build'))
    (directory/'plan.snapshot.json').write_bytes(raw_plan)
    (directory/'controller.py').write_bytes(Path(__file__).read_bytes())
    source, dependency, build = directory/'source', directory/'ggml-source', directory/'build'
    build.mkdir()
    receipt = dict(schema=1, image=IMAGE, devices_exposed=False, source_revision=plan['source_revision'],
                   ggml_revision=plan['ggml']['revision'], source_copy=str(source), ggml_source=str(dependency),
                   build=str(build), plan_snapshot=str(directory/'plan.snapshot.json'),
                   plan_sha256=sha(directory/'plan.snapshot.json'), patches=plan['patches'],
                   builder_sha256=sha(Path(__file__)), commands=[], status='preparing', build_rc=None)
    def run(argv, log):
        receipt['commands'].append(argv)
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        with (directory/'build.log').open('w') as log:
            run(['git','clone','--local','--no-hardlinks','--no-checkout',str(SOURCE),str(source)],log)
            run(['git','-C',str(source),'checkout','--detach',plan['source_revision']],log)
            if args.ggml_source is not None:
                ggml = args.ggml_source.resolve()
                if output(['git','-C',str(ggml),'rev-parse','HEAD']) != plan['ggml']['revision'] or output(['git','-C',str(ggml),'status','--porcelain']):
                    raise RuntimeError('Complete dependency checkout must be clean at 3cf03257')
                run(['git','clone','--local','--no-hardlinks','--no-checkout',str(ggml),str(dependency)],log)
                run(['git','-C',str(dependency),'checkout','--detach',plan['ggml']['revision']],log)
            else:
                run(['git','init',str(dependency)],log)
                run(['git','-C',str(dependency),'remote','add','origin',plan['ggml']['url']],log)
                run(['git','-C',str(dependency),'fetch','--depth','1','origin',plan['ggml']['revision']],log)
                run(['git','-C',str(dependency),'checkout','--detach',plan['ggml']['revision']],log)
            if not (dependency/'ggml/CMakeLists.txt').is_file() or not (dependency/'gguf-py/gguf/quants.py').is_file():
                raise RuntimeError('Dependency source incomplete: headers alone cannot link ggml-cpu/base')
            for patch in plan['patches']:
                path = REPO/patch['path']
                run(['git','-C',str(source),'apply','--check',str(path)],log)
                run(['git','-C',str(source),'apply',str(path)],log)
            receipt['patched_source_sha256'] = {name:sha(source/name) for name in plan['overlay_files']}
            receipt['source_copy_status'] = output(['git','-C',str(source),'status','--porcelain'])
            receipt['ggml_tree'] = output(['git','-C',str(dependency),'rev-parse','HEAD^{tree}'])
            configure=['cmake','-S','/src/sycl','-B','/build','-G','Ninja']
            configure += ['-D'+key+'='+value for key,value in sorted(plan['cmake_flags'].items())]
            script='\n'.join(['set -eo pipefail','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh; fi',
                'icpx --version > /build/compiler-version.txt','dpkg-query -W > /build/packages.txt',shlex.join(configure),
                'cmake --build /build --target '+shlex.join(plan['build_targets'])+' -j '+str(args.jobs)])
            receipt['container_script']=script
            run(['docker','run','--rm','--network','none','--user',str(os.getuid())+':'+str(os.getgid()),
                 '-v',str(source)+':/src:ro','-v',str(dependency)+':/ggml:ro','-v',str(build)+':/build',IMAGE,script],log)
        binaries=[build/name for name in plan['build_targets']]
        if not all(path.is_file() for path in binaries):
            raise RuntimeError('Required full executable missing')
        receipt['binary_sha256']={str(path):sha(path) for path in binaries}
        receipt['cmake_cache_sha256']=sha(build/'CMakeCache.txt')
        receipt['status']='built; native upload, complete model fidelity, PLE and split lifecycle unqualified'
        receipt['build_rc']=0
    except Exception as error:
        receipt.update(status='failed', build_rc=1, error=str(error))
    finally:
        receipt['external_source_unchanged']=(output(['git','-C',str(SOURCE),'rev-parse','HEAD'])==plan['source_revision'] and not output(['git','-C',str(SOURCE),'status','--porcelain']))
        receipt['plan_snapshot_unchanged']=sha(directory/'plan.snapshot.json')==receipt['plan_sha256']
        if not receipt['external_source_unchanged'] or not receipt['plan_snapshot_unchanged']:
            receipt.update(status='failed isolation',build_rc=1)
        (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii')
    print(json.dumps({'receipt':str(directory/'receipt.json'),'status':receipt['status']}))
    return receipt['build_rc']


if __name__=='__main__':
    raise SystemExit(main())
