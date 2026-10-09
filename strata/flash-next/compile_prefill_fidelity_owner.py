#!/usr/bin/env python3
"""Actual consumed prefill/verify TU compilation in immutable0017 overlay, no devices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T075303Z-3vcjlgw9')
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
PATCH=ROOT/'strata/flash-next/patches/0017-sycl-prefill-fidelity-impl-owner.patch'
PATCH_SHA='f92143d988517cc32ac60e32d864d9816186d82a46515a25bf3455d3822ba6b8'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--leased',action='store_true');a=p.parse_args()
    if not a.leased:
        os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
    for fd,card in [(8,0),(9,1)]:assert os.path.samefile('/proc/self/fd/'+str(fd),'/mnt/vm_8tb/b70/gpu.lock.'+str(card))
    assert sha(PATCH)==PATCH_SHA
    prior=json.loads((BASE/'receipt.json').read_text());assert prior['plan_sha256']=='827f3631b3feb857d6a91d76cab060c5b373832934b17e82dce0cc349e124639'
    out=Path(tempfile.mkdtemp(prefix='strata-prefill-owner0017-',dir='/mnt/vm_8tb/b70/build'))
    source=out/'source';shutil.copytree(BASE/'source',source,symlinks=True)
    subprocess.run(['git','apply','--check',str(PATCH)],cwd=source,check=True)
    subprocess.run(['git','apply',str(PATCH)],cwd=source,check=True)
    (out/'controller.py').write_bytes(Path(__file__).read_bytes())
    common=['icpx','-DSTRATA_NATIVE_EXPERTS=1','-DSTRATA_SYCL_Q8_HC_BUILT=1','-DSTRATA_VERSION="0.1.41"',
            '-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-default-sub-group-size=32',
            '-fsycl-device-code-split=per_kernel','-fp-model=precise','-Wno-unused-parameter',
            '-Wno-unused-variable','-Wno-deprecated-declarations','-Wall','-Wextra','-qmkl=sequential',
            '-I/src/sycl/include','-I/src/include','-I/src/src/core','-I/ggml/ggml/include']
    def compile(name):
        file='sycl/src/'+('prefill/prefill.cpp' if name=='prefill' else 'core/verify.cpp')
        flags=common+['-c','/src/'+file,'-o','/work/'+name+'.o']
        cmd=['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
             '--entrypoint','/bin/bash','-v',str(source)+':/src:ro','-v',str(BASE/'ggml-source')+':/ggml:ro',
             '-v',str(out)+':/work',IMAGE,'-lc',shlex.join(flags)]
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        return dict(tu=file,command=cmd,exit_code=r.returncode,seconds=time.monotonic()-start,
                    source_sha256=sha(source/file),object_sha256=sha(out/(name+'.o')) if (out/(name+'.o')).exists() else None)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(compile,['prefill','verify']))
    receipt=dict(scope='actual consumed SYCL TUs compile only; no devices/execution',image=IMAGE,devices_exposed=False,
                 base_source=str(BASE/'source'),base_plan_sha256=prior['plan_sha256'],overlay=str(source),
                 patch=str(PATCH),patch_sha256=PATCH_SHA,results=results,
                 consumed_headers={str(p.relative_to(source)):sha(p) for p in [source/'sycl/include/strata/core/verify.hpp',
                     source/'include/strata/prefill/prefill.hpp',source/'sycl/include/strata/core/fidelity_observer.hpp']},
                 passed=all(r['exit_code']==0 for r in results))
    (out/'compile-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('RECEIPT',out/'compile-receipt.json')
    for result in results:print(result['tu'],'exit',result['exit_code'],'seconds',round(result['seconds'],3))
    if not receipt['passed']:raise SystemExit(1)


if __name__=='__main__':main()
