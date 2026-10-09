#!/usr/bin/env python3
"""Compile/link actual NativeDense upload oracle against a qualified full build; no devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

REPO=Path(__file__).resolve().parents[2]
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def sha(p):
    d=hashlib.sha256()
    with p.open('rb') as h:
        for b in iter(lambda:h.read(1024*1024),b''):d.update(b)
    return d.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-receipt',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=REPO/'strata/flash-next/native-source-upload-plan-full.json',help='Immutable full source generation plan, e.g. full-v6')
    args=p.parse_args()
    engine=json.loads(args.engine_receipt.read_text())
    if engine.get('image')!=IMAGE or engine.get('source_revision')!='fb58e0dbc8399662c0e47c76578c6e878b14f6cf' or engine.get('ggml_revision')!='3cf03257f219afbe7334045ff7c6a06ac68c627d':
        raise RuntimeError('Pinned engine source/dependency/image generation required')
    if engine.get('build_rc')!=0 or not engine.get('status','').startswith('built;') or not engine.get('external_source_unchanged') or engine.get('devices_exposed') is not False:
        raise RuntimeError('Passing complete isolated engine build receipt required')
    source=Path(engine['source_copy']);build=Path(engine['build']);ggml=Path(engine['ggml_source'])
    for name,expected in engine['patched_source_sha256'].items():
        if sha(source/name)!=expected:raise RuntimeError('Patched source changed')
    for name,expected in engine['binary_sha256'].items():
        if sha(Path(name))!=expected:raise RuntimeError('Built executable changed')
    if sha(build/'CMakeCache.txt')!=engine['cmake_cache_sha256']:raise RuntimeError('Engine build config changed')
    libraries=[build/('libstrata_'+name+'.a') for name in ['engine','prefill','spec','kernels','core','kernels_cpu']]
    libraries+=[build/'ggml/src/libggml-cpu.a',build/'ggml/src/libggml-base.a']
    if not all(path.is_file() for path in libraries):raise RuntimeError('Actual full engine/static GGML libraries required')
    library_hashes={str(path):sha(path) for path in libraries}
    out=Path(tempfile.mkdtemp(prefix='strata-source-upload-oracle-full-',dir='/mnt/vm_8tb/b70/build'))
    fixture=REPO/'strata/flash-next/native_source_upload_gpu_oracle_full.cpp';(out/'oracle.cpp').write_bytes(fixture.read_bytes())
    plan_path=args.plan.resolve();plan_raw=plan_path.read_bytes();plan=json.loads(plan_raw)
    if sha(out/'oracle.cpp')!=plan['oracle_source_sha256']:raise RuntimeError('Oracle source differs from preregistered snapshot')
    (out/'plan.snapshot.json').write_bytes(plan_raw)
    compiler=['icpx','-std=c++20','-O2','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-DSTRATA_NATIVE_EXPERTS=1','-DSTRATA_SYCL_Q8_HC_BUILT=1','-I/src/sycl/include','-I/src/include','-I/src/third_party/ggml','-I/ggml/ggml/include','/work/oracle.cpp','-Wl,--start-group']
    compiler+=[str(path).replace(str(build),'/engine') for path in libraries]
    compiler+=['-Wl,--end-group','-lze_loader','-lcrypto','-qmkl=sequential','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/work/source-upload-oracle']
    command=['docker','run','--rm','--network','none','--user',str(os.getuid())+':'+str(os.getgid()),'-v',str(source)+':/src:ro','-v',str(build)+':/engine:ro','-v',str(ggml)+':/ggml:ro','-v',str(out)+':/work',IMAGE,'exec '+shlex.join(compiler)]
    receipt={'schema':1,'scope':'Actual source-upload oracle source compilation/link only; no devices or inference','image':IMAGE,'devices_exposed':False,'engine_receipt':str(args.engine_receipt.resolve()),'engine_receipt_sha256':sha(args.engine_receipt),'oracle_source_sha256':sha(out/'oracle.cpp'),'plan_sha256':sha(out/'plan.snapshot.json'),'library_sha256':library_hashes,'command':command,'passed':False}
    with (out/'build.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipt['build_rc']=result.returncode
    if result.returncode==0 and (out/'source-upload-oracle').is_file():receipt.update(passed=True,binary_sha256=sha(out/'source-upload-oracle'))
    receipt['libraries_unchanged']={str(path):sha(path) for path in libraries}==library_hashes
    if not receipt['libraries_unchanged']:receipt['passed']=False
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii')
    print(json.dumps({'passed':receipt['passed'],'receipt':str(out/'receipt.json')}))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
