#!/usr/bin/env python3
"""Apply full native HC series and compile/run policy and metadata on CPU only."""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/vm_8tb/github/strata')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
FILES = ['sycl/CMakeLists.txt', 'include/strata/core/weights.hpp',
         'sycl/src/core/native_dense.cpp', 'sycl/src/core/layer.cpp',
         'sycl/src/core/mtp.cpp', 'sycl/src/core/verify.cpp', 'sycl/src/prefill/prefill.cpp',
         'sycl/src/program/generate.cpp']
PATCHES = ['0001-sycl-native-q8-hc-projection-primitives.patch',
           '0002-sycl-native-hc-composition.patch',
           '0003-sycl-native-hc-source-owner-decode.patch']


def main():
    with tempfile.TemporaryDirectory(prefix='strata-native-hc-owner-cpu-') as temp:
        work = Path(temp)
        for name in FILES:
            target = work/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((SOURCE/name).read_bytes())
        for name in PATCHES:
            patch = ROOT/'strata/flash-next/patches'/name
            subprocess.run(['git', 'apply', '--check', str(patch)], cwd=work, check=True)
            subprocess.run(['git', 'apply', str(patch)], cwd=work, check=True)
        (work/'policy.cpp').write_text(r'''
#include "strata/core/native_hc_policy.hpp"
#include "strata/core/weights.hpp"
#include <cassert>
#include <cstdio>
#include <type_traits>
int main(int argc,char** argv) {
    using namespace strata::core;
    static_assert(std::is_same_v<decltype(WeightRef{}.hc_source_f32),const float*>);
    static_assert(std::is_same_v<decltype(WeightRef{}.hc_source_q8),const uint8_t*>);
    WeightRef w; assert(w.hc_source_type==-1 && !w.hc_source_f32 && !w.hc_source_q8 && !w.hc_source_bytes);
    std::string err;
    const bool ok=native_hc_policy(err);
    assert(argc==2 && ok==(argv[1][0]=='1'));
    assert(ok?err.empty():!err.empty());
    std::printf("PASS policy expected%d: %s\n",int(ok),err.c_str());
}
''')
        commands = [
            'g++ -std=c++17 -O1 -fsanitize=address,undefined -I/work/sycl/include -I/work/include /work/policy.cpp -o /work/off',
            'g++ -std=c++17 -O1 -fsanitize=address,undefined -DSTRATA_SYCL_Q8_HC_BUILT=1 -I/work/sycl/include -I/work/include /work/policy.cpp -o /work/on',
            '/work/off 1', '/work/on 1',
            'STRATA_SYCL_NATIVE_HC=0 /work/off 1', 'STRATA_SYCL_NATIVE_HC=0 /work/on 1',
            'STRATA_SYCL_NATIVE_HC=1 /work/off 0', 'STRATA_SYCL_NATIVE_HC=1 /work/on 1',
            'STRATA_SYCL_NATIVE_HC=yes /work/off 0', 'STRATA_SYCL_NATIVE_HC=yes /work/on 0',
        ]
        for flag in ['STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_QFUSE','STRATA_GR_V3']:
            commands.append(f'STRATA_SYCL_NATIVE_HC=1 {flag}=1 /work/on 0')
            commands.append(f'STRATA_SYCL_NATIVE_HC=1 {flag}=0 /work/on 1')
        command = 'set -e\n'+'\n'.join(commands)
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
                        '--entrypoint','/bin/bash','-v',str(work)+':/work',IMAGE,'-lc',command],check=True)
    patch=ROOT/'strata/flash-next/patches'/PATCHES[-1]
    print('patch0003 SHA256',hashlib.sha256(patch.read_bytes()).hexdigest())
    print('PASS sequential apply and CPU policy/typed metadata syntax;16 policy cases')
    print('SYCL translation units, model routes, uploaded bytes and GPU arithmetic remain unqualified.')


if __name__ == '__main__':
    main()
