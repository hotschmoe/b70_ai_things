#!/usr/bin/env python3
"""CPU-only descriptor validation; no device discovery or SYCL compilation."""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/vm_8tb/github/strata')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def main():
    patches = [ROOT / 'strata/flash-next/patches' / name for name in [
        '0001-sycl-native-q8-hc-projection-primitives.patch',
        '0002-sycl-native-hc-composition.patch']]
    with tempfile.TemporaryDirectory(prefix='strata-hc-composition-cpu-') as directory:
        directory = Path(directory)
        cmake = directory / 'sycl/CMakeLists.txt'
        cmake.parent.mkdir()
        cmake.write_bytes((SOURCE / 'sycl/CMakeLists.txt').read_bytes())
        for patch in patches:
            subprocess.run(['git', 'apply', '--check', str(patch)], cwd=directory, check=True)
            subprocess.run(['git', 'apply', str(patch)], cwd=directory, check=True)
        (directory / 'descriptor.cpp').write_text(r'''
#include "strata/kernels/hc_native_composition.hpp"
#include <cassert>
#include <cstdio>
#include <limits>
int main() {
    using namespace strata::kernels;
    // Pointers are never dereferenced: this test validates host preflight only.
    float f; uint8_t q;
    int valid=0, rejected=0;
    for (int T=1;T<=8;++T) for (bool apply : {false,true}) for (bool inject : {false,true}) {
        HcNativeArgs a;
        a.tokens=T; a.w.norm=&f; a.w.norm_floats=10240;
        a.w.down=&q; a.w.down_bytes=3481600; a.w.up=&q; a.w.up_bytes=3481600;
        a.R=&f; a.R_floats=T*10240; a.xn=&f; a.xn_floats=T*10240;
        a.lo=&f; a.lo_floats=T*320; a.gate=&f; a.gate_floats=T*10240;
        a.rs=&f; a.rs_floats=T*4; a.mixed=&f; a.mixed_floats=T*2560;
        if (apply) { a.apply=true; a.bo_prev=&f; a.bo_floats=T*2560;
            a.inj_prev=&f; a.prev_floats=T*4; a.R_out=&f; a.R_out_floats=T*10240; }
        if (inject) { a.w.inject=&f; a.w.inject_floats=40960; a.inject_out=&f; a.inject_out_floats=T*4; }
        assert(hc_native_descriptor_ok(a)); ++valid;
        auto reject=[&](HcNativeArgs b){ assert(!hc_native_descriptor_ok(b)); ++rejected; };
        auto b=a; b.n_embd=1; reject(b);
        b=a; b.hc=3; reject(b); b=a; b.lr=32; reject(b);
        b=a; b.tokens=0; reject(b); b=a; b.tokens=9; reject(b);
        b=a; b.eps=0; reject(b); b=a; b.eps=std::numeric_limits<float>::quiet_NaN(); reject(b);
        b=a; b.eps=std::numeric_limits<float>::infinity(); reject(b);
#define REJECT_NULL(field) b=a; b.field=nullptr; reject(b)
#define REJECT_SHORT(field) b=a; --b.field; reject(b)
        REJECT_NULL(w.norm); REJECT_SHORT(w.norm_floats);
        REJECT_NULL(w.down); REJECT_SHORT(w.down_bytes);
        REJECT_NULL(w.up); REJECT_SHORT(w.up_bytes);
        REJECT_NULL(R); REJECT_SHORT(R_floats);
        REJECT_NULL(xn); REJECT_SHORT(xn_floats);
        REJECT_NULL(lo); REJECT_SHORT(lo_floats);
        REJECT_NULL(gate); REJECT_SHORT(gate_floats);
        REJECT_NULL(rs); REJECT_SHORT(rs_floats);
        REJECT_NULL(mixed); REJECT_SHORT(mixed_floats);
        if (apply) { REJECT_NULL(bo_prev); REJECT_SHORT(bo_floats);
            REJECT_NULL(inj_prev); REJECT_SHORT(prev_floats);
            REJECT_NULL(R_out); REJECT_SHORT(R_out_floats); }
        else { b=a; b.bo_prev=&f; reject(b); b=a; b.bo_floats=1; reject(b);
            b=a; b.inj_prev=&f; reject(b); b=a; b.prev_floats=1; reject(b);
            b=a; b.R_out=&f; reject(b); b=a; b.R_out_floats=1; reject(b); }
        if (inject) { REJECT_SHORT(w.inject_floats); REJECT_NULL(inject_out); REJECT_SHORT(inject_out_floats); }
        else { b=a; b.w.inject_floats=1; reject(b); b=a; b.inject_out=&f; reject(b);
            b=a; b.inject_out_floats=1; reject(b); }
    }
    std::printf("PASS composition descriptor: %d valid cases, %d rejected cases\n",valid,rejected);
}
'''.replace('#include <cstdio>', '#include <cstdio>\n#include <initializer_list>'))
        command = ('g++ -std=c++17 -O1 -fsanitize=address,undefined -I/work/sycl/include '
                   '/work/descriptor.cpp -o /work/descriptor && /work/descriptor')
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--user',
                        f'{os.getuid()}:{os.getgid()}', '--entrypoint', '/bin/bash',
                        '-v', str(directory)+':/work', IMAGE, '-lc', command], check=True)
    for patch in patches:
        print(patch.name, hashlib.sha256(patch.read_bytes()).hexdigest())
    print('No SYCL compilation, device discovery, GPU arithmetic, or model-route validation performed.')


if __name__ == '__main__':
    main()
