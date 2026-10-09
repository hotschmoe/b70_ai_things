#!/usr/bin/env python3
"""CPU checks for actual descriptor helper and synthetic Q8_0 fixture decoding.

Does not compile SYCL or validate GPU arithmetic; those remain leased gates.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/vm_8tb/github/strata')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def decode_scale(bits):
    sign = -1 if bits & 0x8000 else 1
    exponent, fraction = (bits >> 10) & 31, bits & 1023
    if exponent == 31:
        raise ValueError('fixture excludes nonfinite scales')
    return sign * math.ldexp(fraction if exponent == 0 else 1024 + fraction,
                             -24 if exponent == 0 else exponent - 25)


def main():
    # Validate fixture layout/reference independently of the device implementation.
    finite_scales = 0
    for bits in range(65536):
        if ((bits >> 10) & 31) == 31:
            continue
        scale = struct.unpack('<e', struct.pack('<H', bits))[0]
        assert scale == decode_scale(bits)
        codes = [-128, -1, 0, 1, 127] * 6 + [-128, 127]
        block = struct.pack('<H32b', bits, *codes)
        assert len(block) == 34
        for lane in [0, 1, 2, 3, 4, 31]:
            raw = block[2 + lane]
            signed = raw if raw < 128 else raw - 256
            decoded = scale * signed
            assert decoded == decode_scale(bits) * codes[lane]
            assert struct.unpack('<f', struct.pack('<f', decoded))[0] == decoded
        finite_scales += 1
    # Deliberately retain F32 injection values that a BF16 storage path changes.
    injection = struct.unpack('<f', struct.pack('<f', 1.001))[0]
    bits = struct.unpack('<I', struct.pack('<f', injection))[0]
    truncated_bf16 = struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]
    assert injection != truncated_bf16

    plan = json.loads((ROOT / 'strata/flash-next/native-q8-hc-primitive-plan.json').read_text())
    patch = ROOT / plan['patch']
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == plan['patch_sha256']
    with tempfile.TemporaryDirectory(prefix='strata-hc-cpu-') as directory:
        directory = Path(directory)
        cmake = directory / 'sycl/CMakeLists.txt'
        cmake.parent.mkdir()
        cmake.write_bytes((SOURCE / 'sycl/CMakeLists.txt').read_bytes())
        subprocess.run(['git', 'apply', str(patch)], cwd=directory, check=True)
        (directory / 'descriptor.cpp').write_text(r'''
#include "strata/kernels/hc_native_projection.hpp"
#include <cassert>
#include <climits>
#include <cstdio>
int main() {
    using strata::kernels::hc_projection_detail::shape_ok;
    int cases = 0;
    for (int tokens : {1,2,4,8}) {
        assert(shape_ok(10240,320,tokens,10240*size_t(tokens),320*size_t(tokens)));
        assert(shape_ok(320,10240,tokens,320*size_t(tokens),10240*size_t(tokens)));
        assert(shape_ok(10240,4,tokens,10240*size_t(tokens),4*size_t(tokens)));
        cases += 3;
    }
    assert(!shape_ok(0,1,1,0,1));
    assert(!shape_ok(-32,1,1,32,1));
    assert(!shape_ok(33,1,1,33,1));
    assert(!shape_ok(INT_MAX,1,1,size_t(-1),1));
    assert(!shape_ok(32,0,1,32,0));
    assert(!shape_ok(32,10241,1,32,10241));
    assert(!shape_ok(32,1,0,32,1));
    assert(!shape_ok(32,1,9,288,9));
    assert(!shape_ok(32,1,1,31,1));
    assert(!shape_ok(32,1,1,32,0));
    std::printf("PASS actual descriptor helper: %d shapes and10 rejection cases\n",cases);
}
'''.replace('#include <climits>', '#include <climits>\n#include <initializer_list>'))
        command = 'g++ -std=c++17 -O1 -fsanitize=address,undefined -I/work/sycl/include /work/descriptor.cpp -o /work/descriptor && /work/descriptor'
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
                        '--entrypoint','/bin/bash','-v',str(directory)+':/work',IMAGE,'-lc',command],check=True)
    print(f'PASS synthetic Q8 fixture decoding: {finite_scales} finite FP16 scales, signed extremes, F32 injection preservation')
    print('No SYCL compilation, device discovery, or GPU arithmetic was performed.')


if __name__ == '__main__':
    main()
