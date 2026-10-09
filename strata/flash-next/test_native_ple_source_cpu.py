#!/usr/bin/env python3
"""Actual PLE source descriptor/header checks and independent conv-history oracle.

CPU only: no SYCL compilation, device discovery, GPU operations or model proof.
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


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def main():
    patches = sorted((ROOT / 'strata/flash-next/patches').glob('000[1-4]-*.patch'))
    patches += [ROOT / 'strata/flash-next/patches/0006-sycl-source-preserving-ple.patch']
    with tempfile.TemporaryDirectory(prefix='native-ple-cpu-') as tmp:
        tmp = Path(tmp)
        paths = {'include/strata/kernels/ple.hpp'}
        for patch in patches:
            paths.update(line[6:] for line in patch.read_text().splitlines()
                         if line.startswith('+++ b/'))
        for relative in paths:
            source = SOURCE / relative
            if source.exists():
                target = tmp / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        for patch in patches:
            subprocess.run(['git', 'apply', str(patch)], cwd=tmp, check=True)
        (tmp / 'descriptor.cpp').write_text(r'''
#include "strata/kernels/ple.hpp"
#include <cassert>
#include <cstdio>
int main() {
    using namespace strata::kernels;
    PleWeights w;
    assert(!ple_exact_sources_ok(w));
    w.source_exact=true;
    w.key_source_q8=reinterpret_cast<const uint8_t*>(0x1000);w.key_source_bytes=27852800;
    w.value_source_q8=reinterpret_cast<const uint8_t*>(0x2000);w.value_source_bytes=6963200;
    w.conv_source_f32=reinterpret_cast<const float*>(0x3000);w.conv_source_floats=40960;
    assert(ple_exact_sources_ok(w));
    assert(!w.value_bf16 && !w.conv1d_f16 && !w.key_bf16);
    int rejected=1;
    {auto a=w;a.source_exact=false;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;a.key_source_q8=nullptr;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;--a.key_source_bytes;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;++a.key_source_bytes;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;a.value_source_q8=nullptr;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;--a.value_source_bytes;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;++a.value_source_bytes;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;a.conv_source_f32=nullptr;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;--a.conv_source_floats;assert(!ple_exact_sources_ok(a));++rejected;}
    {auto a=w;++a.conv_source_floats;assert(!ple_exact_sources_ok(a));++rejected;}
    std::printf("PASS actual PLE exact-source helper: 1 valid and%d rejection cases\n",rejected);
}
''')
        command = 'g++ -std=c++17 -O1 -fsanitize=address,undefined -I/work/include /work/descriptor.cpp -o /work/descriptor && /work/descriptor'
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
                        '--entrypoint', '/bin/bash', '-v', str(tmp) + ':/work', IMAGE, '-lc', command], check=True)

    # GGML k-fastest weights, row-fastest history; independent scalar and batch
    # oracles must pick exactly the same 9/6/3/0-back taps across chunk boundaries.
    d, history_rows = 10240, 9
    weights = [fp32((i % 23 - 11) * 0.00100013) for i in range(4*d)]
    initial = [[fp32((c*7+r*13) % 31 * 0.015631) for r in range(history_rows)] for c in range(d)]
    changed = sum(struct.unpack('<e', struct.pack('<e', w))[0] != w for w in weights)
    assert changed > 0
    tested = 0
    max_rounding_control = 0.0
    for rows in [1, 2, 4, 8, 9, 16, 17]:
        values = [[fp32(((t*37+c*19) % 127 - 63)*0.031251) for c in range(d)] for t in range(rows)]
        history = [row.copy() for row in initial]
        for t in range(rows):
            for c in range(d):
                scalar = 0.0
                for k in range(4):
                    scalar += weights[4*c+k] * (values[t][c] if k == 3 else history[c][3*k])
                batch = 0.0
                narrowed = 0.0
                for k in range(4):
                    previous = t - 9 + 3*k
                    x = values[previous][c] if previous >= 0 else initial[c][9+previous]
                    batch += weights[4*c+k]*x
                    narrowed += struct.unpack('<e', struct.pack('<e', weights[4*c+k]))[0]*x
                assert scalar == batch
                max_rounding_control = max(max_rounding_control, abs(batch-narrowed))
            for c in range(d):
                history[c] = history[c][1:] + [values[t][c]]
            tested += 1
        for c in range(d):
            expected = (initial[c] + [values[t][c] for t in range(rows)])[-9:]
            assert history[c] == expected
    assert max_rounding_control > 0
    result = {'mode': 'CPU_ONLY', 'source_descriptor_valid': 1, 'source_descriptor_rejections': 11,
              'conv_rows_checked': tested, 'conv_width': d, 'f32_weights_changed_by_f16': changed,
              'max_f16_control_difference': max_rounding_control,
              'patch_sha256': hashlib.sha256(patches[-1].read_bytes()).hexdigest(),
              'gpu_validated': False, 'pass': True}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
