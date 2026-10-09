#!/usr/bin/env python3
"""CPU-only tests of patch004 helper using a fake tensor-readback backend."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/vm_8tb/github/llama.cpp-flashnext')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def main():
    plan_path = ROOT / 'llamacpp/flash-next/cache-byte-invariant-plan.json'
    plan = json.loads(plan_path.read_text())
    patch = ROOT / plan['patch']
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == plan['patch_sha256']
    with tempfile.TemporaryDirectory(prefix='cache-helper-cpu-') as directory:
        directory = Path(directory)
        for name in plan['source_file_sha256']:
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / name, target)
        subprocess.run(['git', 'apply', str(patch)], cwd=directory, check=True)
        probe = directory / 'probe'
        probe.mkdir()
        shutil.copy2(directory / 'src/llama-moe-cache-verify.h', probe)
        (probe / 'ggml-backend.h').write_text(r'''
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
struct ggml_tensor { const char * name; void * data; size_t size; int type; int64_t ne[4]; size_t nb[4]; };
inline size_t read_calls = 0;
inline size_t largest_read = 0;
inline size_t ggml_nbytes(const ggml_tensor * t) { return t->size; }
inline void ggml_backend_tensor_get(const ggml_tensor * t, void * dst, size_t off, size_t n) {
    if (off > t->size || n > t->size-off) { throw std::runtime_error("fake backend bounds"); }
    ++read_calls; if (n > largest_read) { largest_read = n; }
    std::memcpy(dst, static_cast<const uint8_t *>(t->data)+off, n);
}
''')
        (probe / 'llama-impl.h').write_text(r'''
#pragma once
#include <cstdio>
#include <stdexcept>
#define GGML_ASSERT(x) do { if (!(x)) throw std::runtime_error("assert: " #x); } while (0)
#define GGML_ABORT(...) do { char msg[2048]; std::snprintf(msg,sizeof(msg),__VA_ARGS__); throw std::runtime_error(msg); } while (0)
#define LLAMA_LOG_INFO(...) std::printf(__VA_ARGS__)
''')
        (probe / 'other.cpp').write_text(r'''
#include "llama-moe-cache-verify.h"
void other_translation_unit() { ++llama_moe_verify_counts().calls[4]; }
''')
        (probe / 'main.cpp').write_text(r'''
#include "llama-moe-cache-verify.h"
#include <iostream>
void other_translation_unit();
template<class F> void reject(F operation, const char * fragment) {
    bool rejected = false;
    try { operation(); } catch (const std::runtime_error & e) {
        rejected = std::strstr(e.what(), fragment) != nullptr;
    }
    if (!rejected) throw std::runtime_error("missing expected diagnostic rejection");
}
int main() {
    constexpr size_t size = 8*1024*1024+19;
    std::vector<uint8_t> host(size), gpu(size+11);
    for (size_t i=0;i<size;++i) host[i] = uint8_t(i*17+3);
    std::memcpy(gpu.data()+11,host.data(),size);
    ggml_tensor t {"synthetic",gpu.data(),gpu.size(),7,{256,16,5120,1},{1,2,4096,0}};
    if (llama_moe_verify_enabled()) throw std::runtime_error("diagnostic unexpectedly enabled by default");
    llama_moe_verify_range("selected-bank",&t,11,host.data(),size,511,4095);
    if (read_calls != 3 || largest_read != 4*1024*1024) throw std::runtime_error("chunk bound failed");
    if (llama_moe_verify_counts().calls[1] != 1 || llama_moe_verify_counts().bytes[1] != size)
        throw std::runtime_error("positive coverage failed");
    gpu[11] ^= 1;
    reject([&]{llama_moe_verify_range("selected-bank",&t,11,host.data(),size,511,4095);},"offset=11");
    gpu[11] ^= 1; gpu.back() ^= 1;
    reject([&]{llama_moe_verify_range("selected-bank",&t,11,host.data(),size,511,4095);},"MOE_VERIFY FAIL");
    gpu.back() ^= 1;
    reject([&]{llama_moe_verify_range("slotmap",&t,t.size+1,host.data(),0,-1,-1);},"assert:");
    reject([&]{llama_moe_verify_range("slotmap",&t,1,host.data(),t.size,-1,-1);},"assert:");
    llama_moe_verify_range("slotmap",&t,t.size,host.data(),0,-1,-1);
    ggml_tensor matching=t; llama_moe_verify_layout(&t,&matching);
    matching.type++; reject([&]{llama_moe_verify_layout(&t,&matching);},"assert:"); matching=t;
    matching.ne[0]++; reject([&]{llama_moe_verify_layout(&t,&matching);},"assert:"); matching=t;
    matching.ne[1]++; reject([&]{llama_moe_verify_layout(&t,&matching);},"assert:"); matching=t;
    matching.nb[2]++; reject([&]{llama_moe_verify_layout(&t,&matching);},"assert:");
    other_translation_unit();
    if (llama_moe_verify_counts().calls[4] != 1) throw std::runtime_error("coverage did not merge across translation units");
    llama_moe_verify_coverage();
    std::cout << "PASS positive/chunking/first-last mismatch/bounds/layout/default-off/cross-TU coverage\n";
}
''')
        script = '\n'.join([
            'set -euo pipefail', 'unset LLAMA_MOE_VERIFY_BYTES',
            'g++ -std=c++17 -fsyntax-only -I/src/include -I/src/ggml/include -I/src/src /work/src/llama-moe-cache.cpp /work/src/llama-context.cpp',
            'g++ -std=c++17 -O1 -fsanitize=address,undefined -fno-omit-frame-pointer /work/probe/main.cpp /work/probe/other.cpp -o /work/probe/test',
            '/work/probe/test',
        ])
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--user',
                        str(os.getuid()) + ':' + str(os.getgid()),
                        '--entrypoint', '/bin/bash', '-v', str(SOURCE) + ':/src:ro',
                        '-v', str(directory) + ':/work', IMAGE, '-lc', script], check=True)
    print('CPU-only tests passed; no GPU devices or GGML runtime libraries loaded.')


if __name__ == '__main__':
    main()
