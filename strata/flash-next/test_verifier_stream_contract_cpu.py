#!/usr/bin/env python3
"""CPU-only actual queue contract negatives and source constructor regression."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import hashlib

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T080647Z-6_ja3xd5/source')
PATCH=ROOT/'strata/flash-next/patches/0018-sycl-verifier-init-owned-streams.patch'
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def main():
    with tempfile.TemporaryDirectory(prefix='verifier-stream-cpu-') as temp:
        work=Path(temp)
        for line in PATCH.read_text().splitlines():
            if line.startswith('+++ b/'):
                name=line[6:];p=SOURCE/name
                if p.exists():(work/name).parent.mkdir(parents=True,exist_ok=True);(work/name).write_bytes(p.read_bytes())
        subprocess.run(['git','apply',str(PATCH)],cwd=work,check=True)
        header=(work/'sycl/include/strata/core/verify.hpp').read_text()
        for field in ['ext_stream_','cs_','sh_cs_','copy_']:
            assert re.search(r'dpct::queue_ptr\s+'+field+r'\s*=\s*nullptr\s*;',header),field
        body=(work/'sycl/src/core/verify.cpp').read_text()
        assert 'ext_stream_ != &dpct::get_in_order_queue()' not in body
        assert 'if(!init_capture_streams(err)) return false;' in body
        assert 'hits.stage_mirror->bind(hits.d_res,g.n_layers,g.n_expert,*cs_,err)' in body
        assert 'hits_.stage_mirror->matches(hits_.d_res,g.n_layers,g.n_expert,*cs)' in body
        mock=work/'mock/sycl/sycl.hpp';mock.parent.mkdir(parents=True)
        mock.write_text(r'''
#pragma once
namespace sycl {namespace property {namespace queue {struct in_order {};}}
struct queue {int context,device;bool ordered;queue(int c,int d,bool o=true):context(c),device(d),ordered(o){}
int get_context() const{return context;}int get_device() const{return device;}
template<class P>bool has_property() const{return ordered;}};}
''')
        (work/'test.cpp').write_text(r'''
#include "strata/core/verifier_stream_contract.hpp"
#include <cassert>
#include <cstdio>
int main(){using strata::core::verifier_stream_contract;std::string error;
sycl::queue q0(10,0),q1(20,1),same(20,1),foreign_context(21,1),foreign_device(20,0),unordered(20,1,false);
assert(verifier_stream_contract(nullptr,q0,error));assert(verifier_stream_contract(nullptr,q1,error));
assert(verifier_stream_contract(&q1,q1,error));assert(verifier_stream_contract(&same,q1,error));
assert(!verifier_stream_contract(&q0,q1,error));assert(!verifier_stream_contract(&foreign_context,q1,error));
assert(!verifier_stream_contract(&foreign_device,q1,error));assert(!verifier_stream_contract(&unordered,q1,error));
// This is the exact former port bug: construction's GPU0 pointer is mistakenly
// interpreted as an explicit external stream when initialized on GPU1.
sycl::queue* old_external=&q0;assert(old_external!=&q1);assert(!verifier_stream_contract(old_external,q1,error));
// Nullable default remains unspecified across construction/init device changes.
sycl::queue* corrected_external=nullptr;assert(verifier_stream_contract(corrected_external,q1,error));
std::printf("PASS actual queue contract four acceptances/five negatives plus nullable-construction regression\n");}
''')
        command='g++ -std=c++17 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all -I/work/mock -I/work/sycl/include /work/test.cpp -o /work/test && /work/test'
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
                        '--entrypoint','/bin/bash','-v',str(work)+':/work',IMAGE,'-lc',command],check=True)
    print(json.dumps({'mode':'CPU_EXPLICIT_SYCL_MOCK','passed':True,'four_stream_fields_null_before_init':True,
        'engine_uses_actual_control_initializer':True,'mirror_guard_preserved':True,
        'patch_sha256':hashlib.sha256(PATCH.read_bytes()).hexdigest(),'gpu_capture_qualified':False},sort_keys=True))


if __name__=='__main__':main()
