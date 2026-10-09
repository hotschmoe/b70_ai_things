#!/usr/bin/env python3
"""CPU-only shared native HC bind/chunk contracts; mocked device launchers."""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile
import test_native_hc_owner_cpu as base
import native_hc_artifact as artifact

PATCHES = base.PATCHES+['0004-sycl-native-hc-prefill-verifier-routes.patch']


def main():
    with tempfile.TemporaryDirectory(prefix='strata-native-hc-routes-cpu-') as temp:
        work=Path(temp)
        for name in base.FILES:
            target=work/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((base.SOURCE/name).read_bytes())
        for name in PATCHES:
            patch=base.ROOT/'strata/flash-next/patches'/name
            subprocess.run(['git','apply','--check',str(patch)],cwd=work,check=True)
            subprocess.run(['git','apply',str(patch)],cwd=work,check=True)
        (work/'hc_fixture.hpp').write_text(artifact.fixture_header())
        (work/'dispatch.cpp').write_text(r'''
#include "strata/core/native_hc_dispatch.hpp"
#include "strata/core/native_hc_source.hpp"
#include "hc_fixture.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
struct Launch { float* R; const float* bo; const float* inj; int n; };
std::vector<Launch> calls;
namespace strata::kernels {
// CPU mocks capture descriptors only. No data access or GPU arithmetic.
bool hc_native_write_f32(float* R,size_t r,const float* bo,size_t b,const float* inj,size_t i,int n,void*) {
    assert(hc_native_write_shape_ok(r,b,i,n)); calls.push_back({R,bo,inj,n}); return true;
}
bool hc_native_read_f32(const HcNativeArgs&,void*) { assert(false); return false; }
}
int main() {
    using namespace strata::core; using namespace strata::kernels;
    float f; uint8_t q;
    auto from_artifact=[&](const std::string& name) {
        WeightRef w;
        for (const auto& e:artifact_hc) if (name==e.name) {
            assert(native_hc_source_shape(e.role,e.type,e.rank==1?std::vector<uint64_t>{e.ne0}:std::vector<uint64_t>{e.ne0,e.ne1}));
            assert(native_hc_source_bytes(e.role)==e.bytes);
            assert(native_hc_source_bounds(e.data_start,e.relative,e.bytes,e.file_size));
            assert(e.absolute==e.data_start+e.relative);
            w.hc_source_type=e.type; w.hc_source_rank=e.rank;
            w.hc_source_ne0=e.ne0; w.hc_source_ne1=e.ne1; w.hc_source_bytes=e.bytes;
            if (e.type==0) w.hc_source_f32=&f; else w.hc_source_q8=&q;
            return w;
        }
        assert(false); return w;
    };
    WeightRef norm=from_artifact("blk.0.hc_attn_norm.weight"),down=from_artifact("blk.0.hc_attn_down.weight"),
        up=from_artifact("blk.0.hc_attn_up.weight"),inject=from_artifact("blk.0.hc_attn_inject.weight");
    std::string err; HcNativeArgs a;
    assert(native_hc_bind(&norm,&down,&up,&inject,a,err));
    assert(a.w.norm==&f && a.w.inject==&f && a.w.down==&q && a.w.up==&q);
    HcNativeArgs head;
    assert(native_hc_bind(&norm,&down,&up,nullptr,head,err));
    assert(!head.w.inject && !head.w.inject_floats);
    int64_t stage_lo,stage_hi;
    assert(native_hc_effective_bounds(0,-1,0,24,stage_lo,stage_hi) && stage_lo==0 && stage_hi==24);
    assert(native_hc_effective_bounds(0,-1,24,48,stage_lo,stage_hi) && stage_lo==24 && stage_hi==48);
    assert(native_hc_effective_bounds(0,24,24,48,stage_lo,stage_hi) && stage_lo==0 && stage_hi==24);
    assert(native_hc_effective_bounds(0,-1,-1,-1,stage_lo,stage_hi) && stage_lo==0 && stage_hi==-1);
    assert(!native_hc_effective_bounds(0,0,-1,-1,stage_lo,stage_hi));
    assert(!native_hc_effective_bounds(0,-1,24,24,stage_lo,stage_hi));
    assert(!native_hc_effective_bounds(-1,24,-1,-1,stage_lo,stage_hi));
    int first_owned=0,last_owned=0;
    for (const auto& e:artifact_hc) {
        const std::string name=e.name;
        if (name.rfind("blk.",0)==0) {
            const int layer=std::stoi(name.substr(4));
            if (layer<24) ++first_owned; else ++last_owned;
        } else ++last_owned; // head only belongs to final stage
    }
    assert(first_owned==192 && last_owned==195);
    int artifact_bindings=0;
    for (int layer=0;layer<48;++layer) for (const char* half : {"attn","ffn"}) {
        const std::string pre="blk."+std::to_string(layer)+".hc_"+half+"_";
        auto n=from_artifact(pre+"norm.weight"),d=from_artifact(pre+"down.weight"),u=from_artifact(pre+"up.weight"),i=from_artifact(pre+"inject.weight");
        HcNativeArgs actual; assert(native_hc_bind(&n,&d,&u,&i,actual,err)); ++artifact_bindings;
    }
    auto hn=from_artifact("output_hc_norm.weight"),hd=from_artifact("output_hc_down.weight"),hu=from_artifact("output_hc_up.weight");
    assert(native_hc_bind(&hn,&hd,&hu,nullptr,head,err)); ++artifact_bindings;
    assert(artifact_bindings==97);
    assert(!native_hc_source_shape(0,0,{2560,4}));
    assert(!native_hc_source_shape(0,0,{10240,1}));
    assert(!native_hc_source_bounds(100,100,10,109));
    assert(!native_hc_source_bounds(100,0,10,99));
    int rejected=0;
    for (int role=0;role<4;++role) for (int defect=0;defect<3;++defect) {
        WeightRef n=norm,d=down,u=up,i=inject; WeightRef* roles[]={&n,&d,&u,&i};
        if (defect==0) roles[role]->hc_source_type=-1;
        if (defect==1) roles[role]->hc_source_ne0=1;
        if (defect==2) ++roles[role]->hc_source_ne1;
        HcNativeArgs bad; assert(!native_hc_bind(&n,&d,&u,&i,bad,err)); ++rejected;
    }
    for (int n=1;n<=8;++n) {
        assert(hc_native_write_shape_ok(size_t(n)*10240,size_t(n)*2560,size_t(n)*4,n));
        assert(!hc_native_write_shape_ok(size_t(n)*10240-1,size_t(n)*2560,size_t(n)*4,n));
        assert(!hc_native_write_shape_ok(size_t(n)*10240,size_t(n)*2560-1,size_t(n)*4,n));
        assert(!hc_native_write_shape_ok(size_t(n)*10240,size_t(n)*2560,size_t(n)*4-1,n));
    }
    assert(!hc_native_write_shape_ok(0,0,0,0));
    assert(!hc_native_write_shape_ok(9*10240,9*2560,9*4,9));
    std::vector<float> R(17*10240),bo(17*2560),inj(17*4);
    for (int rows : {1,2,8,9,17}) {
        calls.clear(); assert(native_hc_write_rows(R.data(),bo.data(),inj.data(),rows,nullptr,err));
        int seen=0;
        for (auto launch : calls) {
            assert(launch.R==R.data()+size_t(seen)*10240);
            assert(launch.bo==bo.data()+size_t(seen)*2560 && launch.inj==inj.data()+size_t(seen)*4);
            assert(launch.n>=1 && launch.n<=8 && seen+launch.n<=rows);
            seen+=launch.n;
        }
        assert(seen==rows);
    }
    calls.clear();
    assert(!native_hc_write_rows(nullptr,bo.data(),inj.data(),1,nullptr,err));
    assert(!native_hc_write_rows(R.data(),nullptr,inj.data(),1,nullptr,err));
    assert(!native_hc_write_rows(R.data(),bo.data(),nullptr,1,nullptr,err));
    assert(!native_hc_write_rows(R.data(),bo.data(),inj.data(),0,nullptr,err));
    assert(calls.empty());
    std::printf("PASS CPU contracts:387 artifact descriptors,97 actual bindings,7 effective-bound cases,192/195 stage source ownership counts,%d rejected bindings,8 write shapes,26 shape rejections,5 chunk layouts,4 prelaunch rejections\n",rejected);
}
''')
        command=('g++ -std=c++17 -O1 -fsanitize=address,undefined -DSTRATA_SYCL_Q8_HC_BUILT=1 '
                 '-I/work/sycl/include -I/work/include /work/dispatch.cpp -o /work/dispatch && STRATA_SYCL_NATIVE_HC=1 /work/dispatch')
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
                        '--entrypoint','/bin/bash','-v',str(work)+':/work',base.IMAGE,'-lc',command],check=True)
    patch=base.ROOT/'strata/flash-next/patches'/PATCHES[-1]
    print('patch0004 SHA256',hashlib.sha256(patch.read_bytes()).hexdigest())
    print('Model execution, SYCL translation units and GPU write/read arithmetic remain unqualified.')


if __name__ == '__main__':
    main()
