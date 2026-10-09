#!/usr/bin/env python3
"""CPU global-budget and stage-affinity/lifetime tests using an explicit SYCL mock."""
from pathlib import Path
import hashlib
import os
import subprocess
import tempfile
import test_native_hc_owner_cpu as base

EXTRA=['sycl/include/strata/core/gguf_expert_source.hpp','sycl/src/core/gguf_expert_source.cpp',
       'include/strata/core/verify.hpp','sycl/src/program/generate.cpp',
       'sycl/include/strata/kernels/resident_plan_mirror.hpp','sycl/src/kernels/cuda/verify_kernels.dp.cpp']
PATCHES=base.PATCHES+['0004-sycl-native-hc-prefill-verifier-routes.patch',
                     '0005-sycl-stage-owned-expert-mirrors.patch']


def main():
    with tempfile.TemporaryDirectory(prefix='strata-stage-mirror-cpu-') as temp:
        work=Path(temp)
        for name in base.FILES+EXTRA:
            path=work/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((base.SOURCE/name).read_bytes())
        for name in PATCHES:
            patch=base.ROOT/'strata/flash-next/patches'/name
            subprocess.run(['git','apply','--check',str(patch)],cwd=work,check=True)
            subprocess.run(['git','apply',str(patch)],cwd=work,check=True)
        mock=work/'mock/sycl/sycl.hpp'
        mock.parent.mkdir(parents=True)
        mock.write_text(r'''
#pragma once
#include <vector>
#include <cstdint>
namespace sycl {
struct Freed { void* pointer; int context,device; };
inline std::vector<Freed> freed;
inline int waits=0;
struct queue {
    int context,device;
    queue(int c=0,int d=0):context(c),device(d){}
    int get_context() const { return context; }
    int get_device() const { return device; }
    void wait_and_throw() { ++waits; }
};
inline void free(void* pointer,const queue& q) { freed.push_back({pointer,q.context,q.device}); }
}
''')
        (work/'contracts.cpp').write_text(r'''
#include "strata/core/stage_mirror_budget.hpp"
#include "strata/core/stage_expert_mirror.hpp"
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
int main() {
    using namespace strata::core;
    constexpr uint64_t G=uint64_t(1)<<30;
    const uint64_t reserves[]={8*G,8*G,8*G,8*G};
    StageMirrorBudget budget;
    assert(StageMirrorBudget::make(121*G,UINT64_MAX,reserves,4,1024,budget));
    assert(budget.cap==89*G-1024 && budget.used==0);
    const uint64_t stage0=budget.remaining()/2;
    assert(budget.commit(stage0));
    assert(budget.remaining()==89*G-1024-stage0); // reserves counted once
    assert(!budget.commit(budget.remaining()+1));
    assert(budget.commit(budget.remaining()) && budget.remaining()==0);
    assert(!budget.commit(1));
    assert(StageMirrorBudget::make(121*G,32<<20,reserves,4,0,budget) && budget.cap==32<<20);
    assert(StageMirrorBudget::make(8*G,UINT64_MAX,reserves,4,0,budget) && budget.cap==0);
    assert(StageMirrorBudget::make(121*G,0,reserves,4,0,budget) && budget.cap==0);
    const uint64_t overflow[]={UINT64_MAX};
    assert(!StageMirrorBudget::make(UINT64_MAX,UINT64_MAX,overflow,1,1,budget));
    std::string err; uint64_t bytes=0;
    assert(stage_mirror_mib(nullptr,123,bytes,err) && bytes==123);
    assert(stage_mirror_mib("0",999,bytes,err) && bytes==0);
    assert(stage_mirror_mib("1024",0,bytes,err) && bytes==G);
    for (const char* bad : {"","-1","+1","1x","1.5","18446744073709551616","17592186044416"})
        assert(!stage_mirror_mib(bad,0,bytes,err));
    assert(stage_mirror_mib("17592186044415",0,bytes,err));
    std::vector<int32_t> res0(48*512),res1(48*512);
    std::vector<unsigned long long> table0(48*512),table1(48*512);
    uint8_t host0,host1;
    auto a=std::make_shared<StageExpertMirror>(sycl::queue(10,0));
    auto b=std::make_shared<StageExpertMirror>(sycl::queue(20,1));
    a->lb=0;a->le=24;a->n_layers=48;a->n_expert=512;a->table=table0.data();a->host=&host0;
    b->lb=24;b->le=48;b->n_layers=48;b->n_expert=512;b->table=table1.data();b->host=&host1;
    assert(a->bind(res0.data(),48,512,sycl::queue(10,0),err));
    assert(b->bind(res1.data(),48,512,sycl::queue(20,1),err));
    assert(a->bind(res0.data(),48,512,sycl::queue(10,0),err));
    assert(!a->bind(res1.data(),48,512,sycl::queue(10,0),err));
    assert(!a->bind(res0.data(),48,512,sycl::queue(11,0),err));
    assert(!a->bind(res0.data(),48,512,sycl::queue(10,1),err));
    assert(!a->bind(res0.data(),47,512,sycl::queue(10,0),err));
    assert(!a->bind(res0.data(),48,511,sycl::queue(10,0),err));
    assert(!a->bind(nullptr,48,512,sycl::queue(10,0),err));
    assert(a->matches(res0.data(),48,512,sycl::queue(10,0)));
    assert(!a->matches(res1.data(),48,512,sycl::queue(20,1)));
    assert(a->layer_table(0)==table0.data() && a->layer_table(23)==table0.data()+23*512);
    assert(!a->layer_table(-1) && !a->layer_table(24));
    assert(!b->layer_table(23) && b->layer_table(24)==table1.data()+24*512 && !b->layer_table(48));
    // Simulate stage/source/graph shared ownership. No frees until last graph
    // releases its owner, even after source and stage drop their own references.
    auto graph=a; auto source=a; a.reset(); source.reset();
    assert(sycl::freed.empty());
    graph.reset(); assert(sycl::freed.size()==2);
    assert(sycl::freed[0].pointer==table0.data() && sycl::freed[1].pointer==&host0);
    assert(sycl::freed[0].context==10 && sycl::freed[1].context==10);
    b->release(); assert(sycl::freed.size()==4 && !b->host && !b->table);
    assert(sycl::freed[2].context==20 && sycl::freed[3].device==1);
    b.reset(); assert(sycl::freed.size()==4); // checked release is idempotent
    std::printf("PASS CPU global cap/reserves/overflow/parser, two-stage affinity and mocked graph-owner teardown\n");
}
''')
        cmd=('g++ -std=c++17 -O1 -fsanitize=address,undefined -I/work/mock -I/work/sycl/include '
             '/work/contracts.cpp -o /work/contracts && /work/contracts')
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
                        '--entrypoint','/bin/bash','-v',str(work)+':/work',base.IMAGE,'-lc',cmd],check=True)
    patch=base.ROOT/'strata/flash-next/patches'/PATCHES[-1]
    print('patch0005 SHA256',hashlib.sha256(patch.read_bytes()).hexdigest())
    print('SYCL source compilation, actual contexts, GPU-consumed bytes, tier counters and live teardown remain unqualified.')


if __name__=='__main__':
    main()
