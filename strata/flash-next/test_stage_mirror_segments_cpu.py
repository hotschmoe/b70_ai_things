#!/usr/bin/env python3
"""Actual segmented planner/budget/owner tests; explicit CPU mock, no GPU."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ENGINE = Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T052930Z-dzzo14o3')
PACK = Path('/mnt/vm_8tb/b70/models/flashnext-native-source-pack-20261009-v3')
IMAGE = 'sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
PATCH = ROOT / 'strata/flash-next/patches/0012-sycl-segmented-stage-expert-mirrors.patch'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    profile = (ENGINE / 'source/data/expert-profile.bin').read_bytes()
    require_profile = struct.unpack_from('<5I', profile, 4)
    assert profile[:4] == b'STRP' and require_profile[1:3] == (48, 512)
    pairs = [struct.unpack_from('<HH', profile, 24+4*i) for i in range(require_profile[4])]
    resident = set(pairs[:8083])
    sizes = {int(line.split()[0]): int(line.split()[4]) for line in (PACK / 'native_experts.txt').read_text().splitlines()
             if line and not line.startswith('#')}
    missing = [(l, e) for l, e in pairs if (l, e) not in resident]
    assert len(missing) == 16493 and len(set(missing)) == len(missing)
    padded = [(sizes[l]+255)//256*256 for l, e in missing]
    needed = sum(padded)
    assert needed == 51688243200
    with tempfile.TemporaryDirectory(prefix='stage-segments-cpu-') as tmp:
        tmp = Path(tmp)
        names = [line[6:] for line in PATCH.read_text().splitlines() if line.startswith('+++ b/')]
        for name in names:
            original = ENGINE / 'source' / name
            if original.exists():
                target = tmp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
        subprocess.run(['git', 'apply', '--check', str(PATCH)], cwd=tmp, check=True)
        subprocess.run(['git', 'apply', str(PATCH)], cwd=tmp, check=True)
        (tmp / 'actual-padded.bin').write_bytes(b''.join(struct.pack('<Q', n) for n in padded))
        mock = tmp / 'mock/sycl/sycl.hpp'
        mock.parent.mkdir(parents=True)
        mock.write_text(r'''
#pragma once
#include <vector>
#include <stdexcept>
namespace sycl {
struct Freed {void* pointer;int context,device;};
inline std::vector<Freed> freed;inline void* fail_free=nullptr;inline int waits=0;
struct queue {int context,device;queue(int c=0,int d=0):context(c),device(d){}
    int get_context() const{return context;}int get_device() const{return device;}
    void wait_and_throw(){++waits;}};
inline void free(void* p,const queue& q){if(p==fail_free) throw std::runtime_error("injected free failure");freed.push_back({p,q.context,q.device});}
}
''')
        (tmp / 'contracts.cpp').write_text(r'''
#include "strata/core/stage_mirror_segments.hpp"
#include "strata/core/stage_mirror_budget.hpp"
#include "strata/core/stage_expert_mirror.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <vector>
int main() {
    using namespace strata::core;constexpr uint64_t G=uint64_t(1)<<30;std::string error;
    std::ifstream source("/work/actual-padded.bin",std::ios::binary);std::vector<uint64_t> production;uint64_t value;
    while(source.read(reinterpret_cast<char*>(&value),8)) production.push_back(value);
    assert(production.size()==16493);
    for(uint64_t limit:{64ull<<20,256ull<<20,512ull<<20,1024ull<<20}) {
        StageMirrorSegmentPlan plan;assert(StageMirrorSegmentPlan::make(production,limit,plan,error));
        assert(plan.bytes==51688243200ull && plan.locations.size()==production.size());
        assert(std::accumulate(plan.sizes.begin(),plan.sizes.end(),uint64_t(0))==plan.bytes);
        std::vector<uint64_t> filled(plan.sizes.size());
        for(size_t i=0;i<production.size();++i) {auto loc=plan.locations[i];assert(loc.offset==filled[loc.segment]);assert(loc.offset%256==0);assert(production[i]<=plan.sizes[loc.segment]-loc.offset);filled[loc.segment]+=production[i];}
        assert(filled==plan.sizes);for(auto n:plan.sizes) assert(n>0 && n<=limit && n%256==0);
        std::printf("PASS actual16493-expert plan: limit%llu segments%zu bytes%llu\n",(unsigned long long)limit,plan.sizes.size(),(unsigned long long)plan.bytes);
    }
    StageMirrorSegmentPlan keep;assert(StageMirrorSegmentPlan::make({256},1024,keep,error));
    for(uint64_t limit:std::vector<uint64_t>{0,255,1025,G+256}) assert(!StageMirrorSegmentPlan::make({256},limit,keep,error));
    for(auto sizes:std::vector<std::vector<uint64_t>>{{0},{257},{1280},{UINT64_MAX}}) assert(!StageMirrorSegmentPlan::make(sizes,1024,keep,error));
    assert(keep.bytes==256 && keep.sizes.size()==1 && keep.locations.size()==1);
    StageMirrorSegmentPlan empty;assert(StageMirrorSegmentPlan::make({},1024,empty,error) && empty.bytes==0 && empty.sizes.empty());
    // Original source bytes preserved across arbitrary noncontiguous buffers.
    const std::vector<uint64_t> raw={257,128,600,255,700},pad={512,256,768,256,768};StageMirrorSegmentPlan small;
    assert(StageMirrorSegmentPlan::make(pad,1024,small,error));std::vector<std::vector<uint8_t>> segments;
    for(uint64_t n:small.sizes) segments.emplace_back(n,0xa5);std::vector<unsigned long long> table;
    for(size_t i=0;i<raw.size();++i) {auto loc=small.locations[i];auto* p=segments[loc.segment].data()+loc.offset;table.push_back(reinterpret_cast<unsigned long long>(p));for(size_t b=0;b<raw[i];++b) p[b]=uint8_t((i*37+b*19)%256);}
    for(size_t i=0;i<raw.size();++i) {auto* p=reinterpret_cast<const uint8_t*>(table[i]);for(size_t b=0;b<raw[i];++b) assert(p[b]==uint8_t((i*37+b*19)%256));for(size_t b=raw[i];b<pad[i];++b) assert(p[b]==0xa5);}
    const uint64_t reserve[]={8*G,8*G,8*G,8*G};StageMirrorBudget budget;
    assert(StageMirrorBudget::make(121*G,64*G,reserve,4,1024,budget));
    assert(budget.covers_all({40*G,10*G}) && budget.covers_all({0,64*G}) && !budget.covers_all({40*G,25*G}));
    assert(!budget.covers_all({UINT64_MAX,1}) && budget.used==0);assert(budget.commit(40*G));assert(budget.covers_all({10*G}));assert(!budget.covers_all({25*G}));
    // Owning context/device and graph lifetimes remain actual owner methods.
    uint8_t a,b,c;unsigned long long device_table;
    auto owner=std::make_shared<StageExpertMirror>(sycl::queue(10,0));owner->segments={{&a,256},{&b,256},{&c,256}};owner->table=&device_table;
    auto graph=owner;owner.reset();assert(sycl::freed.empty());graph.reset();assert(sycl::freed.size()==4);
    for(auto free:sycl::freed) assert(free.context==10 && free.device==0);
    auto partial=std::make_shared<StageExpertMirror>(sycl::queue(20,1));partial->segments={{&a,256},{&b,256},{&c,256}};partial->table=&device_table;
    sycl::fail_free=&b;bool failed=false;try{partial->release();}catch(const std::runtime_error&){failed=true;}assert(failed && !partial->table && !partial->segments[0].base && partial->segments[1].base==&b);
    assert(sycl::freed.size()==6);sycl::fail_free=nullptr;partial->release();assert(sycl::freed.size()==8 && partial->segments.empty());partial->release();partial.reset();assert(sycl::freed.size()==8);
    for(size_t i=4;i<8;++i) assert(sycl::freed[i].context==20 && sycl::freed[i].device==1);
    std::printf("PASS actual planner/source-pointer bytes/global-budget/graph-lifetime/context/free-retry CPU contracts\n");
}
''')
        cmd = 'g++ -std=c++17 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all -I/work/mock -I/work/sycl/include /work/contracts.cpp -o /work/contracts && /work/contracts'
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '--entrypoint','/bin/bash','-v',str(tmp)+':/work',IMAGE,'-lc',cmd],check=True)
    print(json.dumps({'mode':'CPU_EXPLICIT_SYCL_MOCK','patch_sha256':sha(PATCH),'actual_missing_experts':len(missing),
        'actual_required_padded_bytes':needed,'profile_sha256':sha(ENGINE/'source/data/expert-profile.bin'),
        'native_experts_sha256':sha(PACK/'native_experts.txt'),'gpu_allocation_limit_proven':False,'passed':True},sort_keys=True))


if __name__ == '__main__':
    main()
