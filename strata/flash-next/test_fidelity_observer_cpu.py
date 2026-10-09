#!/usr/bin/env python3
"""CPU-only frozen-v6 observer contracts with explicit mock SYCL allocation/copies."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'


def main():
    plan=json.loads((ROOT/'strata/flash-next/fidelity-observer-source-plan.json').read_text())
    source=Path(plan['base_source']);patch=ROOT/plan['patch']
    assert hashlib.sha256(patch.read_bytes()).hexdigest()==plan['patch_sha256']
    with tempfile.TemporaryDirectory(prefix='strata-fidelity-cpu-') as name:
        work=Path(name)
        for file,expected in plan['base_files'].items():
            raw=(source/file).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
            out=work/file;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
        subprocess.run(['git','apply','--check',str(patch)],cwd=work,check=True)
        subprocess.run(['git','apply',str(patch)],cwd=work,check=True)
        p=work/'mock/sycl/sycl.hpp';p.parent.mkdir(parents=True)
        p.write_text(r'''
#pragma once
#include <cstring>
#include <cstdlib>
#include <stdexcept>
namespace sycl {
inline int allocs=0,frees=0,copies=0,waits=0;
struct event{void wait_and_throw()const{++waits;}};
struct queue{int context,device;queue(int c=1,int d=0):context(c),device(d){}
 int get_context()const{return context;}int get_device()const{return device;}
 void wait_and_throw(){++waits;}
 event memcpy(void* dst,const void* src,size_t bytes){++copies;std::memcpy(dst,src,bytes);return {};}
};
template<class T>T* malloc_device(size_t count,queue&){++allocs;return static_cast<T*>(std::malloc(count*sizeof(T)));}
inline void free(void* p,queue&){++frees;std::free(p);}
}
''')
        (work/'captures').mkdir();(work/'arm').write_text('CPU only\n')
        (work/'observer.cpp').write_text(r'''
#include "strata/core/fidelity_observer.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
using namespace strata::core::fidelity_diag;
int main(int argc,char** argv){
 assert(argc==2);bool on=!std::strcmp(argv[1],"on");
 assert(geometry(2560,4,48,248320));assert(geometry(2560,4,48,0));
 assert(!geometry(2560,1,48,248320));assert(!geometry(4096,4,48,248320));assert(!geometry(2560,4,48,1));
 size_t off=0;assert(slot_offset(0,24,0,off)&&off==0);assert(slot_offset(24,48,47,off)&&off==23*10240);
 assert(!slot_offset(0,24,24,off));assert(!slot_offset(24,48,23,off));assert(!slot_offset(-1,24,0,off));
 assert(first_row(1,0,20,21));assert(!first_row(2,0,20,21));assert(!first_row(1,1,20,21));assert(!first_row(1,0,19,21));
 begin({11,12},true);
 if(!on){assert(!current().active && !sycl::allocs && !sycl::copies);std::puts("PASS observer default off: no mock allocation/copy");return 0;}
 assert(current().active && current().ordinal==1);resumed(0);
 sycl::queue q(17,1);
 std::vector<uint32_t> residual(D,0x3f800000u),logits(VOCAB,0x40000000u);
 residual[0]=0x80000000u;residual[1]=0x7fc12345u;
 std::string err;
 {snapshot snap(q,1,24,48,true);assert(sycl::allocs==1);
  for(int l=24;l<48;++l)snap.layer(l,reinterpret_cast<const float*>(residual.data()),q);
  snap.logits(reinterpret_cast<const float*>(logits.data()),q);
  assert(sycl::copies==25);assert(snap.dump("first_window_residual",1,12,true,err));
  bool rejected=false;try{sycl::queue bad(18,1);snap.layer(24,reinterpret_cast<const float*>(residual.data()),bad);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
 }
 assert(sycl::frees==1);
 int n=0;for(const auto& entry:std::filesystem::directory_iterator(std::getenv("STRATA_FIDELITY_DIAG_DIR"))){
  ++n;auto size=std::filesystem::file_size(entry.path());assert(size==ROW_BYTES || size==size_t(VOCAB)*4);
  if(size==ROW_BYTES){std::ifstream in(entry.path(),std::ios::binary);char first[8];in.read(first,8);
   assert(uint8_t(first[3])==0x80 && uint8_t(first[4])==0x45 && uint8_t(first[7])==0x7f);}
 }
 assert(n==25);for(int i=0;i<6;++i)begin({11,12},true);assert(!current().active && current().ordinal==7);
 std::puts("PASS CPU shape/stage/first-row bounds, mock context ownership and canonical byte snapshots");
}
''')
        cmd=('g++ -std=c++17 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all '
             '-I/work/mock -I/work/sycl/include /work/observer.cpp -o /work/observer\n'
             '/work/observer off\n'
             'STRATA_FIDELITY_DIAG=1 STRATA_FIDELITY_DIAG_ACTIVATIONS=1 '
             'STRATA_FIDELITY_DIAG_ARM=/work/arm STRATA_FIDELITY_DIAG_DIR=/work/captures /work/observer on')
        result=subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '--entrypoint','/bin/bash','-v',str(work)+':/work',IMAGE,'-lc','set -e\n'+cmd],check=True,capture_output=True,text=True)
        # Keep the many mocked raw-vector records out of terminal progress; their
        # exact capture count/byte sizes are asserted above.
        print(result.stdout.strip())
    print('Patch SHA256',plan['patch_sha256'])
    print('Actual SYCL TUs/graphs, GPU state, full model outputs and off/on equivalence remain unqualified.')


if __name__=='__main__':main()
