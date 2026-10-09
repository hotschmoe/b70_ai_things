#!/usr/bin/env python3
"""0021 reconstructed source + actual helper under mock SYCL; no GPU/full TU."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from prepare_layer0_capture_patch_v1 import construct,BASE

HERE=Path(__file__).resolve().parent
MOCK=r'''#pragma once
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#include <array>
namespace sycl {
namespace property { namespace queue {struct in_order{};}}
inline int allocations=0,frees=0,copies=0,waits=0;
struct event{void wait_and_throw()const{++waits;}};
struct state{int context=1,device=0,key=-1;bool in_order=true;std::array<std::vector<std::function<void()>>,4>graphs;};
class queue {
 std::shared_ptr<state>s_;
public:
 queue(int context=1,int device=0,bool ordered=true):s_(std::make_shared<state>()){s_->context=context;s_->device=device;s_->in_order=ordered;}
 int get_context()const{return s_->context;}int get_device()const{return s_->device;}
 template<class T>bool has_property()const{return s_->in_order;}
 void wait_and_throw()const{++waits;}
 event memcpy(void*dst,const void*src,std::size_t n){++copies;auto f=[=]{std::memcpy(dst,src,n);};if(s_->key<0)f();else s_->graphs[s_->key].push_back(f);return {};}
 void begin(int k){s_->graphs[k].clear();s_->key=k;}void end(){s_->key=-1;}void replay(int k){for(auto&f:s_->graphs[k])f();}
};
template<class T>T*malloc_device(std::size_t n,const queue&){++allocations;return static_cast<T*>(std::calloc(n,sizeof(T)));}
inline void free(void*p,const queue&){++frees;std::free(p);}
}
'''
TEST=r'''#include "strata/core/layer0_numerical_observer.hpp"
#include <cassert>
#include <iostream>
using namespace strata::core::layer0_diag;
int main(int argc,char**argv){sycl::queue q;bool off=argc==2&&std::string(argv[1])=="off";
 if(off){snapshot s(q,0);begin({1},true);s.copy("invalid",nullptr,0,q);s.release();assert(sycl::allocations==0&&sycl::copies==0&&sycl::frees==0);std::cout<<"OFF_PASS\n";return 0;}
 snapshot s(q,0);std::vector<std::vector<std::uint8_t>>raw(fields.size());for(std::size_t i=0;i<fields.size();++i)raw[i].assign(fields[i].bytes,std::uint8_t(i+1));int rejected=0;
 auto reject=[&](auto f){try{f();}catch(const std::exception&){++rejected;return;}assert(false);};
 auto build=[&](int key){s.begin_roster(key);q.begin(key);for(std::size_t i=0;i<fields.size();++i)if(i<24||i>=31)s.copy(fields[i].name,raw[i].data(),raw[i].size(),q);s.stamp(q);q.end();s.seal_roster(key);};
 build(0);begin({10},true);resumed(0);std::string err;
 reject([&]{s.arm(1,1,0,10,q);}); // no constructed graph1
 reject([&]{s.arm(0,1,1,10,q);}); // wrong input position
 s.arm(0,1,0,10,q);assert(!s.dump(0,1,0,10,err));++rejected; // no replay
 q.replay(0);assert(s.dump(0,1,0,10,err));assert(!s.dump(0,1,0,10,err));++rejected; // duplicate
 build(2);begin({10,11},true);resumed(0);assert(!s.dump(2,1,1,11,err));++rejected; // previous request snapshot
 s.arm(2,1,1,11,q);q.replay(0);assert(!s.dump(2,1,1,11,err));++rejected; // wrong combined0013 graph
 q.replay(2);assert(s.dump(2,1,1,11,err));
 s.begin_roster(3);reject([&]{s.seal_roster(3);}); // empty roster cannot inherit seen26
 reject([&]{s.stamp(q);}); // partial/empty graph cannot mark complete
 reject([&]{s.copy("router_ids",raw[22].data(),39,q);});
 sycl::queue foreign(2,1);reject([&]{s.copy("router_ids",raw[22].data(),40,foreign);});
 begin({10,11,12,13},true);resumed(1);assert(!current().active);assert(s.dump(2,1,3,13,err));++rejected; // reused -> no publication
 current().active=false;assert(s.dump(2,1,3,13,err));++rejected; // cancellation -> no publication
 begin({10,11,12,13,14,15,16,17},true);resumed(0);s.arm(2,1,7,17,q);assert(!s.dump(2,1,7,17,err));++rejected; // stale nonce cleared
 q.replay(2);assert(s.dump(2,1,7,17,err));
 begin({10},true);assert(!current().active);++rejected; // four-request bound; no fifth frame
 s.release();assert(sycl::allocations==1&&sycl::frees==1);std::cout<<"ON_PASS rejected="<<rejected<<" fields=26 unobserved=7\n";
}
'''


def main():
    original,changed,added=construct();patch=HERE/'patches/0021-sycl-bounded-layer0-numerical-capture.patch'
    source_headers={p:hashlib.sha256(text.encode()).hexdigest() for p,text in {**changed,**added}.items()}
    with tempfile.TemporaryDirectory(prefix='layer0-capture-cpu-') as temporary:
        work=Path(temporary);overlay=work/'overlay';overlay.mkdir()
        for p,text in original.items():dest=overlay/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        subprocess.run(['git','apply','--check',str(patch)],cwd=overlay,check=True,capture_output=True)
        subprocess.run(['git','apply',str(patch)],cwd=overlay,check=True,capture_output=True)
        for p,text in {**changed,**added}.items():assert (overlay/p).read_text()==text
        v=changed['sycl/src/core/verify.cpp'];g=changed['sycl/src/program/generate.cpp']
        assert 'layer0_exec_[(ar_off_?1:0)+(fidelity_first_?2:0)]' in v
        assert 'STRATA_VERIFY_EAGER' in v and '!one_token_self_commit()' in v and 'layer0_diag::first_position' in g
        hooks=set(__import__('re').findall(r'layer0_snapshot_->(?:copy|pair)\("([^"]+)"',v));assert len(hooks)==26
        mock=work/'sycl/sycl.hpp';mock.parent.mkdir();mock.write_text(MOCK);(work/'test.cpp').write_text(TEST)
        source=work/'strata/core';source.mkdir(parents=True)
        for p,text in added.items():(source/Path(p).name).write_text(text)
        image='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
        cmd=['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}','-v',str(work)+':/work',image,'icpx -std=c++17 -O1 -I/work /work/test.cpp -o /work/test']
        build=subprocess.run(cmd,capture_output=True,text=True)
        if build.returncode:raise RuntimeError(build.stdout+build.stderr)
        off=subprocess.run([str(work/'test'),'off'],capture_output=True,text=True,check=True);assert 'OFF_PASS' in off.stdout
        output=work/'output';output.mkdir();arm=work/'ARM';arm.touch();env={**os.environ,'STRATA_LAYER0_Q8_DIAG':'1','STRATA_LAYER0_Q8_DIAG_ARM':str(arm),'STRATA_LAYER0_Q8_DIAG_DIR':str(output),'STRATA_LAYER0_Q8_DIAG_BINDING_SHA256':'a'*64}
        on=subprocess.run([str(work/'test'),'on'],env=env,capture_output=True,text=True)
        if on.returncode:raise RuntimeError(on.stdout+on.stderr)
        reports=[json.loads(p.read_bytes()) for p in output.glob('*.json')];assert len(reports)==3
        for report in reports:
            assert report['request_replay_marker_verified'] and report['full_model_math_qualified'] is False
            assert sum(f['observed'] for f in report['fields'])==26 and len(report['fields'])==33
            for f in report['fields']:
                if f['observed']:assert Path(f['file']).stat().st_size==f['bytes']
                else:assert f['file']=='' and f['provenance']=='UNOBSERVED_no_producer_hook'
        result={'mode':'CPU_MOCK_SYCL_LAYER0_CAPTURE_ONLY','passed':True,'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'source_reconstruction_exact':True,'consumed_source_sha256':source_headers,'default_off_zero_allocations_copies_frees':True,'mock_result':on.stdout.strip(),'successful_frames':3,'observed_fields':26,'unobserved_fields':7,'nonce_and_graph_key_replay_guard':True,'cpu_compile_image':image,'devices_exposed':False,'actual_sycl_tu_compiled':False,'gpu_executed':False,'full_model_math_qualified':False}
        (HERE/'layer0-capture-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:value for k,value in result.items() if k!='consumed_source_sha256'},sort_keys=True))


if __name__=='__main__':main()
