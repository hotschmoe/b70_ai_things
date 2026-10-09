#!/usr/bin/env python3
"""0023 actual helper / source / GPU-mapping semantics under CPU mock only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from prepare_layer0_producer_patch_v1 import construct
from test_layer0_numerical_capture_cpu_v1 import MOCK
HERE=Path(__file__).resolve().parent
MOCK2=MOCK.replace(' event memcpy(void*dst', ''' template<class F>event single_task(F f){if(s_->key<0)f();else s_->graphs[s_->key].push_back(f);return {};}
 event memset(void*dst,int value,std::size_t n){auto f=[=]{std::memset(dst,value,n);};if(s_->key<0)f();else s_->graphs[s_->key].push_back(f);return {};}
 event memcpy(void*dst''')+'\n#include <cmath>\nnamespace sycl::native {inline float exp(float value){return std::exp(value);}}\n'
TEST=r'''#include "strata/core/layer0_numerical_observer.hpp"
#include <cassert>
#include <iostream>
using namespace strata::core::layer0_diag;using namespace strata::kernels::numerical_diag;
int main(int argc,char**argv){sycl::queue q;if(argc==2&&std::string(argv[1])=="off"){snapshot s(q,0);s.copy("ignored",nullptr,0,q);s.release();assert(sycl::allocations==0&&sycl::copies==0);std::cout<<"OFF_PASS\n";return 0;}
 snapshot s(q,0);std::vector<std::vector<std::uint8_t>>raw(fields.size());for(std::size_t i=0;i<fields.size();++i)raw[i].assign(fields[i].bytes,std::uint8_t(i+1));
 std::array<std::int32_t,10>ids{{11,37,3,41,9,22,8,17,29,5}},dst{{6,1,8,3,0,9,2,7,4,5}},tok{};std::array<std::int32_t,512>res;res.fill(-1);
 std::array<unsigned long long,10>slot_offsets;std::array<unsigned long long,512>mirror{};std::vector<std::uint8_t>cache(640),ram(640);
 for(int rank=0;rank<10;++rank){slot_offsets[rank]=rank*64;if(rank<5)res[ids[rank]]=rank;else mirror[ids[rank]]=reinterpret_cast<unsigned long long>(ram.data()+rank*64);}
 std::array<unsigned long long,5>gp0{},gp1{};std::array<std::int32_t,6>starts0{{0,1,2,3,4,5}},starts1{{5,6,7,8,9,10}};int count0=5,count1=5;
 for(int e=0;e<10;++e){int rank=dst[e];auto address=res[ids[rank]]>=0?reinterpret_cast<unsigned long long>(cache.data()+slot_offsets[res[ids[rank]]]):mirror[ids[rank]];(e<5?gp0[e]:gp1[e-5])=address;}
 std::vector<float>gate(6400),up(6400),shared_g(640),shared_u(640);std::vector<std::uint8_t>hq(7200),shared_hq(720,44);
 auto fill=[&](float seed){for(int e=0;e<10;++e){for(int j=0;j<640;++j){gate[e*640+j]=seed+dst[e]*.01f+j*.0001f;up[e*640+j]=.7f+dst[e]*.02f;}for(int b=0;b<720;++b)hq[e*720+b]=std::uint8_t(50+dst[e]);}for(int j=0;j<640;++j){shared_g[j]=.2f;shared_u[j]=.3f;}};
 entry_binding binding{ids.data(),res.data(),mirror.data(),slot_offsets.data(),cache.data(),10,64};
 expert_view first{gate.data(),up.data(),hq.data(),gp0.data(),starts0.data(),&count0,dst.data(),tok.data(),10,640};expert_view second=first;second.group_ptr=gp1.data();second.group_start=starts1.data();second.groups=&count1;
 s.begin_roster(0);q.begin(0);for(std::size_t i=0;i<fields.size();++i)if(i<24||i>=31)s.copy(fields[i].name,raw[i].data(),raw[i].size(),q);
 q.single_task([&]{fill(.1f);});s.observe_shared_gu(shared_g.data(),shared_u.data(),1,640,q);s.observe_shared_hq(shared_hq.data(),1,640,q);
 s.observe_expert(first,binding,false,q);s.observe_expert(first,binding,true,q);
 q.single_task([&]{fill(.9f);});s.observe_expert(second,binding,false,q);s.observe_expert(second,binding,true,q);
 s.stamp(q);q.end();s.seal_roster(0);begin({10},true);resumed(0);s.arm(0,1,0,10,q);q.replay(0);std::string err;assert(s.dump(0,1,0,10,err));int rejected=0;
 auto run_bad=[&](auto mutate,auto restore){current().ordinal=2;current().active=true;s.arm(0,1,0,10,q);mutate();q.replay(0);assert(!s.dump(0,1,0,10,err));++rejected;restore();};
 int saved=dst[0];run_bad([&]{dst[0]=dst[1];},[&]{dst[0]=saved;});
 run_bad([&]{tok[0]=1;},[&]{tok[0]=0;});
 auto pointer=gp0[0];run_bad([&]{gp0[0]^=64;},[&]{gp0[0]=pointer;});
 int old_count=count1;run_bad([&]{count1=0;},[&]{count1=old_count;});
 int old_start=starts1[0];run_bad([&]{starts1[0]=4;},[&]{starts1[0]=old_start;});
 int expert=ids[dst[0]];auto old_mirror=mirror[expert];int old_res=res[expert];res[expert]=-1;run_bad([&]{mirror[expert]=0;},[&]{mirror[expert]=old_mirror;res[expert]=old_res;});
 int original_id=ids[dst[0]];run_bad([&]{ids[dst[0]]=512;},[&]{ids[dst[0]]=original_id;});
 run_bad([&]{count1=11;},[&]{count1=old_count;});
 current().ordinal=2;current().active=true;s.arm(0,1,0,10,q);assert(!s.dump(0,1,0,10,err));++rejected;
 q.replay(0);assert(!s.dump(2,1,0,10,err));++rejected;
 bool shape_rejected=false;try{s.observe_shared_gu(shared_g.data(),shared_u.data(),2,640,q);}catch(const std::invalid_argument&){shape_rejected=true;}assert(shape_rejected);++rejected;
 current().active=false;assert(s.dump(0,1,0,10,err));s.release();assert(sycl::allocations==1&&sycl::frees==1);
 std::cout<<"ON_PASS mapping_rejections="<<rejected<<" fields=33 raw_hidden_unobserved=1\n";}
'''


def main():
    base,changed,added=construct();patch=HERE/'patches/0023-sycl-layer0-numerical-producer-hooks-draft.patch'
    with tempfile.TemporaryDirectory(prefix='layer0-producer-cpu-') as temporary:
        work=Path(temporary);overlay=work/'overlay';overlay.mkdir()
        for p,text in base.items():dest=overlay/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        subprocess.run(['git','apply','--check',str(patch)],cwd=overlay,check=True,capture_output=True);subprocess.run(['git','apply',str(patch)],cwd=overlay,check=True,capture_output=True)
        for p,text in {**changed,**added}.items():assert (overlay/p).read_text()==text
        mock=work/'sycl/sycl.hpp';mock.parent.mkdir();mock.write_text(MOCK2);(work/'test.cpp').write_text(TEST)
        paths={'strata/core/layer0_numerical_contract.hpp':(HERE/'layer0_numerical_contract_v1.hpp').read_text(),'strata/core/layer0_numerical_observer.hpp':(HERE/'layer0_numerical_observer_v2.hpp').read_text(),'strata/kernels/layer0_producer_hooks.hpp':(HERE/'layer0_producer_hooks_v1.hpp').read_text()}
        for p,text in paths.items():dest=work/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        image='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
        build=subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}','-v',str(work)+':/work',image,'icpx -std=c++17 -O1 -I/work /work/test.cpp -o /work/test'],capture_output=True,text=True)
        if build.returncode:raise RuntimeError(build.stdout+build.stderr)
        off=subprocess.run([str(work/'test'),'off'],capture_output=True,text=True,check=True);assert 'OFF_PASS' in off.stdout
        output=work/'output';output.mkdir();arm=work/'ARM';arm.touch();env={**os.environ,'STRATA_LAYER0_Q8_DIAG':'1','STRATA_LAYER0_Q8_DIAG_ARM':str(arm),'STRATA_LAYER0_Q8_DIAG_DIR':str(output),'STRATA_LAYER0_Q8_DIAG_BINDING_SHA256':'b'*64}
        on=subprocess.run([str(work/'test'),'on'],env=env,capture_output=True,text=True)
        if on.returncode:raise RuntimeError(on.stdout+on.stderr)
        reports=[json.loads(p.read_bytes()) for p in output.glob('*.json')];assert len(reports)==1;report=reports[0]
        assert len(report['fields'])==33 and all(f['observed'] for f in report['fields']) and report['raw_fused_hidden_observed'] is False
        assert report['expert_tiers'].count(1)==5 and report['expert_tiers'].count(2)==5
        fields={f['name']:f for f in report['fields']};import numpy as np
        mapping=np.frombuffer(Path(fields['expert_entry_map']['file']).read_bytes(),dtype='<i4').reshape(10,3)
        assert list(mapping[:,0])==[11,37,3,41,9,22,8,17,29,5] and list(mapping[:,1])==[0]*10 and list(mapping[:,2])==list(range(10))
        gu=np.frombuffer(Path(fields['expert_gate_up']['file']).read_bytes(),dtype='<f4').reshape(10,2,640)
        # Entries0..4 preserve their first producer values despite deliberate full scratch overwrite.
        first_ranks=[6,1,8,3,0]
        for rank in range(10):assert abs(float(gu[rank,0,0])-(.1 if rank in first_ranks else .9)-rank*.01)<1e-6
        for name in ['shared_hidden_DERIVED','expert_hidden_DERIVED']:assert fields[name]['provenance']=='DERIVED_gpu_native_exp_from_actual_gate_up'
        result={'mode':'CPU_MOCK_PRODUCER_MAPPING_ONLY','passed':True,'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'source_reconstruction_exact':True,'consumed_source_sha256':{p:hashlib.sha256(t.encode()).hexdigest() for p,t in {**changed,**added}.items()},'mock_result':on.stdout.strip(),'off_zero_allocations_copies':True,'actual_entry_slot_mapping_checked':True,'resident_and_mirror_pointer_binding_checked':True,'scratch_overwrite_boundary_checked':True,'observed_fields':33,'raw_hidden_observed':False,'derived_hidden':True,'devices_exposed':False,'actual_sycl_tu_compiled':False,'gpu_executed':False,'full_model_math_qualified':False}
        (HERE/'layer0-producer-capture-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='consumed_source_sha256'},sort_keys=True))


if __name__=='__main__':main()
