#!/usr/bin/env python3
"""Mock allocator/queue controls for actual new owner header/helper. No GPU compile."""
from pathlib import Path
import json,subprocess,tempfile,hashlib
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'strata/flash-next/serial-lifecycle-slot-owner-engine-build-plan-v1.json'
PLAN_SHA='05041bd19eed14a9758ab00c9b02784308a126ce7a6b6414727668bda6c9fa1c'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
with tempfile.TemporaryDirectory(prefix='strata-slot-owner-cpu-') as name:
 w=Path(name)
 assert sha(PLAN)==PLAN_SHA,'Frozen combined plan changed'
 plan=json.loads(PLAN.read_text());base=Path(plan['source_root'])
 assert subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()==plan['source_revision']
 assert not subprocess.check_output(['git','-C',str(base),'status','--porcelain'],text=True).strip()
 for rel,expected in plan['source_file_sha256'].items():assert sha(base/rel)==expected,rel
 OVERLAY=w/'reconstructed-source'
 subprocess.run(['git','clone','--local','--no-hardlinks','--no-checkout',str(base),str(OVERLAY)],check=True,capture_output=True)
 subprocess.run(['git','-C',str(OVERLAY),'checkout','--detach',plan['source_revision']],check=True,capture_output=True)
 for row in plan['patches']:
  patch=ROOT/row['path'];assert sha(patch)==row['sha256'],row['path']
  subprocess.run(['git','-C',str(OVERLAY),'apply','--check',str(patch)],check=True,capture_output=True)
  subprocess.run(['git','-C',str(OVERLAY),'apply',str(patch)],check=True,capture_output=True)
 for rel,expected in plan['expected_patched_source_sha256'].items():assert sha(OVERLAY/rel)==expected,rel
 assert plan['patches'][-1]['sha256']=='695689b799a4f8fb340a21340db78ece349844642f1a82e91c845ab1a0361ed5'
 print('PASS exact pristine base +20 tracked patches and38 consumed reconstructed source hashes; untracked overlay unused')
 def put(rel,text):p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='ascii')
 put('mock/sycl/sycl.hpp',r'''#pragma once
#include <cstdlib>
#include <cassert>
#include <map>
namespace sycl {
inline int waits=0,allocs=0,frees=0;inline bool fail_device=false;inline std::map<void*,int> live;
struct queue {int device=0;void wait_and_throw(){++waits;}};
inline void* malloc_device(size_t n,const queue& q){if(fail_device)return nullptr;void* p=std::malloc(n);assert(p);live[p]=q.device;++allocs;return p;}
inline void* malloc_host(size_t n,const queue& q){void* p=std::malloc(n);assert(p);live[p]=q.device;++allocs;return p;}
inline void free(void* p,const queue& q){assert(p && live.count(p) && live[p]==q.device);live.erase(p);std::free(p);++frees;}
}
''')
 put('mock/dpct/dpct.hpp',r'''#pragma once
#include <sycl/sycl.hpp>
namespace dpct {inline int current=0;inline sycl::queue queues[2]={{0},{1}};inline sycl::queue& get_in_order_queue(){return queues[current];}}
''')
 put('mock/strata/core/on_device.hpp',r'''#pragma once
#include <dpct/dpct.hpp>
namespace strata::core {struct OnDevice {int old;OnDevice(int dev):old(dpct::current){assert(dev>=0 && dev<2);dpct::current=dev;}~OnDevice(){dpct::current=old;}};}
''')
 put('mock/strata/core/session.hpp',r'''#pragma once
#include <dpct/dpct.hpp>
#include <cstdint>
#include <stdexcept>
namespace strata {
inline sycl::queue* q_of(void* p){return p?static_cast<sycl::queue*>(p):&dpct::get_in_order_queue();}
namespace kernels {struct KvHostPools{};inline int rope_released=0;inline float* active_rope[2]={};inline int rope_scaling(){return 0;}inline void rope_table_set(float* c,float*,int,int){active_rope[dpct::current]=c;}inline void rope_table_release(float* p){++rope_released;for(auto& x:active_rope)if(x==p)x=nullptr;}}
namespace core {
struct ModelGeometry {};struct Shapes{int n_rot=64;};inline Shapes qsa_shapes(const ModelGeometry&){return {};}
struct QsaState {bool owns_rope=false;int64_t max_cells=2048;float* cos_tab=nullptr;float* sin_tab=nullptr;int32_t* host_step=nullptr;int32_t* host_pos=nullptr;void* owned_host_kv=nullptr;uint64_t owned_host_kv_bytes=0;kernels::KvHostPools host;};
struct SessionState {QsaState* qsa_states=nullptr;int64_t qsa_ord0=0,qsa_alloc=0;};
inline bool qsa_kv_elastic(){return false;}
inline int init_calls=0;inline bool partial_fail=false,throw_init=false;
uint64_t g_kv_host_bytes=0;
void qsa_state_release_owned_host(QsaState&,void*);
inline uint64_t session_init(const ModelGeometry&,int64_t,int64_t,void* base,SessionState& s,int64_t lo,int64_t hi){
 assert(base);++init_calls;s.qsa_ord0=lo/4;s.qsa_alloc=(hi-lo)/4;s.qsa_states=new QsaState[12]();
 for(int j=0;j<s.qsa_alloc;++j){auto& q=s.qsa_states[s.qsa_ord0+j];auto& queue=dpct::get_in_order_queue();
  q.host_step=static_cast<int32_t*>(sycl::malloc_host(64,queue));q.host_pos=static_cast<int32_t*>(sycl::malloc_host(64,queue));q.owned_host_kv=sycl::malloc_host(64,queue);q.owned_host_kv_bytes=64;g_kv_host_bytes+=64;q.owns_rope=j==0;q.cos_tab=static_cast<float*>(base);q.sin_tab=q.cos_tab;if(q.owns_rope)kernels::rope_table_set(q.cos_tab,q.sin_tab,2048,0);
  if(partial_fail)return 0;if(throw_init)throw std::runtime_error("mock partial init exception");
 }
 return 1;
}
}}
''')
 layer=(OVERLAY/'sycl/src/core/layer.cpp').read_text();a=layer.index('void qsa_state_register_existing_rope(');b=layer.index('\nvoid qsa_state_zero(',a);release=layer[a:b]
 put('test.cpp',r'''#include "strata/core/slot_session_arena.hpp"
#include <memory>
#include <vector>
namespace strata::core {
'''+release+r'''
}
int main(){using namespace strata::core;ModelGeometry g;std::string err;
 // Constructor oncard0, allocation/init/free owned bystage1; global QSAordinal8.
 {SlotSessionArena owner(1);assert(dpct::current==0);assert(owner.initialize(g,2048,10,32,48,128,err));assert(owner.qsa_ord0==8);assert(sycl::live.size()==13);owner.release();owner.release();assert(sycl::live.empty() && g_kv_host_bytes==0 && dpct::current==0);}
 int calls=init_calls;sycl::fail_device=true;{SlotSessionArena owner(1);assert(!owner.initialize(g,2048,10,32,48,128,err));assert(init_calls==calls && sycl::live.empty());}sycl::fail_device=false;
 partial_fail=true;{SlotSessionArena owner(1);assert(!owner.initialize(g,2048,10,32,48,128,err));assert(sycl::live.empty() && g_kv_host_bytes==0);}partial_fail=false;
 throw_init=true;{SlotSessionArena owner(1);assert(!owner.initialize(g,2048,10,32,48,128,err));assert(sycl::live.empty() && g_kv_host_bytes==0);}throw_init=false;
 // A later stage fits fewer sessions: resize frees every discarded stage0arena.
 std::vector<std::unique_ptr<SlotSessionArena>> slots;
 for(int i=0;i<3;++i){auto p=std::make_unique<SlotSessionArena>(0);assert(p->initialize(g,2048,10,0,32,128,err));slots.push_back(std::move(p));}
 auto live=sycl::live.size();slots.resize(1);assert(sycl::live.size()==live/3);assert(strata::kernels::active_rope[0]==nullptr);qsa_state_register_existing_rope(slots[0]->qsa_states[0],g);assert(strata::kernels::active_rope[0]==slots[0]->base());slots.clear();assert(sycl::live.empty());
 assert(sycl::allocs==sycl::frees && g_kv_host_bytes==0 && sycl::waits>0);
 std::puts("PASS CPU mock: correctstage context/global QSAordinal, null noinit, partial/throw rollback, double release, resize fallback, allowned host/device frees");
}
''')
 # Ordinary host compiler against mocks; no SYCL flags/devices/runtime linking.
 cmd='g++ -std=c++20 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all -I/work/mock -I/overlay/sycl/include /work/test.cpp -o /work/test && /work/test'
 r=subprocess.run(['docker','run','--rm','--network','none','--entrypoint','/bin/bash','-v',str(w)+':/work','-v',str(OVERLAY)+':/overlay:ro',IMAGE,'-lc',cmd],capture_output=True,text=True)
 print(r.stdout.strip());print(r.stderr[:1000] if r.returncode else 'No GPU/SYCL device-code compilation or execution.');raise SystemExit(r.returncode)
