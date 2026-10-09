#!/usr/bin/env python3
"""Actual new SlotSessionArena body with host failure/ownership mocks only."""
import hashlib,json,os,subprocess,tempfile
from pathlib import Path
from audit_slot_owner_trace_v1 import audit
HERE=Path(__file__).resolve().parent
MOCK=r'''#pragma once
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
namespace sycl{enum class backend{ext_oneapi_level_zero};inline int allocations=0,frees=0,waits=0,queries=0;inline bool null_alloc=false;
struct queue{int device=0;queue(int d=0):device(d){};int get_context()const{return 0x100+device;}void wait_and_throw(){++waits;}};
inline void* alloc(size_t bytes,const queue&q,const char*kind){++allocations;void*p=std::malloc(bytes);std::fprintf(stderr,"<--- %s(.hContext = 0x%x, .size = %zu, .ppMem = 0x222 (%p)) -> UR_RESULT_SUCCESS;\n",kind,q.get_context(),bytes,p);return p;}
inline void*malloc_device(size_t bytes,const queue&q){if(null_alloc)return nullptr;return alloc(bytes,q,"urUSMDeviceAlloc");}
inline void*malloc_host(size_t bytes,const queue&q){return alloc(bytes,q,"urUSMHostAlloc");}
inline void free(void*p,const queue&q){if(!p)return;++frees;std::fprintf(stderr,"<--- urUSMFree(.hContext = 0x%x, .pMem = %p) -> UR_RESULT_SUCCESS;\n",q.get_context(),p);std::free(p);}
template<backend B>inline void*get_native(int context){++queries;void*n=reinterpret_cast<void*>(uintptr_t(context+0x1000));std::fprintf(stderr,"<--- urContextGetNativeHandle(.hContext = 0x%x, .phNativeContext = 0x333 (%p)) -> UR_RESULT_SUCCESS;\n",context,n);return n;}}
'''
SESSION=r'''#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
namespace strata::core{inline int active_device=0,mode=0;struct ModelGeometry{int n_head=2;};struct QsaState{void*host_step=nullptr,*host_pos=nullptr,*owned_host_kv=nullptr,*host_kv=nullptr;uint64_t owned_host_kv_bytes=0;};struct SessionState{QsaState*qsa_states=nullptr;int64_t qsa_ord0=0,qsa_alloc=0;};inline bool qsa_kv_elastic(){return false;}
inline uint64_t session_init(const ModelGeometry&,int64_t,int64_t,void*,SessionState&s,int64_t,int64_t){sycl::queue q(active_device);s.qsa_states=new QsaState[1];s.qsa_alloc=1;auto&k=s.qsa_states[0];k.host_step=sycl::malloc_host(20,q);k.host_pos=sycl::malloc_host(8,q);k.owned_host_kv=sycl::malloc_host(32,q);k.host_kv=k.owned_host_kv;k.owned_host_kv_bytes=32;if(mode==2)throw std::runtime_error("controlled partial-init throw");return mode==1?0:64;}
inline void qsa_state_release_owned_host(QsaState&s,void*opaque){auto&q=*static_cast<sycl::queue*>(opaque);sycl::free(s.host_step,q);sycl::free(s.host_pos,q);sycl::free(s.owned_host_kv,q);s={};}}
'''
CPP=r'''#include <cassert>
#include "strata/core/slot_session_arena.hpp"
int main(int argc,char**argv){using namespace strata::core;std::string mode=argc>1?argv[1]:"normal";core_unused:;if(mode=="partial")strata::core::mode=1;if(mode=="throw")strata::core::mode=2;if(mode=="null")sycl::null_alloc=true;ModelGeometry g;SlotSessionArena owner(0,0,1);std::string error;bool ok=owner.initialize(g,1,1,0,4,64,error);
if(mode=="null"){assert(!ok&&!sycl::allocations&&!sycl::frees&&!sycl::queries);return 0;}if(mode=="partial"||mode=="throw"){assert(!ok&&sycl::allocations==4&&sycl::frees==4);return 0;}assert(ok);owner.trace_graphs_bound();if(mode!="stale")owner.trace_graphs_retired();owner.release();owner.release();assert(sycl::allocations==4&&sycl::frees==4&&sycl::waits==1);if(!std::getenv("STRATA_SLOT_OWNER_TRACE"))assert(!sycl::queries);return 0;}
'''
def main():
 plan=json.loads((HERE/'slot-owner-trace-source-draft-v1.json').read_bytes());source=Path(plan['overlay']);assert hashlib.sha256((HERE.parent.parent/plan['patch']).read_bytes()).hexdigest()==plan['patch_sha256']
 for n,digest in plan['expected_source_sha256'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==digest
 with tempfile.TemporaryDirectory(prefix='slot29-host-mocks-') as name:
  w=Path(name)
  for path,text in [('mock/sycl/sycl.hpp',MOCK),('mock/sycl/ext/oneapi/backend/level_zero.hpp','#pragma once\n'),('mock/strata/core/session.hpp',SESSION),('mock/strata/core/on_device.hpp','#pragma once\n#include "session.hpp"\nnamespace strata::core{struct OnDevice{int old;OnDevice(int d):old(active_device){active_device=d;}~OnDevice(){active_device=old;}};}\n'),('mock/dpct/dpct.hpp','#pragma once\n#include <sycl/sycl.hpp>\n#include "strata/core/session.hpp"\nnamespace dpct{inline sycl::queue get_in_order_queue(){return sycl::queue(strata::core::active_device);}}\n'),('mock/strata/kernels/qsa.hpp','#pragma once\nnamespace strata::kernels{inline unsigned qsa_step_bytes(){return 16;}}\n'),('test.cpp',CPP.replace('core_unused:;',''))]:
   p=w/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
  docker=['docker','run','--rm','--network','none','--user','1000:1000','--memory','512m','--memory-swap','512m','-v',str(w)+':/work','-v',str(source)+':/source:ro','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7']
  command='LC_ALL=C g++ -std=c++17 -Wall -Wextra -Werror -Wno-misleading-indentation -fsanitize=address,undefined -I/work/mock -I/source/sycl/include /work/test.cpp -o /work/test';p=subprocess.run(docker+[command],capture_output=True,text=True);assert p.returncode==0,p.stderr
  traces={}
  for mode in ['off','normal','partial','throw','null','stale']:
   cmd=('env -u STRATA_SLOT_OWNER_TRACE' if mode=='off' else 'env STRATA_SLOT_OWNER_TRACE=1')+' /work/test '+('normal' if mode=='off' else mode);p=subprocess.run(docker+[cmd],capture_output=True,text=True);assert p.returncode==0,p.stderr;traces[mode]=p.stderr
  assert 'SLOT_USM' not in traces['off'];assert audit(traces['normal'],[(0,1)])['passed'];assert audit(traces['partial'])['passed'] and audit(traces['throw'])['passed'];assert not audit(traces['stale'])['passed'];assert not audit(traces['null'])['passed']
  good=traces['normal'];lines=good.splitlines();reg=next(l for l in lines if l.startswith('SLOT_USM ') and 'owner_register' in l);free=next(l for l in lines if 'urUSMFree' in l);retired=next(l for l in lines if 'graphs_retired"' in l and '"event":"graphs_retired"' in l);returned=next(l for l in lines if '"event":"release_returned"' in l)
  bad=[good.replace(free,'',1),good+'\n'+free,good+'\n'+reg,good.replace(retired,'',1),good.replace(returned,'',1),good.replace('urUSMFree(.hContext = 0x100','urUSMFree(.hContext = 0x999',1),good.replace('"generation":1','"generation":2',1)]
  for trace in bad:assert not audit(trace,[(0,1)])['passed']
 result={'CONFIG':'actual new arena header/initialize/release with host ASanUBSan mocks and raw UR logical traces; no SYCL/SDK/devices/weights','COMMAND':'python3 strata/flash-next/test_slot_owner_trace_cpu_v1.py','RESULT':{'off_existing_alloc_free_wait_counts_unchanged_no_nativecontext_queries':True,'normal_four_unique_owners_and_alias_not_double_registered':True,'partial_zero_throw_cleanup':True,'null_not_qualified_as_owned_success':True,'stale_missing_wrongcontext_doublefree_controls':len(bad)+2,'patch_sha256':plan['patch_sha256']},'VERDICT':'PASS CPU owned marker/control source only; actual slots/graph retirement/allocator/source/serving unqualified'}
 (HERE/'slot-owner-trace-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS actual slot header mock lifetimes; no GPU/source payload')
if __name__=='__main__':main()
