#!/usr/bin/env python3
"""Pinned reconstructed real mirror header/factory with CPU mock allocator; no GPU."""
import hashlib,json,subprocess,tempfile,os,shlex
from pathlib import Path
import audit_stage_mirror_owner_trace_v1 as parser
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(root,path,text):p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='ascii')
def main():
 planpath=HERE/'serving-mirror-owner-engine-build-plan-v1.json';plan=json.loads(planpath.read_text())
 with tempfile.TemporaryDirectory(prefix='strata-mirror31-cpu-') as name:
  w=Path(name);source=w/'source';base=Path(plan['source_root'])
  assert subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()==plan['source_revision']
  for n,d in plan['source_file_sha256'].items():assert sha(base/n)==d,n
  subprocess.run(['git','clone','--local','--no-hardlinks','--no-checkout',str(base),str(source)],check=True,capture_output=True);subprocess.run(['git','-C',str(source),'checkout','--detach',plan['source_revision']],check=True,capture_output=True)
  for row in plan['patches']:
   patch=ROOT/row['path'];assert sha(patch)==row['sha256'];subprocess.run(['git','-C',str(source),'apply','--check',str(patch)],check=True,capture_output=True);subprocess.run(['git','-C',str(source),'apply',str(patch)],check=True,capture_output=True)
  for n,d in plan['expected_patched_source_sha256'].items():assert sha(source/n)==d,n
  for h in plan['added_header_payloads']:assert sha(source/h['path'])==h['sha256']
  text=(source/'sycl/src/core/gguf_expert_source.cpp').read_text();start=text.index('std::shared_ptr<StageExpertMirror> GgufExpertSource::mirror_stage(');end=text.index('\n} catch (const std::exception& ex)',start);end=text.index('\n',end+1);factory=text[start:end]
  close=text[text.index('void GgufExpertSource::close() {'):text.index('\nbool GgufExpertSource::open(')]
  assert 'stage_alias_.clear();' in close and close.index('trace_graphs_retired("source_aliases",this)')<close.index('stage_mirrors_.clear()')
  gen=(source/'sycl/src/program/generate.cpp').read_text();retire=gen[gen.index('                // Drain every stage context'):]
  assert 'gs.expert_mirror->trace_expect_verifier(&gs.ver)' in gen and 'first_stage_mirror->trace_expect_verifier(&ver)' in gen and 'o.pipeline_windows>0' in gen and '!o.serve' in gen
  assert retire.index('token_graph_free(tgraph)')<retire.index('trace_serving_graphs_retired()') and retire.index('stage->ver.release_stage_mirror_graphs()')<retire.index('trace_serving_graphs_retired()')<retire.index('gguf_src.close()')<retire.index('owner->release()')
  verify=(source/'sycl/src/core/verify.cpp').read_text();body=verify[verify.index('void Verifier::release_stage_mirror_graphs()'):verify.index('Verifier::~Verifier()')]
  assert body.index('exec_bm_.clear()')<body.index('trace_graphs_retired("verifier",this)')<body.index('hits_.stage_mirror.reset()')
  put(w,'mock/sycl/ext/oneapi/backend/level_zero.hpp','#pragma once\n')
  put(w,'mock/sycl/sycl.hpp',r'''#pragma once
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <map>
#include <stdexcept>
#include <cstdint>
namespace sycl {
inline int native_queries=0,allocations=0,frees=0,waits=0,fail_after=-1;inline bool fail_read=false;
struct context {int id;bool operator==(context b)const{return id==b.id;}bool operator!=(context b)const{return id!=b.id;}};
struct device {int id;bool operator==(device b)const{return id==b.id;}bool operator!=(device b)const{return id!=b.id;}};
struct entry{int device;bool host;size_t bytes;};inline std::map<void*,entry> live;
namespace property {namespace queue {struct in_order{};}}namespace usm {enum class alloc {unknown,device,host};}
enum class backend {ext_oneapi_level_zero};
template<backend B>void* get_native(context c){++native_queries;std::fprintf(stderr,"<--- urContextGetNativeHandle(.hContext = 0x%x, .phNativeContext = 0xf (0x%x)) -> UR_RESULT_SUCCESS;\n",0x10+c.id,0x100+c.id);return reinterpret_cast<void*>(uintptr_t(0x100+c.id));}
template<backend B>void* get_native(device d){++native_queries;std::fprintf(stderr,"<--- urDeviceGetNativeHandle(.hDevice = 0x%x, .phNativeDevice = 0xf (0x%x)) -> UR_RESULT_SUCCESS;\n",0x20+d.id,0x200+d.id);return reinterpret_cast<void*>(uintptr_t(0x200+d.id));}
struct event {void wait_and_throw(){++waits;}};
struct queue {int dev=0;context get_context()const{return {dev};}device get_device()const{return {dev};}template<class P>bool has_property()const{return true;}void wait_and_throw(){++waits;}event memcpy(void* dst,const void* src,size_t bytes){assert(live.count(dst)&&live[dst].device==dev&&live[dst].bytes>=bytes);std::memcpy(dst,src,bytes);return {};}};
inline void* allocate(size_t bytes,const queue&q,bool host){if(fail_after==allocations)return nullptr;void* p=std::malloc(bytes);assert(p);live[p]={q.dev,host,bytes};++allocations;std::fprintf(stderr,"<--- %s(.hContext = 0x%x, .hDevice = 0x%x, .size = %zu, .ppMem = 0xf (%p)) -> UR_RESULT_SUCCESS;\n",host?"urUSMHostAlloc":"urUSMDeviceAlloc",0x10+q.dev,0x20+q.dev,bytes,p);return p;}
template<class T>T* malloc_host(size_t n,const queue&q){return static_cast<T*>(allocate(n*sizeof(T),q,true));}
template<class T>T* malloc_device(size_t n,const queue&q){return static_cast<T*>(allocate(n*sizeof(T),q,false));}
inline usm::alloc get_pointer_type(void* p,context c){assert(live.count(p)&&live[p].device==c.id);return live[p].host?usm::alloc::host:usm::alloc::device;}
inline void free(void* p,const queue&q){assert(live.count(p)&&live[p].device==q.dev);std::fprintf(stderr,"<--- urUSMFree(.hContext = 0x%x, .pMem = %p) -> UR_RESULT_SUCCESS;\n",0x10+q.dev,p);live.erase(p);std::free(p);++frees;}
}
''')
  put(w,'fixture.cpp',r'''#include "strata/core/stage_expert_mirror.hpp"
#include "strata/core/stage_mirror_segments.hpp"
#include <memory>
#include <thread>
#include <algorithm>
namespace strata::kernels::cpu {struct Layout {std::vector<uint64_t> bytes=std::vector<uint64_t>(48,100);};inline const Layout& expert_layout(){static Layout l;return l;}}
namespace dpct {inline sycl::queue get_in_order_queue(){return {0};}}
namespace strata::core {
struct GgufExpertSource {
 uint64_t mirror_bytes_=0;std::vector<int> mirror_off_,layer_first_,fds_,names_,layer_fd_,ring_,ring_key_,where_;int ring_next_=0;
 void* mirror_=nullptr;int64_t n_layers_=48,n_expert_=512;uint64_t stage_bytes_=0;std::vector<std::shared_ptr<StageExpertMirror>> stage_mirrors_;std::vector<uint8_t*> stage_alias_;
 bool read_into(int64_t,int64_t,void* dst,size_t bytes){if(sycl::fail_read)return false;std::memset(dst,7,bytes);return true;}
 std::shared_ptr<StageExpertMirror> mirror_stage(const std::vector<std::pair<int64_t,int64_t>>&,uint64_t,int,int64_t,int64_t,const sycl::queue&,std::string&,uint64_t,int,int);
 void close();
};
'''+close+factory+r'''
}
int main(int argc,char**argv){assert(argc==2);bool enabled=argv[1][0]=='1';assert((std::getenv("STRATA_MIRROR_OWNER_TRACE")!=nullptr)==enabled);
using namespace strata::core;
for(int dev=0;dev<2;++dev){GgufExpertSource src;std::string error;auto owner=src.mirror_stage({{dev*24,0},{dev*24,1},{dev*24,2}},4096,1,dev*24,(dev+1)*24,sycl::queue{dev},error,512,dev,dev);assert(owner&&owner->bytes==768&&owner->segments.size()==2);owner->trace_serving_bound();int c1=1;owner->trace_expect_verifier(&c1);std::thread a([&]{owner->trace_graphs_bound("verifier",&c1);}),b([&]{owner->trace_graphs_bound("verifier",&c1);});a.join();b.join();owner->trace_graphs_retired("verifier",&c1);owner->trace_serving_graphs_retired();src.close();owner->release();owner.reset();assert(sycl::live.empty());}
// Partial allocation failure: the real factory retains and frees completed segments.
{GgufExpertSource src;std::string error;sycl::fail_after=sycl::allocations+1;auto owner=src.mirror_stage({{0,0},{0,1},{0,2}},4096,1,0,24,sycl::queue{0},error,512,0,0);assert(!owner&&!error.empty()&&sycl::live.empty());sycl::fail_after=-1;}
// Gather failure: source aliases/table are never published; every host owner frees.
{GgufExpertSource src;std::string error;sycl::fail_read=true;auto owner=src.mirror_stage({{0,0},{0,1}},4096,1,0,24,sycl::queue{0},error,512,0,0);assert(!owner&&!error.empty()&&sycl::live.empty());sycl::fail_read=false;}
// Descriptor cap is preflighted before any owner allocation, never after malloc.
if(enabled){GgufExpertSource src;std::string error;std::vector<std::pair<int64_t,int64_t>> pairs;for(int e=0;e<256;++e)pairs.push_back({0,e});int before=sycl::allocations;auto owner=src.mirror_stage(pairs,65536,1,0,24,sycl::queue{0},error,256,0,0);assert(!owner&&!error.empty()&&sycl::allocations==before&&sycl::live.empty());}
assert(sycl::frees==sycl::allocations);if(!enabled)assert(sycl::native_queries==0);std::fprintf(stdout,"CPU_MIRROR_BODY_PASS allocations=%d frees=%d native_queries=%d enabled=%d\n",sycl::allocations,sycl::frees,sycl::native_queries,int(enabled));}
''')
  exe=w/'fixture';command=['g++','-std=c++17','-O1','-g','-pthread','-fno-omit-frame-pointer','-fsanitize=address,undefined','-fno-pie','-no-pie','-I'+str(w/'mock'),'-I'+str(source/'sycl/include'),'-I'+str(source/'include'),str(w/'fixture.cpp'),'-o',str(exe)];
  docker=['docker','run','--rm','--network','none','--user','1000:1000','--memory','512m','--memory-swap','512m','-v',str(w)+':/work','-v',str(source)+':/source:ro',plan['image']]
  inside=lambda value:value.replace(str(source),'/source').replace(str(w),'/work')
  compiled=subprocess.run(docker+[shlex.join([inside(x) for x in command])],capture_output=True,text=True);assert compiled.returncode==0,compiled.stderr
  off=subprocess.run(docker+['env -u STRATA_MIRROR_OWNER_TRACE /work/fixture 0'],capture_output=True,text=True);assert off.returncode==0,off.stderr;assert 'MIRROR_USM ' not in off.stderr and 'native_queries=0' in off.stdout
  on=subprocess.run(docker+['STRATA_MIRROR_OWNER_TRACE=1 /work/fixture 1'],capture_output=True,text=True);assert on.returncode==0,on.stderr;good=parser.audit(on.stderr,[(0,0,0,24),(1,1,24,48)]);assert good['passed'],good['errors']
  negatives={};lines=on.stderr.splitlines()
  def bad(name,changed):result=parser.audit('\n'.join(changed),[(0,0,0,24),(1,1,24,48)]);assert not result['passed'],name;negatives[name]=True
  index=next(i for i,x in enumerate(lines) if '<--- urUSMFree' in x);bad('missing_free',lines[:index]+lines[index+1:]);bad('double_free',lines[:index+1]+[lines[index]]+lines[index+1:]);bad('failed_free',[x.replace('UR_RESULT_SUCCESS','UR_RESULT_ERROR_UNKNOWN') if i==index else x for i,x in enumerate(lines)])
  for event in ('verifier_expected','graphs_retired','serving_graphs_retired','release_begin','free_begin','free_returned','release_returned'):
   idx=next(i for i,x in enumerate(lines) if x.startswith('MIRROR_USM ') and json.loads(x[11:])['event']==event);bad('missing_'+event,lines[:idx]+lines[idx+1:])
  for field,value in [('device',9),('stage',9),('allocation_generation',999999),('bytes',1),('role','borrowed_alias'),('ze_context','0x999'),('segment',999)]:
   changed=list(lines);idx=next(i for i,x in enumerate(lines) if x.startswith('MIRROR_USM ') and json.loads(x[11:])['event']=='owner_register');mark=json.loads(changed[idx][11:]);mark[field]=value;changed[idx]='MIRROR_USM '+json.dumps(mark);bad('wrong_'+field,changed)
  bad('no_verifier_bindings',[x for x in lines if not (x.startswith('MIRROR_USM ') and json.loads(x[11:]).get('consumer_kind')=='verifier')])
  bad('missing_one_of_two_stage_verifier_pairs',[x for x in lines if not (x.startswith('MIRROR_USM ') and json.loads(x[11:]).get('stage')==0 and json.loads(x[11:]).get('consumer_kind')=='verifier')])
  bad('empty',[])
  result={'CONFIG':'Pristine fb58 +29+31 tracked source/60 ledger/24 headers; real mirror header and extracted actual factory; mock CPU queues/allocators/source bytes only','COMMAND':command,'RESULT':{'off':off.stdout.strip(),'on':on.stdout.strip(),'negative_controls':negatives,'source_plan_sha256':sha(planpath),'source_body_sha256':hashlib.sha256(factory.encode()).hexdigest(),'source_close_body_sha256':hashlib.sha256(close.encode()).hexdigest(),'ASAN_UBSAN':True,'owner_count':len(good['owners'])},'VERDICT':'PASS CPU source-body ownership only; no SDK/GPU/model/physicalreclamation qualification','actual_gpu_lifecycle_qualified':False}
  (HERE/'stage-mirror-owner-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS actual reconstructed mirror header/factory ASAN+UBSAN, off0nativequeries, rollback+quota and strict trace negatives; CPU only')
if __name__=='__main__':main()
