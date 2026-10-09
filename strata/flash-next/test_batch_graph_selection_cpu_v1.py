#!/usr/bin/env python3
"""Verbatim actual capture-selection prefixes against host cache mocks only."""
import hashlib,json,subprocess,tempfile,shlex,os
from pathlib import Path
HERE=Path(__file__).resolve().parent

def block(text,start):
 a=text.index(start);brace=text.index('{',a);depth=1;at=brace+1
 while depth:
  depth+=(text[at]=='{')-(text[at]=='}');at+=1
 return text[a:at]

def main():
 plan=json.loads((HERE/'solo-migration-observer-source-draft-v1.json').read_bytes());source=Path(os.environ.get('STRATA_OBSERVER26_CPU_OVERLAY',plan['overlay']));cpp=(source/'sycl/src/core/verify.cpp').read_text();header=(source/'sycl/include/strata/core/verify.hpp').read_text()
 for name,h in plan['expected_source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h
 common=Path(plan['base_source'])/'include/strata/core/verify.hpp';assert hashlib.sha256(common.read_bytes()).hexdigest()==plan['unchanged_source_sha256']['include/strata/core/verify.hpp'];keys=block(common.read_text(),'    std::vector<int> bkey(')+block(common.read_text(),'    static std::vector<int> batch_key(')
 prefix=[]
 for start in ['bool Verifier::capture(int T,','bool Verifier::capture_batch(']:
  a=cpp.index(start);b=cpp.index('    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_))',a);prefix.append(cpp[a:b]+'return false;}catch(...){throw;}\n')
 fixture=r'''#include <cassert>
#include <vector>
#include <map>
#include <array>
#include <string>
#include <memory>
#include <cstdlib>
#include "strata/core/batch_fidelity_contract.hpp"
namespace batch_fidelity=strata::core::batch_fidelity;
using strata::core::batch_fidelity::graph_layout;
inline void*ptr(int n){return reinterpret_cast<void*>(uintptr_t(n));}
namespace dpct::experimental{using command_graph_exec_ptr=void*;}
struct OnDevice{explicit OnDevice(int){}};
struct Snapshot{std::vector<graph_layout> roster;void begin_roster(const graph_layout& layout,int&){roster.push_back(layout);}};
struct Verifier{int device_=0,q=0;int*cs_=&q;bool batch_observe_=false,batch_admission_=false,ar_off_=false,layer0_first_=false,fidelity_first_=false;int batch_phase_=0;
 std::array<void*,9>exec_{},exec_nr_{};std::array<void*,4>layer0_exec_{};std::array<void*,2>fidelity_exec_{},batch_admission_exec_{},batch_migration_exec_{};
 std::map<std::vector<int>,void*>batch_observe_exec_,exec_bm_;std::shared_ptr<Snapshot>batch_snapshot_=std::make_shared<Snapshot>();
 bool capture(int T,std::string&err);bool capture_batch(const int*rows,int S,int hbase,std::string&err);
'''+keys+'};\n'+''.join(prefix)+r'''
int main(){Verifier v;std::string e;int rows[2]={0,1},reverse[2]={1,0};v.exec_[1]=ptr(11);v.exec_bm_[v.bkey(rows,2,0)]=ptr(12);
 assert(v.capture(1,e)&&v.capture_batch(rows,2,0,e)&&v.batch_snapshot_->roster.empty()); // Real unarmed normal-cache hit.
 v.batch_observe_=v.batch_admission_=true;v.batch_phase_=1;assert(!v.capture(1,e));assert(v.batch_snapshot_->roster.back().words[1]==1);v.batch_admission_exec_[0]=ptr(21);assert(v.capture(1,e)&&v.batch_snapshot_->roster.size()==1);
 v.batch_phase_=2;assert(!v.capture(1,e));assert(v.batch_snapshot_->roster.back().words[1]==2);v.batch_migration_exec_[0]=ptr(22);assert(v.capture(1,e));
 v.batch_admission_=false;v.batch_phase_=0;assert(!v.capture_batch(rows,2,0,e));assert(v.batch_snapshot_->roster.back().words[1]==0);v.batch_observe_exec_[v.bkey(rows,2,0)]=ptr(23);assert(v.capture_batch(rows,2,0,e));
 assert(!v.capture_batch(reverse,2,0,e));v.batch_observe_exec_[v.bkey(reverse,2,0)]=ptr(24);v.ar_off_=true;assert(!v.capture_batch(rows,2,0,e));
 v.batch_observe_=false;v.ar_off_=false;assert(v.capture_batch(rows,2,0,e)&&v.exec_bm_[v.bkey(rows,2,0)]==ptr(12));
}
'''
 with tempfile.TemporaryDirectory(prefix='batch26-selection-cpu-') as name:
  w=Path(name);(w/'fixture.cpp').write_text(fixture);command='LC_ALL=C g++ -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-misleading-indentation -I/source/sycl/include /work/fixture.cpp -o /work/test && env -u STRATA_VERIFY_EAGER /work/test'
  result=subprocess.run(['docker','run','--rm','--network','none','--user','1000:1000','--memory','512m','--memory-swap','512m','-v',str(w)+':/work','-v',str(source)+':/source:ro','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7',command],capture_output=True,text=True);assert result.returncode==0,result.stderr
 result={'CONFIG':'verbatim actual capture/capture_batch selection prefixes and bkey; host mocks/no SYCL/GPU/SDK','COMMAND':'python3 strata/flash-next/test_batch_graph_selection_cpu_v1.py','RESULT':{'unarmed_cached_normal_not_armed_observer_graph':True,'admission_vs_solo_migration_graph_rosters_distinct':True,'cached_observer_does_not_reseal':True,'row_order_and_AR_variants_distinct':True,'ARMoff_returns_normal_map':True,'patch_sha256':plan['patch_sha256']},'VERDICT':'PASS CPU selection source branches only; actual warm->ARM replay/context/producer/math requires GPU qualification'}
 (HERE/'batch-graph-selection-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS actual graph-cache selection branches; no GPU/SDK')
if __name__=='__main__':main()
