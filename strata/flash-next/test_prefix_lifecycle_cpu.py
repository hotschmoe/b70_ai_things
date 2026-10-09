#!/usr/bin/env python3
"""Actual portable cache owner/header CPU tests and frozen-source binding; no GPU."""
import hashlib,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'strata/flash-next/prefix-lifecycle-source-plan.json'
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'

def main():
 plan=json.loads(PLAN.read_text());base=Path(plan['base_source']);source=Path(plan['overlay']);patch=ROOT/plan['patch']
 assert hashlib.sha256(patch.read_bytes()).hexdigest()==plan['patch_sha256']
 for name,expected in plan['base_files'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==expected
 with tempfile.TemporaryDirectory(prefix='prefix-lifecycle-cpu-') as directory:
  work=Path(directory);overlay=work/'source'
  import shutil
  shutil.copytree(base/'include',overlay/'include')
  for name in plan['base_files']:
   p=overlay/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((base/name).read_bytes())
  subprocess.run(['git','apply','--check',str(patch)],cwd=overlay,check=True);subprocess.run(['git','apply',str(patch)],cwd=overlay,check=True)
  for name,expected in plan['patched_files'].items():assert hashlib.sha256((overlay/name).read_bytes()).hexdigest()==expected
  (work/'arm').touch()
  (work/'test.cpp').write_text(r'''
#include "strata/core/conversation_cache.hpp"
#include <cassert>
#include <string>
using namespace strata::core;
namespace lc=strata::core::prefix_lifecycle;
int main(int argc,char**argv){
 bool on=argc>1;std::vector<int64_t> input{11,12,13};lc::begin(input,true);
 assert(lc::active()==on);lc::Sha256 hash;char empty[65];hash.finish(empty);assert(std::string(empty)=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
 char digest[65];lc::token_digest(input,digest);std::printf("DIGEST %s\n",digest);std::vector<int64_t> long_ids;for(int i=0;i<8192;++i)long_ids.push_back(i);lc::token_digest(long_ids,digest);std::printf("LONGDIGEST %s\n",digest);
 auto image=[](int id){SavedConversation s;s.layer_lo=0;s.layer_hi=48;s.live.ids={id,id+1};return s;};
 ConversationCache cache(4096,1);auto a=image(11);auto ad=lc::describe(a);assert(cache.put(std::move(a)));assert(cache.size()==1&&cache.bytes()<=4096);
 auto b=image(21);assert(cache.put(std::move(b)));assert(cache.size()==1&&cache.evictions()==1&&cache.bytes()<=4096);
 auto taken=cache.take(0);assert(taken.live.ids==std::vector<int32_t>({21,22}));assert(cache.size()==0);
 auto cp=image(1);cp.live.ids.clear();ConversationCheckpoint checkpoint;checkpoint.ids={17,18};cp.checkpoints.push_back(std::move(checkpoint));
 auto describe=lc::describe(cp);if(on){assert(describe.tokens==2&&std::string(describe.key)!=std::string(ad.key));}
 lc::state("committed_live",taken.live.ids,true,false,"complete","stop");
 // A pinned entry remains protected: instrumentation cannot alter admission.
 auto pin=image(31);ConversationCheckpoint pinned;pinned.ids={31};pinned.pinned=true;pin.checkpoints.push_back(std::move(pinned));assert(cache.put(std::move(pin)));auto refused=image(41);assert(!cache.put(std::move(refused)));assert(cache.size()==1&&cache.evictions()==1);
 ConversationCache many(65536,4);assert(many.put(image(51)));assert(many.put(image(61)));assert(many.put(image(71)));auto middle=many.take(1);assert(middle.live.ids[0]==61);assert(many.evict_oldest());auto last=many.take(0);assert(last.live.ids[0]==71);
 std::puts("PASS actual portable cache ownership/budget/pin policy and lifecycle SHA256");
}
''')
  command=['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}', '--entrypoint','/bin/bash','-v',str(work)+':/work',IMAGE,'-lc',
    'set -e\ng++ -std=c++20 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all -I/work/source/include /work/test.cpp -o /work/test\n/work/test\nSTRATA_PREFIX_LIFECYCLE_DIAG=1 STRATA_PREFIX_DIAG_ARM=/work/arm /work/test on']
  run=subprocess.run(command,capture_output=True,text=True)
  if run.returncode:print(run.stdout+run.stderr);raise RuntimeError('Portable CPU fixture failed')
  records=[json.loads(line[4:]) for line in run.stderr.splitlines() if line.startswith('PCL ')]
  expected=hashlib.sha256(b''.join(i.to_bytes(4,'little') for i in [11,12,13])).hexdigest();assert ('DIGEST '+expected) in run.stdout;expected_long=hashlib.sha256(b''.join(i.to_bytes(4,'little') for i in range(8192))).hexdigest();assert ('LONGDIGEST '+expected_long) in run.stdout
  cache=[r for r in records if r['event']=='cache'];admit=[r for r in cache if r['action']=='admit'];evict=[r for r in cache if r['action']=='evict'];take=[r for r in cache if r['action']=='take'];assert len(admit)==6 and len(evict)==2 and len(take)==3
  assert evict[0]['snapshot_instance']==admit[0]['snapshot_instance'] and evict[0]['snapshot_metadata_sha256']==admit[0]['snapshot_metadata_sha256']
  assert take[0]['snapshot_instance']==admit[1]['snapshot_instance'];assert take[1]['snapshot_instance']==admit[4]['snapshot_instance'];assert evict[1]['snapshot_instance']==admit[3]['snapshot_instance'];assert take[2]['snapshot_instance']==admit[5]['snapshot_instance'];assert admit[0]['snapshot_instance']!=admit[1]['snapshot_instance']
  state=next(r for r in records if r['event']=='committed_live');assert state['chain_updated'] and not state['live_reusable'] and not state['published']
  assert all(r['retained_bytes']<=r['budget'] and r['entries']<=r['slots'] for r in cache)
  assert run.stdout.count('PASS actual portable')==2
  receipt={'CONFIG':'actual portable ConversationCache and draft0019 header; pinned GCC CPU container, no devices','COMMAND':'python3 strata/flash-next/test_prefix_lifecycle_cpu.py','RESULT':{'source_binding':True,'canonical_sha256':True,'actual_instance_victim_and_take':True,'middle_erase_identity_preserved':True,'pin_policy_unchanged':True,'updated_not_reusable_not_published':True,'off_on_cpu_policy_outputs_equal':True,'cpu_log':run.stdout},'VERDICT':'PASS portable CPU owner/telemetry; actual SYCL TU compilation/GPU model/lifecycle unqualified','source_plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest()}
  (ROOT/'strata/flash-next/prefix-lifecycle-cpu-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print('PASS actual portable cache/header under fatal ASan/UBSan and source reconstruction; no SYCL/GPU')
if __name__=='__main__':main()
