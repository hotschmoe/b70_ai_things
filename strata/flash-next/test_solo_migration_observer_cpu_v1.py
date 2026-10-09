#!/usr/bin/env python3
"""Actual new diagnostic headers with host graph mocks; no SDK/GPU/model."""
import ast,hashlib,json,os,subprocess,tempfile,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 plan=json.loads((HERE/'solo-migration-observer-source-draft-v1.json').read_bytes());source=Path(os.environ.get('STRATA_OBSERVER26_CPU_OVERLAY',plan['overlay']))
 assert sha(ROOT/plan['patch'])==plan['patch_sha256']
 for name,digest in plan['expected_source_sha256'].items():assert sha(source/name)==digest
 old=ast.parse((HERE/'test_batch_fidelity_observer_v2_cpu.py').read_text());constants={n.targets[0].id:ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
 cpp=constants['CPP']
 cpp=cpp.replace('if(argc>1&&std::string(argv[1])=="coverage")','if(argc>1&&(std::string(argv[1])=="coverage"||std::string(argv[1])=="migration"))')
 cpp=cpp.replace('auto id=ids[row];s0.arm','auto id=ids[row];admitting()=id;s0.arm')
 cpp=cpp.replace('  records().complete(ids[0],false);records().complete(ids[1],false);s0.release();',r'''  records().complete(ids[0],false);records().complete(ids[1],false);
  if(std::string(argv[1])=="migration"){
   bad([&]{begin(11,0,-1,102);}); // No actual cancelled slot completion.
   records().closed(11,1,0,false);bad([&]{begin(11,0,-1,102);});
   for(int row=0;row<2;++row){close_slot(ids[row].rid,ids[row].slotgen,ids[row].slot,true);auto id=begin(ids[row].rid,0,-1,ids[row].prompt+2);assert(id.phase==2&&id.slot==ROWS&&id.source_slot==row&&id.source_slotgen==ids[row].slotgen);id.pos=id.prompt-1;id.token=41+row;colour[0]=row;
    std::fprintf(stderr,"SBF resume pid=%ld rid=%llu enginegen=%llu requestgen=%llu slot=6 slotgen=%llu reused=%lld read_from=%lld reread_to=%lld\n",long(getpid()),(unsigned long long)id.rid,(unsigned long long)id.enginegen,(unsigned long long)id.generation,(unsigned long long)id.slotgen,(long long)id.pos,(long long)id.pos,(long long)id.pos);auto solo=graph_layout::make(nullptr,1,2);if(row==0){build(s0,q0,5,0,32,false,solo);build(s1,q1,5,32,48,true,solo);}
    s0.arm(solo,&id,selected,1,2,q0);q0.replay(5);
    if(row==0){auto changed=id;changed.source_slot=1;failed([&]{return s0.dump(&changed,selected,1,2,error);});s0.discard();s0.arm(solo,&id,selected,1,2,q0);q0.replay(5);
     ::unlink(settings().arm);failed([&]{return s0.dump(&id,selected,1,2,error);});int fd=::open(settings().arm,O_CREAT|O_WRONLY,0600);assert(fd>=0);::close(fd);failed([&]{return s0.dump(&id,selected,1,2,error);});s0.discard();s0.arm(solo,&id,selected,1,2,q0);q0.replay(5);}
    assert(s0.dump(&id,selected,1,2,error));s1.arm(solo,&id,selected,1,2,q1);q1.replay(5);assert(s1.dump(&id,selected,1,2,error));
    for(int stage=0;stage<2;++stage)span(id,stage,stage?32:0,stage?48:32,id.pos,id.pos+1,"solo_migration_target_verify");records().complete(id,2);
    std::fprintf(stderr,"SBF solo_event pid=%ld rid=%llu enginegen=%llu requestgen=%llu event=%llu pos=%lld token=%d rows=1 main_session=1 completed=1\n",long(getpid()),(unsigned long long)id.rid,(unsigned long long)id.enginegen,(unsigned long long)id.generation,(unsigned long long)id.event,(long long)id.pos,id.token);
    assert(!begin(ids[row].rid,0,-1,id.prompt+1).rid&&admitting().rid==0);
   }
  }
  s0.release();''')
 # Separate exact begin/ARM-loss scenario, no graph or allocation.
 cpp=cpp.replace(' unsigned rejects=0;',r''' if(argc>1&&std::string(argv[1])=="stale"){
  auto id=begin(77,1,0,10);assert(admitting().rid==77);assert(!begin(88,0,-1,10).rid&&admitting().rid==0);span(id,0,0,48,0,10,"prefill_chunk");
  admitting()=id;::unlink(settings().arm);assert(!begin(89,1,0,10).rid&&admitting().rid==0);span(id,0,0,48,0,10,"prefill_chunk");assert(!sycl::allocations&&!sycl::copies&&!sycl::waits);return 0;
 }
 unsigned rejects=0;''')
 with tempfile.TemporaryDirectory(prefix='observer26-header-cpu-') as name:
  w=Path(name);mock=w/'mock/sycl';mock.mkdir(parents=True);(mock/'sycl.hpp').write_text(constants['MOCK']);(w/'test.cpp').write_text(cpp);capture=w/'capture';capture.mkdir();arm=w/'ARM';arm.write_text('ARM\n')
  docker=['docker','run','--rm','--network','none','--user','1000:1000','--memory','1g','--memory-swap','1g','-v',str(w)+':/work','-v',str(source)+':/source:ro','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7']
  command=['env','LC_ALL=C','g++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-Wno-parentheses','-fsanitize=address,undefined','-g','-I/work/mock','-I/source/sycl/include','/work/test.cpp','-o','/work/test']
  compiled=subprocess.run(docker+[shlex.join(command)],capture_output=True,text=True);assert compiled.returncode==0,compiled.stderr
  logs={}
  for mode in ['off','unarmed','stale','migration']:
   env=dict(os.environ);env.pop('STRATA_BATCH_FIDELITY_DIAG',None);env.pop('STRATA_BATCH_FIDELITY_ARM',None);env.pop('STRATA_BATCH_FIDELITY_DIR',None)
   arm.write_text('ARM\n')
   if mode!='off':env.update(STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/work/ABSENT' if mode=='unarmed' else '/work/ARM',STRATA_BATCH_FIDELITY_DIR='/work/capture')
   runenv=['env','-u','STRATA_BATCH_FIDELITY_DIAG','-u','STRATA_BATCH_FIDELITY_ARM','-u','STRATA_BATCH_FIDELITY_DIR']+[k+'='+env[k] for k in ['STRATA_BATCH_FIDELITY_DIAG','STRATA_BATCH_FIDELITY_ARM','STRATA_BATCH_FIDELITY_DIR'] if k in env]+['/work/test','off' if mode in ('off','unarmed') else mode]
   result=subprocess.run(docker+[shlex.join(runenv)],capture_output=True,text=True);assert result.returncode==0,result.stderr;logs[mode]=result.stderr
   if mode=='stale':assert 'SBF span ' not in result.stderr
  trace=logs['migration'];assert trace.count('SBF migration_begin ')==2 and trace.count('SBF solo_event ')==2 and trace.count('SBF vector ')==294
  # Preserve successful synthetic producer files for independent collector tests.
  receipt_dir=Path(tempfile.mkdtemp(prefix='observer26-mock-output-',dir='/mnt/vm_8tb/b70/build'));import shutil;shutil.copytree(capture,receipt_dir/'capture');(receipt_dir/'trace.log').write_text(trace)
 result={'CONFIG':'exact new source5files/actualheaders hostASanUBSan graphmocks; no SDK/GPU/model','COMMAND':'python3 strata/flash-next/test_solo_migration_observer_cpu_v1.py','RESULT':{'defaultoff_unarmed_zero_alloc_copy_wait':True,'actual_begin_unknown_solo_and_ARMloss_clear_current':True,'stale_priorRID_span_suppressed':True,'ARMloss_poison_after_reappearance':True,'two_RID_three_samples_two_stages_raw_vectors':294,'synthetic_output_observed':True,'patch_sha256':plan['patch_sha256']},'VERDICT':'PASS CPU diagnostic source controls only; actual migration/body/source/math/cache qualification absent','full_model_math_qualified':False}
 latest={'output':str(receipt_dir),'patch_sha256':plan['patch_sha256'],'trace_sha256':sha(receipt_dir/'trace.log')}
 Path('/mnt/vm_8tb/b70/build/strata-observer26-cpu-latest-v1.json').write_text(json.dumps(latest,indent=2)+'\n')
 (HERE/'solo-migration-observer-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS observer26 actual headers host mocks; GPU/migration math unqualified')
if __name__=='__main__':main()
