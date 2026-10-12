"""Source-only bounded host timing over genuine teacher-head/V6 source."""
import argparse,hashlib,json
from pathlib import Path
import prepare_original_mtp53_teacher_head_compile_v1 as parent
HERE=Path(__file__).resolve().parent;PLAN=HERE/'original-mtp53-paired-host-timing-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def one(s,a,b):
 if s.count(a)!=1:raise ValueError('Unique timing source anchor required '+a[:100])
 return s.replace(a,b,1)
def target_source(s):
 s='#include "strata/kernels/original_mtp53_phase_timing.hpp"\n'+s
 a=s.index('bool Verifier::run(int T,');b=s.index('\nvoid Verifier::set_plan_slot',a);body=s[a:b]
 body=one(body,'    const OnDevice on_device(device_);','    strata::kernels::original_mtp53::TargetHostSpan53 host_stage(this,device_,int(lb_),int(le_),T,pos0);\n    const OnDevice on_device(device_);')
 body=one(body,'    const uint64_t h36_ReplaySubmit_id=', '    host_stage.submitting();\n    const uint64_t h36_ReplaySubmit_id=')
 body=one(body,'    ple_input33::submit_returned(ple_input_publication,int(le));','    ple_input33::submit_returned(ple_input_publication,int(le));\n    host_stage.submitted(le==0);')
 body=one(body,'        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);','        const bool host_stage_ok=next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);host_stage.returned(host_stage_ok);return host_stage_ok;')
 body=one(body,'    progress_beat();\n    return true;','    progress_beat();\n    host_stage.returned(true);return true;')
 return s[:a]+body+s[b:]
def paired_source(s):
 s=one(s,'if(phase_flag && std::strcmp(phase_flag,"0")!=0)throw std::runtime_error("pair53 real shape requires phase timing OFF");','if(phase_flag && std::strcmp(phase_flag,"0")!=0 && std::strcmp(phase_flag,"1")!=0)throw std::runtime_error("pair53 phase timing expects0/1");')
 s=one(s,'if(!main.run(1,&last,position,pool,user,&first,error)||!main.commit(1,error)||!main.wait_commit(error))throw std::runtime_error(error);++position;', '{PhaseScope53 phase(&timing,"target_run");if(!main.run(1,&last,position,pool,user,&first,error))throw std::runtime_error(error);}{PhaseScope53 phase(&timing,"target_commit");if(!main.commit(1,error))throw std::runtime_error(error);}{PhaseScope53 phase(&timing,"target_wait_commit");if(!main.wait_commit(error))throw std::runtime_error(error);}++position;')
 s=one(s,'  const bool prompt_ok=first_prefill.run(prompt.data()+cut_start,end-cut_start,cut_start,error);', '  bool prompt_ok;{TargetHostSpan53 host_part(&first_prefill,-1,0,48,int(end-cut_start),cut_start,"target_prefill_part_run");prompt_ok=first_prefill.run(prompt.data()+cut_start,end-cut_start,cut_start,error);host_part.returned(prompt_ok);}')
 s=one(s,'if(!main.wait_commit(error))throw std::runtime_error(error);if(enabled)resident.set_timing(nullptr);','{PhaseScope53 phase(&timing,"final_target_wait_commit");if(!main.wait_commit(error))throw std::runtime_error(error);}if(enabled)resident.set_timing(nullptr);')
 return s
def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for name,want in plan['files'].items():
  if sha(HERE.parents[1]/name)!=want:raise ValueError('Exact timing source changed '+name)
 out=Path(out);record=parent.prepare(root,receipt,out);overlay=out/'overlay'
 for name,target in plan['overlay_sources'].items():(overlay/target).write_bytes((HERE.parents[1]/name).read_bytes())
 main=overlay/'sycl/src/program/original_mtp53_two_stage_diagnostic.cpp';text=main.read_text()
 text=one(text,'if(phase_timing && std::strcmp(phase_timing,"0")!=0)', 'if(phase_timing && std::strcmp(phase_timing,"0")!=0 && std::strcmp(phase_timing,"1")!=0)')
 text=text.replace('MTP53 pair phase timing must be OFF; old2048 row limit is not this scope','MTP53 pair bounded host phase timing expects0/1')
 main.write_text(text,encoding='ascii');verify=overlay/'sycl/src/core/verify.cpp';verify.parent.mkdir(parents=True,exist_ok=True);verify.write_text(target_source((Path(root)/'source/sycl/src/core/verify.cpp').read_text()),encoding='ascii')
 flags=record['compile_commands'][0]['argv'];base=flags[:flags.index('/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp')]
 link=flags[flags.index('-Wl,--start-group'):];link[link.index('/sdk/build/libstrata_engine.a')]='/out/libstrata_engine.without_verify.a';link[-1]='/out/original-mtp53-host-timing'
 record.update(compile_commands=[{'name':name,'argv':base+['-c',source,'-o','/out/'+name]}for source,name in[('/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp','main.host-timing.o'),('/overlay/sycl/src/core/verify.cpp','verify.host-timing.o')]],link_argv=[base[0],'-fsycl','/out/main.host-timing.o','/out/verify.host-timing.o',*link],archive_filter={'original':'/sdk/build/libstrata_engine.a','copy':'/out/libstrata_engine.without_verify.a','remove_members':['verify.cpp.o'],'before_counts':[1],'after_counts':[0]},host_timing_source_plan_sha256=sha(PLAN),host_timing_default_OFF=True,host_phase_capacity=32768,host_spans_not_additive=True,Verifier_public_private_layout_changed=False,new_GPU_waits_queries_copies=0,native_MAIN_two_stage_sha256=sha(main),host_timing_overlay_sha256={str(p.relative_to(overlay)):sha(p)for p in sorted(overlay.rglob('*'))if p.is_file()},actual_compile_receipt_ELF_required=True,model_or_speed_qualified=False)
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--sdk',type=Path,required=True);parser.add_argument('--sdk-receipt-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();print(json.dumps(prepare(args.sdk,args.sdk_receipt_sha256,args.output),ensure_ascii=True))
