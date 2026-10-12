"""Default-OFF first-kept-row own-head omission; source preparation only."""
import argparse, hashlib, json
from pathlib import Path
import prepare_original_mtp53_teacher_head_compile_v1 as parent
HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-mtp53-decode-head-skip-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('Exact decode-head source seam missing: '+a)
 return s.replace(a,b,1)
def transformed_step(source):
 s='#include "strata/kernels/original_mtp53_decode_head_skip.hpp"\n'+source
 s=once(s,'uint64_t head_calls_=0,head_skips_=0;', 'uint64_t head_calls_=0,head_skips_=0,decode_rows_=0,decode_skips_=0,decode_windows_=0;')
 s=once(s,'    bool head_ready()const noexcept{return head_ready_;}', '    uint64_t decode_rows()const noexcept{return decode_rows_;}\n    uint64_t decode_skips()const noexcept{return decode_skips_;}\n    uint64_t decode_windows()const noexcept{return decode_windows_;}\n    bool head_ready()const noexcept{return head_ready_;}')
 a='{PhaseScope53 phase(timing_,"draft_record_wait");record(device_token,main.final_R_all()+size_t(row)*10240,w,s);q.wait_and_throw();}'
 b='{PhaseScope53 phase(timing_,"draft_record_wait");record(device_token,main.final_R_all()+size_t(row)*10240,w,s,decode_head_compute53(decode_head_skip53(),keep,row));q.wait_and_throw();}'
 s=once(s,a,b)
 a='if(state_.committed_cells()!=base_position+keep){main_failed_=true;error="draft53 catch-up did not reach target prefix";return false;}return true;'
 # Only bounded actual method changes; old unbounded helper remains byte-identical.
 begin=s.index('    bool verify_bounded_and_catchup(')
 prefix,tail=s[:begin],s[begin:]
 tail=once(tail,a,'if(state_.committed_cells()!=base_position+keep || !head_ready_){main_failed_=true;error="draft53 catch-up did not reach target prefix/fresh head";return false;}decode_rows_+=keep;decode_skips_+=decode_head_skip53() && keep==2; ++decode_windows_;return true;')
 return prefix+tail
def transformed_loaded(source):
 s=once(source,'    const strata::core::OnDevice on(device_);\n    const char* skip=', '    const strata::core::OnDevice on(device_);\n    (void)decode_head_skip53(); // Validate BEFORE original tensor upload.\n    const char* skip=')
 s=once(s,'teacher_head_skip_=skip && std::strcmp(skip,"1")==0;', 'teacher_head_skip_=skip && std::strcmp(skip,"1")==0;if(teacher_head_skip_)throw std::runtime_error("decode-head experiment requires teacher head skip OFF");')
 a='~LoadedMainDraft53(){const strata::core::OnDevice on(device_);owner.retire();}'
 b='~LoadedMainDraft53(){const strata::core::OnDevice on(device_);owner.retire();if(!failed && draft){std::printf("MTP53_DECODE_HEAD_COUNTS mode=%d windows=%llu rows=%llu skips=%llu calls=%llu total_calls=%llu total_skips=%llu fresh_head=%d committed_cells=%lld\\n",int(decode_head_skip53()),(unsigned long long)draft->decode_windows(),(unsigned long long)draft->decode_rows(),(unsigned long long)draft->decode_skips(),(unsigned long long)(draft->decode_rows()-draft->decode_skips()),(unsigned long long)draft->head_calls(),(unsigned long long)draft->head_skips(),int(draft->head_ready()),(long long)cells());std::fflush(stdout);}}'
 return once(s,a,b)
def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for name,want in plan['files'].items():
  if sha(HERE.parents[1]/name)!=want:raise ValueError('Frozen decode-head source changed '+name)
 out=Path(out);record=parent.prepare(root,receipt,out);overlay=out/'overlay'
 targets={'sycl/include/strata/kernels/original_mtp53_draft_step.hpp':transformed_step,'sycl/include/strata/kernels/original_mtp53_loaded_main_adapter.hpp':transformed_loaded}
 for name,fn in targets.items():
  path=overlay/name;path.write_text(fn(path.read_text()),encoding='ascii')
 helper='sycl/include/strata/kernels/original_mtp53_decode_head_skip.hpp'
 (overlay/helper).write_bytes((HERE/'original-mtp53/decode_head_skip_v1.hpp').read_bytes())
 record.update(decode_head_source_plan_sha256=sha(PLAN),decode_head_overlay_sha256={n:sha(overlay/n)for n in list(targets)+[helper]},only_matched_delta='STRATA_ORIGINAL_MTP53_DECODE_FIRST_HEAD_SKIP=0/1',teacher_head_skip_required=0,default_OFF=True,all_waits_and_state_settle_preserved=True,only_bounded_accepted_first_of_two_skipped=True,runtime_qualified=False)
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
