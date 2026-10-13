"""Named failed-parent baseline component; no old binary/current identity authority."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-mtp53-serve-baseline-history-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def need(v,s):
 if not v:raise ValueError(s)
def default0_source_equivalence(old_main,new_main,new_verify,sdk_verify):
 # Explicit ONLY-original branches vanish at actual serving flag0. Verify the
 # complete remaining MAIN and complete verifier source, not a few keywords.
 s=new_main
 old='gs.head.loaded() ? &gs.head : nullptr, std::max(o.spec, o.batch), err);'
 new='gs.head.loaded() ? &gs.head : nullptr, original_serving?strata::kernels::kVerifyMaxT:std::max(o.spec, o.batch), err);'
 need(s.count(new)==1,'Exact LAST capacity conditional required');s=s.replace(new,old,1)
 new='                    if(original_serving)gs.ver.set_batch_graph_limit(64);\n';need(s.count(new)==1,'Exact LAST graph conditional required');s=s.replace(new,'',1)
 begin=s.index('            if(originalServe && (line=="ORIGINAL_EVICT"');end=s.index('            if(originalServe && (line.rfind("ORIGINAL_SAVE"',begin)if '            if(originalServe && (line.rfind("ORIGINAL_SAVE"'in s[begin:]else s.index('            if(originalServe && (line.rfind("ORIGINAL_SAVE ",',begin)
 s=s[:begin]+s[end:];need(s==old_main,'Default0 entire constructed MAIN differs')
 v=new_verify;include='#include "strata/core/original_mtp53_paired_slot_rows.hpp"\n';need(v.startswith(include),'Exact paired helper include required');v=v[len(include):]
 start=v.index('    auto original_flag=[]');end=v.index('        err = "verify: batch rows out of range (init_slots)";',start)
 replacement='''    if (S < 1 || S > max_t_ || hbase < 0 ||
        (next_ != nullptr && hbase + S > (int) slots_.size())) {
'''
 v=v[:start]+replacement+v[end:];v=v.replace('rows[t] == rows[t - 1] && next_ != nullptr && !paired_original_rows)','rows[t] == rows[t - 1] && next_ != nullptr)',1)
 need(v==sdk_verify,'Default0 entire actual verifier source differs')
 return {'default0_MAIN_control_flow_body_unchanged':True,'default0_verifier_body_unchanged':True,'target_public_layout_changed':False,'new_binary_runtime_qualified':False}
def finalized_binding(prior,expected_report_sha,new_build,current_runtime,current_inputs):
 prior=Path(prior);need(prior.is_absolute()and prior.resolve()==prior and sha(prior/'report.json')==expected_report_sha,'Authentic failed parent required')
 plan=read(PLAN)
 for p,h in plan['files'].items():need(sha(HERE.parents[1]/p)==h,'Baseline history source changed')
 report=read(prior/'report.json');need(report['passed']is False and report['source_plan_sha256']==sha(HERE/'original-mtp53-serve-qualification-source-plan-v2.json'),'Original failed V2 parent must stayfailed')
 artifacts=report['artifacts'];rows={}
 for rel,want in artifacts.items():
  if rel.startswith('target-baseline/')or rel=='intake.json':need(sha(prior/rel)==want,'Original baseline artifact changed '+rel);rows[rel]=want
 baseline=read(prior/'target-baseline/result.json');need(baseline['passed']is True and baseline['errors']==[] and baseline['owned_removed']is True and baseline['original_EOF']is True and baseline['forced_cleanup']is False and type(baseline['original_return_code'])is int and baseline['original_return_code']==0,'Closed normal baseline component required')
 capture=read(prior/'target-baseline/native.capture.json');need(capture['passed']is True and capture['forced_cleanup']is False and capture['return_code']==0,'Original baseline exact pipe EOF required')
 terminal=read(prior/'target-baseline/terminal.json');st=terminal['State'];need(type(st['ExitCode'])is int and st['ExitCode']==0 and all(st[k]is False for k in ['Running','Paused','Restarting','OOMKilled','Dead'])and st['Error']=='','Original baseline normal terminal required')
 for label in ['pre-health','pre-collective','post-health','post-collective','post-kernel']:
  evidence=read(prior/('target-baseline/'+label+'.result.json'));need(type(evidence['return_code'])is int and evidence['return_code']==0,'Original baseline health/kernel required')
  for stream in ['stdout','stderr']:need(sha(prior/('target-baseline/'+label+'.'+stream))==evidence[stream+'_sha256'],'Original baseline check raw changed')
 old_intake=read(prior/'intake.json');old_receipt=Path(old_intake['actual_compile_receipt']);need(sha(old_receipt)==report['actual_compile_receipt_sha256'],'Original baseline compile receipt changed');old_build=read(old_receipt);need(old_build['passed']is True and old_build['actual58']==new_build['actual58'],'Same actual58 target archives/source required')
 old_root=old_receipt.parent.parent;new_root=Path(current_runtime['actual_compile_receipt']).parent.parent
 old_main=old_root/'overlay/sycl/src/program/generate.cpp';new_main=new_root/'overlay/sycl/src/program/generate.cpp';new_verify=new_root/'overlay/sycl/src/core/verify.cpp';sdk=Path(new_build['actual58']['SDK']);old_source=read(HERE/'original-mtp53-serve-source-plan-v3.json');new_source=read(HERE/'original-mtp53-serve-source-plan-v5.json')
 need(sha(old_main)==old_source['constructed_generate_sha256']and sha(new_main)==new_source['constructed_generate_sha256']and sha(new_verify)==new_source['files']['strata/flash-next/original-mtp53/paired_slot_verify_v1.cpp']and sha(sdk/'source/sycl/src/core/verify.cpp')==new_source['actual58_verify_sha256'],'Actual compiled default0 source bytes required')
 proof=default0_source_equivalence(old_main.read_text(),new_main.read_text(),new_verify.read_text(),(sdk/'source/sycl/src/core/verify.cpp').read_text())
 need(old_intake['native_argv']==current_runtime['native_argv'],'Same actual native config required');old_env=dict(old_intake['native_env']);new_env=dict(current_runtime['native_env']);old_env.pop('STRATA_ORIGINAL_MTP53_DRAFT_CONTRACT_SHA256');new_env.pop('STRATA_ORIGINAL_MTP53_DRAFT_CONTRACT_SHA256');need(old_env==new_env,'Default0 environment mustmatch exceptunused original contract pin')
 need(len(baseline['cohorts'])==3 and all(len(c)==4 for c in baseline['cohorts']),'Original complete baseline histories required')
 for cohort in baseline['cohorts']:
  for i,row in enumerate(cohort):need(row['input_ids']==current_inputs[i%2]['ids'],'Same authentic tokenized input required')
 for p,want in rows.items():need(sha(prior/p)==want,'Closing baseline raw changed')
 return {**baseline,'historical_baseline_component':True,'original_parent_passed':False,'original_parent_sha256':expected_report_sha,'source_equivalence':proof,'new_binary_or_model_admission_transferred':False,'performance_qualified':False}
