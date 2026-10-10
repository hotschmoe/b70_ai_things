#!/usr/bin/env python3
"""Metadata/command/scope controls only; never compiles or executes a device workload."""
import copy,json,re
from pathlib import Path
import prepare_gdn_state_transition35_v1 as p

def reject(fn):
 try:fn()
 except (ValueError,KeyError):return
 raise AssertionError('Invalid source control accepted')
foundation=p.read(p.FOUNDATION);inventory=p.read(foundation['inventory']);g=p.geometry(inventory,foundation)
assert g['state_bytes']==128*48*128*4 and g['history_bytes']==10240*3*4
for key,value in [('qwen4exp.ssm.state_size',64),('qwen4exp.ssm.group_count',8),('qwen4exp.ssm.time_step_rank',32),('qwen4exp.ssm.conv_kernel',3),('qwen4exp.ssm.inner_size',4096)]:
 bad=copy.deepcopy(inventory);bad['files'][0]['metadata'][key]['value']=value;reject(lambda:p.geometry(bad,foundation))
bad=copy.deepcopy(inventory);bad['inventory_complete']=False;reject(lambda:p.geometry(bad,foundation))
bad=copy.deepcopy(inventory);bad['errors']=['header mismatch'];reject(lambda:p.geometry(bad,foundation))
bad=copy.deepcopy(inventory)
for f in bad['files']:
 for t in f['tensors']:
  if t['name']=='blk.0.ssm_conv1d.weight':t['shape_ggml_order']=[4,8192]
reject(lambda:p.geometry(bad,foundation))
planpath=p.ROOT/'strata/flash-next/gdn-state-transition35-v1-compile-run-plan.json';plan=p.read(planpath)
assert plan['geometry']==g and plan['expected_raw_comparisons']==27 and plan['model_math_qualified'] is False
assert plan['real_row3_GDN_inputs_observed'] is False and plan['backend_patch_required'] is False
assert plan['required_absent_environment']==['STRATA_VERIFY_EAGER'] and plan['environment']['STRATA_GDN_SPLIT']=='0'
assert len(plan['actual_eight_SDK_ELFs'])==8 and plan['engine_plan_sha256']==p.ENGINE_PLAN_SHA
assert plan['leaf_source_sha256']==p.sha(plan['leaf_source']) and plan['producer_sha256']==p.sha(Path(p.__file__))
assert all(p.sha(row['path'])==row['sha256'] for row in plan['current_SDK_dependencies'])
compilecmd=plan['compile_argv_inside_pinned_image'];assert '-fsycl' in compilecmd and '-fsycl-default-sub-group-size=32' in compilecmd and '-fp-model=precise' in compilecmd
assert '/sdk/build/libstrata_kernels.a' in compilecmd and '/sdk/build/libstrata_core.a' in compilecmd
source=Path(plan['leaf_source']).read_text()
for marker in ['keep=0;keep<=2','gdn_conv_commit(cwst,dq,C,nk','HK,HV,2,nullptr','HK,HV,2,nk,&q,2','q.memset(st,0,ST*4)','forward-state-unmodified','forward-conv-unmodified','accepted-state','accepted-conv','carry-state','carry-conv','carry-output','modified_state_output_detected','auto own_h=scalar_conv(qkv,cw,STEPS)','scalar_state(own_h,gate,beta,STEPS)','numerical_tolerance_qualified=0','for(auto p:owned) sycl::free(p,q)']:
 assert marker in source,marker
assert 'scalar_state(h_ref' not in source,'Independent arithmetic must not receive native normalized qkv'
assert '#include <cuda' not in source and 'cudaMalloc' not in source and 'hip' not in source
print('PASS CPU/source GDN35: actual locked header geometry +8 negatives, pinned current SYCL archive/header/source/eightELF closure, fresh precise subgroup32 compile proposal, zeroF32 T2commit0/1/2+T1carry, perturbedstate negative contract, independent synthetic FP64 diagnostic, explicit no math tolerance. No compile/device/Docker/model payload touched.')
