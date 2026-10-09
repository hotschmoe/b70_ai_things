#!/usr/bin/env python3
"""Explicit frozen21 dispatch and blocked integrated29 admission; no payload IO."""
import builtins,copy,io,json,tempfile
from pathlib import Path
from unittest.mock import patch
import layer0_numerical_qualification_v9 as q
HERE=Path(__file__).resolve().parent

def main():
 root=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008');oldplan=root/'f08-20261009/layer0-numerical-onecard-prepared-v2/plan.json';child=root/'f08-20261009/layer0-numerical-onecard-v2/child/report.json';parent=root/'f08-20261009/layer0-numerical-onecard-v2/parent-qualification.json'
 reference=q.read(oldplan);prepared=q.read(Path(reference['prepared'])/'prepared.json');old28=root/'f10-20261009/c1-onecard-combined28-prepared-v6'
 plan={'reference_plan':str(oldplan),'reference_plan_sha256':q.sha(oldplan),'reference_qualification':str(child),'reference_qualification_sha256':q.sha(child),'reference_parent_qualification':str(parent),'reference_parent_qualification_sha256':q.sha(parent),'engine_root':str(Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T185206Z-d5q32zc7')),'prepared':str(old28)}
 prohibited=[Path(row['path']).resolve() for row in prepared['model_shards']];pack=Path(prepared['pack']).resolve();model_attempts=[];real_io=io.open;real_builtin=builtins.open
 def guarded(real):
  def open_file(file,*args,**kw):
   if not isinstance(file,int):
    path=Path(file).resolve()
    if path in prohibited or path==pack or path.is_relative_to(pack):model_attempts.append(str(path));raise AssertionError('CPU metadata dispatch cannot read model/pack payload')
   return real(file,*args,**kw)
  return open_file
 negatives=0
 def bad(fn):
  nonlocal negatives
  try:fn()
  except (ValueError,AssertionError,KeyError,FileNotFoundError):negatives+=1;return
  raise AssertionError('Wrong generation/lane accepted')
 with patch.object(io,'open',guarded(real_io)),patch.object(builtins,'open',guarded(real_builtin)):
  sdk=q.engine_binding(Path(plan['engine_root']),q.PLAN_SOURCE_SHA);assert sdk['build_rc']==0 and len(sdk['patched_source_sha256'])==60 and len(sdk['binary_sha256'])==8
  bad(lambda:q.engine_binding(Path(q.read(old28/'prepared.json')['engine_receipt']).parent,q.PLAN_SOURCE_SHA))
  lane=q.arm_lane_metadata(plan,'reference21',Path(reference['engine_root']),Path(reference['prepared']),False)
  assert lane['lane']=='reference21_frozen_V2' and lane['validator_source']==str(Path(q.reference_parent.ctrl.c1.__file__))
  bad(lambda:q.arm_lane_metadata(plan,'reference21',Path(plan['engine_root']),old28,False))
  bad(lambda:q.arm_lane_metadata(plan,'reference21',Path(reference['engine_root']),Path(reference['prepared']),True))
  bad(lambda:q.arm_lane_metadata(plan,'candidatecombined_on',Path(reference['engine_root']),Path(reference['prepared']),True))
  bad(lambda:q.arm_lane_metadata(plan,'candidatecombined_on',Path(plan['engine_root']),old28,False))
  bad(lambda:q.arm_lane_metadata(plan,'candidatecombined_off',Path(plan['engine_root']),old28,False))
  bad(lambda:q.c1.validate_prepared(old28))
  broken=copy.deepcopy(plan);broken['reference_parent_qualification_sha256']='0'*64;bad(lambda:q.arm_lane_metadata(broken,'reference21',Path(reference['engine_root']),Path(reference['prepared']),False))
  # The frozen source plan remains available, without pretending a genuine C18
  # qualification exists while the root's live C18 run is incomplete.
  for name in q.NUMERICAL_BASELINE_OFF:bad(lambda:q.numerical_baseline_gate({name:'1'}))
  q.numerical_baseline_gate({})
  # Candidate C18 final proof must reject BEFORE validate_prepared's source pages.
  with patch.object(q,'engine_binding',return_value={'scope':'CPU mocked SDK only'}),patch.object(q.c1,'metadata_admission_gate',return_value={'scope':'CPU mocked upload only'}),patch.object(q,'validate_final_source_proof',side_effect=ValueError('Missing actual C18 final proof')),patch.object(q.c1,'validate_prepared',side_effect=AssertionError('Model scan must not happen before C18 final proof')):
   bad(lambda:q.candidate_binding(plan))
 assert not model_attempts
 result={'CONFIG':'Actual old21 one V2 metadata dispatch; old28 candidate refused; C18 final proof gate simulated absent. All original model/pack opens denied','COMMAND':'python3 strata/flash-next/test_numerical_arm_dispatch_cpu_v9.py','RESULT':{'actual_integrated29_SDK_source_metadata_verified':True,'actual_reference21_metadata_verified':True,'actual_candidate_C1parent8_proof_verified':False,'wrong_generation_lane_flag_proof_negative_controls':negatives,'model_payload_read_attempts':0,'reference_plan_sha256':q.sha(oldplan),'reference_child_sha256':q.sha(child),'reference_parent_sha256':q.sha(parent)},'VERDICT':'PASS CPU dispatch only; no new genuine preparation/runtime/math claim','full_model_math_qualified':False}
 q.write(HERE/'numerical-arm-dispatch-cpu-receipt-v9.json',result);print('PASS V9 frozen21 dispatch and failclosed C18/source/flag gates; no payload IO')
if __name__=='__main__':main()
