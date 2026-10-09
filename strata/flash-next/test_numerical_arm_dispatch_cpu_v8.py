#!/usr/bin/env python3
"""Actual old reference metadata + actual new C1 proof; reject model payload IO."""
import builtins,copy,io,json
from pathlib import Path
from unittest.mock import patch
import layer0_numerical_qualification_v8 as q
HERE=Path(__file__).resolve().parent

def main():
 root=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008');oldplan=root/'f08-20261009/layer0-numerical-onecard-prepared-v2/plan.json';child=root/'f08-20261009/layer0-numerical-onecard-v2/child/report.json';parent=root/'f08-20261009/layer0-numerical-onecard-v2/parent-qualification.json';new=root/'f10-20261009/c1-onecard-combined28-prepared-v6'
 reference=q.read(oldplan);candidate=q.read(new/'prepared.json');old=q.read(Path(reference['prepared'])/'prepared.json');engine=Path(candidate['engine_receipt']).parent
 plan={'reference_plan':str(oldplan),'reference_plan_sha256':q.sha(oldplan),'reference_qualification':str(child),'reference_qualification_sha256':q.sha(child),'reference_parent_qualification':str(parent),'reference_parent_qualification_sha256':q.sha(parent),'engine_root':str(engine),'prepared':str(new)}
 prohibited=[Path(row['path']).resolve() for row in old['model_shards']+candidate['model_shards']];packs=[Path(m['pack']).resolve() for m in [old,candidate]];real_io=io.open;real_builtin=builtins.open;model_attempts=[]
 def guarded(real):
  def open_file(file,*a,**kw):
   if not isinstance(file,int):
    path=Path(file).resolve()
    if path in prohibited or any(path==pack or path.is_relative_to(pack) for pack in packs):model_attempts.append(str(path));raise AssertionError('CPU dispatch test must not read model/pack payload')
   return real(file,*a,**kw)
  return open_file
 negatives=0
 def bad(fn):
  nonlocal negatives
  try:fn()
  except (ValueError,AssertionError,KeyError):negatives+=1;return
  raise AssertionError('Wrong lane/reference proof accepted')
 with patch.object(io,'open',guarded(real_io)),patch.object(builtins,'open',guarded(real_builtin)):
  ref=q.arm_lane_metadata(plan,'reference21',Path(reference['engine_root']),Path(reference['prepared']),False);assert ref['validator_source']==str(Path(q.reference_parent.ctrl.c1.__file__)) and ref['lane']=='reference21_frozen_V2'
  for name,enabled in [('candidatecombined_off',False),('candidatecombined_on',True)]:
   got=q.arm_lane_metadata(plan,name,engine,new,enabled);assert got['validator_source']==str(Path(q.c1.__file__)) and got['lane']=='candidate_combined28_strict_C1v6'
  bad(lambda:q.arm_lane_metadata(plan,'reference21',engine,new,False));bad(lambda:q.arm_lane_metadata(plan,'reference21',Path(reference['engine_root']),Path(reference['prepared']),True));bad(lambda:q.arm_lane_metadata(plan,'candidatecombined_on',Path(reference['engine_root']),Path(reference['prepared']),True));bad(lambda:q.arm_lane_metadata(plan,'candidatecombined_on',engine,new,False));bad(lambda:q.arm_lane_metadata(plan,'arbitrary',engine,new,False))
  # Real generation validators reject the actual preparation from the other lane
  # at their source/controller gate before any model/pack payload operation.
  bad(lambda:q.c1.validate_prepared(Path(reference['prepared'])));bad(lambda:q.reference_parent.ctrl.c1.validate_prepared(new))
  # Real failed old paired report remains inadmissible, despite preserved math.
  pairplan=root/'f08-20261009/layer0-numerical-twocard-prepared-v2/plan.json';pairchild=root/'f08-20261009/layer0-numerical-twocard-v2/child/report.json';pairparent=root/'f08-20261009/layer0-numerical-twocard-v2/parent-qualification.json';pair=q.read(pairplan);failed=copy.deepcopy(plan);failed.update(reference_plan=str(pairplan),reference_plan_sha256=q.sha(pairplan),reference_qualification=str(pairchild),reference_qualification_sha256=q.sha(pairchild),reference_parent_qualification=str(pairparent),reference_parent_qualification_sha256=q.sha(pairparent));bad(lambda:q.arm_lane_metadata(failed,'reference21',Path(pair['engine_root']),Path(pair['prepared']),False))
  broken=copy.deepcopy(plan);broken['reference_parent_qualification_sha256']='0'*64;bad(lambda:q.arm_lane_metadata(broken,'reference21',Path(reference['engine_root']),Path(reference['prepared']),False))
 assert not model_attempts,'A supposedly metadata-only check attempted model reads'
 result={'CONFIG':'actual frozen21 one V2 prepared/engine/source/qualified child+parent; actual import28 one C1parent6 strong final proof; all model/pack opens denied','COMMAND':'python3 strata/flash-next/test_numerical_arm_dispatch_cpu_v8.py','RESULT':{'actual_reference21_metadata_verified':True,'actual_candidate_C1parent6_proof_verified':True,'wrong_lane_negative_controls':negatives,'model_payload_read_attempts':0,'reference_plan_sha256':q.sha(oldplan),'reference_child_sha256':q.sha(child),'reference_parent_sha256':q.sha(parent),'candidate_prepared_sha256':q.sha(new/'prepared.json'),'candidate_final_qualification_sha256':q.sha(new/'qualification.json')},'VERDICT':'PASS CPU actual metadata/dispatch proof only; no numerical preparation/execution/GPU/model payload reads; failed21 pair remains inadmissible','full_model_math_qualified':False}
 q.write(HERE/'numerical-arm-dispatch-cpu-receipt-v8.json',result);print('PASS actual old/new lane metadata and strong C1proof; no payload reads')
if __name__=='__main__':main()
