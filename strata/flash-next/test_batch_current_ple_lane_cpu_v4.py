#!/usr/bin/env python3
"""Current-PLE32 exact lane/source guards; synthetic metadata, no weights/GPU."""
import ast,json,tempfile
from pathlib import Path
import batch_numerical_proofs_v4 as p
import qualify_batch_numerical_v4 as parent
HERE=Path(__file__).resolve().parent

def bad(fn):
 try:fn()
 except (ValueError,AssertionError,KeyError,FileNotFoundError):return
 raise AssertionError('Historical wrong-PLE lane accepted')

def function(file,name):return ast.dump(next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)

def main():
 spec=p.lane_contract('source32');assert (spec['source_files'],spec['headers'],spec['patches'],spec['generation'])==(62,26,32,10);c1,proof=p.providers('source32');assert c1.__name__=='c1_serve_controller_combined_v10' and proof.__name__=='qualify_c1_serving_combined_v10'
 controls=0
 for lane in ('source29','source31','old28'):bad(lambda:p.lane_contract(lane));controls+=1
 for file,name in [('full_buffered_identity','full_buffered_identity'),('stat_signature','stat_signature'),('finalizable','finalizable')]:assert function(HERE/'qualify_batch_numerical_v3.py',file)==function(HERE/'qualify_batch_numerical_v4.py',name)
 with tempfile.TemporaryDirectory() as name:
  root=Path(name)
  for oldplan in ('94aef305dbc6a31f023afa55573d52a9ce6af9443cfc0182eb7c324255159094','82951242c998acecd0b0a138466991c607c1ab89abbf3b4501c001e0feaae96e'):
   (root/'receipt.json').write_text(json.dumps({'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'plan_sha256':oldplan}));bad(lambda:p.engine_binding(root,'source32'));controls+=1
 policy={(1,'admission'):{'reused':0,'read_from':0,'reread_to':-1}};trace='SBF resume rid=1 slot=0 reused=0 read_from=0 reread_to=-1';assert p.exact_producer_counters(trace,{1},policy)
 for key,value in [('reused','1'),('read_from','1'),('reread_to','0')]:bad(lambda:p.exact_producer_counters(trace.replace(key+'='+('-1' if key=='reread_to' else '0'),key+'='+value),{1},policy));controls+=1
 plan=parent.source_plan_binding('source32');assert len(plan['patches'])==32 and len(plan['expected_patched_source_sha256'])==62 and len(plan['added_header_payloads'])==26
 for n in (2,4,6):
  case=p.read(HERE/('batch-numerical-case'+str(n)+'-source32-v1.json'));old=p.read(HERE/('batch-numerical-case'+str(n)+'-source29-v1.json'));assert case['source32_semantic_current_PLE_required'] and case['tokens']==old['tokens'] and case['api_token_ids']==old['api_token_ids'] and case['messages']==old['messages'] and case['source_lane']=='source32'
 print(json.dumps({'passed':True,'negative_controls':controls,'source32_62_26_32_eight_C110_required':True,'historical_wrong_PLE29_31_rejected':True,'parent_source_health_full4_finalization_AST_unchanged':True,'original_weight_reads':0,'actual_GPU_Docker_runtime_writes_or_genuine_prepares':0},ensure_ascii=True))
if __name__=='__main__':main()
