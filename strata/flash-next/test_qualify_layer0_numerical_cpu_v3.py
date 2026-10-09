#!/usr/bin/env python3
"""Retained parent lifecycle and changed page/extent controls; CPU source only."""
import ast,copy,json
from pathlib import Path
import qualify_layer0_numerical_v3 as q

def main():
 source=q.source_plan_binding();assert len(source['patches'])==24 and len(source['expected_patched_source_sha256'])==51
 assert q.sha(Path(q.ctrl.__file__))==q.L0_NUMERICAL_SHA
 parent={'child_return_code':0,'interrupted':False,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[]};child={'numerical_and_teardown_passed':True,'finished_epoch':1};post={'passed':True,'started':2};assert q.finalizable(parent,child,post)
 for key,value in [('child_return_code',1),('interrupted',True),('owned_containers_terminal',False),('forced_cleanup',True),('pre_health_passed',False),('post_health_passed',False),('kernel_fault_gate_passed',False),('errors',['source-page/fullhash failed'])]:
  bad=copy.deepcopy(parent);bad[key]=value;assert not q.finalizable(bad,child,post)
 assert not q.finalizable(parent,child,{'passed':True,'started':0});assert not q.finalizable(parent,child,{'passed':False,'started':2})
 current=ast.parse(Path(q.__file__).read_text());old=ast.parse(Path(q.__file__).with_name('qualify_layer0_numerical_v2.py').read_text())
 names=['stat_signature','full_buffered_identity','finalizable','command','health','faults','containers','stop','save']
 for name in names:
  a=next(n for n in ast.walk(current) if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name==name);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
 text=Path(q.__file__).read_text()
 for marker in ['preserve_source_pages(shards[2],out,label)','candidatecombined_on/engine.combined.log','pass_fds=(8,9)','ctrl.c1.leased([0,1])','label=b70.prefix.plan=','verify_layer0_packets_v3.py','--lifecycle-receipt']:assert marker in text
 result={'CONFIG':'pure CPU AST source/lifecycle negatives; no current model payload or runtime accesses','COMMAND':'python3 strata/flash-next/test_qualify_layer0_numerical_cpu_v3.py','RESULT':{'retained_parent_helper_ast_equal':names,'finalization_negative_controls':10,'new_two_page_preservation_hook':True,'same_merged_producer_stream':True},'VERDICT':'PASS CPU parent contracts only; genuine combined SDK/C1/full390 preparation and actual GPU numerical evidence absent','full_model_math_qualified':False}
 Path(__file__).with_name('layer0-parent-orchestration-cpu-receipt-v3.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS V3 retained parent gates and failclosed controls; CPU source only')
if __name__=='__main__':main()
