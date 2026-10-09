#!/usr/bin/env python3
"""Retained per-process parent/source/artifact semantics; no model/GPU execution."""
import ast,json,tempfile,time
from pathlib import Path
import qualify_batch_numerical_v3 as p

def function(path,name):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
def main():
 here=Path(__file__).resolve().parent
 for name in ('full_buffered_identity','stat_signature','finalizable'):assert function(here/'qualify_batch_numerical_v2.py',name)==function(here/'qualify_batch_numerical_v3.py',name)
 assert p.source_plan_binding('source29')['expected_patched_source_sha256'] and len(p.source_plan_binding('source31')['expected_patched_source_sha256'])==62
 parent={'child_return_code':0,'interrupted':False,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[],'post_health_finished_epoch':2};child={'collection_and_teardown_passed':True,'finished_epoch':1};identity={'passed':True,'started':3};assert p.finalizable(parent,child,identity)
 for key,value in [('child_return_code',1),('interrupted',True),('owned_containers_terminal',False),('forced_cleanup',True),('pre_health_passed',False),('post_health_passed',False),('kernel_fault_gate_passed',False),('errors',['actualfailure'])]:
  bad=dict(parent);bad[key]=value;assert not p.finalizable(bad,child,identity)
 assert not p.finalizable(parent,child,{'passed':True,'started':1.5})
 print(json.dumps({'passed':True,'negative_controls':9,'V2_parent_full4_stat_finalization_AST_unchanged':True,'explicit_both_source_plans_checked':True,'actual_model_payload_reads':0,'actual_GPU_executions':0}))
if __name__=='__main__':main()
