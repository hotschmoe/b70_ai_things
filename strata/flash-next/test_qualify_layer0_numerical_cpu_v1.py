#!/usr/bin/env python3
"""CPU parent identity and lifecycle invariants; no health/container/model run."""
import ast,copy,json
from pathlib import Path
import qualify_layer0_numerical_v1 as q


def main():
 source=q.source_plan_binding();assert len(source['patches'])==21
 assert q.sha(Path(q.ctrl.__file__))==q.L0_NUMERICAL_SHA
 parent={'child_return_code':0,'interrupted':False,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[]};child={'numerical_and_teardown_passed':True,'finished_epoch':1};post={'passed':True,'started':2};assert q.finalizable(parent,child,post)
 for key,value in [('child_return_code',1),('interrupted',True),('owned_containers_terminal',False),('forced_cleanup',True),('pre_health_passed',False),('post_health_passed',False),('kernel_fault_gate_passed',False),('errors',['failed'])]:
  bad=copy.deepcopy(parent);bad[key]=value;assert not q.finalizable(bad,child,post)
 assert not q.finalizable(parent,child,{'passed':True,'started':0})
 current=ast.parse(Path(q.__file__).read_text());old=ast.parse(Path(q.__file__).with_name('qualify_serial_prefix_v6.py').read_text())
 for name in ['stat_signature','full_buffered_identity','finalizable','command','health','faults','containers','source_watch','stop','save']:
  a=next(node for node in ast.walk(current) if isinstance(node,ast.FunctionDef) and node.name==name);b=next(node for node in ast.walk(old) if isinstance(node,ast.FunctionDef) and node.name==name);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
 text=Path(q.__file__).read_text()
 for marker in ['pass_fds=(8,9)','ctrl.c1.leased([0,1])','label=b70.prefix.plan=','--packet-receipt','--lifecycle-receipt','layer0-logical-lifecycle.json','packet-contract-cpu']:assert marker in text
 plan=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f08-20261009/layer0-numerical-onecard-prepared-v1/plan.json');actual=q.prepared_chain_binding(q.read(plan));assert actual['candidate']['source_plan_sha256']==q.ENGINE_PLAN_SHA
 result={'CONFIG':'actual genuine21 onecard preparation/source binding; retained parent gates pure CPU/source checks','COMMAND':'python3 strata/flash-next/test_qualify_layer0_numerical_cpu_v1.py','RESULT':{'actual21_and_matched20_identity_chain':True,'old_parent_control_helpers_ast_equal':True,'all_parent_negative_finalization_gates':True,'fd8_9_packet_lifecycle_manifest_guards':True},'VERDICT':'PASS CPU source/parent invariants; real numerical GPU/lifecycle/packet evidence absent','full_model_math_qualified':False};Path(__file__).with_name('layer0-parent-orchestration-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS actual21+20 identity andretained parent gates; no GPU/numerical claims')
if __name__=='__main__':main()
