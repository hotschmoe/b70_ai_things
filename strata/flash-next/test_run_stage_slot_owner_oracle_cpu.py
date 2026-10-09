#!/usr/bin/env python3
"""CPU-only case/parent lifecycle controls, every process/device boundary mocked."""
import copy,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import run_stage_slot_owner_oracle as q
PLAN=q.ROOT/'strata/flash-next/stage-slot-owner-oracle-plan.json'
plan=q.read(PLAN);assert len(q.cases(plan))==3
bad=copy.deepcopy(plan);bad['cases'][1]['mask']='0'
try:q.cases(bad)
except ValueError:pass
else:raise AssertionError('wrong physical card accepted')
base={'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'owned_containers_terminal':True,'interrupted':False,'errors':[],'forced_cleanup':False,'cases':[{'case':r['id'],'passed':True,'removed':True,'terminal':{'ExitCode':0,'Running':False,'OOMKilled':False}} for r in plan['cases']]}
assert q.can_pass(base)
for key,value in [('pre_health_passed',False),('post_health_passed',False),('kernel_fault_gate_passed',False),('owned_containers_terminal',False),('interrupted',True),('forced_cleanup',True),('errors',['failure'])]:
 b=copy.deepcopy(base);b[key]=value;assert not q.can_pass(b)
b=copy.deepcopy(base);b['cases'].pop();assert not q.can_pass(b)
with tempfile.TemporaryDirectory(prefix='owner-runner-cpu-') as td:
 root=Path(td)
 for rel in ['vllm/int4/diagnostics/xpu_health_strict.sh','bin/xpu-collective-health','bin/xpu-collective-health.py']:
  p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU mock source\n')
 for failing in [False,True]:
  out=root/('failure' if failing else 'success');containers={};calls=[];receipt=root/'build/receipt.json';receipt.parent.mkdir(exist_ok=True);receipt.write_text('{}')
  class FakeProcess:
   def __init__(self,cmd,**kwargs):
    self.pid=424242;self.returncode=0;calls.append((cmd,kwargs.get('pass_fds')))
    if cmd[:3]==['docker','run','-d']:
     name=cmd[cmd.index('--name')+1];label=name.split('-424242-')[-1] if '-424242-' in name else name.split('-'+str(os.getpid())+'-')[-1]
     labels={cmd[i+1].split('=',1)[0]:cmd[i+1].split('=',1)[1] for i,x in enumerate(cmd) if x=='--label'}
     containers[name]={'Name':'/'+name,'Config':{'Labels':labels,'Image':q.IMAGE},'State':{'Running':False,'ExitCode':0,'OOMKilled':False}}
     profile=next(x for x in plan['cases'] if x['id']==label)
     q.write(out/(label+'.json'),{'owner_and_probe_passed':True,'owner_registrations':profile['expected_owner_registrations'],'stages':len(profile['stages']),'cells':64,'model_weights_loaded':False,'inference_or_graph_retirement_qualified':False})
    if cmd[:3]==['docker','rm','-f']:containers.pop(cmd[3],None)
    if hasattr(kwargs.get('stdout'),'write'):kwargs['stdout'].write('CPU mock journal/health/client\n')
   def wait(self,timeout=None):return self.returncode
   def poll(self):return self.returncode
   def terminate(self):self.returncode=-15
  def output(cmd,**kwargs):
   if cmd[:3]==['docker','ps','-aq']:
    match=cmd[cmd.index('--filter')+1]
    if match.startswith('name=^/slot-owner-'):return '\n'.join(n for n in containers if match=='name=^/'+n+'$')
    return ''
   raise AssertionError('Unexpected external call '+str(cmd))
  def fake_run(cmd,**kwargs):
   assert cmd[:2]==['docker','logs'];kwargs['stdout'].write('CPU mock complete UR trace\n')
   return type('Result',(),{'returncode':0})()
  def fake_collect(raw,log,recipe,label):return {'passed':not failing,'CPU_SYNTHETIC':True,'case':label}
  argv=['run_stage_slot_owner_oracle.py','--oracle-receipt',str(receipt),'--plan',str(PLAN),'--output',str(out),'--leased']
  with patch.object(q,'ROOT',root),patch.object(q.lifecycle,'leased',lambda cards:None),patch.object(q,'validate_build',lambda *a: ({'engine_receipt':'CPU_FAKE','engine_receipt_sha256':'CPU_FAKE'},plan)),patch.object(q,'collect',fake_collect),patch.object(q.lifecycle,'inspected',lambda name:containers[name]),patch.object(q.subprocess,'Popen',FakeProcess),patch.object(q.subprocess,'check_output',output),patch.object(q.subprocess,'run',fake_run),patch.object(q.os,'stat',lambda *a,**k:type('S',(),{'st_gid':1})()),patch.object(q.sys,'argv',argv):
   code=q.main()
  r=q.read(out/'receipt.json');assert code==(1 if failing else 0) and r['passed']==(not failing) and not containers
  assert r['post_health_passed'] and r['owned_containers_terminal']
  assert len(r['cases'])==(1 if failing else 3)
  assert all(fds==(8,9) for cmd,fds in calls)
  assert (out/'owner_card0-logical-free.json').exists()
 print('PASS CPU actual main mocked3cases success+tracefailure, fd8/9, exactroster/mask andfailuregates, ownedterminalcleanup and posthealthretention; no GPU/process execution')
