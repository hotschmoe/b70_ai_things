#!/usr/bin/env python3
"""Synthetic parent lifecycle and tiny buffered identity controls. No GPU touches."""
import copy,hashlib,io,json,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
import qualify_serial_prefix as q

def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid control accepted')

with tempfile.TemporaryDirectory(prefix='prefix-parent-cpu-') as td:
 root=Path(td)
 for rel in ['vllm/int4/diagnostics/xpu_health_strict.sh','bin/xpu-collective-health','bin/xpu-collective-health.py']:
  path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU mock source only\n')
 lockpath=root/'strata/flash-next/model-lock.json';lockpath.parent.mkdir(parents=True)
 files=[];shards=[]
 for i in range(4):
  rel='UD-Q4_K_XL/part%d.gguf'%i;p=root/'models'/rel;p.parent.mkdir(parents=True,exist_ok=True);raw=bytes([i+1])*16384;p.write_bytes(raw);shards.append(p);files.append({'path':rel,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 lock={'revision':'CPU_SYNTHETIC','destination':'models','files':files};q.write(lockpath,lock)
 result=q.full_buffered_identity(lockpath,lock,shards,root/'full.json',time.time()-1)
 assert result['passed'] and len(result['rows'])==4 and result['started']>=result['after_child_terminal_epoch']
 bad=copy.deepcopy(lock);bad['files'][2]['sha256']='0'*64
 result=q.full_buffered_identity(lockpath,bad,shards,root/'bad.json',time.time()-1)
 assert not result['passed'] and len(result['rows'])==4 and not result['rows'][2]['passed'] and result['rows'][3]['passed']
 reject(lambda:q.full_buffered_identity(lockpath,lock,shards,root/'future.json',time.time()+10))
 parent={'child_return_code':0,'interrupted':False,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'errors':[]};child={'numerical_and_teardown_passed':True,'finished_epoch':1};post={'passed':True,'started':2}
 assert q.finalizable(parent,child,post)
 for key,value in [('child_return_code',1),('interrupted',True),('owned_containers_terminal',False),('forced_cleanup',True),('pre_health_passed',False),('post_health_passed',False),('kernel_fault_gate_passed',False),('errors',['preserved failure'])]:
  p=copy.deepcopy(parent);p[key]=value;assert not q.finalizable(p,child,post)
 assert not q.finalizable(parent,{'numerical_and_teardown_passed':False,'finished_epoch':1},post)
 assert not q.finalizable(parent,child,{'passed':False,'started':2})
 assert not q.finalizable(parent,child,{'passed':True,'started':0})
 # Execute the actual wrapper main with all external/device boundaries mocked.
 # Popen/check_output never run a real health, Docker or model process.
 for failing in [False,True]:
  out=root/('failure' if failing else 'success');source_identity=root/'prepared-identity.json';q.write(source_identity,{'CPU_SYNTHETIC':True})
  plan={'cards':[0],'args':[],'controller_sha256':q.V5_SHA,'model_identity':{'path':str(source_identity),'sha256':q.sha(source_identity)}};planpath=root/'plan.json';q.write(planpath,plan)
  calls=[]
  class FakeProcess:
   def __init__(self,cmd,**kwargs):
    self.pid=424242;self.returncode=0;self.stdout=io.StringIO('CPU mock child stdout\n');calls.append((cmd,kwargs.get('pass_fds')))
    if 'run' in cmd and str(Path(q.ctrl.__file__)) in cmd:
     childdir=Path(cmd[cmd.index('--output')+1]);childdir.mkdir();q.write(childdir/'plan.snapshot.json',plan);q.write(childdir/'report.json',{'numerical_and_teardown_passed':not failing,'finished_epoch':time.time(),'passed':False});self.returncode=1 if failing else 0
    elif 'finalize' in cmd:
     childdir=Path(cmd[cmd.index('--output')+1]);r=q.read(childdir/'report.json');r['passed']=True;q.write(childdir/'report.json',r)
    elif kwargs.get('stdout') is not None and hasattr(kwargs['stdout'],'write'):kwargs['stdout'].write('CPU synthetic healthy journal/command\n')
   def wait(self,timeout=None):return self.returncode
   def poll(self):return self.returncode
   def terminate(self):self.returncode=-15
   def send_signal(self,sig):self.returncode=-sig
   def kill(self):self.returncode=-9
  argv=['qualify_serial_prefix.py','--plan',str(planpath),'--output',str(out),'--leased']
  with patch.object(q,'ROOT',root),patch.object(q.ctrl.c1,'leased',lambda cards:None),patch.object(q.ctrl,'verify_model_identity',lambda *args: {'CPU_SYNTHETIC':True}),patch.object(q.subprocess,'Popen',FakeProcess),patch.object(q.subprocess,'check_output',lambda *args,**kwargs:''),patch.object(q.sys,'argv',argv):
   code=q.main()
  r=q.read(out/'parent-qualification.json');assert code==(1 if failing else 0) and r['passed']==(not failing)
  assert r['post_health_passed'] and r['owned_containers_terminal'] and q.read(out/'post-model-identity.json')['passed']
  assert q.read(out/'post-model-identity.json')['started']>=q.read(out/'child/report.json')['finished_epoch']
  assert any(fds==(8,9) for cmd,fds in calls if 'run' in cmd)
  assert any('finalize' in cmd for cmd,fds in calls)==(not failing)
  assert 'CPU mock child stdout' in (out/'child-supervisor.log').read_text()
 print('PASS CPU parent: buffered all4 hashes/failure preservation, terminal-time gate, finalization negatives, actual main mocked success+child failure, fd8/9 forwarding, stdout forwarding, post-health retained. No devices/processes launched.')
