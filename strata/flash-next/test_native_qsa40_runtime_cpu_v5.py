"""Producer-shaped CPU controls; all model/Docker/GPU boundaries mocked."""
import copy,json,sys,tempfile,unittest,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import native_qsa40_runtime_v5 as d
import native_qsa40_reader_v5 as raw
import native_qsa40_evidence_v5 as evidence
from native_qsa40_protocol_v5 import Protocol
import qualify_native_qsa40_v5 as parent
class Controls(unittest.TestCase):
 def config(self,root):
  (root/'prepared.json').write_text('{}');(root/'server-config.json').write_text(json.dumps({'args':['--no-prefill-borrow'],'env':{}}));return {'prepared':{'cards':[0],'pack':str(root/'pack'),'engine_receipt':str(root/'sdk/receipt.json'),'runtime':{'image':'sha256:CPU_FAKE'}}}
 def test_real_shaped_prepare_and_complete_current_manifest(self):
  with tempfile.TemporaryDirectory()as tmp:
   base=Path(tmp);prepared=base/'prepared';prepared.mkdir();binding=self.config(prepared)
   with patch.object(d.spec,'candidate_binding',return_value=binding),patch.object(d,'own_binding',return_value={'independent_original':True}),patch.object(d,'closure',return_value='a'*64),patch.object(d,'kernel_binding',return_value={'actual_source40_flags':'CPU_MOCK'}):
    plan=d.prepare(SimpleNamespace(prepared=prepared,own_producer=base/'own',output=base/'new'));self.assertEqual(plan['cards'],[0]);d.manifest(plan)
    for key,value in [('image','FOREIGN'),('args',plan['args']+['--wrong-math','1']),('base_env',dict(plan['base_env'],STRATA_UNKNOWN='1')),('registry_sha256','0'*64),('source_plan_sha256','0'*64)]:
     changed=copy.deepcopy(plan);changed[key]=value;self.assertRaises(ValueError,d.manifest,changed)
 def test_wrong_candidate_refused_before_own_or_output(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);out=root/'never'
   with patch.object(d.spec,'candidate_binding',side_effect=ValueError('oldC137/source39')),patch.object(d,'own_binding')as own:
    self.assertRaises(ValueError,d.prepare,SimpleNamespace(prepared=root,own_producer=root,output=out));own.assert_not_called();self.assertFalse(out.exists())
 def test_actual_merged_fake_producer_pinabsence_diagnostics_and_EOF(self):
  script="import sys\nprint('READY CPU',flush=True)\nfor line in sys.stdin:\n if line.startswith('QUIT'):break\n print('SFD request CPU',flush=True)\n print('PCL '+chr(123)+'\\\"event\\\":\\\"finish\\\"'+chr(125),flush=True)\n print('T 7',flush=True)\n print('DONE 1 4',flush=True)\n"
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);p=Protocol([sys.executable,'-u','-c',script],out,lambda:None,ready_timeout=5);request=p.request('row',[1,2,3,4],fresh=1,max_new=1,pin=None);self.assertIsNone(request['pin']);self.assertNotIn('pin=',request['command']);self.assertIn('SFD request CPU',request['stderr']);self.assertEqual(p.close(),0);self.assertTrue(p.capture['passed']);self.assertTrue(p.capture['eof']);self.assertTrue((out/'engine.combined.log').exists())
 def test_truncated_semantic_fake_producer_not_qualified(self):
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);p=Protocol([sys.executable,'-u','-c',"import sys;print('READY CPU',flush=True);sys.stdin.readline();sys.stdout.write('T 7');sys.stdout.flush()"],out,lambda:None,ready_timeout=5);self.assertRaises(ValueError,p.request,'row',[1,2,3,4],fresh=1,max_new=1);p.close(failed=True);self.assertFalse(p.capture['passed']);self.assertIsNotNone(p.capture['error'])
 def test_consumed_payload_extra_symlink_and_changed_bytes(self):
  with tempfile.TemporaryDirectory()as tmp:
   f=Path(tmp)/'a';f.write_bytes(b'1234');self.assertEqual(raw.consume(f,4),b'1234');self.assertRaises(ValueError,raw.consume,f,3);alias=Path(tmp)/'alias';alias.symlink_to(f);self.assertRaises(ValueError,raw.consume,alias,4)
 def test_finite_chronology_refuses_bool_and_nan(self):
  for value in(True,float('nan'),float('inf'),-1):self.assertRaises(ValueError,evidence.finite,[value])
 def test_parent_Popen_failure_recorded_without_model_or_health(self):
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);planfile=out/'plan.json';planfile.write_text('{}')
   with patch.object(parent.subprocess,'Popen',side_effect=OSError('synthetic_launch_refusal')),patch.object(parent,'health')as health:
    self.assertRaises(OSError,parent.run_child,out,planfile,{},False,1,3);health.assert_not_called();self.assertFalse(parent.read(out/'off-child-launch-failure.json')['actual_child_started'])
 def test_cleanup_already_absent_never_inspects_or_removes(self):
  with patch.object(d.c,'absent',return_value=True),patch.object(d.c,'inspected')as inspect,patch.object(parent.subprocess,'run')as run:
   self.assertFalse(parent.cleanup(Path('/tmp/noexec'),{},123,False));inspect.assert_not_called();run.assert_not_called()
 def test_original_Docker_recipe_mutation_is_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);out=root/'off';out.mkdir();(root/'plan.snapshot.json').write_text('{}');plan={'engine_root':str(root/'sdk'),'pack':str(root/'pack'),'image':'sha256:CPU_FAKE','args':['--no-prefill-borrow'],'base_env':{}}
   argv,env,name=d.command(plan,out,123,False,1000);mounts=[]
   for i,v in enumerate(argv):
    if v=='-v':a,b,c=argv[i+1].rsplit(':',2);mounts.append({'Source':a,'Destination':b,'RW':c=='rw','Type':'bind'})
   obj={'Name':'/'+name,'Image':plan['image'],'Config':{'Image':plan['image'],'Labels':{'b70.qsa40.plan':d.sha(root/'plan.snapshot.json')},'User':'1000:1000','Entrypoint':['/bin/bash'],'Cmd':argv[argv.index(plan['image'])+1:],'OpenStdin':True,'Tty':False,'Env':[k+'='+v for k,v in env.items()]},'HostConfig':{'NetworkMode':'none','Privileged':False,'Memory':105*1024**3,'MemorySwap':105*1024**3,'NanoCpus':0,'DeviceRequests':[],'Devices':[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}],'GroupAdd':['1000']},'Mounts':mounts}
   d.inspection(obj,argv,plan)
   for group,key,value in [('Config','Image','foreign'),('Config','Cmd',['foreign']),('HostConfig','Memory',1),('HostConfig','NetworkMode','host'),('HostConfig','Devices',[])]:
    changed=copy.deepcopy(obj);changed[group][key]=value;self.assertRaises(ValueError,d.inspection,changed,argv,plan)
 def test_normal_parent_RC0_retained_writer_forces_owned_cleanup_failure(self):
  import os,signal
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);pidfile=root/'writer.pid';script="import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);Path(sys.argv[1]).write_text(str(p.pid));print('READY CPU',flush=True);sys.stdin.readline()"
   cleanup=[]
   def retire():cleanup.append(True);os.kill(int(pidfile.read_text()),signal.SIGTERM)
   p=Protocol([sys.executable,'-u','-c',script,str(pidfile)],root,retire,ready_timeout=5,drain_timeout=.05);self.assertEqual(p.close(),0);self.assertEqual(cleanup,[True]);self.assertTrue(p.capture['eof']);self.assertTrue(p.capture['reader_retired']);self.assertFalse(p.capture['passed']);self.assertTrue(p.capture['forced_cleanup'])
 def test_cleanup_exception_does_not_skip_actual_child_reader_retirement(self):
  import io,subprocess
  class Child:
   pid=123;stdout=io.StringIO('actual own child output\n')
   def __init__(self):self.calls=0;self.terminated=False
   def wait(self,timeout=None):
    self.calls+=1
    if self.calls==1:raise subprocess.TimeoutExpired('CPU_mock_child',timeout)
    return 0
   def terminate(self):self.terminated=True
   def poll(self):return None
   def kill(self):pass
  with tempfile.TemporaryDirectory()as tmp:
   out=Path(tmp);directory=out/'off';planfile=out/'plan.json';planfile.write_text('{}');ready={'pid':{'pid':123},'parent':{'pid':os.getpid()},'plan_sha256':parent.sha(planfile),'source_plan_sha256':'a'*64};child=Child()
   def launch(*args,**kwargs):parent.write(directory/'READY.json',ready);return child
   with patch.object(parent.subprocess,'Popen',side_effect=launch),patch.object(parent,'pid_identity',side_effect=lambda pid:{'pid':pid}),patch.object(d,'closure',return_value='a'*64),patch.object(parent,'health',return_value={'finished_epoch':1}),patch.object(parent,'journal',return_value={'sha256':'CPU_FAKE'}),patch.object(parent,'cleanup',side_effect=ValueError('foreign image refused')):
    self.assertRaises(ValueError,parent.run_child,out,planfile,{},False,1,3)
   row=parent.read(out/'off-child.receipt.json');self.assertTrue(row['stdout_capture']['reader_retired']);self.assertTrue(row['stdout_capture']['eof']);self.assertTrue(row['forced_cleanup']);self.assertIn('foreign image refused',row['error'])
if __name__=='__main__':unittest.main()
