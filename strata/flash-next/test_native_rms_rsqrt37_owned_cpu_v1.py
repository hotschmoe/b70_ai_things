"""Owned producer-shaped metadata/source negatives; no GPU/compiler/model calls."""
import copy,json,unittest,ast,tempfile
from pathlib import Path
from unittest.mock import patch
import qualify_native_rms_rsqrt37_v1 as q
class Tests(unittest.TestCase):
 def command(self):return ['docker','run','--name','owned','-v','/tmp/build:/out:rw','-e','ZE_AFFINITY_MASK=0','-e','ONEAPI_DEVICE_SELECTOR=level_zero:gpu','image','-c','exec leaf']
 def obj(self):return {'Name':'/owned','Image':'image','Config':{'Image':'image','Labels':{'b70.rms37.plan':q.sha(q.PLAN)},'User':'1000:1000','Entrypoint':['/bin/bash'],'Cmd':['-c','exec leaf'],'Env':['ZE_AFFINITY_MASK=0','ONEAPI_DEVICE_SELECTOR=level_zero:gpu']},'HostConfig':{'NetworkMode':'none','Privileged':False,'Memory':2<<30,'MemorySwap':2<<30,'NanoCpus':2000000000,'PidsLimit':256,'DeviceRequests':None,'Devices':[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}]},'Mounts':[{'Source':'/tmp/build','Destination':'/out','RW':True,'Type':'bind'}]}
 def test_actual_shaped_ownership(self):q.observed_container(self.obj(),'owned','image',self.command())
 def test_image_device_mount_user_and_eager_negatives(self):
  for key in ('image','device','mount','user','eager'):
   r=self.obj()
   if key=='image':r['Image']='other'
   elif key=='device':r['HostConfig']['Devices']=[]
   elif key=='mount':r['Mounts'].append({'Source':'/model','Destination':'/model','RW':False,'Type':'bind'})
   elif key=='user':r['Config']['User']='root'
   else:r['Config']['Env'].append('STRATA_VERIFY_EAGER=0')
   with self.assertRaises(ValueError):q.observed_container(r,'owned','image',self.command())
 def report(self):return {'run_command':{'finished_epoch':10.},'GPU_terminal_epoch':11.,'post_health':{'finished_epoch':12.},'post_journal':{'started_command_epoch':13.,'finished_epoch':14.},'post_full4':{'started':16.,'finished':20.},'before_post_full4_pages':{'epoch':15.},'post_pages':{'epoch':21.},'finished_epoch':22.}
 def test_complete_actual_chronology(self):q.chronology(self.report())
 def test_original_scan_cannot_precede_journal_or_terminal(self):
  for key,val in [('started',13.),('finished',15.)]:
   r=self.report();r['post_full4'][key]=val
   with self.assertRaises(ValueError):q.chronology(r)
 def test_nonfinite_or_boolean_epoch(self):
  for value in (True,float('nan'),float('inf')):
   r=self.report();r['GPU_terminal_epoch']=value
   with self.assertRaises(ValueError):q.chronology(r)
 def test_maps_after_actual_leaf_and_no_math_edit(self):
  s=Path(q.__file__).with_name('native_rms_rsqrt37_owned_entry_v1.cpp').read_text();self.assertLess(s.index('rms37_frozen_proposal_main(argc,argv)'),s.index('postexecution mapped-library'));self.assertIn('.loaded-maps-before',s);self.assertIn('.loaded-maps',s)
 def test_timeout_leases_retained_until_process_and_EOF(self):
  s=Path(q.__file__).read_text();self.assertIn('while proc.poll() is None:',s);self.assertIn('while reader.is_alive():reader.join(1)',s);self.assertIn('Posthealth/source cannot outrun actual command/EOF retirement',s);ast.parse(s)
 def test_intended_runtime_and_production_wrapper_depth(self):
  self.assertEqual(Path(q.CONTAINER_RUNNER).parents[2],Path('/harness'));self.assertIn('c388186',q.RUNTIME_IMAGE);s=Path(q.__file__).read_text();self.assertIn('/opt/b70-c1-python/bin/python /harness/strata/flash-next/qualifier.py',s)
 def test_proposal_kept_frozen(self):self.assertEqual(q.sha(Path(q.__file__).with_name('native-rms-rsqrt37-source-plan-v1.json')),q.PROPOSAL_SHA)
 def test_noEOF_failed_report_rejected_before_numerics(self):
  with patch.object(q,'read',return_value={'passed':False}) as read:
   with self.assertRaises(ValueError):q.finalized_binding('/CPU_SYNTHETIC')
 def test_actual_shaped_EOF_receipt_mutations(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);log=root/'stdout.log';cmd=root/'command.json';log.write_text('CPU_PRODUCER_TERMINAL\n');q.write(cmd,['CPU_ONLY']);row=dict(passed=True,reader_retired=True,eof=True,error=None,command_error=None,return_code=0,path=str(log),sha256=q.sha(log),command=['CPU_ONLY'],command_sha256=q.sha(cmd));q.command_binding(row,log,cmd)
   for key,value in [('eof',False),('reader_retired',False),('error','BrokenPipeError'),('command_error','forced cleanup'),('return_code',1),('sha256','changed')]:
    bad=dict(row);bad[key]=value
    with self.assertRaises(ValueError):q.command_binding(bad,log,cmd)
 def test_forced_or_nonzero_terminal_rejected(self):
  good={'Running':False,'ExitCode':0,'OOMKilled':False,'Error':''};q.terminal_binding(good)
  for key,value in [('Running',True),('ExitCode',137),('OOMKilled',True),('Error','forced')]:
   bad=dict(good);bad[key]=value
   with self.assertRaises(ValueError):q.terminal_binding(bad)
 def test_source(self):q.source_binding()
if __name__=='__main__':unittest.main()
