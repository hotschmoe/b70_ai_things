"""Actual tiny owned-pipe timeout plus synthetic exact health/ownership negatives."""
import copy,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import native_rms_phase_supervisor_v3 as supervisor
import native_rms_health_journal_binding_v3 as h
import qualify_native_rms_rsqrt37_v3 as q
class Tests(unittest.TestCase):
 def test_actual_owned_writer_cleanup_unblocks_EOF_and_timeout_notPASS(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);script=root/'client.py';pid=root/'writer.pid';script.write_text("import os,signal,subprocess,sys\nsignal.signal(signal.SIGTERM,lambda *a:None)\np=subprocess.Popen([sys.executable,'-c','import time; print(\\\"OWNED_WRITER\\\",flush=True);time.sleep(60)'])\nopen(sys.argv[1],'w').write(str(p.pid))\np.wait()\nprint('OWNED_DRAIN',flush=True)\n")
   cleaned=[]
   def owned_cleanup():
    end=time.monotonic()+5
    while not pid.exists() and time.monotonic()<end:time.sleep(.01)
    os.kill(int(pid.read_text()),signal.SIGTERM);cleaned.append(True)
   started=time.monotonic();row=supervisor.supervise([sys.executable,str(script),str(pid)],root/'stdout.log',.5,owned_cleanup);self.assertLess(time.monotonic()-started,10);self.assertEqual(cleaned,[True]);self.assertTrue(row['reader_retired']);self.assertTrue(row['eof']);self.assertFalse(row['passed']);self.assertIn('TimeoutExpired',row['command_error']);self.assertIn('OWNED_DRAIN',(root/'stdout.log').read_text())
 def test_normal_parent_exit_retained_writer_cleanup_before_EOF(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);script=root/'normal.py';pid=root/'writer.pid';script.write_text("import subprocess,sys\np=subprocess.Popen([sys.executable,'-c','import time;print(\\\"LATE_WRITER\\\",flush=True);time.sleep(60)'])\nopen(sys.argv[1],'w').write(str(p.pid))\nprint('NORMAL_PARENT_EXIT',flush=True)\n")
   cleaned=[]
   def cleanup():os.kill(int(pid.read_text()),signal.SIGTERM);cleaned.append(True)
   start=time.monotonic();row=supervisor.supervise([sys.executable,str(script),str(pid)],root/'stdout.log',5,cleanup,drain_timeout=.1);self.assertLess(time.monotonic()-start,5);self.assertEqual(cleaned,[True]);self.assertEqual(row['return_code'],0);self.assertTrue(row['reader_retired']);self.assertTrue(row['eof']);self.assertFalse(row['passed']);self.assertIn('stdout drain deadline',row['command_error'])
 def command(self):return ['docker','run','--name','owned','-v','/tmp/build:/out:rw','-e','ZE_AFFINITY_MASK=0','-e','ONEAPI_DEVICE_SELECTOR=level_zero:gpu','image','-c','exec leaf']
 def obj(self):return {'Name':'/owned','Image':'image','Config':{'Image':'image','Labels':{'b70.rms37.plan':q.sha(q.PLAN)},'User':'1000:1000','Entrypoint':['/bin/bash'],'Cmd':['-c','exec leaf'],'Env':['ZE_AFFINITY_MASK=0','ONEAPI_DEVICE_SELECTOR=level_zero:gpu']},'HostConfig':{'NetworkMode':'none','Privileged':False,'Memory':2<<30,'MemorySwap':2<<30,'NanoCpus':2000000000,'PidsLimit':256,'DeviceRequests':None,'Devices':[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}]},'Mounts':[{'Source':'/tmp/build','Destination':'/out','RW':True,'Type':'bind'}]}
 def test_foreign_image_or_recipe_cannot_authorize_cleanup(self):
  for kind in ('image','command','mount'):
   obj=self.obj()
   if kind=='image':obj['Image']='foreign'
   elif kind=='command':obj['Config']['Cmd']=['-c','foreign']
   else:obj['Mounts'][0]['RW']=False
   with self.assertRaises(ValueError):q.observed_container(obj,'owned','image',self.command())
 def report(self):
  row=lambda cmd,start,end:dict(command=cmd,started_command_epoch=start,finished_epoch=end,return_code=0,error=None,command_error=None,eof=True,reader_retired=True,passed=True)
  cmds=[['/CPU/vllm/int4/diagnostics/xpu_health_strict.sh','--img','CPU_IMAGE'],['/CPU/bin/xpu-collective-health','--img','CPU_IMAGE','--p2p','0','--timeout','180']];r={'started_epoch':1.,'GPU_terminal_epoch':20.,'compile_command':{'started_command_epoch':10.}}
  for stage,offset in [('pre',2.),('post',21.)]:r[stage+'_health']={'passed':True,'image':'CPU_IMAGE','rows':[row(cmds[0],offset,offset+1),row(cmds[1],offset+2,offset+3)],'finished_epoch':offset+4};r[stage+'_journal']=row(['journalctl','-k','--since','@1','--no-pager'],offset+5,offset+6)
  return r
 def test_health_journal_exact_positive(self):h.admit(self.report(),Path('/CPU'),'CPU_IMAGE')
 def test_wrongP2P_image_since_and_aggregate_negatives(self):
  for kind in ('P2P','image','since','aggregate','postbeforeGPU'):
   r=self.report()
   if kind=='P2P':r['post_health']['rows'][1]['command'][-3]='1'
   elif kind=='image':r['pre_health']['rows'][0]['command'][-1]='FOREIGN'
   elif kind=='since':r['post_journal']['command'][3]='@2'
   elif kind=='aggregate':r['post_health']['finished_epoch']=22.
   else:r['post_health']['rows'][0]['started_command_epoch']=19.
   with self.assertRaises(ValueError):h.admit(r,Path('/CPU'),'CPU_IMAGE')
 def test_frozen_V2_preserved(self):self.assertEqual(q.sha(Path(q.__file__).with_name('native-rms-rsqrt37-owned-source-plan-v2.json')),'c842ccc477a637700d5009b09ea21211fbc169b9c496defbe22966b30e052524')
if __name__=='__main__':unittest.main()
