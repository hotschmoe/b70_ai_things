"""Tiny synthetic proc/cgroup/ownership controls; no actual runtime probes."""
import copy,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import observe_cpu_swap_attribution_v2 as obs
class Controls(unittest.TestCase):
 def proc(self,root,pid=123,start=900,swap=4):
  p=root/str(pid);p.mkdir(exist_ok=True);(p/'stat').write_text(str(pid)+' (name with spaces) '+' '.join(['S']+['0']*18+[str(start)]+['0']*5));(p/'status').write_text('VmSwap: '+str(swap)+' kB\n');(p/'comm').write_text('CPU-only-test\n');return p
 def globalroot(self,r):
  (r/'meminfo').write_text('MemAvailable: 100 kB\nSwapFree: 80 kB\nSwapTotal: 1000 kB\n');(r/'vmstat').write_text('pswpin 7\npswpout 9\n')
 def test_host_and_perPID_delta_with_PID_reuse(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.globalroot(r);p=self.proc(r);a,previous=obs.global_sample(r);self.assertEqual(a['swap_processes'][0]['VmSwap'],4096)
   self.proc(r,swap=6);(r/'vmstat').write_text('pswpin 8\npswpout 12\n');b,previous=obs.global_sample(r,previous);self.assertEqual(b['host_delta']['pswpout'],3);self.assertEqual(b['swap_processes'][0]['VmSwap_delta'],2048)
   self.proc(r,start=901,swap=1);c,_=obs.global_sample(r,previous);self.assertIsNone(c['swap_processes'][0]['VmSwap_delta'])
 def test_missing_kernel_fields_are_not_zero_attribution(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.globalroot(r);p=self.proc(r);(p/'status').write_text('Name: kernel\n');v,_=obs.global_sample(r);self.assertEqual(v['process_coverage_errors'][0]['error'],'missing_or_malformed_VmSwap');self.assertFalse(v['swap_processes'])
 def test_swap_process_row_bound_fails_without_truncation(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.globalroot(r);self.proc(r)
   with patch.object(obs,'MAX_SWAP_ROWS',0):self.assertRaises(ValueError,obs.global_sample,r)
 def test_no_environment_or_cmdline_needed(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.proc(r);self.assertEqual(obs.process_metadata(r,123)['comm'],'CPU-only-test')
 def fixture(self,r):
  plan={'image':'image','runner_sha256':'label','memory_cap_bytes':116<<30};command=['docker','run','--name','CPU','--network','host','--memory',str(116<<30),'--memory-swap',str(116<<30),'--cpus','8','--user','1000:1000','-w','/results/empty','image','-i','/usr/bin/python3','/test']
  obj={'Name':'/CPU','Image':'image','Config':{'Image':'image','Labels':{'b70.api-overlap.cpu-screen':'label'},'Cmd':command[command.index('image')+1:],'Entrypoint':['/usr/bin/env'],'User':'1000:1000','WorkingDir':'/results/empty'},'HostConfig':{'NetworkMode':'host','Memory':116<<30,'MemorySwap':116<<30,'NanoCpus':8*10**9,'PidsLimit':256,'Devices':[],'DeviceRequests':None,'Privileged':False,'GroupAdd':None},'Mounts':[],'State':{'Running':True,'Pid':123}}
  p=self.proc(r);(p/'smaps_rollup').write_text('\n'.join(k+': 1 kB' for k in ('Rss','Pss','Private_Clean','Private_Dirty','Shared_Clean','Shared_Dirty','Anonymous','Swap','SwapPss')));(p/'cgroup').write_text('0::/owned\n');cg=r/'cg/owned';cg.mkdir(parents=True)
  for name,value in {'memory.current':'1','memory.peak':'2','memory.max':str(116<<30),'memory.swap.current':'0','memory.swap.max':'0','memory.events':'oom 0','memory.stat':'anon 1\nfile 2'}.items():(cg/name).write_text(value)
  return plan,command,obj,r/'cg'
 def test_exact_owned_smaps_and_memory_stat_scopes(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);plan,cmd,obj,cg=self.fixture(r);v=obs.owned_sample(obj,cmd,plan,r,cg);self.assertEqual(v['smaps_rollup_bytes']['Anonymous'],1024);self.assertIn('file 2',v['cgroup']['memory.stat']);self.assertFalse(v['devices_granted'])
 def test_foreign_owner_image_device_and_guard_change_refused(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);plan,cmd,obj,cg=self.fixture(r)
   for mutate in ('owner','image','device','swap'):
    bad=copy.deepcopy(obj)
    if mutate=='owner':bad['Config']['Labels']['b70.api-overlap.cpu-screen']='foreign'
    elif mutate=='image':bad['Image']='foreign'
    elif mutate=='device':bad['HostConfig']['Devices']=[{'PathOnHost':'/dev/dri'}]
    else:bad['HostConfig']['MemorySwap']+=1
    with self.subTest(mutate=mutate):self.assertRaises(ValueError,obs.owned_sample,bad,cmd,plan,r,cg)
 def test_immutable_output_duplicate_and_nonfinite_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'row';obs.write_new(p,{'x':1});self.assertRaises(FileExistsError,obs.write_new,p,{'x':2});self.assertRaises(ValueError,obs.write_new,Path(d)/'bad',{'x':float('nan')})
 def test_runtime_idle_receipt_is_rejoined_not_digest_only(self):
  source=Path(obs.__file__).read_text();self.assertIn("sha(idle_path)==report['idle_receipt_sha256']",source);self.assertIn('finalized_binding(idle_path.parent)',source);self.assertIn("idle['finished_epoch']<=report['started_epoch']<=idle['finished_epoch']+300",source);self.assertIn("Observer final current source changed",source)
 def test_idle_window_rejects_live_owned_screen(self):
  from types import SimpleNamespace
  with patch.object(obs.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='id\n')),patch.object(obs,'inspect',return_value={'State':{'Running':True},'Name':'/CPU'}):self.assertRaises(ValueError,obs.idle_owned_absence,{'runner_sha256':'label'})
 def test_readonly_sample_mutation_and_idle_chronology(self):
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);h=base/'h';h.mkdir();root=base/'run';root.mkdir();screen=h/'screen';screen.write_text('{}');source=h/'cpu-swap-attribution-source-plan-v2.json';source.write_text('{\"files\":{}}');(root/'observer-source-plan.snapshot.json').write_bytes(source.read_bytes());(root/'source-plan.snapshot.json').write_bytes(screen.read_bytes());binding={'CPU_fixture':'true'}
   rows=[]
   for index in range(2):
    row={'index':index,'started_epoch':100+index*30,'finished_epoch':101+index*30,'host':dict(MemAvailable=10,SwapFree=20,SwapTotal=30,pswpin=0,pswpout=0),'host_delta':{k:None if index==0 else 0 for k in ('pswpin','pswpout','SwapFree','MemAvailable')},'swap_processes':[],'pid_census_count':0,'owned':[],'idle_owned_absence':{'owned_screen_running':[],'census_epoch':100+index*30}};p=root/('sample-'+str(index).zfill(5)+'.json');obs.write_new(p,row);rows.append(p)
   report={'schema':1,'passed':True,'mode':'idle','seconds':30,'duration_completed':True,'started_epoch':99,'finished_epoch':132,'source_binding':binding,'observer_source_plan_sha256':obs.sha(source),'memory_guards_changed':False,'inference_settings_changed':False,'actual_GPU_touch':False,'model_payload_read':False,'causal_attribution_qualified':False,'sample_sha256':{p.name:obs.sha(p) for p in rows},'sample_count':2,'owned_sample_count':0};obs.write_new(root/'report.json',report)
   with patch.object(obs,'HERE',h),patch.object(obs,'ROOT',base),patch.object(obs,'SCREEN_SHA',obs.sha(screen)),patch.object(obs,'source_binding',return_value=({},binding)):
    self.assertEqual(obs.finalized_binding(root)['sample_count'],2);rows[0].write_text('{}');self.assertRaises(ValueError,obs.finalized_binding,root)
 def test_frozen_screen_source_binding_only(self):
  plan,binding=obs.source_binding();self.assertEqual(binding['screen_plan_sha256'],obs.SCREEN_SHA);self.assertEqual(plan['sampling_seed'],1234)
if __name__=='__main__':unittest.main()
