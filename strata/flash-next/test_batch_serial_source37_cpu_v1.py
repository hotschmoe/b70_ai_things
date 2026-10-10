"""Tiny source37 cacheOFF lifecycle and real merged-pipe controls; no GPU/model."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import extract_serial_cacheoff_numerical_v1 as extractor
import run_batch_serial_source37_v1 as adapter
import batch_serial_source37_v1 as ctrl

class Serial37Tests(unittest.TestCase):
 def setUp(self):
  self.plan={'args':['--batch','2','--prompt-cache','0','--conversation-cache-mib','0','--suffix-draft','0','--lookup-chain','0'],'env':{}}
  self.args,self.env=adapter.config(self.plan);ids=[1,2];digest=extractor.old.le32_digest(ids)
  begin={'event':'begin','pid':5,'request':3,'tokens':2,'input_sha256_le32':digest}
  commit={'event':'committed_live','pid':5,'request':3,'phase':'complete','finish':'length','published':False,'chain_updated':True,'live_reusable':False,'ids_truncated':False,'tokens':2,'ids':ids,'sha256_le32':digest}
  spans=[{'event':'stage_span','pid':5,'request':3,'phase':'prefill','device':d,'lb':lo,'le':hi,'lo':0,'hi':2,'complete':done} for d,(lo,hi) in enumerate([(0,32),(32,48)]) for done in (False,True)]
  self.raw={'ids':ids,'fresh':1,'pin':None,'command':extractor.protocol.request_command(ids,1,1,None),'cancel_requested':None,'stop_sent':False,'output_ids':[7],'done':'DONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2 0','stderr':['SFD request pid=5 request=3 tokens=2']+['PCL '+json.dumps(e) for e in [begin,*spans,commit]]}
  self.numeric={'ledger':{'actual_reused':0,'candidate_resume':0,'evaluated_prompt_rows':2,'evaluated_decode_rows':0,'generated':1,'cancelled':False,'finish':'length'},'selection':{**dict.fromkeys(('actual_reused','candidate_resume','parked_bytes','parked_entries','evictions','checkpoints'),0),'reread':False}}
 def parse(self,raw=None):
  with patch.object(extractor.old,'extract_numeric',return_value=copy.deepcopy(self.numeric)):
   return extractor.extract(self.raw if raw is None else raw,Path('/tmp'),self.args,self.env,[(0,32),(32,48)])
 def test_source_tuple_true_update_false_reuse(self):
  m=self.parse();self.assertFalse(m['numerical_cacheOFF_lifecycle']['cache_qualification_granted']);self.assertTrue(m['numerical_cacheOFF_lifecycle']['commit']['chain_updated'])
 def test_false_update_and_live_reuse_reject(self):
  for change in ({'chain_updated':False},{'live_reusable':True},{'published':True},{'phase':'decode'}):
   r=copy.deepcopy(self.raw);e=json.loads(r['stderr'][-1][4:]);e.update(change);r['stderr'][-1]='PCL '+json.dumps(e)
   with self.assertRaises(ValueError):self.parse(r)
 def test_missing_second_stage_return_rejects(self):
  r=copy.deepcopy(self.raw);r['stderr'].pop(-2)
  with self.assertRaises(ValueError):self.parse(r)
 def test_legacy_pin_zero_is_not_future_producer_admitted(self):
  r=copy.deepcopy(self.raw);r.update(pin=0,command=extractor.protocol.request_command(r['ids'],1,1,0))
  with self.assertRaises(ValueError):self.parse(r)
 def test_concrete_driver_error_propagates_without_native_action(self):
  n,p=adapter.shared.numerical,adapter.shared.MergedProtocol
  with patch.object(adapter.shared,'run',side_effect=ValueError('CPU mock failure')):
   with self.assertRaises(ValueError):adapter.run(self.plan,Path('/tmp'),Path('/tmp'),Path('/tmp'),0)
  self.assertIs(n,adapter.shared.numerical);self.assertIs(p,adapter.shared.MergedProtocol)
 def test_new_controller_rejects_foreign_and_oldplan_before_source(self):
  for p in ({'kind':'serial','driver_sha256':'x'},{'serial_source37_generation':1,'kind':'native','driver_sha256':'x'}):
   with self.assertRaises(ValueError):ctrl.manifest_binding(p)
 def test_real_single_pipe_absent_pin_and_diagnostics(self):
  with tempfile.TemporaryDirectory(prefix='serial37-CPU-') as d:
   root=Path(d);s=root/'mock.py';s.write_text("import sys\nprint('READY 2048 stop',flush=True)\nfor line in sys.stdin:\n if line.startswith('QUIT'):break\n print('SFD request pid=1 request=1 tokens=2',file=sys.stderr,flush=True)\n print('T 7',flush=True)\n print('DONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2 0',flush=True)\n")
   p=adapter.CacheOffMergedProtocol([sys.executable,str(s)],root)
   try:
    r=p.request('tiny',[1,2],fresh=1,max_new=1);self.assertIsNone(r['pin']);self.assertNotIn('pin=',r['command']);self.assertEqual(r['output_ids'],[7]);self.assertTrue(any(x.startswith('SFD ') for x in r['stderr']))
   finally:
    self.assertEqual(p.close(),0);p.p.stdin.close();p.p.stdout.close()
class RecoveryTests(unittest.TestCase):
 def test_known_failure_actual_initial_shape_and_othererror_refusal(self):
  import adjudicate_serial37_cacheoff_first_v1 as a
  p={'passed':False,'parent_generation':40,'child_return_code':1,'errors':['BATCH_NUMERICAL child failed 1'],'interrupted':False,'forced_cleanup':False,'owned_containers_terminal':True}
  t={'passed':False,'error':'ValueError: Noncancelled live chain was not reusable','engine_rc':0,'removed':True,'state':{'ExitCode':0,'Running':False,'OOMKilled':False},'failure_traceback':'run_batch_serial_controls_v40.py serial_prefix_qualification_v6.py ValueError: Noncancelled live chain was not reusable'}
  a.known_failure(p,t)
  for key,value in [('engine_rc',1),('removed',False),('error','ValueError: other failure')]:
   bad=copy.deepcopy(t);bad[key]=value
   with self.assertRaises(ValueError):a.known_failure(p,bad)
  p['passed']=True
  with self.assertRaises(ValueError):a.known_failure(p,t)
 def test_complete_selected_roster_and_wrongbudget(self):
  from serial37_selection_v1 import selected_jobs
  p={'group_index':0,'actual_serial_selected_indices':[0,1,2,3],'first_job_adjudication':None,'actual_serial_job_count':4,'actual_serial_submission_budgets':[1]*4}
  self.assertEqual(selected_jobs(p,[0,1,2,3]),[0,1,2,3]);p['actual_serial_submission_budgets']=[64]*4
  with self.assertRaises(ValueError):selected_jobs(p,[0,1,2,3])
 def test_partial_selection_requires_named_recovery(self):
  from serial37_selection_v1 import selected_jobs
  p={'group_index':0,'actual_serial_selected_indices':[1,2,3],'first_job_adjudication':None,'actual_serial_job_count':3,'actual_serial_submission_budgets':[1]*3}
  with self.assertRaises((ValueError,TypeError)):selected_jobs(p,[0,1,2,3])
 def test_new_concrete_source_uses_strict_cacheoff_and_absentpin(self):
  s=Path(adapter.shared.__file__).read_text();self.assertIn('meta=extract(raw',s);self.assertNotIn('numerical.base.extract(',s);self.assertIn('CacheOffMergedProtocol(command,directory)',s);self.assertIn('for job in selected_jobs(plan,jobs)',s)
 def test_future_parent_real_generation_and_origin_preserved(self):
  s=Path(ctrl.__file__).with_name('qualify_batch_serial_source37_v1.py').read_text();self.assertIn("'parent_generation':41",s);self.assertIn('kernel_journal_receipts',s);self.assertIn('child_interpreter',s)
if __name__=='__main__':unittest.main()
