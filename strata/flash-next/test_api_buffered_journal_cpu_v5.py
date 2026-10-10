"""Exact original journal/EOF/epoch evidence controls, no actual journal/GPU call."""
import copy,tempfile,time,unittest,ast
from pathlib import Path
import api_journal_exact_admission_v5 as q
import qualify_batch_api_cache_positive_buffered_v5 as parent_source
from batch_numerical_proofs_v7 import write,sha
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.parent={'started_epoch':100.25,'finished_epoch':200,'child_launch_started_epoch':125,'child_terminal_epoch':150,'pre_health_finished_epoch':110,'post_health_finished_epoch':170,'journal_admission_generation':5,'kernel_journal_rows':{},'child_stdout_reader':{'completed':True,'error':None,'eof_epoch':149},'post_model_identity':{'started':180,'finished':190}}
  (self.root/'child-supervisor.log').write_text('CPU_SYNTHETIC\n');self.parent['child_stdout_log_sha256']=sha(self.root/'child-supervisor.log');write(self.root/'post-model-identity.json',{'started':180,'finished':190});cmd=['journalctl','-k','--since','@100','--no-pager']
  for stage,epoch in [('pre',120),('post',175)]:
   write(self.root/(stage+'-health.json'),{'passed':True,'started_epoch':105 if stage=='pre' else 160,'finished_epoch':self.parent[stage+'_health_finished_epoch']})
   label=stage+'-kernel-journal';log=self.root/(label+'.log');log.write_text('CPU_NO_FAULT_CONTROL\n');command=self.root/(label+'.command.json');write(command,cmd);self.parent['kernel_journal_rows'][label]={'path':str(log),'command':cmd,'command_file_sha256':sha(command),'sha256':sha(log),'return_code':0,'error':None,'observed_epoch':epoch,'started_epoch':epoch-2,'finished_epoch':epoch-1,'supervisor_pid':777}
 def tearDown(self):self.tmp.cleanup()
 def test_exact_original_rows_and_epoch_positive(self):self.assertFalse(q.journal_gate(self.root,self.parent)['original_rows_or_errors_rewritten'])
 def test_rc0_error_still_failure_originalrow_not_erased(self):
  self.parent['kernel_journal_rows']['post-kernel-journal']['error']='CPU_TIMEOUT_WITH_RC0';before=copy.deepcopy(self.parent)
  with self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
  self.assertEqual(self.parent,before)
 def test_changed_argv_log_hash_or_epoch_or_identity_copy_refused(self):
  for mutation in ('argv','log','epoch','identity','EOF'):
   saved=copy.deepcopy(self.parent);log=self.root/'post-kernel-journal.log';raw=log.read_bytes()
   if mutation=='argv':self.parent['kernel_journal_rows']['pre-kernel-journal']['command']=['FALSE_JOURNAL']
   if mutation=='log':log.write_bytes(raw+b'CHANGED\n')
   if mutation=='epoch':self.parent['kernel_journal_rows']['post-kernel-journal']['observed_epoch']=149
   if mutation=='identity':self.parent['post_model_identity']['started']=179
   if mutation=='EOF':self.parent['child_stdout_reader']['eof_epoch']=151
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
   self.parent=saved;log.write_bytes(raw)
 def test_saved_leaf_stdoutSHA_mustmatch_actual_current_original(self):
  (self.root/'child-supervisor.log').write_text('CPU_MUTATED_ORIGINAL\n')
  with self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
 def test_exact_pinned_source_adapter_contains_two_EOF_guards_and_originalrows(self):
  code=parent_source.adapted_source();ast.parse(code);self.assertEqual(code.count("Actual child stdout EOF/drain failed"),2);self.assertIn("row['error'] is None",code);self.assertIn("parent.setdefault('kernel_journal_rows'",code);self.assertIn('import batch_api_cache_positive_buffered_v5 as ctrl',code);self.assertIn("'started':identity['started']",code)
 def test_actual_identity_scan_before_postjournal_refused(self):
  self.parent['post_model_identity']={'started':171,'finished':174};write(self.root/'post-model-identity.json',{'started':171,'finished':174})
  with self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
 def test_actual_health_epoch_cannot_borrow_parent_copy(self):
  health={'passed':True,'started_epoch':160,'finished_epoch':172};write(self.root/'post-health.json',health)
  with self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
 def test_prejournal_after_actual_launch_refused(self):
  self.parent['child_launch_started_epoch']=119
  with self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
 def test_missing_nonfinite_or_reversed_actual_command_times_refused(self):
  original=copy.deepcopy(self.parent)
  for key,value in [('started_epoch',float('nan')),('finished_epoch',176),('started_epoch',176)]:
   self.parent=copy.deepcopy(original);self.parent['kernel_journal_rows']['post-kernel-journal'][key]=value
   with self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
  self.parent=original;del self.parent['kernel_journal_rows']['pre-kernel-journal']['started_epoch']
  with self.assertRaises(KeyError):q.journal_gate(self.root,self.parent)
 def test_old_generation_and_missing_launch_refused(self):
  self.parent['journal_admission_generation']=4
  with self.assertRaises(ValueError):q.journal_gate(self.root,self.parent)
  self.parent['journal_admission_generation']=5;del self.parent['child_launch_started_epoch']
  with self.assertRaises(KeyError):q.journal_gate(self.root,self.parent)
 def test_actual_adapted_command_produces_reader_required_schema(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  tree=ast.parse(parent_source.adapted_source());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');node=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='command');namespace={'out':self.root,'write':write,'sha':sha,'print':lambda *a,**k:None,'time':time,'subprocess':SimpleNamespace(STDOUT=-2,Popen=lambda *a,**k:SimpleNamespace(pid=777,wait=lambda **k:0),TimeoutExpired=TimeoutError)}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-parent-command','exec'),namespace);row=namespace['command'](['CPU_ONLY_SYNTHETIC'],'cpu-command');self.assertEqual(row['return_code'],0);self.assertIsNone(row['error']);self.assertEqual(row['supervisor_pid'],777);self.assertLessEqual(row['started_epoch'],row['finished_epoch']);self.assertEqual(row['command_file_sha256'],sha(self.root/'cpu-command.command.json'))
if __name__=='__main__':unittest.main()
