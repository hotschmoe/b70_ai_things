"""Tiny original receipt/log admission controls; no device or model boundaries."""
import copy,json,tempfile,unittest
from pathlib import Path
import validate_hc_composition35_v3 as reader
class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='hc35-journal-CPU-');self.root=Path(self.tmp.name);self.p={'started_epoch':10.,'finished_epoch':40.,'pre_health_finished_epoch':11.,'post_health_finished_epoch':25.,'child_started_epoch':15.,'child_terminal_epoch':20.,'kernel_journal_receipts':{}}
  for stage,start,finish in [('pre',12.,13.),('post',26.,27.)]:
   path=self.root/(stage+'-kernel-journal.log');path.write_text('CPU_SYNTHETIC_COMPLETE_KERNEL_JOURNAL\n');cmd=self.root/(stage+'-kernel-journal.command.json');argv=['journalctl','-k','--since','@10','--no-pager'];cmd.write_text(json.dumps(argv));row={'path':str(path),'sha256':reader.sha(path),'stdout_sha256':reader.sha(path),'command':argv,'command_file_sha256':reader.sha(cmd),'return_code':0,'error':None,'started_epoch':start,'finished_epoch':finish};self.p['kernel_journal_receipts'][stage]=row;self.save(stage)
 def tearDown(self):self.tmp.cleanup()
 def save(self,stage):
  (self.root/(stage+'-kernel-receipt.json')).write_text(json.dumps(self.p['kernel_journal_receipts'][stage]))
 def test_complete_exact_original_pre_post_receipts(self):
  for stage in ('pre','post'):self.assertEqual(reader.journal_receipt_binding(self.root,self.p,stage),self.p['kernel_journal_receipts'][stage])
 def test_changed_or_truncated_journal_rejected(self):
  path=self.root/'pre-kernel-journal.log'
  for value in ('CPU_MODIFIED_JOURNAL\n',''):
   path.write_text(value)
   with self.assertRaises(ValueError):reader.journal_receipt_binding(self.root,self.p,'pre')
 def test_timeout_error_rczero_cannot_be_complete(self):
  self.p['kernel_journal_receipts']['post']['error']='Command deadline exceeded; CPU_SYNTHETIC';self.save('post')
  with self.assertRaises(ValueError):reader.journal_receipt_binding(self.root,self.p,'post')
 def test_nonzero_or_missing_receipt_refused(self):
  self.p['kernel_journal_receipts']['pre']['return_code']=1;self.save('pre')
  with self.assertRaises(ValueError):reader.journal_receipt_binding(self.root,self.p,'pre')
  (self.root/'post-kernel-receipt.json').unlink()
  with self.assertRaises(FileNotFoundError):reader.journal_receipt_binding(self.root,self.p,'post')
 def test_before_after_chronology_and_nonfinite_failclosed(self):
  original=copy.deepcopy(self.p)
  for stage,key,value in [('pre','started_epoch',10.),('pre','finished_epoch',16.),('post','started_epoch',19.),('post','finished_epoch',float('nan')),('post','finished_epoch',41.)]:
   self.p=copy.deepcopy(original);self.p['kernel_journal_receipts'][stage][key]=value;self.save(stage)
   with self.subTest(stage=stage,key=key),self.assertRaises(ValueError):reader.journal_receipt_binding(self.root,self.p,stage)
 def test_changed_actual_argv_rejected_even_updated_command_sha(self):
  row=self.p['kernel_journal_receipts']['pre'];row['command'][-1]='--since';cmd=self.root/'pre-kernel-journal.command.json';cmd.write_text(json.dumps(row['command']));row['command_file_sha256']=reader.sha(cmd);self.save('pre')
  with self.assertRaises(ValueError):reader.journal_receipt_binding(self.root,self.p,'pre')
 def test_device_FP_log_sha_bound_and_unique_actual_lines(self):
  path=self.root/'leaf.log';text='HC35_COMPOSITION_DEVICE backend=level_zero vendor=CPU_SYNTHETIC driver=CPU_SYNTHETIC affinity=0 selector=level_zero:gpu\nHC35_COMPOSITION_CONFIG synthetic=1 compiled_composition=1 shadows_separate=1 subgroup=32 eps=1e-6 graph_and_queue=1 normal_model_graph_qualified=0 device_intrinsics_qualified=0 model_math_qualified=0\nHC35_COMPOSITION_FP_CONFIG flags=CPU_SYNTHETIC observed_device_flags_only=1 compiler_lowering_unobserved=1\n';path.write_text(text);saved={'log_sha256':reader.sha(path)};r=reader.leaf_log_binding(self.root,saved);self.assertFalse(r['device_general_FP_mode_qualified']);path.write_text(text.replace('vendor=CPU_SYNTHETIC','vendor=CHANGED'))
  with self.assertRaises(ValueError):reader.leaf_log_binding(self.root,saved)
  path.write_text(text+text.splitlines()[0]+'\n');saved['log_sha256']=reader.sha(path)
  with self.assertRaises(ValueError):reader.leaf_log_binding(self.root,saved)
 def test_faults_stores_original_receipts_and_rejects_error(self):
  import inspect,qualify_hc_composition35_v3 as parent
  source=inspect.getsource(parent.main);self.assertIn("parent.setdefault('kernel_journal_receipts',{})[stage]=row",source);self.assertIn("row['error'] is None and not FAULT",source);self.assertIn("parent['child_started_epoch']=time.time()",source)
if __name__=='__main__':unittest.main()
