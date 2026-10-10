"""Hashed producer-shaped fault logs cannot qualify via passed metadata."""
import hashlib,tempfile,unittest
from pathlib import Path
import native_rms_kernel_journal_v5 as k
class Tests(unittest.TestCase):
 def fixture(self,root):
  r={'passed':True,'artifact_sha256':{}}
  for s in ('pre','post'):
   p=root/(s+'-kernel.log');p.write_text('xe: normal completion\n',encoding='ascii');sha=hashlib.sha256(p.read_bytes()).hexdigest();r[s+'_journal']={'passed':True,'return_code':0,'error':None,'eof':True,'reader_retired':True,'path':str(p),'sha256':sha};r['artifact_sha256'][p.name]=sha
  return r
 def test_actual_shaped_clean_pre_post(self):
  with tempfile.TemporaryDirectory() as t:self.assertTrue(k.admit(Path(t),self.fixture(Path(t))))
 def test_forged_pass_and_updated_fault_hashes_rejected(self):
  for stage in ('pre','post'):
   for fault in ('Fault response','CAT error','GPU HANG','GPU coredump','Job 9 timed out','GT0 reset failed'):
    with tempfile.TemporaryDirectory() as t:
     root=Path(t);r=self.fixture(root);p=root/(stage+'-kernel.log');p.write_text(fault.lower()+'\n',encoding='ascii');sha=hashlib.sha256(p.read_bytes()).hexdigest();r[stage+'_journal']['sha256']=sha;r['artifact_sha256'][p.name]=sha
     with self.assertRaises(ValueError):k.admit(root,r)
 def test_actual_log_path_and_hash_required(self):
  for what in ('path','hash','roster'):
   with tempfile.TemporaryDirectory() as t:
    root=Path(t);r=self.fixture(root)
    if what=='path':r['pre_journal']['path']='/foreign'
    elif what=='hash':r['pre_journal']['sha256']='foreign'
    else:del r['artifact_sha256']['pre-kernel.log']
    with self.assertRaises(ValueError):k.admit(root,r)
 def test_producer_and_public_reader_use_same_refusal(self):
  import qualify_native_rms_rsqrt37_v5 as q
  s=Path(q.__file__).read_text();self.assertIn('reject_faults((out/(stage+',s);self.assertIn('kernel_admit(root,r)',s)
 def test_saved_terminal_inspection_cannot_contradict_report(self):
  import qualify_native_rms_rsqrt37_v5 as q
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'inspection.json';obj={'State':{'Running':False,'ExitCode':0,'OOMKilled':False,'Error':'','FinishedAt':'ACTUAL_SAVED'},'Image':'PINNED_IMAGE'};q.write(p,obj);q.terminal_receipt_binding(obj,p)
   for field in ('FinishedAt','ExitCode'):
    bad={'State':dict(obj['State']),'Image':obj['Image']};bad['State'][field]='CHANGED' if field=='FinishedAt' else False
    with self.assertRaises(ValueError):q.terminal_receipt_binding(bad,p)
 def test_current_ELF_before_after_subset_and_maps_required(self):
  import copy,qualify_native_rms_rsqrt37_v5 as q
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'CPU_ELF';p.write_bytes(b'CPU_ONLY');before={'helper_sha256':q.sha(p),'libraries':{'CPU_LIB':{'sha256':'ORIGINAL','bytes':1}}};after=dict(helper_sha256=q.sha(p),libraries=dict(before['libraries']),mapped_GPU_libraries_included=True);q.runtime_receipt_binding(before,after,p)
   for field in ('beforeELF','afterELF','library','maps'):
    a=copy.deepcopy(after);b=copy.deepcopy(before)
    if field=='beforeELF':b['helper_sha256']='CHANGED'
    elif field=='afterELF':a['helper_sha256']='CHANGED'
    elif field=='library':a['libraries']['CPU_LIB']['sha256']='CHANGED'
    else:a['mapped_GPU_libraries_included']=False
    with self.assertRaises(ValueError):q.runtime_receipt_binding(b,a,p)
if __name__=='__main__':unittest.main()
