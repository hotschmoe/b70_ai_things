import hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import source41_off_deferred_identity_v1 as p

class Controls(unittest.TestCase):
 def context(self,root,failed=False):
  off=root/'off';off.mkdir();(off/'report.json').write_text('{"phase":"off-equivalence"}')
  with patch.object(p,'source_binding',return_value='source'):
   value=p.DeferredOffIdentity(off)
   with patch.object(p.original,'_finalized_binding',side_effect=ValueError('original failure')if failed else None,return_value={'OFF_equivalence_passed':True}):value.recollect()
  return value
 def scan(self,root,ctx,before_read_boundary=False):
  lock={'revision':'synthetic','destination':str(root),'files':[]};rows=[]
  for i in range(4):
   path=root/'UD-Q4_K_XL'/str(i);path.parent.mkdir(exist_ok=True);path.write_bytes(bytes([i]));sig=p.original.c.stat_signature(path);sha=hashlib.sha256(bytes([i])).hexdigest();lock['files'].append({'path':'UD-Q4_K_XL/'+str(i),'size':1,'sha256':sha});rows.append({'path':str(path),'bytes':1,'sha256':sha,'expected_sha256':sha,'passed':True,'stat_before':sig,'stat_after':sig})
  started=(ctx.finished-.001)if before_read_boundary else time.time()+.001
  scan={'passed':True,'lock_sha256':'lock','model_revision':'synthetic','rows':rows,'started':started,'finished':started+.001};path=root/'first-arm-pre-full4.json';path.write_text(json.dumps(scan));before={'epoch':started-.0001};after={'epoch':started+.002};return lock,path,before,after
 def close(self,ctx,lock,path,before,after):
  original_read=p.read_unique
  with patch.object(p,'read_unique',side_effect=lambda f:lock if Path(f).name=='model-lock.json'else original_read(f)),patch.object(p,'sha',return_value='lock'),patch.object(p,'source_binding',return_value='source'),patch.object(p.original,'pages'),patch.object(p.time,'time',return_value=after['epoch']+.001):return ctx.close_with_caller_scan(path,before,after)
 def test_no_standalone_old_public_scan_claim(self):
  with tempfile.TemporaryDirectory()as t:
   ctx=self.context(Path(t));view=ctx.view();self.assertTrue(view['identity_deferred']);self.assertFalse(view['standalone_model_or_runtime_qualified']);self.assertFalse(view['separate_V3_current_public_full4_rehashed_claimed'])
   with self.assertRaises(ValueError):ctx.require_closed()
 def test_exact_caller_closing_scan_after_all_reads(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);ctx=self.context(root)
   with patch.object(p,'source_binding',return_value='source'):ctx.mark_prerequisite_reads_finished(lambda:None)
   lock,path,before,after=self.scan(root,ctx);view=self.close(ctx,lock,path,before,after);self.assertFalse(view['identity_deferred']);self.assertIsNotNone(ctx.require_closed()['caller_identity_scan']);self.assertFalse(view['full_model_math_qualified'])
 def test_scan_before_final_prerequisite_reads_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);ctx=self.context(root)
   with patch.object(p,'source_binding',return_value='source'):ctx.mark_prerequisite_reads_finished(lambda:None)
   lock,path,before,after=self.scan(root,ctx,True)
   with self.assertRaises(ValueError):self.close(ctx,lock,path,before,after)
 def test_failed_predicate_requires_scan_and_stays_failed_after(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);ctx=self.context(root,True)
   with self.assertRaises(ValueError):ctx.require_closed()
   with patch.object(p,'source_binding',return_value='source'):ctx.mark_prerequisite_reads_finished(lambda:None)
   lock,path,before,after=self.scan(root,ctx);self.close(ctx,lock,path,before,after)
   with self.assertRaisesRegex(ValueError,'Original predicate failure'):ctx.require_closed()
 def test_source_callback_failure_still_requires_close(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);ctx=self.context(root)
   with patch.object(p,'source_binding',return_value='source'):ctx.mark_prerequisite_reads_finished(lambda:(_ for _ in()).throw(ValueError('source failure')))
   self.assertIsNotNone(ctx.reads_finished)
   with self.assertRaises(ValueError):ctx.require_closed()
 def test_owner_and_duplicate_recollection_refused(self):
  with tempfile.TemporaryDirectory()as t:
   ctx=self.context(Path(t))
   with self.assertRaises(ValueError):ctx.recollect()
   with patch.object(p.os,'getpid',return_value=ctx.owner+1),self.assertRaises(ValueError):ctx.require_closed()
 def test_wrong_publisher_hash_refused(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);ctx=self.context(root)
   with patch.object(p,'source_binding',return_value='source'):ctx.mark_prerequisite_reads_finished(lambda:None)
   lock,path,before,after=self.scan(root,ctx);body=json.loads(path.read_text());body['rows'][0]['sha256']='wrong';path.write_text(json.dumps(body))
   with self.assertRaises(ValueError):self.close(ctx,lock,path,before,after)
if __name__=='__main__':unittest.main()
