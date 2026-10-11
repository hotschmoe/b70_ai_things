"""Producer-shaped PCL transport and lifecycle controls without captured payloads."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source43_num10_numeric_v2 as q
class Controls(unittest.TestCase):
 def fixture(self):
  pid=17;request=1;digest=q.original.r.numeric.le32_digest([248045]);events=[{'event':'begin','pid':pid,'request':request,'tokens':1,'input_sha256_le32':digest},{'event':'committed_live','pid':pid,'request':request,'published':False,'chain_updated':True,'live_reusable':False,'phase':'complete','finish':'length','ids_truncated':False,'tokens':1,'ids':[248045],'sha256_le32':digest}]
  for complete in(False,True):events.append({'event':'stage_span','pid':pid,'request':request,'phase':'prefill','device':0,'lb':0,'le':48,'lo':0,'hi':1,'complete':complete})
  raw={'ids':[248045],'command':q.original.r.request_command((248045,),1,1,None),'fresh':1,'pin':None,'cancel_requested':None,'stop_sent':False,'output_ids':[846],'done':'DONE 1 1 0.1 0.2 length 0 0 0 0 0 0 0 0 1','stderr':['PCL '+json.dumps(e)for e in events]}
  meta={'ledger':{'actual_reused':0,'candidate_resume':0,'evaluated_prompt_rows':1,'evaluated_decode_rows':0,'generated':1,'cancelled':False,'finish':'length'},'selection':{'actual_reused':0,'candidate_resume':0,'parked_bytes':0,'parked_entries':0,'evictions':0,'checkpoints':0,'reread':False},'logits':[{'pid':str(pid),'request':str(request)}]};return raw,meta
 def run_fixture(self,kind):
  raw,meta=self.fixture()
  if kind=='missing':raw['stderr']=raw['stderr'][1:]
  elif kind=='foreign':e=json.loads(raw['stderr'][1][4:]);e['pid']=99;raw['stderr'][1]='PCL '+json.dumps(e)
  elif kind=='counter':meta['ledger']['evaluated_prompt_rows']=4
  elif kind=='commit':e=json.loads(raw['stderr'][1][4:]);e['tokens']=4;raw['stderr'][1]='PCL '+json.dumps(e)
  elif kind=='span':raw['stderr']=raw['stderr'][:-1]
  with patch.object(q.original.r.numeric,'extract_numeric',return_value=meta):
   if kind:
    with self.assertRaises(ValueError):q.numeric(raw,Path('/CPU_SYNTHETIC'),[(0,48)])
   else:self.assertIs(q.numeric(raw,Path('/CPU_SYNTHETIC'),[(0,48)]),meta)
 def test_real_shaped_PCL_JSON_prefix_survives(self):self.run_fixture(None)
 def test_missing_begin_refused(self):self.run_fixture('missing')
 def test_foreign_owner_refused(self):self.run_fixture('foreign')
 def test_old_fourrow_counter_refused(self):self.run_fixture('counter')
 def test_old_fourrow_commit_refused(self):self.run_fixture('commit')
 def test_missing_returned_span_refused(self):self.run_fixture('span')
 def test_AST_exact_unchanged_transport_slice_and_seven_literals(self):
  fn,counts,node=q.port();self.assertEqual(counts,{'prompt_length':7,'PCL_prefix_slice':1});slices=[n for n in ast.walk(node)if isinstance(n,ast.Slice)];self.assertEqual(len(slices),1);self.assertEqual(slices[0].lower.value,4)
 def test_v1_real_counterexample_is_not_accepted_as_fix(self):
  import source43_num10_numeric_v1 as old
  raw,meta=self.fixture()
  with patch.object(q.original.r.numeric,'extract_numeric',return_value=meta):
   with self.assertRaises(json.JSONDecodeError):old.numeric(raw,Path('/CPU_SYNTHETIC'),[(0,48)])
if __name__=='__main__':unittest.main()
