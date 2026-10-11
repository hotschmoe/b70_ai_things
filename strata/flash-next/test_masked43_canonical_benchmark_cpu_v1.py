import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import benchmark_masked43_canonical_v1 as b
class Controls(unittest.TestCase):
 def data(self):
  view={'files':{'own':{'bytes':32,'stat5':[45,2,32,4,5],'sha256':'a'*64}},'types':[1,1.,True,-0.]};w={'schema':1,'calls':6,'owner_pid':47,'boundaries':[{'label':label,'owner_pid':47,'view':copy.deepcopy(view)}for label in b.LABELS]};return w
 def test_original_three_and_local_reuse_calls_all_seven(self):
  rows=self.data()['boundaries'];fn=b.canonical
  for method,count in((b.baseline,21),(b.within_call,9)):
   with patch.object(b,'canonical',wraps=fn)as call:self.assertEqual(method(rows,copy.deepcopy(rows[0]['view'])),[True]*7);self.assertEqual(call.call_count,count)
 def test_typed_negative_controls_last_comparison_not_skipped(self):self.assertTrue(all(b.negatives().values()))
 def test_initial_comparison_mutation_not_saved_pass(self):
  rows=self.data()['boundaries'];initial=copy.deepcopy(rows[0]['view']);initial['types'][2]=1
  self.assertEqual(b.baseline(rows,initial),[False]*7);self.assertEqual(b.within_call(rows,initial),[False]*7)
 def test_no_cross_call_cached_string(self):
  rows=self.data()['boundaries'];initial=copy.deepcopy(rows[0]['view']);self.assertTrue(all(b.within_call(rows,initial)));rows[-1]['view']['files']['own']['bytes']=33;self.assertFalse(b.within_call(rows,initial)[-1])
 def test_duplicate_serialized_json_keys_refused(self):
  with self.assertRaises(ValueError):b.read_unique(b'{"view":{"x":1,"x":2}}')
 def test_raw_mutation_and_alias_refused(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'raw';p.write_bytes(b'original');h=hashlib.sha256(p.read_bytes()).hexdigest();b.read_bound(p,h);p.write_bytes(b'changed!')
   with self.assertRaises(ValueError):b.read_bound(p,h)
   alias=Path(t)/'alias';alias.symlink_to(p)
   with self.assertRaises(ValueError):b.read_bound(alias,h)
 def test_round_ABBA_end_to_end_saved_fixture_scope(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor=root/'actor';actor.mkdir();w=actor/'producer-operation-witness.json';w.write_text(json.dumps(self.data()));h=hashlib.sha256(w.read_bytes()).hexdigest();r=actor/'report.json';r.write_text(json.dumps({'producer_pid':47,'producer_operation_witness_sha256':h,'artifact_sha256':{'producer-operation-witness.json':h},'passed':True,'errors':[]}));rh=hashlib.sha256(r.read_bytes()).hexdigest()
   with patch.object(b,'source_binding',return_value='CPU_SOURCE'):result=b.run(w,h,r,rh,root/'new',1)
   self.assertEqual([v['method']for v in result['trials']],['A','B','B','A']);self.assertTrue(result['all_seven_comparisons_exact']);self.assertFalse(result['current_semantic_or_model_admission_authority']);self.assertEqual(result['witness_before'],result['witness_after'])
 def test_saved_report_witness_crossbind_refused(self):
  with self.assertRaises(ValueError):b.validate(self.data(),{'producer_pid':48,'producer_operation_witness_sha256':'a','artifact_sha256':{'producer-operation-witness.json':'a'},'passed':True,'errors':[]},'a')
if __name__=='__main__':unittest.main()
