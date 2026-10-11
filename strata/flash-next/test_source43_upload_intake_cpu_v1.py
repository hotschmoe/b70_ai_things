import copy,json,tempfile,unittest
from pathlib import Path
import source43_sdk_binding_v1 as s
import admit_source_upload43_v1 as u
class Controls(unittest.TestCase):
 def fixture(self):
  plan=json.loads(s.PLAN.read_bytes());root=s.SDK;receipt={'image':s.IMAGE,'devices_exposed':False,'build_rc':0,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'status':'built; fresh','source_revision':plan['source_revision'],'ggml_revision':plan['ggml']['revision'],'plan_sha256':s.PLAN_SHA,'source_copy':str(root/'source'),'build':str(root/'build'),'plan_snapshot':str(root/'plan.snapshot.json'),'patched_source_sha256':dict(plan['expected_patched_source_sha256']),'patches':plan['patches'],'binary_sha256':{str(root/'build'/name):'a'*64 for name in s.BINARY_NAMES}}
  return receipt,plan
 def test_real43_shaped_metadata_exact70_and8(self):
  r,p=self.fixture();value=s.metadata(r,p,s.SDK);self.assertEqual(value['actual_receipt_source_rows'],70);self.assertEqual(value['independently_required_source_rows'],70);self.assertEqual(value['required_earlier_added_headers'],32)
 def test_old41_shape_or_source_plan_refused(self):
  for kind in('sourcecount','patch','plan','foreignbinary'):
   r,p=self.fixture()
   if kind=='sourcecount':r['patched_source_sha256'].pop(next(iter(r['patched_source_sha256'])))
   elif kind=='patch':r['patches']=r['patches'][:-2]
   elif kind=='plan':r['plan_sha256']='0'*64
   else:r['binary_sha256']['/foreign/strata']=r['binary_sha256'].pop(str(s.SDK/'build/strata'))
   with self.subTest(kind=kind),self.assertRaises(ValueError):s.metadata(r,p,s.SDK)
 def test_actual_layout43_and_old_generation_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'log';row='UPLOAD43_LAYOUT sizeof_Verifier=100 align_Verifier=8 sizeof_QsaBuffers=128 align_QsaBuffers=8 half_offset=16 expanded_offset=24 idx_raw_offset=32 mode=0 source_generation=43\n';p.write_text(row);self.assertEqual(u.layout_markers(p)['source_generation'],43);p.write_text(row.replace('source_generation=43','source_generation=41'))
   with self.assertRaises(ValueError):u.layout_markers(p)
if __name__=='__main__':unittest.main()
