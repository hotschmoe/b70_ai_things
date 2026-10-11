"""Metadata/current-namespace/layout negatives; no ELF/library/model execution."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import source41_sdk_binding_v1 as sdk
import admit_source_upload41_v1 as a
import build_source_upload41_oracle_v1 as build
import run_source_upload41_oracle_v1 as run
class Controls(unittest.TestCase):
 def actual_metadata(self):return json.loads((sdk.SDK/'receipt.json').read_bytes()),json.loads(sdk.PLAN.read_bytes())
 def test_actual67_receipt_separate_required69_and_all8_metadata(self):
  r,p=self.actual_metadata();got=sdk.metadata(r,p,sdk.SDK);self.assertEqual(got['actual_receipt_source_rows'],67);self.assertEqual(got['independently_required_source_rows'],69)
 def test_old_source40_and_missing_source_binary_header_refused(self):
  for mode in('plan','source','binary','header','patch'):
   r,p=self.actual_metadata()
   if mode=='plan':r['plan_sha256']='d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6'
   elif mode=='source':p['expected_patched_source_sha256'].pop(next(iter(p['expected_patched_source_sha256'])))
   elif mode=='binary':r['binary_sha256'].pop(next(iter(r['binary_sha256'])))
   elif mode=='header':p['added_header_payloads'].pop()
   else:p['patches'][-1]['sha256']='0'*64;r['patches']=p['patches']
   self.assertRaises(ValueError,sdk.metadata,r,p,sdk.SDK)
 def test_private_namespace_and_fresh_source41_CLI_routing(self):
  self.assertEqual(build.derived().__globals__['__file__'],build.__file__);self.assertEqual(run.derived().__globals__['__file__'],run.__file__);self.assertIn('strata/flash-next/native_source_upload41_gpu_oracle_v1.cpp',build.derived().__code__.co_consts);self.assertIn(41,[x.argval for x in __import__('dis').get_instructions(run.derived())])
 def test_frozen390_body_exact_main_identifier_only(self):
  here=Path(a.__file__).parent;old=(here/'native_source_upload_gpu_oracle_full.cpp').read_text();new=(here/'native_source_upload41_gpu_oracle_v1.cpp').read_text();self.assertIn(old.replace('int main(','int source41_frozen_upload_main('),new);self.assertIn('offsetof(strata::core::QsaBuffers,idx_half_store)',new);self.assertIn('sizeof(strata::core::Verifier)',new)
 def marker(self):return b'UPLOAD41_LAYOUT sizeof_Verifier=1000 align_Verifier=8 sizeof_QsaBuffers=256 align_QsaBuffers=8 half_offset=56 expanded_offset=64 idx_raw_offset=72 mode=0 source_generation=41\n'
 def test_layout_marker_exact_roster_positive(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'log';p.write_bytes(b'UR unrelated\n'+self.marker());self.assertEqual(a.layout_markers(p)['source_generation'],41)
 def test_missing_duplicate_truncated_wrongmode_layout_refused(self):
  for raw in(b'UR only\n',self.marker()*2,self.marker()[:-1],self.marker().replace(b'mode=0',b'mode=1'),self.marker().replace(b'expanded_offset=64',b'expanded_offset=32'),self.marker()+b'UPLOAD41_LAYOUT_ERROR bad\n'):
   with tempfile.TemporaryDirectory()as d:
    p=Path(d)/'log';p.write_bytes(raw);self.assertRaises(ValueError,a.layout_markers,p)
 def test_compile_recipe_actual41_includes_source_headers_libraries_and_nodevices(self):
  r,_=self.actual_metadata();cmd=a.compile_recipe(r,Path('/CPU_SYNTHETIC/out'));self.assertNotIn('--device',cmd);self.assertIn(str(sdk.SDK/'source')+':/src:ro',cmd);self.assertIn(str(sdk.SDK/'build')+':/engine:ro',cmd);self.assertIn('-I/src/sycl/include -I/src/include',cmd[-1]);self.assertEqual(cmd[-1].count('/engine/'),8)
if __name__=='__main__':unittest.main()
