"""Exact append side proof and unmodified old C140 public-reader routing."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import registry_c140_chunk173_association_v1 as r
import full_cache_shared_baseline_admission_v12 as b
import full_cache_shared_registry_source_v5 as source
from strict_registry_yaml_v3 import parse
class Controls(unittest.TestCase):
 def test_exact173_currentbytes_and_original171_rows(self):
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'models.yaml';path.write_bytes(r.PRIOR.read_bytes()+b'\n'+r.rendered_fragment(r.DELTA.read_bytes()));proof=r.association(path);self.assertEqual(proof['actual_models'],173);self.assertFalse(proof['old171_current_global_hash_gate_passed']);self.assertFalse(proof['runtime_or_math_proof_transferred'])
 def test_changed_original_row_extra_alias_or_wrongseparator_refused(self):
  raw=r.PRIOR.read_bytes()+b'\n'+r.rendered_fragment(r.DELTA.read_bytes())
  for bad in(raw.replace(b'hotschmoe-dd',b'foreign-name',1),raw+b'\n',r.PRIOR.read_bytes()+r.rendered_fragment(r.DELTA.read_bytes())):
   with tempfile.TemporaryDirectory()as d:
    path=Path(d)/'models.yaml';path.write_bytes(bad);self.assertRaises(ValueError,r.association,path)
 def test_171_and173_original_prepared_digest_entry_sideproof(self):
  alias=parse(r.C140.read_bytes())[0]['served_model_id']
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'models.yaml';path.write_bytes(r.PRIOR.read_bytes()+b'\n'+r.rendered_fragment(r.DELTA.read_bytes()))
   for digest,count in((r.PRIOR_SHA,171),(r.CURRENT_SHA,173)):
    prepared={'alias':alias,'registry_sha256':digest};before=dict(prepared);proof=r.baseline_registry_binding(prepared,path);self.assertEqual(proof['actual_prepared_models'],count);self.assertEqual(prepared,before);self.assertFalse(proof['prepared_digest_or_original_proof_rewritten'])
   for bad in({'alias':alias,'registry_sha256':r.BASE_SHA},{'alias':'foreign','registry_sha256':r.PRIOR_SHA}):self.assertRaises(ValueError,r.baseline_registry_binding,bad,path)
 def test_original_C140_all_predicates_run_before_side_association(self):
  prepared={'original':'immutable'};proof={'original_C140':'all-source-runtime-predicates'}
  with patch.object(b,'original',return_value=(prepared,proof))as old,patch.object(b,'registry_binding',return_value={'metadata_association_only':True})as side:
   got,binding=b.finalized_binding('/CPU_SYNTHETIC');self.assertIs(got,prepared);self.assertIs(binding['actual_original_C1401403_proof'],proof);old.assert_called_once_with('/CPU_SYNTHETIC');side.assert_called_once_with(prepared);self.assertFalse(binding['new_GPU_health_authorization_transferred'])
 def test_nonregistry_inherited_source_mutation_still_refused_first(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);path=root/'code.py';path.write_bytes(b'original');files={source.REGISTRY:source.BASE_SHA,'code.py':source.sha(path)};path.write_bytes(b'changed')
   with patch.object(source,'association')as gate:self.assertRaises(ValueError,source.verify_files,files,root,root/'unused');gate.assert_not_called()
if __name__=='__main__':unittest.main()
