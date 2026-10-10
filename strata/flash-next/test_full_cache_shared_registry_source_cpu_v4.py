import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_registry_source_v4 as r
from registry_c140_shared_association_v3 import BASE,C140,SHARED,rendered_fragment
class Registry(unittest.TestCase):
 def test_exact171_without_original165_raw_hash_claim(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);(root/'code.py').write_bytes(b'unchanged');current=root/'registry.yaml';current.write_bytes(BASE.read_bytes()+rendered_fragment(C140.read_bytes())+rendered_fragment(SHARED.read_bytes()));files={r.REGISTRY:r.BASE_SHA,'code.py':r.sha(root/'code.py')};proof=r.verify_files(files,root,current);self.assertEqual(proof['current_registry_association']['actual_models'],171);self.assertFalse(proof['original_global_hash_gate_passed']);self.assertFalse(proof['historical_registry_digest_substituted'])
   current.write_bytes(BASE.read_bytes())
   with self.assertRaises(ValueError):r.verify_files(files,root,current)
 def test_nonregistry_mutation_is_not_waived(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);path=root/'code.py';path.write_bytes(b'original');files={r.REGISTRY:r.BASE_SHA,'code.py':r.sha(path)};path.write_bytes(b'changed')
   with patch.object(r,'association')as association,self.assertRaises(ValueError):r.verify_files(files,root,root/'unused')
   association.assert_not_called()
 def test_actual_prepared169_digest_and_entry_required(self):
  from strict_registry_yaml_v3 import parse
  entries=parse(C140.read_bytes());expected=hashlib.sha256(BASE.read_bytes()+rendered_fragment(C140.read_bytes())).hexdigest()
  with patch.object(r,'association',return_value={'actual_models':171}):
   proof=r.prepared169_binding({'registry_sha256':expected,'alias':entries[0]['served_model_id']});self.assertFalse(proof['prepared_digest_rewritten'])
   for value in ({'registry_sha256':r.BASE_SHA,'alias':entries[0]['served_model_id']},{'registry_sha256':expected,'alias':'foreign'}):
    with self.assertRaises(ValueError):r.prepared169_binding(value)
