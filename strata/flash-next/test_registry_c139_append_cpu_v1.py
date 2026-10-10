"""Prospective exact169 semantic parse, no canonical write or runtime proof."""
import tempfile,unittest
from pathlib import Path
import registry_c139_append_v1 as registry
class Controls(unittest.TestCase):
 def test_current165_plus_reviewed4_exact169(self):
  with tempfile.TemporaryDirectory(prefix='C139-registry-CPU-') as d:
   p=Path(d)/'prospective.yaml';p.write_bytes(registry.BASE.read_bytes()+registry.DELTA.read_bytes());r=registry.association(p);self.assertEqual(r['actual_models'],169);self.assertFalse(r['old_global_gates_currently_qualified'])
 def test_duplicate_top_models_key_refused(self):
  raw=registry.BASE.read_bytes()+b'\nmodels:\n'+registry.DELTA.read_bytes()
  with self.assertRaises(ValueError):registry.parse(raw)
 def test_old_row_modified_or_append_reordered_refused(self):
  with tempfile.TemporaryDirectory(prefix='C139-registry-neg-CPU-') as d:
   p=Path(d)/'prospective.yaml'
   for raw in [registry.BASE.read_bytes(),registry.DELTA.read_bytes()+registry.BASE.read_bytes(),registry.BASE.read_bytes()+registry.DELTA.read_bytes()+b'\n']:
    p.write_bytes(raw)
    with self.assertRaises(ValueError):registry.association(p)
if __name__=='__main__':unittest.main()
