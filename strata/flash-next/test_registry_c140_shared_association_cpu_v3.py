import tempfile,unittest
from pathlib import Path
import registry_c140_shared_association_v3 as r

class Registry(unittest.TestCase):
 def test_exact_source40_four_and_shared_two_not_C139_four(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'registry.yaml';base=r.BASE.read_bytes();delta=r.rendered_fragment(r.C140.read_bytes());extra=r.rendered_fragment(r.SHARED.read_bytes());p.write_bytes(base+delta);self.assertEqual(r.association(p)['actual_models'],169)
   p.write_bytes(base+delta+extra);out=r.association(p,True);self.assertEqual(out['actual_models'],171);self.assertFalse(out['original_global_hash_gate_passed']);self.assertFalse(out['source37_or_C139_runtime_proof_transferred'])
 def test_C139_or_original_changed_entry_or_extra_bytes_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'registry.yaml';raw=r.BASE.read_bytes()+r.rendered_fragment(r.C140.read_bytes())+r.rendered_fragment(r.SHARED.read_bytes())
   for wrong in [r.BASE.read_bytes()+(r.HERE/'c1-combined-v139-model-registry-proposal-v2.yaml').read_bytes(),raw+b'\n',raw.replace(b'hotschmoe-dd',b'foreign-model',1)]:
    p.write_bytes(wrong)
    with self.assertRaises(ValueError):r.association(p,True)
if __name__=='__main__':unittest.main()
