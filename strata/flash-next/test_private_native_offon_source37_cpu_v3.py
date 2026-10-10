"""V3 canonical first49 interface plus exact tiny index/byte union controls."""
import json,tempfile,unittest
from pathlib import Path
from test_private_native_offon_source37_cpu_v1 import Protocol,Plans,Serial
from test_private_native_offon_source37_cpu_v2 import MixedSerial
import validate_private_native_offon_source37_v3 as reader
import serial37_canonical_json_v3 as codec

class V3(unittest.TestCase):
 def test_new_schema_roster_and_parent43_source(self):
  self.assertTrue(str(reader.serial_roster_path(Path('/tmp/ON'))).endswith('.serial49-roster-v3.json'));s=Path(reader.__file__).read_text();self.assertIn("parent['parent_generation']==43",s);self.assertIn("source['serial_source37_generation']==3",s)
 def test_actual_first_receipt_json_domain_retains_false_flags(self):
  actual={'raw_meta':{'layer_shape':{0:(4,2560)}},'original_parent_passed':False,'new_absent_PIN_scope_qualified':False}
  with tempfile.TemporaryDirectory(prefix='private37-json-CPU-') as d:
   p=Path(d)/'first.json';p.write_text(json.dumps(actual));self.assertTrue(codec.matches_saved(actual,p));saved=json.loads(p.read_bytes());saved['original_parent_passed']=0;p.write_text(json.dumps(saved));self.assertFalse(codec.matches_saved(actual,p))
 def test_new_union_still_requires_exact_four_indices(self):
  jobs=[{'rid':r,'role':role} for r,role in [(2001,'admission'),(2002,'admission'),(2001,'later'),(2002,'later')]];rows=lambda js:{((j['rid'],j['role']),l):'CPU_mock' for j in js for l in range(-1,48)};first,remaining=rows(jobs[:1]),rows(jobs[1:]);self.assertEqual(len(reader.complete_union(jobs,first,remaining)),196);remaining.update(first)
  with self.assertRaises(ValueError):reader.complete_union(jobs,first,remaining)
if __name__=='__main__':unittest.main()
