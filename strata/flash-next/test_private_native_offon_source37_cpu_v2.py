"""Historical49 + prospective147 index union; tiny actual bytes, no GPU/model."""
import copy,struct,tempfile,unittest
from pathlib import Path
from test_private_native_offon_source37_cpu_v1 import Protocol,Plans,Serial
import validate_private_native_offon_source37_v2 as reader

class MixedSerial(unittest.TestCase):
 def fixture(self):
  jobs=[{'rid':r,'role':role} for r,role in [(2001,'admission'),(2002,'admission'),(2001,'later'),(2002,'later')]]
  rows=lambda subset:{((j['rid'],j['role']),layer):'tiny' for j in subset for layer in range(-1,48)}
  return jobs,rows(jobs[:1]),rows(jobs[1:])
 def test_unique_source_indices_cover_exact196(self):
  jobs,first,rest=self.fixture();self.assertEqual(len(reader.complete_union(jobs,first,rest)),196)
 def test_missing_remaining_row_and_borrowed_first_reject(self):
  jobs,first,rest=self.fixture()
  for change in ('missing','borrow'):
   bad=dict(rest)
   if change=='missing':bad.pop(next(iter(bad)))
   else:bad.update(first)
   with self.assertRaises(ValueError):reader.complete_union(jobs,first,bad)
 def test_wrong_first_index_or_duplicate_selected_job_reject(self):
  jobs,first,rest=self.fixture()
  with self.assertRaises(ValueError):reader.complete_union(jobs,rest,first)
  jobs[1]=jobs[0]
  with self.assertRaises(ValueError):reader.complete_union(jobs,first,rest)
 def test_tiny_actual196_exactbytes_and_onebit_failure(self):
  jobs,first,rest=self.fixture()
  with tempfile.TemporaryDirectory(prefix='private37-union-CPU-') as d:
   a=Path(d)/'a.raw';b=Path(d)/'b.raw';a.write_bytes(struct.pack('<ff',1,2));b.write_bytes(struct.pack('<II',0x3f800001,0x40000000));union=reader.complete_union(jobs,first,rest);batch={k:str(a) for k in union};serial=dict(batch)
   self.assertTrue(reader.compare_all(batch,serial)['passed']);serial[next(iter(rest))]=str(b);result=reader.compare_all(batch,serial);self.assertFalse(result['passed']);self.assertTrue(any(v['first_different_float']==0 for v in result['comparisons'].values()))
 def test_new_canonical_roster_distinct_and_historical_scope_explicit(self):
  self.assertTrue(str(reader.serial_roster_path(Path('/tmp/ON'))).endswith('.serial49-roster-v2.json'))
  s=Path(reader.__file__).read_text();self.assertIn("'full_group_absent_PIN_scope_qualified':False",s);self.assertIn("'historical_supervisor_EOF_producer_field_observed':False",s);self.assertIn("parent['parent_generation']==42",s);self.assertIn('future.finalized_binding(root)==binding',s)
if __name__=='__main__':unittest.main()
