"""Actual JSON representation boundary, exact values, duplicate/nonfinite rejects."""
import json,tempfile,unittest
from pathlib import Path
from test_batch_serial_source37_cpu_v1 import Serial37Tests,RecoveryTests
from test_batch_serial_source37_cpu_v2 import CaptureTests
import serial37_canonical_json_v3 as codec

class Canonical(unittest.TestCase):
 def test_actual_shaped_tuple_and_integer_key_roundtrip(self):
  actual={'raw_meta':{'stage_ranges':[(0,32),(32,48)],'coverage':{0:{'shape':(4,2560)},47:{'shape':(4,2560)}}},'original_parent_passed':False}
  with tempfile.TemporaryDirectory(prefix='serial37-json-CPU-') as d:
   p=Path(d)/'saved.json';p.write_text(json.dumps(actual));saved=json.loads(p.read_bytes());self.assertNotEqual(actual,saved);self.assertTrue(codec.matches_saved(actual,p))
 def test_bool_integer_and_integer_float_are_distinct(self):
  self.assertNotEqual(codec.canonical({'flag':True}),codec.canonical({'flag':1}));self.assertNotEqual(codec.canonical({'n':1}),codec.canonical({'n':1.0}));self.assertNotEqual(codec.canonical({'n':-0.0}),codec.canonical({'n':0.0}))
 def test_every_field_retained_changed_value_rejected(self):
  a={'raw_sha256':'abc','metadata':(1,2),'original_parent_passed':False};b=dict(a);b['raw_sha256']='def';self.assertNotEqual(codec.canonical(a),codec.canonical(b));b=dict(a);b.pop('original_parent_passed');self.assertNotEqual(codec.canonical(a),codec.canonical(b))
 def test_integer_string_key_collision_and_nested_duplicates_reject(self):
  for a in ({1:'a','1':'a'},{'nested':{1:'a','1':'b'}}):
   with self.assertRaises(ValueError):codec.canonical(a)
  with tempfile.TemporaryDirectory(prefix='serial37-dups-CPU-') as d:
   p=Path(d)/'bad.json';p.write_text('{"nested":{"1":1,"1":1}}')
   with self.assertRaises(ValueError):codec.read_unique(p)
 def test_nonfinite_and_unsupported_key_types_reject(self):
  for a in ({'n':float('nan')},{'n':float('inf')},{True:'x'},{1.5:'x'},{None:'x'}):
   with self.assertRaises(ValueError):codec.canonical(a)
 def test_saved_nonfinite_rejects(self):
  with tempfile.TemporaryDirectory(prefix='serial37-nan-CPU-') as d:
   p=Path(d)/'bad.json';p.write_text('{"a":NaN}')
   with self.assertRaises(ValueError):codec.read_unique(p)
 def test_source_consumers_use_strict_canonical_saved_interface(self):
  h=Path(codec.__file__).parent;s=(h/'serial37_selection_v3.py').read_text();self.assertIn('matches_saved(recovery.finalized_binding(),path)',s);s=(h/'validate_private_native_offon_source37_v3.py').read_text();self.assertIn('matches_saved(first.finalized_binding(),receipt)',s);self.assertNotIn('first.finalized_binding()==saved',s);self.assertIn("parent['parent_generation']==43",s)
 def test_preserved_origin_prepare_does_not_call_origin_prepare(self):
  import batch_serial_source37_v3 as ctrl
  from types import SimpleNamespace
  from unittest.mock import patch
  with tempfile.TemporaryDirectory(prefix='serial37-origin-CPU-') as d:
   root=Path(d);p=root/'origin.json';p.write_text('{"kind":"serial","lane":"source37"}');out=root/'prepared';a=SimpleNamespace(origin_plan=p,output=out,first_job_adjudication=None)
   with patch.object(ctrl.origin,'manifest_binding',return_value={'CPU_mock':True}),patch.object(ctrl.origin,'prepare') as forbidden,patch.object(ctrl,'read',side_effect=lambda path:json.loads(Path(path).read_bytes()) if Path(path)==p else {'jobs':[1,2,3,4]}),patch.object(ctrl,'manifest_binding',return_value={'CPU_mock':True}):
    # Complete source recipe fields normally come from the genuine origin; this
    # CPU shape only exercises branch/ownership and never native admission.
    with patch.object(ctrl.origin,'read',return_value={'jobs':[1,2,3,4]}):
     old=ctrl.read
     def declared(path):
      if Path(path)==p:return {'kind':'serial','lane':'source37','batch_parent':str(root),'group_index':0}
      return {'jobs':[1,2,3,4]}
     with patch.object(ctrl,'read',side_effect=declared):ctrl.prepare(a)
    forbidden.assert_not_called();self.assertTrue((out/'plan.json').is_file());self.assertEqual(json.loads((out/'plan.json').read_bytes())['reused_V40_origin_plan']['sha256'],ctrl.sha(p))
if __name__=='__main__':unittest.main()
