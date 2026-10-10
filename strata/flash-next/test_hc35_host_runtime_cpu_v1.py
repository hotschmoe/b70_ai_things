"""Tiny source/fixture/runtime-parser controls; compiled helper never executed."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import qualify_hc35_host_runtime_v1 as q

class SupplementTests(unittest.TestCase):
 def test_all15_known_answers_against_python_analytic_lane(self):
  cases=q.fixture_cases();self.assertEqual(len(cases),15)
  for case in cases:
   fn={'dot':q.host.dot32_f32,'q8dot':q.host.q8dot32_f32,'fma':q.host.fma_f32}[case['op']]
   args=[case['a'],case['b']]+([case['c']] if case['c'] is not None else [])
   self.assertEqual(fn(*args),case['expected'],case['label'])
 def test_preserved_fixture_mutation_and_known_answer_refused(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);case=q.fixture_cases()[0];row=q.preserve_fixture(root,case);q.validate_fixture(root,row,case)
   path=Path(row['files']['expected']['path']);path.write_bytes(b'BAD!')
   with self.assertRaises(ValueError):q.validate_fixture(root,row,case)
 def test_fixture_symlink_escape_refused(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);case=q.fixture_cases()[0];row=q.preserve_fixture(root,case);path=Path(row['files']['a']['path']);raw=path.read_bytes();path.unlink();other=root/'other';other.write_bytes(raw);path.symlink_to(other)
   with self.assertRaises(ValueError):q.validate_fixture(root,row,case)
 def test_ldd_complete_and_missing_scope(self):
  lines=['linux-vdso.so.1 (0x1)']+['%s => /lib/%s (0x1)'%(s,s) for s in ('libm.so.6','libc.so.6','libstdc++.so.6','libgcc_s.so.1')]+['/lib/ld-linux-x86-64.so.2 (0x1)'];parsed=q.parse_ldd('\n'.join(lines));self.assertIn('ELF_interpreter',parsed)
  for text in ('libm.so.6 => not found','linux-vdso.so.1 (0x1)','UNKNOWN x'):
   with self.assertRaises(ValueError):q.parse_ldd(text)
 def test_changed_library_identity_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'lib';p.write_bytes(b'one');before=q.file_binding(p);p.write_bytes(b'two');self.assertNotEqual(before,q.file_binding(p))
 def test_final_failed_report_stops_before_runtime_or_helper(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'report.json').write_text(json.dumps({'passed':False,'errors':['failed']}))
   with patch.object(q,'runtime_binding',side_effect=AssertionError('No runtime/helper call')):
    with self.assertRaises(ValueError):q.finalized_binding(root,root/'helper',root)
 def test_exact_source_bindings(self):
  self.assertEqual(q.source_binding()['V2_source_plan_sha256'],q.V2_PLAN_SHA)
 def test_no_actual_helper_called_in_fixture_creation(self):
  with patch.object(q.subprocess,'run',side_effect=AssertionError('Compiled helper forbidden')):self.assertEqual(len(q.fixture_cases()),15)

if __name__=='__main__':unittest.main()
