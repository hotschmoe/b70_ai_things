import tempfile,unittest
from pathlib import Path
import prepare_hc_rms3_operator_v1 as p
class Controls(unittest.TestCase):
 def test_actual_same_source_and_exact_observer_scope(self):
  source=p.SDK59/'source/sycl/src/kernels/hc_native_composition.cpp';text=source.read_text();generated=p.extract(text);self.assertEqual(p.sha(source),p.COMPOSITION_SHA);self.assertEqual(p.sha(p.SDK61/'source/sycl/src/kernels/hc_native_composition.cpp'),p.COMPOSITION_SHA);self.assertEqual(generated.count('sycl::fma(r,r,sum)'),2);self.assertEqual(generated.count('observed_sum[tc]=square_sum'),1);self.assertNotIn('a.w.down',generated);self.assertNotIn('a.w.up',generated)
 def test_foreign_arithmetic_refused(self):
  text=(p.SDK59/'source/sycl/src/kernels/hc_native_composition.cpp').read_text()
  with self.assertRaises(ValueError):p.extract(text.replace('sycl::rsqrt(','sycl::sqrt('))
 def test_real_source_preparation_no_runtime(self):
  with tempfile.TemporaryDirectory()as d:
   out=Path(d)/'new';recipe=p.source_prepare(out);self.assertTrue(recipe['observer_default_OFF']);self.assertFalse(recipe['compiled_or_GPU_executed']);self.assertEqual(set(recipe['files']),{'fixture.cpp','hc_rms3_generated_v1.inc','hc_native_composition.hpp'});self.assertIn('-fp-model=precise',recipe['compiler_argv']);self.assertIn('-cl-fp32-correctly-rounded-divide-sqrt',recipe['compiler_argv'])
if __name__=='__main__':unittest.main()
