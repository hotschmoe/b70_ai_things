import unittest
from pathlib import Path
import prepare_hc_rms3_operator_v2 as p
class Controls(unittest.TestCase):
 def test_exact_third_source_variant(self):
  text=(p.SDK59/'source/sycl/src/kernels/hc_native_composition.cpp').read_text();base=p.prior.extract(text);result=p.extract(text);self.assertTrue(result.startswith(base));self.assertEqual(result.count('1.0f/sycl::sqrt(argument)'),1);self.assertEqual(result.count('sycl::rsqrt('),2);self.assertEqual(result.count('sycl::fma(r,r,sum)'),3)
 def test_unchanged_normal_guard_and_own_graph_controls(self):
  text=(p.ROOT/'kernels/hc_rms3_operator_fixture_v2.cpp').read_text();self.assertIn('std::memcmp(x0.data(),native.data(),10240*4)',text);self.assertIn('std::memcmp(x2.data(),own_calc.first.data(),40960)',text);self.assertIn('std::memcmp(changed2.data(),changed_expected.first.data(),40960)',text);self.assertIn('buffers.reserve(10)',text);self.assertIn('graphs.reserve(3)',text);self.assertIn('q.wait_and_throw();',text);self.assertIn('std::_Exit(86)',text)
if __name__=='__main__':unittest.main()
