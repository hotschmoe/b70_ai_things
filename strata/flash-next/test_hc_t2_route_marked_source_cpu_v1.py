import unittest
from pathlib import Path
import prepare_hc_t2_route_marked_v1 as p
class Controls(unittest.TestCase):
 def test_actual_verify_record_submit_drain(self):
  s=(p.PARENT/'overlay/sycl/src/core/verify.cpp').read_text();m=p.marked_verify(s)
  self.assertIn('const bool pair=t2_hc && eligible;',m)
  self.assertEqual(m.count('hc_t2_route_proof::seal(this'),1);self.assertEqual(m.count('hc_t2_route_proof::submit(this'),1);self.assertEqual(m.count('hc_t2_route_proof::complete(this'),1)
  self.assertLess(m.index('const dpct::err0 se = DPCT_CHECK_ERROR(cs_->wait());'),m.index('hc_t2_route_proof::complete(this'))
  self.assertLess(m.index('if (le != 0)',m.index('const uint64_t h36_ReplaySubmit_id')),m.index('hc_submitted=hc_t2_route_proof::submit'))
  self.assertEqual(m.count('ext_oneapi_graph'),s.count('ext_oneapi_graph'));self.assertEqual(m.count('wait()'),s.count('wait()'));self.assertEqual(m.count('wait_and_throw()'),s.count('wait_and_throw()'))
  header=(Path(p.__file__).parent/'hc_t2_route_proof_v1.hpp').read_text();self.assertNotIn('sycl::',header);self.assertNotIn('Verifier',header[header.index('#include'):]);self.assertIn('std::array<Row,32>',header)
if __name__=='__main__':unittest.main()
