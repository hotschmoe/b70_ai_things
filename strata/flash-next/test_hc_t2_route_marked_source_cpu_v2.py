import unittest
import prepare_hc_t2_route_marked_v1 as old
import prepare_hc_t2_route_marked_v2 as new
class Controls(unittest.TestCase):
 def test_actual_both_success_returns_and_exact_delta(self):
  source=(new.PARENT/'overlay/sycl/src/core/verify.cpp').read_text();v1=old.marked_verify(source);v2=new.marked_verify(source)
  marker='        hc_record.finish();\n'
  self.assertEqual(v2.count(marker),1);self.assertEqual(v2.replace(marker,'',1),v1)
  start=v2.index('bool Verifier::record_window(');stop=v2.index('std::string Verifier::profile_report()',start);body=v2[start:stop]
  self.assertEqual(body.count('hc_record.finish();'),2)
  branch=body.index("if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head")
  early=body.index('hc_record.finish();',branch);ret=body.index('return true;',branch)
  self.assertLess(early,ret);self.assertLess(early,body.index('// ---- the head, T columns'))
  self.assertEqual(body[:branch].count('hc_record.finish();'),0)
  # Original V1 succeeds early WITHOUT finish: preserve this real source-shaped negative.
  old_body=v1[v1.index('bool Verifier::record_window('):v1.index('std::string Verifier::profile_report()')];b=old_body.index("if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head");end=old_body.index('return true;',b)
  self.assertNotIn('hc_record.finish();',old_body[b:end])
if __name__=='__main__':unittest.main()
