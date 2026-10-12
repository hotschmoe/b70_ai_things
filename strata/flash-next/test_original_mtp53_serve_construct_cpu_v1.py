import unittest
import prepare_original_mtp53_serve_source_v1 as p
class Construction(unittest.TestCase):
 def test_actual_seams_construct(self):
  old=(p.SDK/'source/sycl/src/program/generate.cpp').read_text();new=p.construct(old)
  self.assertTrue(new.isascii());self.assertIn('if(originalServe && !req_fresh)',new)
  self.assertIn('originalServe->slots().commit_and_catchup',new)
  self.assertLess(new.index('originalServe->slots().commit_and_catchup'),new.index('std::printf("BDONE'))
  self.assertIn('ORIGINAL_LOADED compound=1 activation=0',new)
  self.assertIn('slot_generations(bs.size(),0)',new)
 def test_stock_loader_unchanged(self):
  old=(p.SDK/'source/sycl/src/program/generate.cpp').read_text();new=p.construct(old)
  first='    if (!o.mtp.empty()) {';last='    mem_mark("the draft' # Loader start and native source body are retained.
  start=old.index(first,old.index('    strata::core::MtpDrafter mtp;'))
  # Exact initial stock loader branch, including paired refusal and own RT tensor load.
  self.assertIn(old[start:start+1500],new)
