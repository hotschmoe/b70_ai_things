import unittest
from pathlib import Path
import prepare_hc_t2_caller_v2 as p
class StageBank(unittest.TestCase):
 def test_only_guard_extension_no_math_or_handoff_delta(self):
  a=p.original.construct();b=p.construct()
  for name in p.original.TARGETS:
   if name!='sycl/src/core/verify.cpp':self.assertEqual(a[name],b[name])
  new='lb_>=0 && lb_<le_ && le_<=g.n_layers && (lb_==0 || hand_in_) && (le_==g.n_layers || hand_out_) &&\n                    hits_.stage_mirror->lb==lb_ && hits_.stage_mirror->le==le_ && (le_==g.n_layers ? next_==nullptr : next_!=nullptr) && ext_stream_==nullptr'
  self.assertEqual(b['sycl/src/core/verify.cpp'].replace(new,'next_==nullptr && lb_==0 && le_==g.n_layers && ext_stream_==nullptr'),a['sycl/src/core/verify.cpp'])
 def test_actual_stage_local_ownership_and_serial_queue(self):
  sdk=p.original.SDK;s=(sdk/'sycl/src/core/verify.cpp').read_text();g=(sdk/'sycl/src/program/generate.cpp').read_text()
  self.assertIn('parts_ = b.take<float>(T * K * N)',s)
  self.assertIn('cs_=ext_stream_ ? ext_stream_:device.create_queue(true)',s)
  self.assertIn('hits.stage_mirror->lb!=lb_ || hits.stage_mirror->le!=le_',s)
  self.assertIn('if (hits.stage_mirror && always_publish_)',s)
  self.assertIn('next_->run(T, tokens, pos0, pool, next_user_, out, err)',s)
  self.assertIn('v->set_always_publish(true)',g)
  for lb,le,incoming,outgoing,valid in [(0,24,False,True,True),(24,48,True,False,True),(24,48,False,False,False),(0,24,False,False,False),(24,24,True,True,False)]:
   self.assertEqual(0<=lb<le<=48 and (lb==0 or incoming)and(le==48 or outgoing),valid)
