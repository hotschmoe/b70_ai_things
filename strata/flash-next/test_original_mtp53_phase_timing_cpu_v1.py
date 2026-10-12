import unittest,re
from pathlib import Path
import prepare_original_mtp53_greedy64_compile_v3 as m
class ExistingBoundaries(unittest.TestCase):
 def test_gpu_wait_and_query_rosters_unchanged(self):
  for old,new in [('draft_step_v3.hpp','draft_step_v4.hpp'),('loaded_main_adapter_v5.hpp','loaded_main_adapter_v6.hpp'),('sustained_greedy_v1.hpp','sustained_greedy_v2.hpp')]:
   a=(m.HERE/'original-mtp53'/old).read_text();b=(m.HERE/'original-mtp53'/new).read_text()
   for needle in ('wait_and_throw','queues_wait_and_throw','get_device','get_info','main.run(','main.commit(','main.wait_commit('):self.assertEqual(a.count(needle),b.count(needle),(new,needle))
  helper=(m.HERE/'original-mtp53/phase_timing_v1.hpp').read_text();self.assertNotIn('sycl::',helper);self.assertIn('if(enabled)rows.reserve(2048)',helper);self.assertIn('timer(p&&p->enabled?p:nullptr)',helper)
 def test_non_nested_scopes_and_end_buffer_dump(self):
  helper=(m.HERE/'original-mtp53/phase_timing_v1.hpp').read_text();self.assertIn('if(timer->active)',helper)
  s=(m.HERE/'original-mtp53/sustained_greedy_v2.hpp').read_text();self.assertGreater(s.index('timings.dump()'),s.index('const auto done=clock::now()'))
  s=(m.HERE/'original-mtp53/draft_step_v4.hpp').read_text();self.assertIn('"draft_record_wait"',s);self.assertIn('"draft_settle_wait"',s)
