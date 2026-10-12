import unittest
from pathlib import Path
import prepare_original_mtp53_two_stage_compile_v4 as old
import prepare_original_mtp53_two_stage_compile_v5 as new
class Controls(unittest.TestCase):
 def test_actual_source_reset_and_exact_undo(self):
  source=(Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source')/'sycl/src/program/generate.cpp').read_text();generated=new.transform(source);hook=new.reset_hook(source)
  self.assertEqual(generated.replace(hook,'',1),old.transform(source))
  self.assertLess(generated.index('if(!originalPairFreshReset('),generated.index('originalPair->bind_target(ver,last_st->ver)'))
  body=new.fresh_reset_body(source)
  for key in('session_zero(state,g,nullptr,(void*)queue)','queue->get_context()!=current.get_context()','for(size_t k=0;k<=stages.size();++k)','queue->memset(q.idx_block_pos','queue->memcpy(q.page_table','queue->wait_and_throw()'):
   self.assertIn(key,body)
  self.assertNotIn('live.clear()',body)
  with self.assertRaises(ValueError):new.transform(source.replace('auto public_reset_main=','auto removed_public_reset_main=',1))
if __name__=='__main__':unittest.main()
