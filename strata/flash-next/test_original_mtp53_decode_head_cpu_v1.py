import unittest
from pathlib import Path
import prepare_original_mtp53_decode_head_skip_v1 as p
import run_original_mtp53_decode_head_v1 as r
H=Path(__file__).resolve().parent/'original-mtp53'
class DecodeHead(unittest.TestCase):
 def test_actual_bounded_delta_and_final_head(self):
  old=(H/'draft_step_teacher_head_v1.hpp').read_text();new=(H/'draft_step_decode_head_v1.hpp').read_text()
  self.assertEqual(new,p.transformed_step(old))
  # Synthetic/old unbounded helper and recurrence arithmetic remain unchanged.
  a='    bool verify_and_catchup(';b='    bool verify_bounded_and_catchup('
  self.assertEqual(old[old.index(a):old.index(b)],new[new.index(a):new.index(b)])
  a='        eh_hc_front(device_token';b='    uint64_t head_calls()'
  self.assertEqual(old[old.index(a):old.index(b)],new[new.index(a):new.index(b)])
  self.assertEqual(old.count('q.wait_and_throw()'),new.count('q.wait_and_throw()'))
  self.assertEqual(new.count('decode_head_compute53('),1)
  self.assertIn('|| !head_ready_',new)
  self.assertEqual(p.transformed_loaded((H/'loaded_main_adapter_teacher_head_v1.hpp').read_text()),(H/'loaded_main_adapter_decode_head_v1.hpp').read_text())
  self.assertTrue(r.unchanged_decode_source()['all_waits_and_recurrence_preserved'])
 def test_actual_counter_join_and_negative(self):
  windows=[{'committed':2}]*229+[{'committed':1}]*53
  for mode,skips in [('off',0),('on',229)]:
   fields={'mode':int(mode=='on'),'windows':282,'rows':511,'skips':skips,'calls':511-skips,'total_calls':2048+511-skips,'total_skips':skips,'fresh_head':1,'committed_cells':2559}
   text='MTP53_DECODE_HEAD_COUNTS '+' '.join(str(k)+'='+str(v) for k,v in fields.items())+'\n'
   self.assertEqual(r.decode_counts(text,mode,{'windows':windows}),fields)
   with self.assertRaises(ValueError):r.decode_counts(text.replace('fresh_head=1','fresh_head=0'),mode,{'windows':windows})
   with self.assertRaises(ValueError):r.decode_counts(text.replace('calls='+str(511-skips)+' ','calls=0 '),mode,{'windows':windows})
  source=(H/'decode_head_skip_v1.hpp').read_text()
  self.assertIn('return !(enabled && keep==2 && row==0);',source)
  self.assertIn('(keep!=1 && keep!=2) || row<0 || row>=keep',source)
  self.assertIn('throw std::runtime_error("decode first head skip must0/1")',source)
  # EOS and budget tail decide keep1 before the only dispatch point.
  step=(H/'draft_step_decode_head_v1.hpp').read_text()
  self.assertIn('const int keep=accepted && !first_eos?2:1;',step)
  self.assertIn('const int target_rows=remaining>=2?2:1;',step)
 def test_isolated_runtime_flag(self):
  source=Path(r.__file__).read_text()
  self.assertIn("'STRATA_ORIGINAL_MTP53_TEACHER_HEAD_SKIP':'0'",source)
  self.assertIn("'STRATA_ORIGINAL_MTP53_RECORD_SETTLE_CHAIN':'0'",source)
  self.assertIn("'STRATA_VERIFY_HC_T2_REUSE':'0'",source)
  self.assertIn("'STRATA_EXPERT_PAIR_REUSE':'0'",source)
  self.assertIn("'STRATA_ORIGINAL_MTP53_PAIR_MODE':'controls'if mode=='controls'else'on'",source)
  self.assertIn("'STRATA_ORIGINAL_MTP53_DECODE_FIRST_HEAD_SKIP':'1'if mode=='on'else'0'",source)
