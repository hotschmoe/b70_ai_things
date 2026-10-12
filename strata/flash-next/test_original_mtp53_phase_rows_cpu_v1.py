import unittest
import run_original_mtp53_greedy64_v4 as r
class Counts(unittest.TestCase):
 def test_counts_derived_from_actual_windows(self):
  actual={'summary':{'prompt_tokens':'2','mode':'original_mtp_on'},'windows':[{'committed':'2'},{'committed':'1'}]};rows=[]
  for scope,counts in [('serial_prompt',{'target_run':2,'target_commit_wait':2,'draft_prepare_submit':2,'draft_record_wait':2,'draft_settle_wait':2}),('decode',{'target_run':2,'target_commit_wait':2,'draft_prepare_submit':3,'draft_record_wait':3,'draft_settle_wait':3,'own_head_read_argmax':2,'draft_prior_drain':2,'final_existing_drain':1})]:
   for phase,count in counts.items():
    for _ in range(count):rows.append('MTP53_HOST_PHASE ordinal=%d scope=%s phase=%s wall_ms=1.0'%(len(rows)+1,scope,phase))
  text='\n'.join(rows)+'\nMTP53_HOST_PHASE_SCOPE rows=%d additional_GPU_waits=0 additional_GPU_queries=0 timers_nested=0\nMTP53_VERIFIER_DECODE_COUNTERS wait_ms=1 pool_ms=1 host_ms=1 commit_ms=1'%len(rows)
  self.assertEqual(r.phase_rows(text,actual)['host_phase_totals_ms']['decode/draft_record_wait'],3)
  with self.assertRaises(ValueError):r.phase_rows(text.replace('phase=draft_record_wait','phase=unknown',1),actual)
