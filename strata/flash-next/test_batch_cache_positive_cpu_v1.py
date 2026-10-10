"""Synthetic source-only positive handoff admission, no tensor/model/GPU work."""
import copy,unittest
import batch_cache_positive_contract_v1 as q
class Controls(unittest.TestCase):
 def setUp(self):
  self.job={'rid':7,'role':'solo_migration','ids':[11,12,13,14],'position':3,'token':14}
  self.trace='\n'.join(['SBF slot_closed pid=55 rid=7 enginegen=1 requestgen=2 slot=1 slotgen=3 cancelled=1 completed=1','SBF migration_begin pid=55 rid=7 enginegen=1 requestgen=2 slot=6 slotgen=3 source_slot=1 source_slotgen=3 prompt=4 event=2 main_session=1','BCHAIN restored slot=1 rid=7 slotgen=3 active_checkpoint=0 tokens=3 points=2 host_bytes=128','SBF resume pid=55 rid=7 enginegen=1 requestgen=2 slot=6 slotgen=3 reused=3 read_from=3 reread_to=-1'])
 def test_actualpositive_witness_preserves_limited_scope(self):
  r=q.last_live_handoff(self.trace,7,self.job,2);self.assertEqual(r['actual_reused'],3);self.assertFalse(r['full_cached_fresh_math_qualified']);self.assertFalse(r['stale_model_or_session_negative_runtime_qualified'])
 def test_stale_pid_engine_request_slot_source_generation(self):
  for old,new in [('pid=55 rid=7 enginegen=1 requestgen=2 slot=6','pid=56 rid=7 enginegen=1 requestgen=2 slot=6'),('BCHAIN restored slot=1 rid=7 slotgen=3','BCHAIN restored slot=1 rid=7 slotgen=4'),('cancelled=1','cancelled=0'),('source_slot=1','source_slot=0')]:
   with self.subTest(new=new),self.assertRaises(ValueError):q.last_live_handoff(self.trace.replace(old,new),7,self.job,2)
 def test_zero_recompute_or_checkpoint_only_transfer_refused(self):
  for old,new in [('reused=3','reused=0'),('read_from=3','read_from=0'),('reread_to=-1','reread_to=0'),('tokens=3','tokens=2'),('active_checkpoint=0','active_checkpoint=1'),('points=2','points=0'),('host_bytes=128','host_bytes=0')]:
   with self.subTest(new=new),self.assertRaises(ValueError):q.last_live_handoff(self.trace.replace(old,new),7,self.job,2)
 def test_missing_or_reordered_or_duplicate_transfer_refused(self):
  lines=self.trace.splitlines()
  for altered in ('\n'.join(lines[:2]+lines[3:]),'\n'.join([lines[0],lines[2],lines[1],lines[3]]),self.trace+'\n'+lines[2]):
   with self.assertRaises(ValueError):q.last_live_handoff(altered,7,self.job,2)
 def test_wrong_consumed_prefix_shape_or_reset_refused(self):
  j=copy.deepcopy(self.job);j['token']=15
  with self.assertRaises(ValueError):q.last_live_handoff(self.trace,7,j,2)
  with self.assertRaises(ValueError):q.last_live_handoff(self.trace+'\nBCPUBLIC reset rid=7 stages=2 fresh=1',7,self.job,2)
 def test_warmfresh_targetcached_profile_and_no_false_bool(self):
  args=['--batch','2','--batch-groups','1','--prompt-cache','3','--adapt-every','0','--adapt-swaps','0','--no-prefill-borrow','--kv','fp16','--conversation-cache-mib','0'];env=dict(q.fresh_profile.__globals__['PROFILE_ENV']);r=q.profile(args,env,2,{'strata_fresh':True},{'strata_fresh':False});self.assertFalse(r['runtime_qualified'])
  for target in ({'strata_fresh':True},{'strata_fresh':0},{}):
   with self.assertRaises((ValueError,KeyError)):q.profile(args,env,2,{'strata_fresh':True},target)
if __name__=='__main__':unittest.main()
