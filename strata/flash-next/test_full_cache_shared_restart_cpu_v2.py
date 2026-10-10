import copy,unittest
from full_cache_shared_restart_v2 import empty_actor

class Restart(unittest.TestCase):
 def rows(self):
  parent={'passed':True,'errors':[],'child_pid':17};plan={'lane':'source39'};works=[{'rid':i,'actual_reused':0,'actual_new_prompt_tokens':2,'input_ids':[1,2],'completed_prompt_read':True} for i in [1,2]]
  child={'collection_and_teardown_passed':True,'removed':True,'error':None,'producer_pid':17,'phase_receipts':[{'phase':{'name':'warm','policy':{'strata_fresh':True}},'work':works},{'phase':{'name':'prime0'},'work':[{'actual_reused':0,'actual_new_prompt_tokens':3,'input_ids':[1,2,3]}]}]};return parent,plan,child
 def test_actual_cold_reestablishment_scope(self):
  p,v,c=self.rows();r=empty_actor(p,v,c);self.assertFalse(r['full_cache_runtime_qualified']);self.assertFalse(r['whole_process_initial_state_directly_sampled']);self.assertFalse(r['cross_actor_state_borrowed'])
 def test_old_actor_or_nonzero_reuse_cannot_qualify_restart(self):
  for kind in ['lane','warmRID','warmReuse','primeReuse','foreignPID']:
   p,v,c=self.rows()
   if kind=='lane':v['lane']='source37'
   if kind=='warmRID':c['phase_receipts'][0]['work'][0]['rid']=3
   if kind=='warmReuse':c['phase_receipts'][0]['work'][0]['actual_reused']=1
   if kind=='primeReuse':c['phase_receipts'][1]['work'][0]['actual_reused']=1
   if kind=='foreignPID':c['producer_pid']=18
   with self.subTest(kind=kind),self.assertRaises(ValueError):empty_actor(p,v,c)

if __name__=='__main__':unittest.main()
