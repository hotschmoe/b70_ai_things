import copy,unittest
from unittest.mock import patch
import full_cache_shared_suite_v3 as s
from full_cache_shared_phase_contract_v2 import scenario_schedule

class Suite(unittest.TestCase):
 def case(self):
  phases={}
  for name,count,n,fresh,pin in [('warm',2,164,True,None),('prime_shared',1,293,False,272),('target_shared',2,304,False,272),('independent',2,271,True,None),('prefill_cancel',2,680,False,272),('decode_cancel',2,300,False,272),('eviction_unpinned',6,179,False,None),('fresh_shared',2,304,True,None)]:
   rows=[{'ids':list(range(n)),'messages':[{'role':'user','content':name+str(i)}]}for i in range(count)];policy={'strata_fresh':fresh}
   if pin is not None:policy['strata_shared_prefix']={'tokens':pin}
   phases[name]={'rows':rows,'actual_HTTP_body_policy':policy}
  return {'schema':'full-cache-shared-CPU-case-v2','slots':2,'shared_boundary':{'tokens':272},'phases':phases}
 def test_full_eviction_fourteen_requests_twelve_armed_bothwholeactors(self):
  declared=s.declaration(self.case());eviction=[a for a in declared['actors']if a['scenario']=='eviction'];self.assertEqual(len(eviction),2);self.assertEqual([a['entire_actor_HTTP_requests']for a in eviction],[14,14]);self.assertEqual([a['armed_requests_required']for a in eviction],[12,12]);self.assertEqual(sorted(r for a in eviction for r in a['selected_actual_RIDs']),list(range(3,15)));self.assertTrue(all(a['cached_state_from_other_actor_allowed']is False for a in declared['actors']))
 def test_complete_named_families_restart_and_persisted_not_subset(self):
  declared=s.declaration(self.case());self.assertEqual({a['scenario']for a in declared['actors']},set(s.SCENARIOS));self.assertEqual(declared['actors'][-1]['name'],'restart-shared-capture0');self.assertTrue(declared['serial_persisted_family_required']);self.assertFalse(declared['full_cache_runtime_qualified']);self.assertTrue(declared['different_model_live_cache_refusal_separate_from_wrong_disk_fingerprint'])
 def test_actual_fourteen_shape_cannot_drop_uncaptured_requests(self):
  declared=next(a for a in s.declaration(self.case())['actors']if a['scenario']=='eviction');plan={'scenario':'eviction','capture_pass':0,'scenario_binding':{'selected_capture_pass':{'selected_actual_RIDs':declared['selected_actual_RIDs']}}};child={'phase_receipts':[{'phase':{'name':n},'work':[]}for n in declared['entire_actor_phase_roster']]}
  with patch.object(s.ctrl,'read',side_effect=[plan,child]),self.assertRaises(ValueError):s.actual_actor_shape('/tiny',declared)
 def test_zero_or_single_victim_kind_cannot_qualify_whole_eviction(self):
  logical={'victims':[{'actual_victim':{'snapshot_bytes':12}}],'real_parked_victim_observed':True,'real_checkpoint_victim_observed':False};proof={'actual_recollected_resource_accounting':{'actual_logical_source39_accounting':logical}}
  with self.assertRaises(ValueError):s.eviction_victim_binding(proof)
  logical['real_checkpoint_victim_observed']=True;self.assertFalse(s.eviction_victim_binding(proof)['physical_residency_qualified'])
  logical['victims']=[]
  with self.assertRaises(ValueError):s.eviction_victim_binding(proof)
if __name__=='__main__':unittest.main()
