"""Source-only synthetic shared-prefix/cache-policy/memory accounting controls."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_admission_v2 as q
import full_cache_checkpoint_victim_admission_v1 as victim
import full_cache_memory_accounting_v1 as memory
class Controls(unittest.TestCase):
 def fixture(self):
  common=[248045,1,2,248046,248045];return {phase:[{'ids':common+[t,248046,248045,9]} for t in ts] for phase,ts in [('prime_shared',[10]),('target_shared',[11,12]),('fresh_shared',[11,12]),('prefill_cancel',[13,14]),('decode_cancel',[15,16])]}|{'independent':[{'ids':[248045,3,4,248046,248045,20,248046,248045,9]}],'eviction_unpinned':[{'ids':[248045,5,6,248046,248045,21,248046,248045,9]}]}
 def test_shared_boundary_is_actualacceptedtokens_and_divergence(self):
  r=q.shared_boundary(self.fixture());self.assertEqual(r['tokens'],4);self.assertEqual(r['ids'],[248045,1,2,248046]);self.assertTrue(r['runtime_pin_creation_and_reuse_still_required'])
 def test_wrong_prefix_or_shared_independent_or_identical_suffix_refused(self):
  for phase in ('target_shared','independent'):
   f=self.fixture()
   if phase=='target_shared':f[phase][0]['ids'][1]=99
   else:f[phase][0]['ids']=list(f['prime_shared'][0]['ids'])
   with self.assertRaises(ValueError):q.shared_boundary(f)
  f=self.fixture();f['target_shared'][1]=copy.deepcopy(f['target_shared'][0])
  with self.assertRaises(ValueError):q.shared_boundary(f)
 def test_actualbody_conservative_scope_exactbool_pin_not_bool(self):
  self.assertEqual(q.body_policy(False,4),{'strata_fresh':False,'strata_shared_prefix':{'tokens':4}})
  for fresh,pin in [(0,4),(False,True),(True,-1)]:
   with self.assertRaises(ValueError):q.body_policy(fresh,pin)
 def test_realreuse_and_newcounts_require_exactevaluatedtail(self):
  ids=[1,2,3,4,5];r=q.cache_work(ids,3,False,3,3,-1,[4,5]);self.assertEqual(r['actual_new_prompt_tokens'],2);q.cache_work(ids,3,True,0,0,-1,ids)
  for evaluated in (ids,[3,4,5],[4]):
   with self.assertRaises(ValueError):q.cache_work(ids,3,False,3,3,-1,evaluated)
 def test_no_authentic_fixture_failclosed_without_runtime(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(FileNotFoundError):q.fixture_admission(Path(t))
 def test_exactCPUrecipe_no_modelpayload_devices_and_no_shadowing(self):
  c=q.fixture_command('/CPU_ROOT','b70-full-cache-tokenizer-v2-77');self.assertNotIn('--device',c);self.assertFalse(any('.gguf' in w for w in c));self.assertEqual(c[-1],'/harness/render_full_cache_fixture_v2.py');self.assertNotIn('/harness/tokenize.py',c)
 def test_sourceLRUpinnedroot_tail_backstop_policy(self):
  p=[{'used':10,'tail':False,'pinned':True},{'used':1,'tail':False,'pinned':True},{'used':2,'tail':True,'pinned':False},{'used':20,'tail':False,'pinned':False}];self.assertEqual(victim.source_victim(p,3),2);p[2]['tail']=False;self.assertEqual(victim.source_victim(p,3),2);p[2]['pinned']=p[3]['pinned']=True;self.assertEqual(victim.source_victim(p,3),1)
 def test_actualvictim_not_predicted_without_afteridentity_witness(self):
  points=[{'index':i,'tokens':i+1,'sha256_le32':str(i),'used':stamp,'pinned':i==0,'tail':False,'point_capacity_bytes':100,'stages':1} for i,stamp in enumerate([9,1,2,3])];before={'phase':'before_eviction','actor':'main','slot':-1,'rid':7,'generation':1,'cap':3,'chain_capacity_bytes':500,'actual_victim':1,'points':points};after=dict(before,phase='after_eviction',actual_victim=-1,chain_capacity_bytes=400,points=[dict(p,index=i) for i,p in enumerate(points[:1]+points[2:])]);r=victim.actual_eviction(before,after);self.assertFalse(r['source35_existing_uninstrumented_runtime_proof'])
  bad=copy.deepcopy(before);bad['actual_victim']=2
  with self.assertRaises(ValueError):victim.actual_eviction(bad,after)
  bad=copy.deepcopy(after);bad['points'][1]['sha256_le32']='WRONG'
  with self.assertRaises(ValueError):victim.actual_eviction(before,bad)
 def test_logical_owned_peaks_keep_state_and_expert_categories_distinct(self):
  private={'passed':True,'owners':[{'allocations':[{'role':'slot_arena','alloc_line':1,'free_line':9,'bytes':100}]}]};mirrors={'passed':True,'owners':[{'allocations':[{'role':'host_segment','line':2,'free_line':8,'bytes':200}]}]}
  with patch.object(memory,'slot_audit',return_value=private),patch.object(memory,'mirror_audit',return_value=mirrors):r=memory.account('',[(0,0,48)],2)
  self.assertEqual(r['owned_logical_categories']['slot_state:slot_arena']['actual_peak_live_owned_extent_bytes'],100);self.assertEqual(r['owned_logical_categories']['expert_mirror:host_segment']['actual_peak_live_owned_extent_bytes'],200);self.assertFalse(r['expert_layer_residency_qualified']);self.assertFalse(r['PCIe_traffic_inferred']);self.assertFalse(r['physical_RSS_device_residency_or_reclamation_qualified'])
 def test_memory_requires_actual_contextfree_and_boundcapacity(self):
  with self.assertRaises(ValueError):memory.peaks([{'bytes':100,'alloc_line':3,'free_line':2}])
 def test_combined_freshpin_case_refusal_is_not_false_backendclaim(self):
  with self.assertRaisesRegex(ValueError,'sourceimplemented'):q.body_policy(True,4)
  self.assertEqual(q.body_policy(True),{'strata_fresh':True})
 def test_stale_startup_hint_never_overrides_actual_consumed_source(self):
  sdk=q.SDK/'source';r=q.source_support_binding(sdk/'sycl/src/program/generate.cpp',sdk/'sycl/include/strata/core/batch_public_prefix.hpp');self.assertFalse(r['startup_hint_is_runtime_support_authority']);self.assertTrue(r['combinedfreshpin_source_implemented']);self.assertFalse(r['combinedfreshpin_runtime_qualified'])
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'misleading.cpp';path.write_text('public_pin_fresh=unsupported')
   with self.assertRaises(ValueError):q.source_support_binding(path,sdk/'sycl/include/strata/core/batch_public_prefix.hpp')
if __name__=='__main__':unittest.main()
