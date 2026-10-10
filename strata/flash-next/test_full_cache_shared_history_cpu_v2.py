import copy,json,unittest
from unittest.mock import patch
import full_cache_shared_history_v2 as h

class History(unittest.TestCase):
 def setUp(self):
  self.messages=[{'role':'user','content':'Explain the water cycle.'}]
  self.client={'request':{'messages':self.messages},'done_received':True,'cancel_requested':False,'error':None,'events':[{'data':json.dumps({'choices':[{'delta':{'content':'Water circulates.'}}]})},{'data':'[DONE]'}]}
  self.begin={'kind':'engine_begin','rendered_prompt':{'messages':self.messages,'ids':[1,2]},'submitted_ids':[1,2],'submitted_ids_sha256':h.digest([1,2]),'rendered_matches_submitted':True,'engine_pid':10,'call':3,'engine_generation':2}
  self.end={'kind':'engine_end','error':None,'cancelled':False,'engine_pid':10,'call':3,'rid':4,'engine_generation':2,'generated_ids':[9,10],'generated_ids_sha256':h.digest([9,10])}
  self.decode={'ids':[9,10],'text':'Water circulates.','engine_pid':10,'call':3,'independent_original_tokenizer_decode':True}
 def derive(self):return h.own_response_messages(self.messages,self.client,self.begin,self.end,self.decode)
 def rendered(self,d):
  req={'messages':d['messages'],'messages_sha256':d['messages_sha256'],'template_kwargs':{'enable_thinking':False}}
  out={**req,'tools':None,'ids':list(range(300)),'ids_sha256':h.digest(list(range(300))),'engine_pid':10,'original_encode_prompt_called':True}
  return h.render_binding(req,out,10)
 def test_own_reply_and_fixed_counters(self):
  original=copy.deepcopy(self.messages);d=self.derive();r=self.rendered(d)
  p=h.pinned_history_phase(d,r,list(range(293)),272)
  self.assertEqual(p['expected_reused'],[272]);self.assertEqual(p['expected_new_prompt_tokens'],[28]);self.assertEqual(self.messages,original)
  f=h.pinned_history_phase(d,r,list(range(293)),272,True)
  self.assertEqual(f['expected_reused'],[0]);self.assertNotIn('strata_shared_prefix',f['policy']);self.assertFalse(p['full_cache_runtime_qualified'])
 def test_foreign_actor_decode_and_input_rejected(self):
  for obj,key,value in [(self.begin,'engine_pid',11),(self.decode,'text','Other reply'),(self.decode,'ids',[8,10]),(self.end,'generated_ids_sha256','0'*64)]:
   before=copy.deepcopy(obj);obj[key]=value
   with self.assertRaises(ValueError):self.derive()
   obj.clear();obj.update(before)
 def test_cancelled_partial_and_after_terminal_rejected(self):
  self.client['cancel_requested']=True
  with self.assertRaises(ValueError):self.derive()
  self.client['cancel_requested']=False;self.client['events'].append({'data':json.dumps({'choices':[{'delta':{'content':'late'}}]})})
  with self.assertRaises(ValueError):self.derive()
 def test_actual_prefix_and_boundary_not_fitted(self):
  d=self.derive();r=self.rendered(d)
  for b in [271,True,300]:
   with self.assertRaises(ValueError):h.pinned_history_phase(d,r,list(range(293)),b)
  r['ids'][0]=1000;r['ids_sha256']=h.digest(r['ids'])
  with self.assertRaises(ValueError):h.pinned_history_phase(d,r,list(range(293)),272)
 def test_render_owner_template_hash_and_integer_types(self):
  d=self.derive();req={'messages':d['messages'],'messages_sha256':d['messages_sha256'],'template_kwargs':{'enable_thinking':False}};r=self.rendered(d)
  for key,value in [('engine_pid',11),('original_encode_prompt_called',False),('ids_sha256','0'*64),('tools',[])]:
   changed=copy.deepcopy(r);changed[key]=value
   with self.assertRaises(ValueError):h.render_binding(req,changed,10)
  with self.assertRaises(ValueError):h.digest([True])
 def test_history_prototype_has_no_guessed_inputs_and_prior_leaf_rule(self):
  d=self.derive();r=self.rendered(d);r['ids'][280]=248045;r['ids_sha256']=h.digest(r['ids']);original=list(range(293))
  proto=h.history_prototypes()[1];self.assertIsNone(proto['rows'][0]['ids']);proto.update(expected_actual_RID_set=[6],raw_capture_requested=True)
  inventory={'kind':'checkpoint_inventory','phase':'main_body_complete','rid':5,'pid':10,'points':[{'tokens':299,'sha256_le32':h.digest(r['ids'][:-1])},{'tokens':280,'sha256_le32':h.digest(r['ids'][:280])}]}
  phase=h.resolve_history(proto,d,r,original,inventory,5);self.assertEqual(phase['expected_reused'],[299]);self.assertEqual(phase['expected_new_prompt_tokens'],[1]);self.assertEqual(phase['expected_source_turn_boundary'],-1);self.assertEqual(phase['required_prior_turn_boundary'],280)
  for missing in [None,{**inventory,'points':inventory['points'][:1]},{**inventory,'rid':4}]:
   with self.assertRaises((ValueError,TypeError)):h.resolve_history(proto,d,r,original,missing,5)
 def test_whole_history_actor_capture_bounded_and_eviction_scope_unchanged(self):
  import full_cache_shared_phase_contract_v2 as contract
  base=[{'name':'warm','rows':[{},{}],'max_new':[32,32],'policy':{'strata_fresh':True}},{'name':'prime0','rows':[{}],'max_new':[1],'policy':{'strata_fresh':False,'strata_shared_prefix':{'tokens':272}}},{'name':'prime1','rows':[{}],'max_new':[1],'policy':{'strata_fresh':False,'strata_shared_prefix':{'tokens':272}}}]
  with patch.object(contract,'schedules',return_value=base):result=contract.scenario_schedule({},'history')
  self.assertEqual(result['capture_roster']['total_HTTP_requests'],8);self.assertEqual(result['capture_roster']['armed_requests_required'],6);self.assertEqual(result['selected_capture_pass']['selected_actual_RIDs'],[3,4,5,6,7,8]);self.assertLess(result['maximum_source_observer_bytes'],64<<20);self.assertFalse(result['shared_state_transferred_between_actors']);self.assertEqual([p['name'] for p in result['phases'][-4:]],list(h.HISTORY_PHASES))

if __name__=='__main__':unittest.main()
