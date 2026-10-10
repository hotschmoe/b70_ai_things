"""NEW CPU authentic-fixture/source metadata and mocked external-boundary controls."""
import copy,tempfile,unittest,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import api_cache_positive_contract_v2 as q
import batch_api_cache_positive_v2 as ctrl
import generate_api_cache_positive_case_v2 as generator
import run_batch_api_cache_positive_v2 as runner
import registry_cache_positive_association_v2 as registry
from audit_batch_numerical_suite_v7 import snapshot_plan_join

class Controls(unittest.TestCase):
 def setUp(self):
  self.fixture,self.binding=q.fixture_binding();self.args=['--prefill','64','--batch','2','--batch-groups','1','--prompt-cache','3','--adapt-every','0','--adapt-swaps','0','--no-prefill-borrow','--kv','fp16','--conversation-cache-mib','0']+sum(([k,v] for k,v in q.CHECKPOINT_ARGS.items()),[]);self.env=dict(q.PROFILE_ENV,ZE_AFFINITY_MASK='0');self.case=generator.case_from_fixture(18602)
 def test_genuine_fixture_and_accepted_lengths_exceed_lcp(self):
  r=q.checkpoint_nonreuse(self.fixture['fixtures'],self.args,self.env);self.assertEqual([r['minimum_accepted_checkpoint_length'] for r in r['rows']],[224,224,228,228]);self.assertEqual([r['possible_prompt_checkpoint_accepted_lengths'][-1] for r in r['rows']],[230,230,234,234]);self.assertEqual({r['actual_lcp_tokens'] for r in r['pairwise_actual_LCP']},{3,9});self.assertTrue(all(r['minimum_source_reusable_accepted_tokens']==64 for r in r['pairwise_actual_LCP']));self.assertFalse(r['actual_initial_nonreuse_runtime_qualified'])
 def test_inherited_checkpoint_overrides_and_presencezero_refused(self):
  for key in q.CHECKPOINT_ARGS:
   args=list(self.args);args[args.index(key)+1]='0'
   with self.subTest(key=key),self.assertRaises(ValueError):q.lock_checkpoint_defaults(args,self.env)
  for key in ('STRATA_CACHE_MESSAGE_BOUNDARY','STRATA_SPLIT_SMALL_OWN','STRATA_SPLIT_SMALL_MAX','STRATA_PREFILL_EQUAL'):
   with self.subTest(key=key),self.assertRaises(ValueError):q.lock_checkpoint_defaults(self.args,dict(self.env,**{key:'0'}))
  with self.assertRaises(ValueError):q.lock_checkpoint_defaults(self.args+['--prompt-cache-tail'],self.env)
 def test_actual_fixture_mutation_or_shared_checkpoint_refused(self):
  f=copy.deepcopy(self.fixture['fixtures']);f['target'][1]['ids']=list(f['target'][0]['ids'])
  with self.assertRaises(ValueError):q.checkpoint_nonreuse(f,self.args,self.env)
  source=q.read(q.SOURCE_PLAN);source=copy.deepcopy(source);source['actual_tokenizer_fixture']['files']['fixtures.json']='0'*64
  with patch.object(q,'read',lambda p:source if Path(p)==q.SOURCE_PLAN else json.loads(Path(p).read_bytes())):
   with self.assertRaises(ValueError):q.fixture_binding()
 def test_positive_initialzero_and_actualpromptminusone_policy(self):
  candidate=dict(self.case,args=self.args,env=self.env);r=q.profile(candidate);self.assertFalse(r['actual_cached_state_handoff_qualified'])
  for row,key,value in [(0,'reused',1),(2,'reused',0),(2,'read_from',0),(2,'reread_to',0)]:
   changed=copy.deepcopy(candidate);changed['actual_counter_policy'][row]['values'][key]=value
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):q.profile(changed)
 def test_fixture_ids_and_literal_roles_bound_to_plan(self):
  p=dict(self.case,args=self.args,env=self.env);p=copy.deepcopy(p);p['messages']['target'][0][0]['role']='user'
  with self.assertRaises(ValueError):q.profile(p)
  p=dict(self.case,args=self.args,env=self.env);p=copy.deepcopy(p);p['api_token_ids']['target'][0][-1]=1
  with self.assertRaises(ValueError):q.profile(p)
 def test_freshfalse_phase_persistent_every_actual_native_leg(self):
  events=[{'kind':'native_send','call':7,'line':'BGEN 1 64 seed=1 rid=9 fresh=0 11,12'},{'kind':'native_send','call':7,'line':'GEN 64 seed=1 rid=9 fresh=0 11,12,13'}];self.assertEqual(q.actual_phase_legs(events,{7},False)['actual_native_leg_counts'],{'7':2})
  for line in ('GEN 64 seed=1 rid=9 fresh=1 11','GEN 64 seed=1 rid=9 fresh=0 fresh=1 11','GEN 64 seed=1 rid=9 fresh=0 ckpt=0 11','GEN 64 seed=1 rid=9 fresh=0 pin=0 11'):
   with self.assertRaises(ValueError):q.actual_phase_legs([{'kind':'native_send','call':7,'line':line}],{7},False)
 def test_initial_coldreset_allowed_but_migration_reset_refused(self):
  job={'rid':7,'role':'solo_migration','ids':[11,12,13,14],'position':3,'token':14};rows=['BCPUBLIC reset rid=7 stages=2 fresh=0 pin_present=0','SBF slot_closed pid=55 rid=7 enginegen=1 requestgen=2 slot=1 slotgen=3 cancelled=1 completed=1','SBF migration_begin pid=55 rid=7 enginegen=1 requestgen=2 slot=6 slotgen=3 source_slot=1 source_slotgen=3 prompt=4 event=2 main_session=1','BCHAIN restored slot=1 rid=7 slotgen=3 active_checkpoint=0 tokens=3 points=2 host_bytes=128','SBF resume pid=55 rid=7 enginegen=1 requestgen=2 slot=6 slotgen=3 reused=3 read_from=3 reread_to=-1'];self.assertTrue(q.last_live_handoff('\n'.join(rows),7,job,2)['initial_cold_reset_before_migration_permitted'])
  with self.assertRaises(ValueError):q.last_live_handoff('\n'.join(rows+[rows[0]]),7,job,2)
 def test_registry_exact_oldbytes_positive_append_optional_fresh(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'models.yaml';old=registry.BASELINE.read_bytes();extra=registry.APPEND.read_bytes()
   for raw in (old+extra,old+registry.FRESH_APPEND.read_bytes()+extra):p.write_bytes(raw);self.assertFalse(registry.association(p)['old_V7_global_gate_passed'])
   p.write_bytes(old.replace(b'endpoint:',b'changed:',1)+extra)
   with self.assertRaises(ValueError):registry.association(p)
 def test_prepare_counter_refusal_before_old_admission_registry_or_lease(self):
  changed=copy.deepcopy(self.case);changed['actual_counter_policy'][-1]['values']['reused']=0
  with patch.object(ctrl,'read',return_value=changed),patch.object(ctrl,'recipe',return_value=(self.args,self.env)),patch.object(ctrl.base,'prepare') as base,patch.object(q,'registry_gate') as registration,patch.object(ctrl.c1,'leased') as lease:
   with self.assertRaises(ValueError):ctrl.prepare(SimpleNamespace(kind='api',lane='source35',spec=Path('/CPU'),prepared=Path('/CPU'),diagnostic=1))
   base.assert_not_called();registration.assert_not_called();lease.assert_not_called()
 def test_prepare_endtoend_named_case_view_restored_and_strict_producer_snapshot(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);prepared=root/'prepared';prepared.mkdir();spec=root/'case.json';ctrl.write(spec,self.case);cfg={'args':self.args,'env':self.env};ctrl.write(prepared/'server-config.json',cfg);ctrl.write(prepared/'prepared.json',{'runtime':{'image':'CPU_SYNTHETIC_IMAGE'},'pack':'CPU_SYNTHETIC_PACK'});out=root/'prepared-plan';seen=[];old_read=ctrl.base.read;old_api=ctrl.base.api;old_manifest=ctrl.base.manifest_binding
   def original_prepare(a):
    view=ctrl.base.read(a.spec);seen.append(view);self.assertEqual(view['schema'],2);self.assertEqual(view['actual_counter_policy'][-1]['values']['reused'],0)
    args,env=ctrl.recipe(a.prepared,2,True);alias=ctrl.base.api.experimental_alias({'args':args,'env':env},2,True,'source35');plan={'kind':'api','slots':2,'prepared':str(prepared),'image':'CPU_SYNTHETIC_IMAGE','pack':'CPU_SYNTHETIC_PACK','args':args,'env':env,'diagnostic':1,'port':18602,'messages':view['messages'],'api_token_ids':view['api_token_ids'],'actual_counter_policy':view['actual_counter_policy'],'research_alias':alias,'registry_binding':{'CPU':True},'spec_sha256':ctrl.sha(spec)};ctrl.base.manifest_binding(plan);out.mkdir();ctrl.write(out/'plan.json',plan)
   with patch.object(ctrl.base,'prepare',original_prepare),patch.object(q,'registry_gate',return_value={'CPU':True}),patch.object(ctrl,'FROZEN_MANIFEST',return_value={'CPU_NONREGISTRY_SOURCE_GATES':True}):ctrl.prepare(SimpleNamespace(kind='api',lane='source35',spec=spec,prepared=prepared,diagnostic=1,output=out))
   self.assertIs(ctrl.base.read,old_read);self.assertIs(ctrl.base.api,old_api);self.assertIs(ctrl.base.manifest_binding,old_manifest);plan=ctrl.read(out/'plan.json');self.assertEqual(plan['actual_counter_policy'][-1]['values']['reused'],'prompt_minus_one');self.assertEqual(plan['authentic_positive_case'],self.case);child=root/'child';child.mkdir();ctrl.write(child/'plan.snapshot.json',plan);ctrl.write(root/'input-plan.snapshot.json',plan)
   with patch.object(runner,'artifact_bindings',return_value={'CPU_TINY':True}):report=runner.seal_report(plan,child,{'collection_and_teardown_passed':True})
   parent={'plan':str(out/'plan.json'),'plan_sha256':ctrl.sha(out/'plan.json')};snapshot_plan_join(root,parent,plan,report);self.assertEqual(report['schema'],6);self.assertFalse(report['actual_cached_state_handoff_qualified']);report.pop('plan_sha256')
   with self.assertRaises(KeyError):snapshot_plan_join(root,parent,plan,report)
if __name__=='__main__':unittest.main()
