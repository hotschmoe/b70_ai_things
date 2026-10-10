"""CPU synthetic controls, no model, Docker or device boundaries executed."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import api_fresh_migration_contract_v1 as contract
import batch_api_fresh_migration_v1 as ctrl
import run_batch_api_fresh_migration_v1 as runner
import registry_append_association_v1 as registry
from audit_batch_numerical_suite_v7 import snapshot_plan_join

class Controls(unittest.TestCase):
 def setUp(self):
  self.args=['--batch','2','--batch-groups','1','--prompt-cache','3','--adapt-every','0','--adapt-swaps','0','--no-prefill-borrow','--kv','fp16','--conversation-cache-mib','0'];self.env=dict(contract.PROFILE_ENV);self.policy=[{'logical_index':i,'role':'admission','values':{'reused':0,'read_from':0,'reread_to':-1}} for i in range(2)]+[{'logical_index':1,'role':'solo_migration','values':{'reused':0,'read_from':0,'reread_to':-1}}]
 def preflight(self,args=None,env=None,policy=None):return contract.preflight(self.args if args is None else args,self.env if env is None else env,contract.REQUEST_POLICY,self.policy if policy is None else policy,2)
 def test_profile_pass_has_explicit_unobserved_geometry_and_cachedhandoff(self):
  r=self.preflight();self.assertFalse(r['actual_cached_state_handoff_qualified']);self.assertFalse(r['geometry_budget_runtime_qualified'])
 def test_cache_zero_before_any_expensive_boundary(self):
  a=list(self.args);a[a.index('--prompt-cache')+1]='0'
  with self.assertRaises(ValueError):self.preflight(a)
 def test_inherited_limits_elastic_and_presence_toggles_refused(self):
  for key,value in [('STRATA_BATCH_CHAIN_MIB','0'),('STRATA_BATCH_TRANSFER_MIB','1'),('STRATA_KV_GROW','1'),('STRATA_CKPT_REREAD','0'),('STRATA_VERIFY_EAGER','0'),('STRATA_PARALLEL_SOLO','0')]:
   with self.subTest(key=key),self.assertRaises(ValueError):self.preflight(env=dict(self.env,**{key:value}))
 def test_unsupported_stage_async_routes(self):
  for more in (['--adapt-async','0'],['--layer-split','32','--split-device','0'],['--layer-split','16','--split-device','1'],['--pipeline-windows','0'],['--peer-device','1']):
   with self.subTest(more=more),self.assertRaises(ValueError):self.preflight(self.args+more)
  self.preflight(self.args+['--layer-split','32','--split-device','1'])
 def test_counters_never_relaxed(self):
  p=copy.deepcopy(self.policy);p[-1]['values']['reused']=1
  with self.assertRaises(ValueError):self.preflight(policy=p)
 def test_actualstartup_exact_geometry_binding(self):
  line='BCHAIN startup slots=2 cap=3 point_bound=100 required=1800 chain_limit=8589934592 transfer_limit=536870912 public_pin_fresh=unsupported';self.assertEqual(contract.actual_startup(line,2)['required'],'1800')
  for modified in (line.replace('1800','1700'),line.replace('slots=2','slots=4'),line.replace('536870912','1'),line+'\n'+line):
   with self.assertRaises(ValueError):contract.actual_startup(modified,2)
 def test_actual_fresh_all_warm_target_and_migration_legs(self):
  events=[{'kind':'native_send','call':1,'line':'BGEN 0 1 32 fresh=1 1,2'},{'kind':'native_send','call':1,'line':'GEN 64 fresh=1 1,2,3'},{'kind':'native_send','call':2,'line':'GEN 32 fresh=1 8'}];self.assertEqual(contract.actual_fresh_legs(events,{1,2})['leg_counts'],{'1':2,'2':1})
  for bad in ('GEN 64 fresh=0 1','GEN 64 fresh=1 fresh=0 1','GEN 64 fresh=1 pin=0 1','GEN 64 1'):
   with self.assertRaises(ValueError):contract.actual_fresh_legs([{'kind':'native_send','call':1,'line':bad}],{1})
  with self.assertRaises(ValueError):contract.actual_fresh_legs(events,{1,2,3})
 def test_registry_exact_append_and_old_digest_not_current(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'models.yaml';p.write_bytes(registry.BASELINE.read_bytes()+registry.APPEND.read_bytes());r=registry.association(p);self.assertFalse(r['old_global_digest_still_current']);self.assertEqual(len(r['new_aliases']),12);p.write_bytes(p.read_bytes().replace(b'endpoint:',b'changed:',1))
   with self.assertRaises(ValueError):registry.association(p)
 def test_missing_registration_rejects_without_modelhash(self):
  with patch.object(contract,'ROOT',Path('/CPU_ABSENT')):
   with self.assertRaises(FileNotFoundError):contract.registry_gate('unregistered')
 def test_completed_report_realSHA_before_sealing_strict_snapshot_endtoend(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'child';out.mkdir();plan={'schema':5,'harness_generation':8,'CPU_SYNTHETIC':True};external=root/'external.json'
   for path in (external,root/'input-plan.snapshot.json',out/'plan.snapshot.json'):ctrl.write(path,plan)
   seen=[]
   def bindings(directory):
    seen.append(True);return {'CPU_ASSERT_SNAPSHOT':ctrl.sha(directory/'plan.snapshot.json')}
   with patch.object(runner,'artifact_bindings',bindings):child=runner.seal_report(plan,out,{'collection_and_teardown_passed':True})
   parent={'plan':str(external),'plan_sha256':ctrl.sha(external)};self.assertEqual(snapshot_plan_join(root,parent,plan,child),ctrl.sha(external));self.assertTrue(seen);self.assertEqual(child['schema'],5);self.assertFalse(child['actual_cached_state_handoff_qualified'])
   child.pop('plan_sha256')
   with self.assertRaises(KeyError):snapshot_plan_join(root,parent,plan,child)
 def test_completed_report_snapshot_mismatch_refused_before_seal(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t);ctrl.write(out/'plan.snapshot.json',{'bad':True})
   with patch.object(runner,'artifact_bindings') as seal:
    with self.assertRaises(ValueError):runner.seal_report({'CPU':True},out,{})
    seal.assert_not_called()
 def test_controller_completed_run_to_strictsnapshot_with_mock_device_boundaries(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'child';plan={'schema':5,'harness_generation':8,'diagnostic':1,'CPU_SYNTHETIC':True};external=root/'external.json';ctrl.write(external,plan);ctrl.write(root/'input-plan.snapshot.json',plan)
   def external_run(actual,directory,health,diagnostic):
    directory.mkdir();ctrl.write(directory/'plan.snapshot.json',actual)
    with patch.object(runner,'artifact_bindings',return_value={'CPU_ONLY':True}):return runner.seal_report(actual,directory,{'collection_and_teardown_passed':True})
   with patch.object(ctrl,'manifest_binding',return_value={'CPU_ONLY':True}),patch.object(ctrl.c1,'leased') as lease,patch.object(ctrl.api,'run',external_run),patch.object(ctrl.signal,'signal'):
    self.assertEqual(ctrl.run(SimpleNamespace(plan=external,output=out,pre_health=root/'CPU-health.json')),0)
   child=ctrl.read(out/'report.json');snapshot_plan_join(root,{'plan':str(external),'plan_sha256':ctrl.sha(external)},plan,child);lease.assert_called_once_with([0,1])
 def test_exacthealth_positive_and_changedcommand_negative(self):
  import validate_batch_api_fresh_migration_v1 as reader
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);commands=[[str(ctrl.ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',reader.shared_parent.HEALTH],[str(ctrl.ROOT/'bin/xpu-collective-health'),'--img',reader.shared_parent.HEALTH,'--p2p','0','--timeout','180']];rows=[]
   for command,label in zip(commands,('strict','compiled-pair')):
    log=root/('pre-'+label+'.log');log.write_text('CPU_SYNTHETIC_HEALTH\n');cmd=root/('pre-'+label+'.command.json');ctrl.write(cmd,command);rows.append({'path':str(log),'return_code':0,'error':None,'sha256':ctrl.sha(log),'command':command,'command_file_sha256':ctrl.sha(cmd)})
   health={'passed':True,'cards':[0,1],'health_image':reader.shared_parent.HEALTH,'files':rows};ctrl.write(root/'pre-health.json',health);reader.exact_health_gate(root,'pre');rows[1]['command'][-1]='999';ctrl.write(root/'pre-health.json',health)
   with self.assertRaises(ValueError):reader.exact_health_gate(root,'pre')
 def test_prepare_invalid_profile_rejected_before_frozen_baseline(self):
  from types import SimpleNamespace
  args=list(self.args);args[args.index('--prompt-cache')+1]='0'
  with patch.object(ctrl,'read',return_value={'slots':2,'actual_counter_policy':self.policy}),patch.object(ctrl,'recipe',return_value=(args,self.env)),patch.object(ctrl.base,'prepare') as baseline,patch.object(ctrl.c1,'leased') as lease:
   with self.assertRaises(ValueError):ctrl.prepare(SimpleNamespace(kind='api',lane='source35',spec=Path('/CPU'),prepared=Path('/CPU'),diagnostic=1))
   baseline.assert_not_called();lease.assert_not_called()
 def test_actual_allstage_resets_each_native_leg_and_count(self):
  events=[{'kind':'native_send','call':1,'line':'GEN 32 fresh=1 rid=7 1,2'}];line='BCPUBLIC reset rid=7 slotgen=1 stages=2 fresh=1 pin_present=0 pin=0 lookup_ceiling=0 read_from=0';self.assertTrue(contract.actual_resets(line,events,{1},2)['all_stage_zero_resets_observed'])
  for bad in (line.replace('stages=2','stages=1'),line.replace('fresh=1','fresh=0'),line+'\n'+line,''):
   with self.assertRaises(ValueError):contract.actual_resets(bad,events,{1},2)
 def test_named_historical_registry_view_only_one_field_original_preserved(self):
  plan={'registry_binding':{'path':'CPU_SYNTHETIC','sha256':registry.BASELINE_SHA},'CPU_RAW_BYTES_UNCHANGED':'original'};original=copy.deepcopy(plan);seen=[]
  def gate(view):seen.append(copy.deepcopy(view));return {'CPU_OTHER_GATES':True}
  proof={'actual_current_global_sha256':'NEW_ACTUAL_DIGEST','old_global_digest_still_current':False}
  with patch.object(registry,'historical_entry_association',return_value=proof),patch.object(ctrl.base,'manifest_binding',gate):r=registry.admit_historical_v7_plan(plan)
  self.assertEqual(plan,original);want=copy.deepcopy(original);want['registry_binding']['sha256']='NEW_ACTUAL_DIGEST';self.assertEqual(seen,[want]);self.assertFalse(r['original_global_gate_passed']);self.assertFalse(r['measurement_requalified_by_registry_association'])
 def test_native_command_unknown_recipe_mutation_detectable(self):
  # Exact immutable command is reconstructed by the final reader, including PID.
  source=Path(runner.__file__).read_text();self.assertIn('command=command_recipe(plan,out,os.getpid())',source);self.assertIn('actual_startup(trace_path.read_text(),slots)',source)
if __name__=='__main__':unittest.main()
