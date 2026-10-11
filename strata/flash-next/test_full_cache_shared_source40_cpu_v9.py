import ast,copy,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_runtime_v9 as shared
import full_cache_shared_batch0_persisted_v9 as serial
import full_cache_shared_batch0_fresh40_v9 as fresh
import qualify_full_cache_shared_runtime_v9 as parent

class Source40(unittest.TestCase):
 def cfg(self):return {'args':['--batch','0','--max-context','2048','--prefill','64','--prompt-cache','0','--suffix-draft','0','--lookup-chain','0','--conversation-cache-mib','0'],'env':{}}
 def test_exact_corrected_engine_and_current1403_not1392(self):
  self.assertEqual(shared.ENGINE_SHA,'d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6');self.assertTrue(str(shared.baseline.__file__).endswith('full_cache_shared_baseline_admission_v9.py'));self.assertNotIn('c139',Path(shared.__file__).read_text())
 def test_whole_batch0_phase_roster_policy_before_observation(self):
  case={'phases':{'prime_shared':{'rows':[{'ids':list(range(293)),'messages':[{'role':'user','content':'authentic placeholder fixture'}]}]}}};plan=serial.schedule(case);self.assertEqual(plan['preregistered_phase_order'],['prime','save','wrong_restore','after_wrong','valid_restore','after_valid']);self.assertEqual([plan[n]['expected_reused'] for n in ('prime','after_wrong','after_valid')],[0,292,292]);self.assertFalse(plan['restored_or_other_actor_state_used_by_fresh_control'])
 def test_serial_recipe_does_not_forge_batch_identity_or_enableFC38(self):
  with patch.object(shared,'read',return_value=self.cfg()):args,env=serial.recipe(Path('/tiny-prepared'))
  self.assertEqual(args[args.index('--batch')+1],'0');self.assertEqual(env['STRATA_FULL_CACHE_OBSERVER38'],'0');self.assertEqual(env['STRATA_BATCH_FIDELITY_DIAG'],'0');self.assertEqual(env['STRATA_PREFIX_LIFECYCLE_DIAG'],'1');self.assertNotIn('STRATA_FULL_CACHE_CAPTURE_RIDS38',env);self.assertEqual(env['STRATA_QSA3_TARGET'],'0')
 def test_QSA40_capture_controls_presence_rejected(self):
  cfg=self.cfg();cfg['env']['STRATA_QSA3_TARGET_DIR']=''
  with patch.object(shared,'read',return_value=cfg),self.assertRaises(ValueError):serial.recipe(Path('/tiny-prepared'))
 def test_purpose_fresh_uses_PCL_ordinal_not_synthetic_RID(self):
  groups=[{'input_ids':[11,12],'source_lineage':{'actual_native_PCL_pid':17,'actual_native_PCL_request_ordinal':i,'actual_HTTP_call':i+3}} for i in (1,2,3)];roster=fresh.job_roster(groups);self.assertEqual(len(roster['jobs']),1);self.assertEqual(len(roster['cached_group_associations']),3);self.assertTrue(all(a['strict_batch_RID_claimed'] is False and 'rid' not in a for a in roster['cached_group_associations']));self.assertFalse(roster['cached_state_imported'])
 def test_current_parent_bytes_health_handshake_and_native_body_AST(self):
  body=parent.adapted_source();ast.parse(body);self.assertIn('full_cache_shared_runtime_v9 as ctrl',body);self.assertIn("'--expected-plan-sha256',snapshot.sha256",body);self.assertIn("health_path=health('leaf')",body)
  for name in ('run_full_cache_shared_batch0_persisted_v9.py','validate_full_cache_shared_batch0_persisted_v9.py'):ast.parse(Path(shared.HERE,name).read_bytes())
if __name__=='__main__':unittest.main()
