"""Actual fresh_recipe and early fixed budgets/unoptimized refusal controls."""
import copy,os,unittest
from unittest.mock import patch
import full_cache43_runtime_v2 as ctrl
import full_cache43_fresh43_v2 as fresh
class Controls(unittest.TestCase):
 def budgets(self):return {'total_text_budget_bytes':32<<30,'total_text_budget_scope':'trace_and_combined_plus_bounded_overflow_failure_record','capture_disk_quota_bytes':512<<20,'capture_disk_quota_scope':'captures_array_files_only'}
 def test_actual_fresh_recipe_invokes_all_three_set_arg_exports(self):
  plan={'args':['--batch','2','--prompt-cache','3','--conversation-cache-mib','512','--suffix-draft','0','--lookup-chain','0'],'env':{'STRATA_QSA_INDEXER_MATERIALIZED_HALF':'1','STRATA_FULL_CACHE_CAPTURE_RIDS38':'1,2'}};before=copy.deepcopy(plan);args,env=fresh.fresh_recipe(plan)
  for name in('--batch','--prompt-cache','--conversation-cache-mib'):self.assertEqual(args[args.index(name)+1],'0')
  self.assertEqual(env['STRATA_QSA_INDEXER_MATERIALIZED_HALF'],'1');self.assertEqual(env['STRATA_BATCH_FULL_STATE_CHAIN'],'0');self.assertEqual(env['STRATA_FIDELITY_DIAG'],'1');self.assertNotIn('STRATA_FULL_CACHE_CAPTURE_RIDS38',env);self.assertEqual(plan,before)
 def test_actual_manifest_rejects_mutated_budget_before_heavy_reads(self):
  changes=[('total_text_budget_bytes',1),('total_text_budget_bytes',float(32<<30)),('total_text_budget_scope','unbounded'),('capture_disk_quota_bytes',1),('capture_disk_quota_bytes',float(512<<20)),('capture_disk_quota_scope','whole_disk')]
  for key,value in changes:
   plan=self.budgets();plan[key]=value
   with patch.object(ctrl,'source_binding',side_effect=AssertionError('No source/model admission before budget refusal')):self.assertRaises(ValueError,ctrl.manifest_binding,plan)
 def test_valid_fixed_budgets(self):ctrl.budget_gate(self.budgets())
 def test_actual_manifest_and_prepare_refuse_optimized_environment(self):
  with patch.dict(os.environ,{'PYTHONOPTIMIZE':'1'}),patch.object(ctrl,'source_binding',side_effect=AssertionError('No heavy admission')):
   self.assertRaises(ValueError,ctrl.manifest_binding,self.budgets());self.assertRaises(ValueError,ctrl.prepare,None)
if __name__=='__main__':unittest.main()
