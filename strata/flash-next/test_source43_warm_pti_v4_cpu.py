import unittest,copy
from unittest.mock import patch
import run_source43_warm_pti_v4 as a
import validate_source43_warm_off_component_v1 as component
class Controls(unittest.TestCase):
 def test_failed_parent_retained(self):
  report={'passed':False,'errors':['actual ON helper refused'],'results':{component.PHASE:{}},'per_arm':{component.PHASE:{}}};scope=component.parent_scope(report);self.assertFalse(scope['original_parent_passed']);self.assertEqual(scope['original_parent_errors'],report['errors'])
  report['passed']=True
  with self.assertRaises(ValueError):component.parent_scope(report)
 def test_compiled_parent_routes_one_ON_and_real_baseline(self):
  self.assertEqual(a.NS['_PORTS']['PHASES'],('source43-warm-pti-ON',));self.assertIn('set_baseline',a.NS['_PORTS']['main'].__code__.co_names);self.assertIn('baseline_snapshot',a.NS['_PORTS']['main'].__code__.co_names)
  with patch.object(component,'finalized_binding',return_value={'request_views':[{'ids':[7]},{'ids':[7]}],'original_parent_passed':False,'historical_metadata_only':True,'current_model_identity_qualified':False,'current_GPU_health_transferred':False,'fresh_public_full4_rehashed':False})as call:
   a.set_baseline('/tmp/actual-failed-parent','a'*64);call.assert_called_once_with('/tmp/actual-failed-parent','a'*64);self.assertIs(a.NS['_PORTS']['baseline_snapshot'](),a.NS['_BASELINE'])
  self.assertTrue(a.baseline_equivalence([{'ids':[7]},{'ids':[7]}])['H36_OFF_ON_equivalence_passed'])
  with self.assertRaises(ValueError):a.baseline_equivalence([{'ids':[8]},{'ids':[8]}])
 def test_component_does_not_transfer_current_authority(self):
  value={'historical_metadata_only':True,'current_model_identity_qualified':False,'current_GPU_health_transferred':False,'fresh_public_full4_rehashed':False}
  with patch.object(component,'body',return_value=value),patch.object(component.actor,'admission',side_effect=AssertionError('recursive current admission forbidden')),patch.object(component.actor.life.q.upload_v2,'full_buffered_identity',side_effect=AssertionError('historical full scan forbidden')):
   self.assertIs(component.finalized_binding('/tmp/component','a'*64),value)
  import ast
  tree=ast.parse(__import__('pathlib').Path(component.__file__).read_bytes());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
  self.assertNotIn('actor.admission',calls);self.assertNotIn('actor.set_runtime',calls);self.assertFalse(any('full_buffered_identity'in name for name in calls))
  self.assertIs(a.NS['_PORTS']['admission'],a.admission)
  with patch.object(component,'finalized_binding',return_value={**value,'current_model_identity_qualified':True}):
   with self.assertRaises(ValueError):a.set_baseline('/tmp/component','a'*64)
 def test_actual_new_exec_dispatch(self):
  import pti_warm_owned_exec_v3 as new
  self.assertIs(a.NS['temporal_exec'],new);self.assertIs(a.NS['_PORTS']['temporal_exec'],new)
if __name__=='__main__':unittest.main()
