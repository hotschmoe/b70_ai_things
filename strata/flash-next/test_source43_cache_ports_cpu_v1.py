"""Tiny deliberate source routes, exact registry preservation and typed gates."""
import ast,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source43_cache_ports_v1 as ports
import registry_source43_cache175_association_v1 as registry
import full_cache43_runtime_v1 as ctrl
import qualify_full_cache43_runtime_v1 as parent
class Controls(unittest.TestCase):
 def test_all_declared_ports_import(self):
  import importlib
  for old,new in ports.MODULES.items():importlib.import_module(new)
 def test_future175_preserves174_not_current_view(self):
  with tempfile.TemporaryDirectory()as td:
   path=Path(td)/'registry.yaml';path.write_bytes(registry.expected_bytes());proof=registry.association(path);self.assertEqual(proof['actual_models'],175);self.assertFalse(proof['old174_current_global_gate_passed']);path.write_bytes(registry.expected_prior_bytes());self.assertRaises(ValueError,registry.association,path)
 def test_actual_generated_parent_routes_and_three_pages(self):
  raw=parent.adapted_source();ast.parse(raw);self.assertIn('import full_cache43_runtime_v1 as ctrl',raw);self.assertIn('source43_cache_watchdog_v1',raw);self.assertNotIn('from full_cache_shared_health_handoff_v12',raw);self.assertNotIn('import full_cache_shared_runtime_v12',raw)
 def test_typed_native2_mode1_only_no_old_current_gate(self):
  with patch.object(ctrl,'source_binding',return_value={'source_generation':43}):
   for bad in(0,True,2.0,4,6):self.assertRaises(ValueError,ctrl.manifest_binding,{'schema':'full-cache-shared-runtime-v3','driver_sha256':'bad','source_binding':{},'lane':'source43','kind':'api','slots':bad,'diagnostic':1,'source_generation':43,'materialized_mode':1})
 def test_no_unmapped_V12_runtime_imports(self):
  for old,new in ports.MODULES.items():
   if not old.endswith('_v12')or old in('full_cache_shared_runtime_v12','full_cache_shared_baseline_admission_v12'):continue
   tree=ports.transform(ports.source(old))
   for node in ast.walk(tree):
    if isinstance(node,ast.Import):self.assertFalse(any(a.name in ports.MODULES for a in node.names))
    if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,ports.MODULES)
if __name__=='__main__':unittest.main()
