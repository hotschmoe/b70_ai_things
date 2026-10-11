"""Every actual V12 producer/reader terminal and FC39 path selects new gate."""
import ast,unittest
from pathlib import Path
class Controls(unittest.TestCase):
 def test_main_persisted_and_native_lifetime_all_use_same_LF_gate(self):
  here=Path(__file__).resolve().parent
  for name in('run_full_cache_shared_runtime_v12.py','run_full_cache_shared_batch0_persisted_v12.py','validate_full_cache_shared_runtime_v12.py','validate_full_cache_shared_batch0_persisted_v12.py'):
   source=(here/name).read_text();self.assertNotIn('eos_waiter_terminal_source40_v2',source);self.assertNotIn('full_cache_shared_native_lifetime_v5',source)
  source=(here/'full_cache_shared_native_lifetime_v6.py').read_text();self.assertIn('from eos_waiter_terminal_source40_v3 import Terminals',source)
 def test_no_old_exact171_current_registry_predicate_in_new_ports(self):
  here=Path(__file__).resolve().parent
  for name in('full_cache_shared_runtime_v12.py','full_cache_shared_batch0_persisted_v12.py'):
   source=(here/name).read_text();self.assertNotIn('registry_c140_shared_association_v3 import association',source);self.assertIn('registry_c140_chunk173_association_v1 import association',source)
if __name__=='__main__':unittest.main()
