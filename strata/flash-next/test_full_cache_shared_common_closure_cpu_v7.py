"""Real public-reader call-routing/return-shape source regressions."""
import ast,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import full_cache_shared_metadata_retirement_v7 as m

class Closure(unittest.TestCase):
 def test_main_public_reader_uses_same_closed_health_handoff(self):
  import validate_full_cache_shared_runtime_v7 as main
  source=Path(main.__file__).read_text();self.assertIn("parent,plan,child=admit(root,ctrl",source);self.assertNotIn("str(external.resolve())",source)
  import validate_full_cache_shared_batch0_persisted_v7 as persisted
  tree=ast.parse(Path(persisted.__file__).read_text());assignments=[n for n in ast.walk(tree)if isinstance(n,ast.Assign)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='admit'];self.assertEqual(len(assignments),1);self.assertEqual(len(assignments[0].targets[0].elts),3)
 def test_actual_successful_metadata_removal_then_unknown_absence_only_retries_absence(self):
  calls=[];commands=[];obj={'Id':'original','State':{'Running':False}};ledger={};attempts=[obj,ValueError('temporary transport error'),None]
  def inspect(name):
   value=attempts.pop(0);calls.append(value)
   if isinstance(value,Exception):raise value
   return value
  def run(command,**kwargs):commands.append(command);return types.SimpleNamespace(returncode=0)
  with tempfile.TemporaryDirectory()as tmp:result=m.retire('owned',lambda o:None,Path(tmp),ledger,inspect,run,lambda t:None)
  self.assertEqual(result['Id'],'original');self.assertEqual(commands,[['docker','rm','owned']]);self.assertEqual(ledger['retirement_failure_count'],1);self.assertTrue(ledger['owned_terminal_and_removed'])
