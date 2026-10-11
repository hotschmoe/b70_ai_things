import ast,copy,unittest
from pathlib import Path
import current_first_gdn_bulk_observation_v1 as a
class Controls(unittest.TestCase):
 def test_only_return_observation_added(self):
  before=copy.deepcopy(next(n for n in ast.walk(ast.parse(Path(a.original.__file__).read_bytes())) if isinstance(n,ast.FunctionDef)and n.name=='mixer'));fn,after=a.observed_mixer();self.assertEqual(len(after.body),len(before.body)+1);self.assertIsInstance(after.body[-2],ast.Assign);del after.body[-2];self.assertEqual(ast.dump(after),ast.dump(before));self.assertIs(fn.__globals__['np'],a.original.np)
 def test_primary_signature_has_no_target(self):
  import inspect
  self.assertEqual(list(inspect.signature(a.independent_first).parameters),['model'])
if __name__=='__main__':unittest.main()
