"""Recipe and control routing only; no process/host/model/GPU observations."""
import ast,unittest
from pathlib import Path
import run_cpu_overlap_observed_retry_v1 as retry
class Controls(unittest.TestCase):
 def test_same_exact_frozen_recipe_and_only_new_output(self):
  c=retry.command(Path('/tmp/new-screen'));self.assertEqual(c[-3:],['--output','/tmp/new-screen','--leased']);self.assertEqual(c.count('--leased'),1)
  self.assertNotIn('--port',c);self.assertNotIn('--wrapper-preflight',c);self.assertEqual(c[c.index('--fixture')+1],str(retry.BASE/'api-positive-overlap-authentic-corpus-v1'))
 def test_pair_inherited_fds_only(self):
  t=ast.parse(Path(retry.__file__).read_text());calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('run','Popen')]
  self.assertEqual(len(calls),3)
  for c in calls:self.assertTrue(any(k.arg=='pass_fds' and isinstance(k.value,ast.Name) and k.value.id=='fd' for k in c.keywords))
 def test_literal_observer_plan_pin(self):
  self.assertEqual(retry.observer.sha(retry.HERE/'cpu-swap-attribution-source-plan-v3.json'),retry.OBS_SHA)
if __name__=='__main__':unittest.main()
