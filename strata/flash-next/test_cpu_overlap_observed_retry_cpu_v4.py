"""Recipe and control routing only; no process/host/model/GPU observations."""
import ast,unittest
from pathlib import Path
import run_cpu_overlap_observed_retry_v4 as retry
class Controls(unittest.TestCase):
 def test_same_exact_frozen_recipe_and_only_new_output(self):
  c=retry.command(Path('/tmp/new-screen'),{'fixture_root':'/tmp/new-fixture','_prepared_path':'/tmp/prepared-plan'});self.assertEqual(c[-3:],['--output','/tmp/new-screen','--leased']);self.assertEqual(c.count('--leased'),1)
  self.assertNotIn('--port',c);self.assertNotIn('--wrapper-preflight',c);self.assertEqual(c[c.index('--fixture')+1],'/tmp/new-fixture')
 def test_pair_inherited_fds_only(self):
  t=ast.parse(Path(retry.__file__).read_text());calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('run','Popen')]
  self.assertEqual(len(calls),3)
  for c in calls:self.assertTrue(any(k.arg=='pass_fds' and isinstance(k.value,ast.Name) and k.value.id=='fd' for k in c.keywords))
 def test_literal_observer_plan_pin(self):
  self.assertEqual(retry.observer.sha(retry.HERE/'cpu-swap-attribution-finite-source-plan-v5.json'),retry.OBS_SHA)
 def test_retirement_timeout_retains_join_and_records_failure(self):
  from unittest.mock import Mock
  import subprocess
  proc=Mock(pid=123);proc.poll.side_effect=[None,0,0];proc.wait.side_effect=[subprocess.TimeoutExpired('fixture',30),0];report={'errors':[]}
  retry.retire(proc,'observer',report);self.assertEqual(proc.wait.call_args_list[0].kwargs,{'timeout':30});self.assertEqual(proc.wait.call_args_list[1].kwargs,{})
  self.assertTrue(report['errors']);self.assertTrue(report['owned_retirement']['observer']['terminal_confirmed'])
 def test_idle_spawn_inside_protected_scope_and_handlers_before_try(self):
  t=ast.parse(Path(retry.__file__).read_text());main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main');scope=next(n for n in main.body if isinstance(n,ast.Try))
  self.assertIn('idle_proc = subprocess.Popen',ast.unparse(scope));self.assertTrue(any('retire(proc, label, report)' in ast.unparse(n) for n in scope.finalbody))
  before=main.body[:main.body.index(scope)];self.assertTrue(any('signal.signal' in ast.unparse(n) for n in before))
if __name__=='__main__':unittest.main()
