"""Tiny source-only boundaries; no model, Docker or leased actor execution."""
import argparse,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_full_cache_shared_suite_v8 as runner
import test_full_cache_shared_suite_cpu_v8 as fixtures

class Runner(unittest.TestCase):
 def test_actual_shared_prepare_directory_contract(self):
  case=fixtures.Suite().case()
  def shared(args):
   args.output.mkdir();(args.output/'plan.json').write_text(json.dumps({'scenario':args.scenario}))
  def serial(args):args.output.write_text('{}')
  with tempfile.TemporaryDirectory() as tmp,patch.object(runner.ctrl,'source_binding',return_value={'source':40}),patch.object(runner.ctrl,'authentic_case',return_value=case),patch.object(runner.ctrl,'prepare',side_effect=shared),patch.object(runner.persisted,'prepare',side_effect=serial):
   out=Path(tmp)/'new';plan=runner.prepare(argparse.Namespace(output=out,prepared=Path('/tiny/prepared'),model_identity=Path('/tiny/identity'),port=28800))
   for item in plan['actor_roots']:
    path=Path(item['plan']);self.assertEqual(path.name,'plan.json');self.assertEqual(runner.Snapshot(path,item['plan_sha256']).sha256,item['plan_sha256'])
 def test_wrong_expected_digest_fails_before_source_or_lease(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'plan.json';path.write_text('{}')
   with patch.object(runner.ctrl,'source_binding') as source,patch.object(runner.ctrl.c1,'leased') as lease,self.assertRaises(ValueError):runner.run(path,Path(tmp)/'out',True,'0'*64)
   source.assert_not_called();lease.assert_not_called();self.assertFalse((Path(tmp)/'out').exists())
 def test_actual_nested_commands_forward_captured_digest(self):
  import ast
  tree=ast.parse(Path(runner.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.List)]
  lists=[n for n in calls if any(isinstance(x,ast.Constant) and x.value=='--leased' for x in n.elts)]
  self.assertEqual(len(lists),3)
  for node in lists:self.assertTrue(any(isinstance(x,ast.Constant) and x.value=='--expected-plan-sha256' for x in node.elts))
if __name__=='__main__':unittest.main()
