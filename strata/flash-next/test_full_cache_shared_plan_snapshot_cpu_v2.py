import ast,tempfile,unittest
from pathlib import Path
from full_cache_shared_plan_snapshot_v2 import Snapshot,write_runtime
import qualify_full_cache_shared_runtime_v2 as parent

class Plan(unittest.TestCase):
 def test_captured_bytes_survive_exact_parent_child_write(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/'plan.json';raw=b'{ "cards" : [0,1], "counter": 2 }\n';source.write_bytes(raw);s=Snapshot(source);s.write(root/'input-plan.snapshot.json');write_runtime(root/'plan.snapshot.json',s.plan,s.raw)
   self.assertEqual((root/'plan.snapshot.json').read_bytes(),raw)
 def test_changed_plan_duplicate_or_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'plan.json';p.write_text('{"a":1}');s=Snapshot(p);p.write_text('{"a":2}')
   with self.assertRaises(ValueError):s.verify()
   p.write_text('{"a":1,"a":2}')
   with self.assertRaises(ValueError):Snapshot(p)
   link=root/'alias.json';link.symlink_to(p)
   with self.assertRaises(ValueError):Snapshot(link)
 def test_actual_adapted_parent_forwards_captured_snapshot_and_sha(self):
  source=parent.adapted_source();ast.parse(source)
  self.assertIn("'--plan',str(out/'input-plan.snapshot.json'),'--expected-plan-sha256',snapshot.sha256",source)
  self.assertIn("snapshot.write(out/'input-plan.snapshot.json')",source)
  self.assertNotIn('plan=read(a.plan)',source);self.assertIn('Snapshot(a.plan,a.expected_plan_sha256)',source)
if __name__=='__main__':unittest.main()
