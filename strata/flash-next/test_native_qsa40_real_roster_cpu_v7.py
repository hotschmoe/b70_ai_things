"""Authentic failed V3 plan metadata only; no SDK/model/capture bytes read."""
import ast,unittest
from pathlib import Path
from unittest.mock import patch
import native_qsa40_byte_epoch_v7 as e
from serial37_canonical_json_v3 import read_unique
import qualify_native_qsa40_v7 as parent
ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/native-qsa40-v3-one-run-v1')
class Controls(unittest.TestCase):
 def test_actual_plan_complete_metadata_roster_shell_not_evidence(self):
  plan=read_unique(ROOT/'plan.snapshot.json')
  with patch.object(e,'current_row',side_effect=AssertionError('No whole-byte reads permitted')):
   roots,files=e.roots_and_files(plan,[]);self.assertEqual(len(roots),15);self.assertNotIn(Path('/bin/bash'),files);self.assertIn(Path(plan['own_producer_root']),roots);self.assertTrue(any(path.name=='num10-onecard-run'for path in roots));self.assertTrue(any(path.name=='p30v4-onecard-run'for path in roots));self.assertTrue((Path(plan['own_producer_root'])/'reference-work/work-report.json').is_file())
   obj=object.__new__(e.ByteEpoch);obj.roots=roots;obj.files=files;obj.model_paths=[];roster,links=obj.layout();self.assertEqual(links,[]);self.assertEqual(len(roster),18689);self.assertLessEqual(sum(path.stat().st_size for path in roster),e.MAX_TOTAL_BYTES);self.assertGreater((Path(plan['own_producer_root'])/'report.json').stat().st_size,16<<20)
 def test_actual_cheap_preflight_catches_before_full_semantics(self):
  plan=read_unique(ROOT/'plan.snapshot.json')
  with patch.object(e,'current_row',side_effect=AssertionError('No whole payload hashes')):result=e.metadata_preflight(plan['prepared'],plan['own_producer_root'])
  self.assertFalse(result['full_payload_bytes_read']);self.assertEqual(result['layout_files'],18689);self.assertEqual(len(result['roots']),15)
 def test_lease_exec_precedes_any_actual_manifest_in_source(self):
  tree=ast.parse(Path(parent.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');calls=[n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)];execute=next(n.lineno for n in calls if n.func.attr=='execv');manifest=next(n.lineno for n in calls if n.func.attr=='manifest');self.assertLess(execute,manifest);self.assertEqual(sum(n.func.attr=='execv'for n in calls),1)
if __name__=='__main__':unittest.main()
