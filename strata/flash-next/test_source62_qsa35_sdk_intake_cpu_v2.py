"""Actual binder metadata intake reaches later original gate; no ABI/payload proof."""
import ast,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source62_qsa35_sdk_binding_v2 as b
HERE=Path(__file__).resolve().parent
class ReachedOriginalProvenance(Exception):pass
class Controls(unittest.TestCase):
 def builder_kind(self):
  t=ast.parse((HERE/'build_native62_qsa35_incremental_v1.py').read_text());v=[n.value.value for n in ast.walk(t)if isinstance(n,ast.keyword)and n.arg=='external_source_identity_kind'and isinstance(n.value,ast.Constant)];self.assertEqual(v,['immutable_actual62_source_snapshot']);return v[0]
 def fixture(self,root,kind):
  p=json.loads(b.PLAN.read_text());r={'build_rc':0,'devices_exposed':False,'external_source_unchanged':True,'plan_snapshot_unchanged':True,'plan_sha256':b.PLAN_SHA,'binary_sha256':{str(root/'build'/str(i)):'0'*64 for i in range(8)},'build_strategy':'direct62_to_qsa35_isolated_incremental_v1','fresh_all_source_compilation':False,'CMake_reconfiguration_performed':False,'parent_unchanged':True,'parent_root':p['source62_parent_actual_sdk'],'parent_receipt_sha256':p['source62_parent_actual_receipt_sha256'],'external_source_root':str(Path(p['source62_parent_actual_sdk'])/'source'),'external_source_identity_kind':kind};rp=root/'receipt.json';rp.write_text(json.dumps(r));(root/'plan.snapshot.json').write_bytes(b.PLAN.read_bytes());b.configure(root,hashlib.sha256(rp.read_bytes()).hexdigest());actual_sha=b.sha
  def only_original_boundary(path):
   if Path(path)==root/'parent-provenance.json':raise ReachedOriginalProvenance('metadata-only test stops before ABI reads')
   return actual_sha(path)
  return only_original_boundary
 def test_actual_builder_receipt_kind_passes_real_binder_gate(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();fn=self.fixture(root,self.builder_kind())
   with patch.object(b,'sha',side_effect=fn):
    with self.assertRaises(ReachedOriginalProvenance):b.binding()
 def test_old_or_foreign_kind_rejected(self):
  for kind in('immutable_actual QSA35_source_snapshot','foreign'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d).resolve();fn=self.fixture(root,kind)
    with patch.object(b,'sha',side_effect=fn):
     with self.assertRaisesRegex(ValueError,'Direct immutable actual62 parent required'):b.binding()
 def test_exact_one_literal_fix_preserves_all_other_gates(self):
  old=(HERE/'source62_qsa35_sdk_binding_v1.py').read_text();new=(HERE/'source62_qsa35_sdk_binding_v2.py').read_text();self.assertEqual(new.replace('immutable_actual62_source_snapshot','immutable_actual QSA35_source_snapshot'),old)
if __name__=='__main__':unittest.main()
