"""Actual-shaped diagnosis negatives in tmp metadata; no real proof execution."""
import copy,json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
import paired_binding_diagnosis48_v1 as d
class Controls(unittest.TestCase):
 def fixture(self,r):
  source=r/'source';source.write_text('{"files":{}}');original=r/'failed-plan';original.write_text('{}');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  row={'schema':1,'source_plan_sha256':sha(source),'source_binding':{},'passed':True,'typed_canonical_JSON_equal':True,'native_python_equal':False,'actual_binding_canonical_sha256':'a','saved_binding_canonical_sha256':'a','actual_GPU_touch':False,'actual_model_inference':False,'original_parent_failure_rewritten':False,'native_type_differences':[{'path':'$.shape','actual_type':'tuple','saved_type':'list'},{'path':'$.rows.1','actual_key_type':'int','saved_key_type':'str'}],'plan_path':str(original),'plan_sha256':sha(original)}
  receipt=r/'receipt';receipt.write_text(json.dumps(row));return source,receipt,row
 def test_explicit_actual_shape_only_representation_admitted(self):
  with tempfile.TemporaryDirectory() as name:
   r=Path(name);source,receipt,row=self.fixture(r)
   with patch.object(d,'PLAN',source),patch.object(d,'PLAN_SHA',d.sha(source)),patch.object(d,'ACTUAL_RECEIPT_SHA',d.sha(receipt)):self.assertTrue(d.binding(receipt)['typed_canonical_JSON_representation_only_proven'])
 def test_changed_bool_type_value_scope_or_actual_shape_refused(self):
  with tempfile.TemporaryDirectory() as name:
   r=Path(name);source,receipt,row=self.fixture(r)
   with patch.object(d,'PLAN',source),patch.object(d,'PLAN_SHA',d.sha(source)),patch.object(d,'ACTUAL_RECEIPT_SHA',d.sha(receipt)):
    for key,value in [('typed_canonical_JSON_equal',1),('native_python_equal',True),('saved_binding_canonical_sha256','b'),('actual_GPU_touch',True),('native_type_differences',[{'path':'x','actual_type':'bool','saved_type':'int'}])]:
     bad=copy.deepcopy(row);bad[key]=value;receipt.write_text(json.dumps(bad));self.assertRaises(ValueError,d.binding,receipt)
 def test_changed_original_failed_plan_refused(self):
  with tempfile.TemporaryDirectory() as name:
   r=Path(name);source,receipt,row=self.fixture(r);Path(row['plan_path']).write_text('{"changed":1}')
   with patch.object(d,'PLAN',source),patch.object(d,'PLAN_SHA',d.sha(source)),patch.object(d,'ACTUAL_RECEIPT_SHA',d.sha(receipt)):self.assertRaises(ValueError,d.binding,receipt)
 def test_controller_requires_diagnosis_before_pack_factory(self):
  import batch_numerical_execution_v48 as controller
  from types import SimpleNamespace
  with patch.object(controller,'diagnosis_binding',side_effect=ValueError('no actual diagnostic')),patch.object(controller,'for_prepared',side_effect=AssertionError('must not read pack')):self.assertRaises(ValueError,controller.prepare,SimpleNamespace(kind='native',binding_type_diagnosis='CPU_nonexistent'))
if __name__=='__main__':unittest.main()
