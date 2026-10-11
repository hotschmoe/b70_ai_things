import ast,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import own_q81_quantizer43_control_v2 as c
import own_q81_quantizer43_operation_v2 as operation
import qualify_own_q81_quantizer43_v2 as q
import test_own_q81_quantizer43_cpu_v1 as old
class Controls(old.Controls):
 def setUp(self):self.a=patch.object(old,'c',c);self.b=patch.object(old,'q',q);self.a.start();self.b.start()
 def tearDown(self):self.a.stop();self.b.stop()
class OperationControls(unittest.TestCase):
 def fixture(self,root):
  for n in operation.FIXTURE_FILES:(root/n).write_bytes(b'owned')
 def test_missing_fixture_refuses_before_original_entry(self):
  with patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   with self.assertRaises(ValueError):operation.context()
 def test_fixture_context_exact_current_PLAN_and_private_state(self):
  with tempfile.TemporaryDirectory()as t,patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   root=Path(t);self.fixture(root);original_op=operation.original._operation;operation.register_fixture(root);ns=operation.context();self.assertEqual(ns['_fixture_root'],root);self.assertEqual(set(ns['_fixture_names']),set(operation.FIXTURE_FILES));self.assertIsNone(ns['_operation']);self.assertIs(operation.original._operation,original_op)
   # Exact source-generated import/roster instructions, not a mirrored bool.
   code=ast.unparse(ast.parse(Path(operation.__file__).read_text()));self.assertIn('qualify_own_q81_quantizer43_v2 import PLAN',code);self.assertIn('_fixture_root/name',code);self.assertIn('source43_sdk_binding_v1',Path(operation.original.__file__).read_text())
 def test_foreign_pid_refused_before_original_call(self):
  with tempfile.TemporaryDirectory()as t,patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   root=Path(t);self.fixture(root);operation.register_fixture(root)
   with patch.object(operation.os,'getpid',return_value=os.getpid()+1):
    with self.assertRaises(ValueError):operation.context()
 def test_missing_consumed_file(self):
  with tempfile.TemporaryDirectory()as t,patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   root=Path(t);self.fixture(root);(root/'cpu-native.q81').unlink()
   with self.assertRaises(ValueError):operation.register_fixture(root)
 def test_fixture_switch_not_cached_pass(self):
  with tempfile.TemporaryDirectory()as t,patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   root=Path(t);a=root/'a';a.mkdir();b=root/'b';b.mkdir();self.fixture(a);self.fixture(b);operation.register_fixture(a)
   with self.assertRaises(ValueError):operation.register_fixture(b)
 def test_fixture_alias_before_read(self):
  with tempfile.TemporaryDirectory()as t,patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   root=Path(t);self.fixture(root);(root/'own-input.f32').unlink();(root/'own-input.f32').symlink_to(root/'config.txt')
   with self.assertRaises(ValueError):operation.register_fixture(root)
 def test_actual_producer_builds_new_plan_current43_and_fixture_roster(self):
  import native_qsa40_byte_epoch_v9 as epochs
  import source43_sdk_binding_v1 as sdk
  import admit_postboot_original_qsa_reference_v4 as current
  captured=[]
  class Epoch:
   def __init__(self,plan,source,shards):captured.extend(map(Path,source))
   def ready(self):pass
  with tempfile.TemporaryDirectory()as t,patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   root=Path(t);fixture=root/'fixture';fixture.mkdir();self.fixture(fixture);operation.register_fixture(fixture);ns=operation.context();own=root/'own';ns['OWN']=own;ns['_full_producer_binding']=lambda r:{'original_predicate_executed':True};fake_sdk=root/'SDK43';new_plan=root/'NEW_Q81_PLAN.json'
   def read(path):
    path=Path(path)
    if path==ns['ADMISSION']:return {'fresh_native_runtime_authorization':{'baseline_root':str(root/'baseline')}}
    if path==root/'baseline/prepared.json':return {'engine_receipt':str(root/'baselineSDK/receipt.json'),'pack':str(root/'pack')}
    if path==ns['HERE']/'model-lock.json':return {'destination':'CPU_NO_MODEL','files':[{'path':'UD-Q4_K_XL/'+str(i)}for i in range(4)]}
    if path==new_plan:return {'files':{'strata/flash-next/own_q81_quantizer43_gpu_v1.cpp':'fake'},'actual_consumed_source_bindings':{str(fake_sdk/'source/sycl/include/strata/kernels/q8_1_finite.hpp'):'fake'}}
    if path==fake_sdk/'receipt.json':return {'binary_sha256':{str(fake_sdk/'build/strata'):'fake'}}
    if path==fake_sdk/'plan.snapshot.json':return {'expected_patched_source_sha256':{'sycl/src/kernels/cuda/native_mmvq.dp.cpp':'fake'}}
    raise AssertionError(str(path))
   ns['read_unique']=read;ns['sha']=lambda p:'tiny'
   with patch.object(q,'PLAN',new_plan),patch.object(sdk,'SDK',fake_sdk),patch.object(epochs,'ByteEpoch',Epoch),patch.object(current,'source_binding',return_value={'cheap_preflight':True}):
    proof=ns['producer_binding'](own)
   self.assertTrue(proof['original_predicate_executed']);self.assertIn(new_plan,captured)
   for name in operation.FIXTURE_FILES:self.assertIn(fixture/name,captured)
   self.assertIn(fake_sdk/'source/sycl/include/strata/kernels/q8_1_finite.hpp',captured);self.assertIn(fake_sdk/'source/sycl/src/kernels/cuda/native_mmvq.dp.cpp',captured);self.assertIn(fake_sdk/'build/libstrata_kernels.a',captured);self.assertIn(fake_sdk/'build/libstrata_core.a',captured)
 def test_original_witness_predicates_unchanged(self):
  with tempfile.TemporaryDirectory()as t,patch.object(operation,'_fixture',None),patch.object(operation,'_context',None):
   root=Path(t);self.fixture(root);operation.register_fixture(root);ns=operation.context()
   node=next(n for n in ast.parse(Path(operation.original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='admit_operation_witness');self.assertEqual(ns['_derived_AST']['admit_operation_witness'],ast.dump(node))
if __name__=='__main__':unittest.main()
