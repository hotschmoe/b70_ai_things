"""New lifecycle/source preflight controls with all device boundaries mocked."""
import copy,json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import qualify_hc_composition35_v1 as parent
import compile_hc_composition35_v1 as compiler
import prepare_hc_composition35_leaf_v1 as producer

class Controls(unittest.TestCase):
 def test_bad_compile_metadata_never_enters_lease_health_or_creates_output(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'new';argv=['parent','--compile-receipt',str(root/'CPU.json'),'--inputs',str(root/'CPU-package'),'--output',str(out)]
   with patch.object(sys,'argv',argv),patch.object(parent,'source_binding',return_value={}),patch.object(parent,'compile_binding',side_effect=ValueError('CPU badcompile')),patch.object(parent.os,'execv') as lease,patch.object(parent.subprocess,'Popen') as launch:
    with self.assertRaises(ValueError):parent.main()
    lease.assert_not_called();launch.assert_not_called();self.assertFalse(out.exists())
 def test_wrong_corpus_before_lease_no_health_finally(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'new';argv=['parent','--compile-receipt',str(root/'CPU.json'),'--inputs',str(root/'CPU-package'),'--output',str(out)]
   with patch.object(sys,'argv',argv),patch.object(parent,'source_binding',return_value={}),patch.object(parent,'compile_binding',return_value=({'plan':{'CPU_ONLY':True}},{})),patch.object(parent.fixture,'input_binding',return_value={'computed_corpus_counts':{'cases':4,'output_floats':38}}),patch.object(parent.os,'execv') as lease,patch.object(parent.subprocess,'Popen') as launch:
    with self.assertRaises((ValueError,KeyError)):parent.main()
    lease.assert_not_called();launch.assert_not_called();self.assertFalse(out.exists())
 def test_nonpositive_deadline_precedes_compile_or_GPU(self):
  argv=['parent','--compile-receipt','/CPU','--inputs','/CPU','--output','/CPU','--max-runtime','0']
  with patch.object(sys,'argv',argv),patch.object(parent,'compile_binding') as compile_gate,patch.object(parent.os,'execv') as lease:
   with self.assertRaises(ValueError):parent.main()
   compile_gate.assert_not_called();lease.assert_not_called()
 def test_compiler_wrong_plan_before_lease_or_output(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);plan=root/'CPU-plan.json';plan.write_text('{}');out=root/'new'
   with patch.object(sys,'argv',['compiler','--plan',str(plan),'--output',str(out)]),patch.object(compiler.os,'execv') as lease,patch.object(compiler,'leased') as device:
    with self.assertRaises(ValueError):compiler.main()
    lease.assert_not_called();device.assert_not_called();self.assertFalse(out.exists())
 def test_foreign_owner_and_wrong_image_refused(self):
  obj={'Name':'/CPU_OWNED','Config':{'Image':'CPU_IMAGE','Labels':{'b70.hc35comp1.runtime':'CPU_BIND'}}};self.assertTrue(parent.owned_container(obj,'CPU_OWNED','CPU_IMAGE','CPU_BIND'))
  for key in ('name','image','label'):
   changed=copy.deepcopy(obj)
   if key=='name':changed['Name']='/FOREIGN'
   if key=='image':changed['Config']['Image']='FOREIGN'
   if key=='label':changed['Config']['Labels']['b70.hc35comp1.runtime']='FOREIGN'
   self.assertFalse(parent.owned_container(changed,'CPU_OWNED','CPU_IMAGE','CPU_BIND'))
 def test_unsafe_teardown_cannot_qualify(self):
  p={'child_return_code':0,'interrupted':False,'owned_containers_terminal':True,'forced_cleanup':False,'pre_health_passed':True,'post_health_passed':True,'kernel_fault_gate_passed':True,'post_source_unchanged':True,'errors':[],'child_terminal_epoch':10,'post_health_finished_epoch':11};proof={'numerical_and_teardown_passed':True};identity={'passed':True,'rows':[{}]*4,'started':12};self.assertTrue(parent.finalizable(p,proof,identity))
  for k,v in [('interrupted',True),('forced_cleanup',True),('owned_containers_terminal',False),('post_source_unchanged',False),('child_return_code',2)]:self.assertFalse(parent.finalizable(dict(p,**{k:v}),proof,identity))
  self.assertFalse(parent.finalizable(p,proof,dict(identity,started=9)))
 def test_actual_readonly_source35_SDK_recipe_reprepare(self):
  plan=producer.read(producer.ROOT/'strata/flash-next/hc-composition35-leaf-compile-plan-v1.json');self.assertEqual(producer.prepare(plan['engine_root']),plan);self.assertEqual(len(plan['actual_eight_SDK_ELFs']),8);self.assertEqual(plan['computed_corpus_counts']['output_floats'],3802788)
if __name__=='__main__':unittest.main()
