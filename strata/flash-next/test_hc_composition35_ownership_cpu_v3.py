"""Actual launch recipe -> inspection ownership controls, all boundaries mocked."""
import copy,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import qualify_hc_composition35_v3 as p
class Controls(unittest.TestCase):
 def test_actual_launch_command_inspection_owner_and_foreign_negative(self):
  receipt={'binary':'/CPU_SYNTHETIC_BUILD/hc_composition_arithmetic35_gpu_v1','plan':{'environment':{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_UR_TRACE':'2'},'image':'CPU_SYNTHETIC_PINNED_IMAGE'}};name='b70-hc35comp3-runtime-123';binding='CPU_SYNTHETIC_BIND'
  with patch.object(p.os,'stat',return_value=SimpleNamespace(st_gid=1000)):command=p.launch_command(receipt,Path('/CPU_SYNTHETIC_OUTPUT'),name,binding,Path('/CPU_SYNTHETIC_INPUTS'))
  label=command[command.index('--label')+1];key,value=label.split('=',1);self.assertEqual(key,'b70.hc35comp3.runtime');self.assertEqual(value,binding);actual_name=command[command.index('--name')+1];image=command[-3];obj={'Name':'/'+actual_name,'Config':{'Image':image,'Labels':{key:value}}};self.assertTrue(p.owned_container(obj,name,receipt['plan']['image'],binding))
  for kind in ('name','image','label'):
   foreign=copy.deepcopy(obj)
   if kind=='name':foreign['Name']='/FOREIGN'
   if kind=='image':foreign['Config']['Image']='FOREIGN'
   if kind=='label':foreign['Config']['Labels'][key]='FOREIGN'
   with self.subTest(kind=kind):self.assertFalse(p.owned_container(foreign,name,receipt['plan']['image'],binding))
  old=copy.deepcopy(obj);old['Config']['Labels']={'b70.hc35comp1.runtime':binding};self.assertFalse(p.owned_container(old,name,receipt['plan']['image'],binding));old['Config']['Labels']={'b70.hc35comp2.runtime':binding};self.assertFalse(p.owned_container(old,name,receipt['plan']['image'],binding))
 def test_unchanged_primitive_compile_namespace_stays_source_build_only(self):
  source=Path(p.__file__).read_text();self.assertIn("'b70.hc35comp1.compile='+PLAN_SHA",source);self.assertNotIn('b70.hc35comp1.runtime',source);self.assertNotIn('b70.hc35comp2.runtime',source)
if __name__=='__main__':unittest.main()
