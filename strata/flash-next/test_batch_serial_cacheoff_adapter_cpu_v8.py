"""New metadata derivation/parent recipe negatives with external boundaries mocked."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import batch_serial_cacheoff_v8 as q
import qualify_batch_serial_cacheoff_v8 as parent
import run_batch_serial_cacheoff_v8 as runtime

class AdapterTests(unittest.TestCase):
 def base(self):return {'kind':'serial','driver_sha256':'f'*64,'prepared':'CPU_FAKE_C113','args':['CPU_RECIPE'],'env':{},'lane':'source35','slots':2,'group_index':0}
 def supplement(self):
  base=self.base();return dict(copy.deepcopy(base),serial_supplement_generation=8,driver_sha256=q.sha(Path(q.__file__)),V7_source_preparation=base)
 def test_visible_sourcepreparation_and_outerfields_bind_current_source(self):
  plan=self.supplement();source={'files':{}};binding={'CPU_SOURCE_ADMISSION_MOCK':True}
  with patch.object(q,'read',lambda path:source),patch.object(q.preparation,'manifest_binding',return_value=binding) as frozen,patch.object(q,'prepared_config',return_value=(plan['args'],plan['env'])),patch.object(q.serial,'config',return_value=([],{})):
   self.assertEqual(q.manifest_binding(plan),binding);frozen.assert_called_once_with(plan['V7_source_preparation'])
   changed=copy.deepcopy(plan);changed['args']=['UNKNOWN_OVERRIDE']
   with self.assertRaises(ValueError):q.manifest_binding(changed)
   changed=copy.deepcopy(plan);changed['driver_sha256']=plan['V7_source_preparation']['driver_sha256']
   with self.assertRaises(ValueError):q.manifest_binding(changed)
 def test_oldV7_no_supplement_and_sourceclosure_change_rejected(self):
  with patch.object(q.preparation,'manifest_binding') as original:
   with self.assertRaises(ValueError):q.manifest_binding(self.base())
   original.assert_not_called()
  with patch.object(q,'read',lambda path:{'files':{'NONEXISTENT_SOURCE':'f'*64}}),patch.object(q,'sha',lambda path:'0'*64):
   plan=self.supplement();plan['driver_sha256']='0'*64
   with self.assertRaises(ValueError):q.manifest_binding(plan)
 def test_fullcommand_matches_frozen_shared_runtime_recipe(self):
  rawroot=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native2-serial0-run/child');plan=q.read(rawroot/'plan.snapshot.json');saved=q.read(rawroot/'serial-0/command.json');name=saved[saved.index('--name')+1];pid=int(name.split('-')[2]);command=runtime.command_recipe(plan,rawroot,0,pid);self.assertEqual(command,saved)
  for key,value in [('image','FOREIGN_IMAGE'),('pack','FOREIGN_PACK'),('env',dict(plan['env'],STRATA_UNKNOWN_MATH='1'))]:
   changed=copy.deepcopy(plan);changed[key]=value;self.assertNotEqual(runtime.command_recipe(changed,rawroot,0,pid),saved)
 def test_parent_delegation_only_processlocal_and_frozen_template(self):
  module=parent.shared;old_file=module.__file__;old_ctrl=module.ctrl;old_sha=module.BATCH_NUMERICAL_SHA
  with patch.object(q,'read',return_value={'files':{str(parent.TEMPLATE.relative_to(q.ROOT)):q.sha(parent.TEMPLATE)}}),patch.object(module,'__file__',old_file),patch.object(module,'ctrl',old_ctrl),patch.object(module,'BATCH_NUMERICAL_SHA',old_sha),patch.object(module,'main',return_value=0) as lifecycle:
   self.assertEqual(parent.main(),0);lifecycle.assert_called_once();self.assertIs(module.ctrl,q);self.assertEqual(module.__file__,parent.__file__)
  self.assertEqual(module.__file__,old_file);self.assertIs(module.ctrl,old_ctrl)
 def test_newreadonly_reader_perrequest_command_terminal_and_postsource_guards(self):
  text=Path(q.__file__).with_name('validate_batch_serial_cacheoff_v8.py').read_text()
  for marker in ("directory/'result.json'","adapter.command_recipe(","ctrl.read(rawpath)==row['raw']","extract(row['raw']","49*len(jobs)","source_after=final_source_join", "cache_qualification_granted"):
   self.assertIn(marker,text)
if __name__=='__main__':unittest.main()
