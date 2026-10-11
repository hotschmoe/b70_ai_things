import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import source41_fidelity_ur_preflight_v2 as p
import admit_source41_model_fidelity_v3 as a

class Controls(unittest.TestCase):
 def fixture(self,root,trace=True):
  capture=root/'actor/source41-original-fidelity';capture.mkdir(parents=True);env={'SYCL_UR_TRACE':'2'}if trace else{}
  recipe={'engine_generation':41,'materialized_mode':1,'observer_enabled':True,'native_env':env};command=['docker','run']+(['-e','SYCL_UR_TRACE=2']if trace else[]);result={'terminal':{'Config':{'Env':['SYCL_UR_TRACE=2']if trace else[]}}}
  for name,value in [('recipe.json',recipe),('command.json',command),('result.json',result)]:
   (capture/name).write_text(json.dumps(value)+'\n')
  (capture/'requests.json').write_text('[]\n');rows=[];owners=[]
  for i,(kind,size,head)in enumerate([('SFD',1966080,0),('SFD',2959360,1),('P30',47185936,0),('P30',47185936,0)]):
   pointer=hex(0x1000+i*256);rows.append(f'<--- urUSMDeviceAlloc(.hContext = 0x10, .hDevice = 0x20, .size = {size}, .ppMem = 0xf ({pointer})) -> UR_RESULT_SUCCESS;');rows.append(f'{kind} allocation '+('pid=9 'if kind=='P30'else'')+f'stage=0 lb=0 le=48 '+(f'pointer={pointer} 'if kind=='P30'else'')+f'bytes={size}'+(f' residuals1 logits{head}'if kind=='SFD'else''));owners.append((kind,size,pointer))
  for kind,size,pointer in owners:
   if kind=='P30':rows.append(f'P30 release_begin pid=9 stage=0 pointer={pointer} bytes={size}')
   rows.append(f'<--- urUSMFree(.hContext = 0x10, .pMem = {pointer}) -> UR_RESULT_SUCCESS;')
   if kind=='P30':rows.append('P30 release_returned pid=9 stage=0')
  (capture/'engine.combined.log').write_text('\n'.join(rows)+'\n');return capture.parent,capture
 def test_actual_shaped_alloc_free_preflight_has_no_model_authority(self):
  with tempfile.TemporaryDirectory()as t:
   root,capture=self.fixture(Path(t));proof=p.preflight(root);self.assertTrue(proof['audit']['passed']);self.assertFalse(proof['full_model_identity_qualified']);self.assertFalse(proof['actor_public_admission_reexecuted'])
 def test_missing_env_refuses_before_public_or_original_admission(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor,capture=self.fixture(root,False)
   with patch.object(a,'actor')as public,patch.object(a.original,'join')as original:
    with self.assertRaises(ValueError):a.finalized_binding(actor,root/'original',root/'refusal.json')
    public.assert_not_called();original.assert_not_called();record=json.loads((root/'refusal.json').read_text());self.assertTrue(record['metadata_only_refusal']);self.assertFalse(record['new_closing_model_scan_executed'])
 def test_declared_trace_without_actual_free_refuses(self):
  with tempfile.TemporaryDirectory()as t:
   root,capture=self.fixture(Path(t));path=capture/'engine.combined.log';path.write_text('\n'.join(x for x in path.read_text().splitlines()if'urUSMFree'not in x)+'\n')
   with self.assertRaises(ValueError):p.preflight(root)
 def test_one_missing_free_old_audit_still_refuses(self):
  with tempfile.TemporaryDirectory()as t:
   root,capture=self.fixture(Path(t));path=capture/'engine.combined.log';lines=path.read_text().splitlines();lines.pop(next(i for i,x in enumerate(lines)if'urUSMFree'in x));path.write_text('\n'.join(lines)+'\n')
   with self.assertRaises(ValueError):p.preflight(root)
 def test_conflicting_or_foreign_terminal_trace_refuses(self):
  with tempfile.TemporaryDirectory()as t:
   root,capture=self.fixture(Path(t));path=capture/'command.json';row=json.loads(path.read_text());row+=['-e','SYCL_UR_TRACE=0'];path.write_text(json.dumps(row))
   with self.assertRaises(ValueError):p.preflight(root)
 def run_valid(self,failed=False):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);actor_root,capture=self.fixture(root);descriptor={'capture_root':str(capture),'materialized_mode':1,'observer_enabled':True,'public_actor_admission':{'source_plan_sha256':'actor-source'}};public=SimpleNamespace(captured_binding=Mock(side_effect=[descriptor,copy.deepcopy(descriptor)]),a=SimpleNamespace(closure=lambda:'actor-source'))
   with patch.object(a,'source_binding',return_value='source'),patch.object(a,'actor',return_value=public),patch.object(a.raw,'captured_arm',return_value={}),patch.object(a.original,'join',side_effect=ValueError('reference failed')if failed else None,return_value={'full_model_math_qualified':False}):
    output=root/'external.json'
    if failed:
     with self.assertRaises(ValueError):a.finalized_binding(actor_root,root/'original',output)
    else:self.assertTrue(a.finalized_binding(actor_root,root/'original',output)['passed'])
    self.assertEqual(public.captured_binding.call_count,2)
 def test_valid_preflight_keeps_both_full_actor_admissions(self):self.run_valid()
 def test_reference_failure_keeps_after_full_actor_admission(self):self.run_valid(True)
 def test_actual_lazy_dispatch_names_V5_only(self):
  with patch.object(a.importlib,'import_module',return_value='actual-V5')as load:
   self.assertEqual(a.actor(),'actual-V5');load.assert_called_once_with('validate_source41_model_cli_v5')
if __name__=='__main__':unittest.main()
