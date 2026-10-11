import json,unittest
from unittest.mock import patch
from pathlib import Path
import run_source43_warm_pti_v5 as a
import run_source43_shortread0_v1 as short
class Controls(unittest.TestCase):
 def saved(self):
  root=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-warm-pti-short-v3-run-v1')
  return tuple(json.loads((root/name).read_bytes())for name in('prepared.snapshot.json','config.snapshot.json','inputs.json','source43-warm-pti-ON/recipe.json'))
 def test_actual_saved_recipe_command_builder(self):
  m,cfg,inputs,saved=self.saved();a.NS['_RUNTIME']=saved['tool_runtime'];a.NS['_CASE']=('short',111,222)
  _,recipe=a.phase_recipe(a.PHASES[0],m,cfg,inputs,'a'*64)
  # Actual metadata-driven Docker argv construction; nothing is launched.
  descriptor={**m,'source':str(a.intake.SDK/'source'),'executable':str(a.intake.SDK/'build/strata')}
  _,cmd=a.life.contract.native_command(Path('/tmp/new-phase'),descriptor,recipe,111,'a'*64)
  self.assertIn(saved['tool_runtime']['install_root']+':/pti:ro',cmd);self.assertIn(str(a.HERE/'pti_owned_temporal_control_v2.py')+':/pti-temporal.py:ro',cmd)
  self.assertIs(a.life.contract.native_command.__globals__,a.NS);self.assertIs(a.NS['_PORTS']['life'],a.life)
 def test_real_preflight_before_current_admission(self):
  root=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-warm-pti-short-v3-run-v1');plan=json.loads((root/'plan.snapshot.json').read_bytes());saved=json.loads((root/'source43-warm-pti-ON/recipe.json').read_bytes());a.NS['_RUNTIME']=saved['tool_runtime'];a.NS['_CASE']=('short',111,222);sha=a.c.sha
  with patch.object(a.c,'sha',side_effect=lambda path:'a'*64 if Path(path)==a.PLAN else sha(path)):
   proof=a.preflight(plan,Path('/tmp/preflight-only'));self.assertTrue(proof['pure_command_construction_passed']);self.assertFalse(proof['GPU_launch_performed'])
  names=a.NS['_PORTS']['main'].__code__.co_names;self.assertLess(names.index('preflight'),names.index('admission'));self.assertIn('command_preflight',a.NS['_PORTS']['main'].__code__.co_varnames)
 def test_OFF_builder_needs_no_runtime(self):
  m,cfg,inputs,saved=self.saved();recipe={**saved,'PTI_enabled':False,'mode':'source43-warm-pti-OFF','native_argv':['/build/strata','--serve'],'native_env':{}}
  _,cmd=short.life.contract.native_command(Path('/tmp/OFF'),m,recipe,111,'a'*64)
  self.assertNotIn('/pti:ro',' '.join(cmd));self.assertFalse(recipe['PTI_enabled'])
if __name__=='__main__':unittest.main()
