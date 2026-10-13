import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import run_native62_hc_rms_fidelity_v1 as a
import native62_hc_rms_raw_v1 as raw
import native62_hc_rms_comparison_v1 as compare
import native_source41_cli_protocol_v4 as pipe
import test_source59_mode0_fidelity_cpu_v2 as constructor
class Controls(unittest.TestCase):
 def test_actual_protected_protocol_call_not_duplicate_kwargs(self):
  # Same exact protected parent constructor expression, Popen is only mocked.
  invoke=constructor.Constructor().call(a.Protocol);proof=json.loads(Path('/tmp/b70-source59-mode0-fidelity-current33-construction-v1.json').read_text());r=proof['commands'][0]
  with tempfile.TemporaryDirectory()as d:
   directory=Path(d);a.c.write(directory/'recipe.json',r['recipe']);active=[]
   with patch.object(pipe.subprocess,'Popen',side_effect=OSError('owned no-launch CPU sentinel'))as process:
    with self.assertRaisesRegex(OSError,'owned no-launch CPU sentinel'):invoke(r['native_command'],directory,lambda:None,active,r['recipe'])
   self.assertEqual(process.call_count,1);self.assertEqual(active,[])
 def test_actual_private_callbacks(self):
  self.assertIs(a.parent.set_case.__globals__,a.NS);self.assertIs(a.parent.set_runtime.__globals__,a.NS);self.assertIs(a.NS['life'].contract.native_command,a.native_command);self.assertIs(a.parent.configure.__globals__,vars(a.parent));self.assertIsNot(a.parent,a.natural);self.assertIs(a.NS['set_sdk'].__globals__,vars(a.parent));self.assertIs(a.BASE_RUN_PHASE.__globals__,a.NS);self.assertIs(a.BASE_RUN_PHASE.__code__,a.prior.BASE_RUN_PHASE.__code__)
 def test_recipe_only_declared_delta_and_literal_scope(self):
  proof=json.loads(Path('/tmp/b70-source59-mode0-fidelity-current33-construction-v1.json').read_text());prior=proof['commands'][0]['recipe'];source='0'*64
  with patch.object(a.parent,'phase_recipe',return_value=({},copy.deepcopy(prior))):left=a.phase_recipe(a.PHASES[0],{}, {},{},source)
  with patch.object(a.parent,'phase_recipe',return_value=({},copy.deepcopy(prior))):right=a.phase_recipe(a.PHASES[1],{}, {},{},source)
  self.assertTrue(a.matched_recipe_gate(left[1],right[1]));self.assertEqual(left[1]['requests'],[{'ordinal':i,'ids':[248045],'fresh':1,'pin':None,'max_new':1}for i in range(1,5)]);self.assertEqual(left[1]['engine_generation'],62);self.assertEqual(right[1]['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE'],'1')
  changed=copy.deepcopy(right[1]);changed['native_env']['STRATA_HC_PREFILL_TILE4']='0'
  with self.assertRaises(ValueError):a.matched_recipe_gate(left[1],changed)
 def test_no_stale_lp_formatter_or_consume_route(self):
  fn=raw._namespace['numeric_software'].port()[0];formatter=fn.__globals__['r'].request_command;self.assertEqual(formatter([248045],max_new=1,fresh=1),'GEN 1 fresh=1 ckpt=1 temperature=0 248045');self.assertEqual(raw.collect.__globals__['actor'],a)
if __name__=='__main__':unittest.main()
