import unittest,copy
from types import SimpleNamespace
from unittest.mock import patch
import run_source43_shortread0_v1 as a
class Controls(unittest.TestCase):
 def test_real_recipe_only_argument_delta(self):
  inputs={'short':[7]*54,'prefill_about512':[7]*524}
  recipe={'native_argv':['/engine','--prefill','64','--spec','2'],'native_env':{},'requests':[{'ordinal':i+1,'role':role,'ids':inputs[role],'fresh':1,'pin':None,'max_new':64}for i,role in enumerate(('short','prefill_about512','prefill_about512','short'))]}
  a.NS['_RUNTIME']={'image':'image'};a.NS['_CASE']=('short',1,1);a.NS['tool']=SimpleNamespace(inherited_env=lambda env:None)
  with patch.object(a.NS['base'],'phase_recipe',return_value=({},copy.deepcopy(recipe))):
   _,result=a.phase_recipe(a.PHASES[0],{'runtime':{'image':'image'}},{'env':{}},inputs,'a'*64)
  self.assertEqual(result['native_argv'],recipe['native_argv']+['--short-read','0']);self.assertFalse(result['PTI_enabled']);self.assertEqual(len(result['requests']),2);self.assertTrue(all(r['max_new']==4 for r in result['requests']))
 def test_compiled_parent_single_phase(self):
  self.assertEqual(a.NS['_PORTS']['PHASES'],('source43-warm-pti-OFF',));self.assertIs(a.NS['_PORTS']['phase_recipe'],a.phase_recipe)
 def test_comparison_no_crosspath_pass(self):
  result=a.repeat_equivalence([{},{}]);self.assertTrue(result['single_arm_repeat_passed']);self.assertNotIn('H36_OFF_ON_equivalence_passed',result)
  with self.assertRaises(ValueError):a.repeat_equivalence([{},{}],[{},{}])
if __name__=='__main__':unittest.main()
