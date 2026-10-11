import unittest
from pathlib import Path
from unittest.mock import patch
import probe_pti_paused_regular_fixture_v1 as p
class Tests(unittest.TestCase):
 def test_recipe_and_private_namespace(self):
  class Stat:st_gid=44
  original=p.prior.recipe
  with patch.object(Path,'stat',return_value=Stat()):
   recipe=p.recipe_for(Path('/tmp/fixture'),'a'*64,Path('/tmp/properties.json'));cmd=recipe('on',Path('/tmp/results'),'owned','b'*64)
  self.assertEqual(cmd[cmd.index('--label')+1],'b70.pti.paused.fixture='+'b'*64);self.assertIn('--retention on',cmd[-1]);self.assertEqual(sum(x=='-v'for x in cmd),5);self.assertIn(str(p.INSTALL)+':/pti:ro',cmd);self.assertEqual(cmd[-3],p.IMAGE)
  ns=p.utility(recipe);self.assertIs(ns['recipe'],recipe);self.assertIs(p.prior.recipe,original);self.assertIn('b70.pti.paused.fixture',ns['owned'].__code__.co_consts)
 def test_fresh_tool_pins_and_pause_template(self):
  import pti_paused_fixture_runtime_v1 as r
  self.assertEqual(r.LIB_SHA,'c72c6b4f42d8fe1a2c352286f5099280038b2136843924489d84a01887f43c4b');self.assertIn('clock.{pid}.jsonl',Path(r.__file__).read_text());self.assertIn("'--resume',session",Path(r.__file__).read_text())
if __name__=='__main__':unittest.main()

class CompileTests(unittest.TestCase):
 def test_compile_recipe_no_device_private_predicates(self):
  recipe=p.compile_recipe(Path('/tmp/properties.json'));cmd=recipe('compile',Path('/tmp/results'),'owned','b'*64);self.assertNotIn('--device',cmd);self.assertNotIn('--group-add',cmd);self.assertIn('-fsycl -O2 -std=c++17 -fsycl-targets=spir64 /leaf/fixture.cpp',cmd[-1]);ns=p.utility(recipe,compile_only=True);self.assertNotIn('--group-add',ns['owned'].__code__.co_consts);self.assertNotIn('PathOnHost',ns['owned'].__code__.co_consts)
