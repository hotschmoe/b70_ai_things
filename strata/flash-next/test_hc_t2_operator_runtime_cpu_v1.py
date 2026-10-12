import unittest
from pathlib import Path
import run_hc_t2_operator_v1 as r
class OwnedRuntime(unittest.TestCase):
 def test_actual_absence_is_name_bound(self):
  self.assertTrue(r.absent(b'error: no such object: owned\n','owned'))
  self.assertFalse(r.absent(b'error: no such object: foreign\n','owned'))
 def test_foreground_and_recipe_guards(self):
  s=Path(r.__file__).read_text()
  self.assertIn("'/proc/self/fd/8'",s);self.assertIn('ZE_AFFINITY_MASK=0',s)
  self.assertIn("if proc and proc.poll()is None:time.sleep(.25);continue",s)
  self.assertIn('Original creator identity unavailable; lease retained',s)
  self.assertIn("'model_payload_read':False",s);self.assertIn("type(state['ExitCode'])is int",s)
  self.assertIn("build['recipe_sha256']==sha(recipe)",s)
