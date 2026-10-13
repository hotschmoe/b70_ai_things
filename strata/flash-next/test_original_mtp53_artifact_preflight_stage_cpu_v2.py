import unittest
from pathlib import Path
from prepare_original_mtp53_artifact_preflight_stage_v2 import corrected_command,PYTHON
class Interpreter(unittest.TestCase):
 def test_qualified_consumed_interpreter_and_manifest_bind(self):
  old=['docker','run','--name','old','--label','key=owner','-v','/old:/artifact:ro','image','-lc','test ! -e /dev/dri; exec python3 /preflight.py --artifacts /artifact --binding-sha256 '+'a'*64]
  new=corrected_command(old,Path('/old'),Path('/new'));self.assertIn('exec '+PYTHON+' /preflight.py',new[-1]);self.assertTrue(new[-1].endswith('a'*64));self.assertIn('/new:/artifact:ro',new);self.assertNotIn('/old:/artifact:ro',new);self.assertIn('exec python3 /preflight.py',old[-1])
