import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
P=Path(__file__).with_name('prepare.py');s=importlib.util.spec_from_file_location('prepare_agent2',P);prepare=importlib.util.module_from_spec(s);s.loader.exec_module(prepare)
class Pair(unittest.TestCase):
 def test_pair_only_intended_differences(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=prepare.prepare(Path(tmp)/'pair');plans=[]
   for card,label in enumerate(['phase','phase-mrv1fix']):
    dest=root/label;plan=json.loads((dest/'plan.json').read_text());plans.append(plan)
    cmd=plan['server'];self.assertEqual(cmd[cmd.index('--card')+1],str(card));self.assertEqual(cmd[cmd.index('--port')+1],str(18151+card));self.assertEqual(cmd[cmd.index('--tensor-parallel-size')+1],'1');self.assertNotIn('--tp-host-trace',cmd)
    self.assertEqual(len(plan['jobs']),3);self.assertIn('--startup',plan['jobs'][0]['command'])
    cases=json.loads((dest/'pi-manifest.json').read_text())['cases'];self.assertEqual([(c['agent'],c['phase']) for c in cases],[('agent_2','warm'),('agent_2','target')])
   self.assertEqual((root/'phase/config/Config.json').read_bytes(),(root/'phase-mrv1fix/config/Config.json').read_bytes())
   self.assertEqual((root/'phase/fresh-scales.json').read_bytes(),(root/'phase-mrv1fix/fresh-scales.json').read_bytes())
if __name__=='__main__':unittest.main()
