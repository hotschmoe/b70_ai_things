import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import build_hc_rms3_operator_v1 as b
import run_hc_rms3_operator_v1 as r
import hc_rms3_owned_command_v1 as owner
PREPARED=Path('/mnt/vm_8tb/b70/build/hc-rms3-isolated-operator-compile-inputs-v1')
class Controls(unittest.TestCase):
 def test_genuine_prepared_admission_and_exact_compile(self):
  recipe,pins=b.admission(PREPARED,b.sha(PREPARED/'compile-recipe.json'));script=b.compile_shell(recipe);self.assertIn('test ! -e /dev/dri',script);self.assertIn('cmp /sdk/build/packages.txt',script);self.assertIn('-cl-fp32-correctly-rounded-divide-sqrt',script);self.assertEqual(recipe['compiler_argv'][-1],'/out/hc-rms3')
 def test_changed_flags_rejected_before_lease(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)
   for f in PREPARED.iterdir():
    if f.is_file()and f.name in('compile-recipe.json','fixture.cpp','hc_rms3_generated_v1.inc','hc_native_composition.hpp'):(p/f.name).write_bytes(f.read_bytes())
   v=json.loads((p/'compile-recipe.json').read_text());v['compiler_argv'].append('-ffast-math');(p/'compile-recipe.json').write_text(json.dumps(v))
   with self.assertRaises(ValueError):b.admission(p,b.sha(p/'compile-recipe.json'))
 def test_single_card_exact_native_command(self):
  a=SimpleNamespace(build=Path('/actual/fresh-build'),inputs=Path('/actual/original-inputs'),expected_pci='0000:03:00.0')
  with patch.object(Path,'stat',return_value=SimpleNamespace(st_gid=123)):
   argv,expected=r.native_command(a,Path('/actual/new-run'),'unique-owner')
  self.assertEqual(argv.count('--device'),1);self.assertIn('/dev/dri',argv);self.assertEqual(expected['env']['ZE_AFFINITY_MASK'],'0');self.assertEqual(expected['env']['B70_HC_RMS3_OBSERVE'],'1');self.assertIn('exec /harness/hc-rms3 /inputs /out/oracle',argv[-1]);self.assertTrue(all('/models'not in x for x in expected['mounts']))
 def test_unknown_owner_refusal(self):
  argv,e=owner.command(b.IMAGE,'name','label',['/actual:/inputs:ro']);argv+=['--entrypoint','/bin/bash',b.IMAGE,'-lc','exec true'];row={'Config':{'Labels':{'label':'foreign'},'Image':b.IMAGE,'Entrypoint':['/bin/bash'],'Cmd':['-lc','exec true'],'User':e['user'],'WorkingDir':'/'},'HostConfig':{},'Image':b.IMAGE}
  with patch.object(owner.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps([row]).encode(),stderr=b'')):
   with self.assertRaises(ValueError):owner.inspect_owned('name',argv,e)
if __name__=='__main__':unittest.main()
