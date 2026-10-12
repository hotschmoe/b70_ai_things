"""Actual builder script syntax and inherited-oneAPI startup control; CPU only."""
import ast,json,os,shlex,subprocess,unittest
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
class Startup(unittest.TestCase):
 def test_actual_script_has_newlines_and_reuses_inherited_oneapi_environment(self):
  tree=ast.parse((HERE/'build_native56_incremental_from55_v2.py').read_bytes());assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets));plan=json.loads((HERE/'native56-hc-prefill-tile4-engine-build-proposal-v1.json').read_bytes())
  configure=['cmake','-S','/src/sycl','-B','/build','-G','Ninja']+['-D'+k+'='+v for k,v in sorted(plan['cmake_flags'].items())]
  script=eval(compile(ast.Expression(assignment.value),'actual-v2-container-script','eval'),dict(shlex=shlex,child_plan=plan,configure=configure,a=SimpleNamespace(jobs=4)))
  self.assertGreater(script.count('\n'),10);self.assertNotIn('\\n',script);subprocess.run(['bash','-n'],input=script,text=True,check=True)
  guard=next(line for line in script.splitlines() if line.startswith('if [[ ${SETVARS_COMPLETED'))
  # The inherited ready state must skip source entirely. The nonexistent path
  # makes an accidental unconditional re-source fail on this CPU host.
  skipped=guard.replace('/opt/intel/oneapi/setvars.sh','/definitely-missing-incremental-startup-control.sh')
  subprocess.run(['bash','-e','-c',skipped],env={**os.environ,'SETVARS_COMPLETED':'1'},check=True)
if __name__=='__main__':unittest.main()
