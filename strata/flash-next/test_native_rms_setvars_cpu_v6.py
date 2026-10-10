"""Real CPU shell initialization guards; no compiler/container/GPU/model."""
import ast,os,subprocess,tempfile,unittest
from pathlib import Path
import native_rms_setvars_shell_v6 as shell
import qualify_native_rms_rsqrt37_v6 as q
class Tests(unittest.TestCase):
 def run_guard(self,value,script):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'setvars.sh';p.write_text(script,encoding='ascii');env=dict(os.environ);env.pop('SETVARS_COMPLETED',None);env.pop('CPU_TEST_INITIALIZED',None)
   if value is not None:env['SETVARS_COMPLETED']=value
   prefix=shell.PREFIX.replace('/opt/intel/oneapi/setvars.sh',str(p));return subprocess.run(['/bin/bash','-c',prefix+'printf "CONTINUED:%s" "${CPU_TEST_INITIALIZED:-skipped}"'],env=env,capture_output=True,text=True,timeout=5)
 def test_initialized_image_skips_setvars_return3(self):
  row=self.run_guard('1','return 3\n');self.assertEqual(row.returncode,0);self.assertEqual(row.stdout,'CONTINUED:skipped')
 def test_unset_zero_and_other_values_initialize(self):
  for value in (None,'0','2',''):
   row=self.run_guard(value,'export SETVARS_COMPLETED=1 CPU_TEST_INITIALIZED=initialized\n');self.assertEqual(row.returncode,0);self.assertEqual(row.stdout,'CONTINUED:initialized')
 def test_genuine_setvars_failure_stops_phase(self):
  for value in (None,'0'):
   row=self.run_guard(value,'return 2\n');self.assertEqual(row.returncode,2);self.assertEqual(row.stdout,'')
 def test_actual_expected_recipes_share_guard_and_inside_runtime_stays_stdlib(self):
  self.assertEqual(q.SETVARS_PREFIX,shell.PREFIX);s=Path(q.__file__).read_text();self.assertEqual(s.count("SETVARS_PREFIX+'/opt/intel/oneapi/compiler"),2);self.assertEqual(s.count("shell=SETVARS_PREFIX+'/opt/b70-c1-python"),2);self.assertNotIn('set -e; source /opt/intel',s)
  tree=ast.parse(s)
  for node in tree.body:
   if isinstance(node,ast.ImportFrom):self.assertEqual(node.module,'pathlib')
 def test_no_device_mount_or_model_smoke_recipes(self):
  recipes=shell.smoke_recipes('COMPILER_IMAGE','RUNTIME_IMAGE');self.assertEqual(len(recipes),2)
  for row in recipes:
   self.assertIn('--rm',row);self.assertEqual(row[row.index('--network')+1],'none');self.assertEqual(row[row.index('--cpus')+1],'2');self.assertIn(shell.PREFIX,row[-1]);self.assertNotIn('--device',row);self.assertNotIn('--gpus',row);self.assertNotIn('-v',row);self.assertNotIn('--mount',row)
 def test_actual_failure_generation_preserved(self):self.assertEqual(q.sha(Path(q.__file__).with_name('native-rms-rsqrt37-owned-source-plan-v5.json')),'a8488530a6e02fd35b70e64be6dfaa9b0129c94b396653fcaae2fd210e17a2bb')
if __name__=='__main__':unittest.main()
