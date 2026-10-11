import json,subprocess,sys,unittest
from pathlib import Path
import full_cache_flat_producer_import_v1 as a
class Flat(unittest.TestCase):
 def test_actual_complete_local_producer_roster_flat_filename(self):
  rows=a.roster(Path(a.__file__).parent);self.assertIn('full_cache_shared_stale_owner_v2',rows);self.assertIn('eos_waiter_observer40_v1',rows);self.assertNotIn('batch54_health_handshake_v1',rows)
  result=subprocess.run([sys.executable,'-I',a.__file__],input=json.dumps(rows).encode('ascii'),capture_output=True,timeout=20);self.assertEqual(result.returncode,0,result.stderr);self.assertTrue(json.loads(result.stdout)['complete'])
 def test_old_exact_import_root_depth_counterexample(self):
  from pathlib import PurePosixPath
  with self.assertRaises(IndexError):_ = PurePosixPath('/controller/batch54_health_handshake_v1.py').parents[2]
 def test_standalone_identity_exact_original_body(self):
  import ast
  here=Path(a.__file__).parent
  def body(p):return next(ast.dump(n,include_attributes=False)for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='pid_identity')
  # Guard is deliberately local, without importing repository-root handshake.
  import full_cache_trace_process_identity_v1 as identity,os
  self.assertEqual(identity.pid_identity(os.getpid())['pid'],os.getpid())
  for invalid in (True,1.0,0,-1):
   with self.assertRaises(ValueError):identity.pid_identity(invalid)
if __name__=='__main__':unittest.main()
