"""Tiny owned CPU launch and source cleanup controls, never Docker or model."""
import ast
import subprocess
import sys
import unittest
from pathlib import Path
import finite_tokenizer_owned_execution_v2 as owned


class Controls(unittest.TestCase):
    def test_tiny_real_cpu_client_join_and_empty_session(self):
        script = '''import sys,tempfile
from pathlib import Path
import finite_tokenizer_owned_execution_v2 as m
with tempfile.TemporaryDirectory() as root:
 command=[sys.executable,'-c',"print('CPU-only synthetic fixture')",'--name','CPU-test-only']
 result=m.execute(command,Path(root),lambda obj:(_ for _ in ()).throw(AssertionError('Docker touched')))
 assert result['return_code']==0 and result['error']is None
 assert result['launch_descendants']['launch_session_empty']
 assert result['command']==command
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_pending_client_kill_or_timer_absence(self):
        source = Path(owned.__file__).read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(any(n.func.attr in ('kill', 'terminate') for n in calls))
        self.assertIn('launches.retire(stop_owned)', source)
        self.assertIn('while process.poll() is None', source)
        self.assertIn('start_new_session=True', source)


if __name__ == '__main__':
    unittest.main()
