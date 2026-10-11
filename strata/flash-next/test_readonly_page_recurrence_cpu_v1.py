import hashlib,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
import probe_readonly_page_recurrence_v1 as p

class Controls(unittest.TestCase):
 def test_buffered_page_read_preserves_non_atime_identity(self):
  with tempfile.TemporaryDirectory()as t:
   path=Path(t)/'tiny';path.write_bytes(b'x'*8192);before=p.stat5(path)
   raw,sig=p.page(path,4096);self.assertEqual(raw,b'x'*4096);self.assertEqual(sig,before);self.assertEqual(p.stat5(path),before)
 def test_partial_page_refused(self):
  with tempfile.TemporaryDirectory()as t:
   path=Path(t)/'tiny';path.write_bytes(b'x')
   with self.assertRaises(ValueError):p.page(path,0)
 def test_phase_foreign_script_refused(self):
  with tempfile.TemporaryDirectory()as t:
   path=Path(t)/'phases.json';path.write_text(json.dumps([{'name':'foreign','argv':['/usr/bin/python3','/tmp/foreign.py'],'script_sha256':'0'*64,'timeout':1}]))
   with self.assertRaises(ValueError):p.phases(path)
 def test_actual_cpu_child_stdout_stderr_eof(self):
  with tempfile.TemporaryDirectory()as t:
   row={'name':'tiny','argv':['/usr/bin/python3','-c','import sys;print("out");print("err",file=sys.stderr)'],'script_sha256':'synthetic','timeout':2}
   proof=p.run_phase(row,Path(t),time.monotonic()+3);self.assertTrue(proof['passed']);self.assertTrue(proof['reader_threads_retired']);self.assertTrue(all(x['EOF']for x in proof['streams'].values()))
 def test_actual_cpu_timeout_not_pass(self):
  with tempfile.TemporaryDirectory()as t:
   row={'name':'timeout','argv':['/usr/bin/python3','-c','import time;time.sleep(3)'],'script_sha256':'synthetic','timeout':1}
   proof=p.run_phase(row,Path(t),time.monotonic()+.1);self.assertFalse(proof['passed']);self.assertTrue(proof['timeout']);self.assertTrue(proof['reader_threads_retired'])
 def test_bounded_idle_producer_preserves_real_tiny_page(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);path=root/'tiny';path.write_bytes(b'x'*4096);before=p.stat5(path)
   proc=subprocess.run([sys.executable,str(Path(p.__file__)),'--path',str(path),'--page-offset','0','--expected-page-sha256',hashlib.sha256(b'x'*4096).hexdigest(),'--output',str(root/'out'),'--duration','1','--interval','.2'],capture_output=True,text=True,timeout=5)
   self.assertEqual(proc.returncode,0,proc.stderr);report=json.loads((root/'out/report.json').read_text());self.assertTrue(report['passed']);self.assertGreater(report['samples'],0);self.assertFalse(report['cause_attributed']);self.assertEqual(p.stat5(path),before);self.assertEqual(path.read_bytes(),b'x'*4096)

if __name__=='__main__':unittest.main()
