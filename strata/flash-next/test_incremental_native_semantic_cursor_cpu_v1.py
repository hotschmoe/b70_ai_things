"""Synthetic large startup and exact source/epoch/retirement controls."""
import json,os,tempfile,unittest
from pathlib import Path
from buffered_native_trace_epoch_v4 import BufferedTrace
from incremental_native_semantic_cursor_v1 import Cursor
class Controls(unittest.TestCase):
 def fixture(self,root,startup=0):
  trace=root/'trace';sink=BufferedTrace(trace,root/'combined')
  for i in range(startup):sink.emit({'kind':'native_receive','engine_pid':7,'line':' ---> urSynthetic '+('x'*1024)})
  sink.marker('HARNESS FULLCACHE_PHASE index=1 name=warm');row=sink.emit({'kind':'fullcache_phase_begin','index':1,'name':'warm','engine_pid':7},semantic=True);anchor=sink.anchor(row);return sink,trace,anchor
 def test_large_startup_parse_once_only_new_complete_rows(self):
  with tempfile.TemporaryDirectory()as tmp:
   sink,path,anchor=self.fixture(Path(tmp),2048);cursor=Cursor(path,anchor);self.assertEqual(cursor.poll(),[])
   for i in range(30):sink.emit({'kind':'native_receive','engine_pid':7,'line':'PP '+str(i)},semantic=True)
   rows=cursor.poll();self.assertEqual([r['line']for r in rows],['PP '+str(i)for i in range(30)]);self.assertEqual(cursor.parsed_records,30);self.assertLess(cursor.bytes_read,40<<10);self.assertEqual(cursor.poll(),[]);cursor.verify_prefix();sink.close();cursor.retire(True,True)
 def test_foreign_inside_line_or_wrong_sequence_anchor_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   sink,path,anchor=self.fixture(Path(tmp));sink.close()
   for key,value in [('end_offset',anchor['end_offset']-1),('sequence',anchor['sequence']+1),('file',{'device':0,'inode':0})]:
    bad=dict(anchor);bad[key]=value;self.assertRaises(ValueError,Cursor,path,bad)
 def test_prefix_rewrite_and_rotation_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   sink,path,anchor=self.fixture(Path(tmp),2);cursor=Cursor(path,anchor);sink.close();raw=path.read_bytes();path.write_bytes(raw.replace(b'urSynthetic',b'urMutatedXX',1));self.assertRaises(ValueError,cursor.verify_prefix);cursor.close_failure()
   path.write_bytes(raw);cursor=Cursor(path,anchor);old=path.with_name('old');path.rename(old);path.write_bytes(raw);self.assertRaises(ValueError,cursor.poll);cursor.close_failure()
 def test_partial_retired_eof_refused_and_no_unowned_eof(self):
  with tempfile.TemporaryDirectory()as tmp:
   sink,path,anchor=self.fixture(Path(tmp));cursor=Cursor(path,anchor);sink.close()
   with path.open('ab')as target:target.write(b'{"partial":')
   self.assertEqual(cursor.poll(),[]);self.assertRaises(ValueError,cursor.retire,True,True);self.assertRaises(ValueError,cursor.retire,False,True);cursor.close_failure()
 def test_duplicate_nan_and_rewrite_with_append_refused(self):
  from incremental_native_semantic_cursor_v1 import strict
  self.assertRaises(ValueError,strict,b'{"sequence":1,"sequence":2}')
  self.assertRaises(ValueError,strict,b'{"epoch":NaN}')
  with tempfile.TemporaryDirectory()as tmp:
   sink,path,anchor=self.fixture(Path(tmp),2);cursor=Cursor(path,anchor);sink.emit({'kind':'native_receive','engine_pid':7,'line':'PP 64 256'},semantic=True);cursor.poll();sink.close();raw=path.read_bytes();path.write_bytes(raw.replace(b'urSynthetic',b'urChangedXX',1)+b' ');self.assertRaises(ValueError,cursor.verify_prefix);cursor.close_failure()
 def test_total_text_budget_preserves_first_rejected_raw_bytes_and_fails(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);sink=BufferedTrace(root/'trace',root/'combined',max_total_bytes=16<<20);line='UR synthetic '+('z'*(256<<10));failed=False
   for _ in range(40):
    try:sink.emit({'kind':'native_receive','engine_pid':7,'line':line},combined_line=line)
    except ValueError:failed=True;break
   self.assertTrue(failed);self.assertTrue((root/'trace.overflow.raw').exists());self.assertRaises(RuntimeError,sink.close);self.assertFalse(sink.status()['passed']);self.assertLessEqual(sum(p.stat().st_size for p in root.iterdir()),16<<20)
if __name__=='__main__':unittest.main()
