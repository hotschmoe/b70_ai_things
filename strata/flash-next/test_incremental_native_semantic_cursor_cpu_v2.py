"""Tiny real files: metadata suspicion requires content proof, never stat trust."""
import json,os,tempfile,unittest
from pathlib import Path
from buffered_native_trace_epoch_v6 import BufferedTrace
from incremental_native_semantic_cursor_v2 import Cursor
class Controls(unittest.TestCase):
 def fixture(self,root):
  sink=BufferedTrace(root/'trace',root/'combined');sink.emit({'kind':'native_receive','engine_pid':7,'line':'UR before anchor'},semantic=True);sink.marker('HARNESS FULLCACHE_PHASE index=1 name=warm');row=sink.emit({'kind':'fullcache_phase_begin','index':1,'name':'warm','engine_pid':7},semantic=True);cursor=Cursor(root/'trace',sink.anchor(row));sink.close();return root/'trace',cursor
 def touch(self,path):
  st=path.stat();os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns+1000000))
 def test_metadata_only_drift_proves_current_bytes(self):
  with tempfile.TemporaryDirectory()as d:
   path,c=self.fixture(Path(d));self.touch(path);self.assertEqual(c.poll(),[]);receipt=c.metadata_proof_receipt();self.assertEqual(receipt['fallback_count'],1);self.assertGreater(receipt['fallback_bytes_read'],0);self.assertFalse(receipt['stat_only_content_trusted']);self.assertTrue(receipt['fallbacks'][0]['pending_current_bytes_equal']);self.assertEqual(receipt['fallbacks'][0]['previous_stat']['bytes'],receipt['fallbacks'][0]['before_stat']['bytes']);c.retire(True,True)
 def test_prefix_lastline_and_partial_same_size_mutations_refused(self):
  for mode in('prefix','lastline','pending'):
   with tempfile.TemporaryDirectory()as d:
    path,c=self.fixture(Path(d))
    if mode=='pending':
     with path.open('ab')as f:f.write(b'{"partial":')
     self.assertEqual(c.poll(),[])
    raw=path.read_bytes()
    if mode=='prefix':raw=raw.replace(b'UR before anchor',b'UR change anchor',1)
    elif mode=='lastline':raw=raw[:c.last_line_start]+raw[c.last_line_start:].replace(b'"warm"',b'"cold"',1)
    else:raw=raw[:-1]+b'X'
    path.write_bytes(raw);self.assertRaises(ValueError,c.poll);self.assertEqual(c.metadata_proof_receipt()['fallback_count'],0);c.close_failure()
 def test_pending_unchanged_metadata_drift_is_bound(self):
  with tempfile.TemporaryDirectory()as d:
   path,c=self.fixture(Path(d))
   with path.open('ab')as f:f.write(b'{"partial":')
   c.poll();self.touch(path);self.assertEqual(c.poll(),[]);self.assertEqual(c.metadata_proof_receipt()['fallbacks'][0]['pending_bytes'],11);self.assertRaises(ValueError,c.retire,True,True);c.close_failure()
 def test_replacement_truncation_refused(self):
  for mode in('replace','truncate'):
   with tempfile.TemporaryDirectory()as d:
    path,c=self.fixture(Path(d));raw=path.read_bytes()
    if mode=='replace':path.rename(path.with_name('old'));path.write_bytes(raw)
    else:path.write_bytes(raw[:c.offset-1])
    self.assertRaises(ValueError,c.poll);c.close_failure()
 def test_owner_and_sequence_refused_without_metadata_fallback(self):
  for mode in('owner','sequence'):
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);sink=BufferedTrace(root/'trace',root/'combined');sink.marker('HARNESS FULLCACHE_PHASE index=1 name=warm');row=sink.emit({'kind':'fullcache_phase_begin','index':1,'name':'warm','engine_pid':7},semantic=True);c=Cursor(root/'trace',sink.anchor(row))
    if mode=='owner':sink.producer_identity=dict(sink.producer_identity,pid=sink.producer_identity['pid']+1)
    else:sink.sequence+=1
    sink.emit({'kind':'native_receive','engine_pid':7,'line':'PP 1'},semantic=True);self.assertRaises(ValueError,c.poll);c.close_failure();sink.close()
 def test_same_filesystem_metadata_control_when_explicitly_requested(self):
  location=os.environ.get('B70_CURSOR_TEST_ROOT')
  if not location:self.skipTest('Root opt-in tiny fixture on actual result filesystem')
  with tempfile.TemporaryDirectory(prefix='.cursor-v2-cpu-',dir=location)as d:
   path,c=self.fixture(Path(d));self.touch(path);self.assertEqual(c.poll(),[]);self.assertEqual(c.metadata_proof_receipt()['fallback_count'],1);c.retire(True,True)
if __name__=='__main__':unittest.main()

class ReceiptControls(unittest.TestCase):
 fixture=Controls.fixture
 touch=Controls.touch
 def test_saved_fallback_current_prefix_pending_and_aggregate_recollected(self):
  from incremental_native_semantic_cursor_v2 import metadata_receipt_binding
  with tempfile.TemporaryDirectory()as d:
   path,c=self.fixture(Path(d));self.touch(path);c.poll();snapshot={'metadata_timestamp_proof':c.metadata_proof_receipt(),'actual_ACK':c.anchor,'last_sequence':c.sequence,'complete_raw_trace_bytes_consumed':c.offset};self.assertEqual(metadata_receipt_binding(path,snapshot)['actual_metadata_fallbacks_recollected'],1)
   snapshot['metadata_timestamp_proof']['fallback_count']=True;self.assertRaises(ValueError,metadata_receipt_binding,path,snapshot);c.close_failure()
 def test_current_saved_fallback_content_and_count_cost_mutations_refused(self):
  import copy
  from incremental_native_semantic_cursor_v2 import metadata_receipt_binding
  for mode in('prefix','pending','cost','stats','owner'):
   with tempfile.TemporaryDirectory()as d:
    path,c=self.fixture(Path(d))
    with path.open('ab')as f:f.write(b'{"partial":')
    c.poll();self.touch(path);c.poll();snapshot={'metadata_timestamp_proof':c.metadata_proof_receipt(),'actual_ACK':c.anchor,'last_sequence':c.sequence,'complete_raw_trace_bytes_consumed':c.offset};item=snapshot['metadata_timestamp_proof']['fallbacks'][0]
    if mode=='prefix':raw=path.read_bytes();path.write_bytes(raw.replace(b'UR before anchor',b'UR change anchor',1))
    elif mode=='pending':raw=path.read_bytes();path.write_bytes(raw[:-1]+b'X')
    elif mode=='cost':snapshot['metadata_timestamp_proof']['fallback_seconds']+=1.
    elif mode=='stats':item['before_stat']['bytes']+=1
    else:item['complete_consumed_prefix']['producer']=dict(c.owner,pid=c.owner['pid']+1)
    self.assertRaises(ValueError,metadata_receipt_binding,path,snapshot);c.close_failure()
