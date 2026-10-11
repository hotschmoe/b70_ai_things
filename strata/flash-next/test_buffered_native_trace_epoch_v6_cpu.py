"""No-GPU exact V5/V6 sink equivalence and failure/anchor controls."""
import hashlib,io,json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import buffered_native_trace_epoch_v5 as old
import buffered_native_trace_epoch_v6 as new

class Tests(unittest.TestCase):
 def test_decimal_fixed_points_match_whole_V5_encoding(self):
  for start in [0,9,99,999,9999,99999,999999,999999999,9999999999]:
   actual={'kind':'native_receive','line':'Unicode snowman \u2603 and quoted "trace_end_offset": 0','sequence':1,'epoch':1.,'nested':{'trace_end_offset':0},'trace_start_offset':start,'trace_end_offset':start,'trace_file':{'device':1,'inode':2},'trace_prefix_sha256_before':'a'*64}
   expected=dict(actual)
   for _ in range(8):
    raw=(json.dumps(expected,ensure_ascii=True,sort_keys=True,allow_nan=False)+'\n').encode('ascii');end=start+len(raw)
    if expected['trace_end_offset']==end:break
    expected['trace_end_offset']=end
   self.assertEqual(new.serialize_record(actual,start),raw);self.assertEqual(actual,expected)
 def test_full_raw_metadata_marker_anchor_seals_byte_identical(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);all_runs=[]
   for module in [old,new]:
    folder=root/module.__name__;folder.mkdir()
    with patch.object(module.time,'time',return_value=123.5),patch.object(module,'pid_identity',return_value={'pid':42,'start_ticks':7}),patch.object(module.os,'fstat',return_value=SimpleNamespace(st_dev=1,st_ino=2)):
     sink=module.BufferedTrace(folder/'trace',folder/'combined',interval=1)
     sink.emit({'kind':'native_receive','line':'   ---> urUSMSharedAlloc','call':None},combined_line='   ---> urUSMSharedAlloc')
     sink.marker('B70_PHASE_BEGIN index=0')
     row=sink.emit({'kind':'fullcache_phase_begin','name':'phase','index':0,'namespace':{'id':4},'render_observation':{'text':'nonascii \u2603'}},semantic=True);anchor=sink.anchor(row)
     sink.emit({'kind':'native_receive','line':'PCL complete','payload':{'trace_end_offset':9}},combined_line='PCL complete',semantic=True)
     status=sink.close();self.assertTrue(status['passed']);self.assertTrue(status['periodic_thread_retired']);all_runs.append(((folder/'trace').read_bytes(),(folder/'combined').read_bytes(),anchor,status))
   self.assertEqual(all_runs[0],all_runs[1])
   prefix=b''
   for raw in all_runs[1][0].splitlines(keepends=True):
    value=json.loads(raw);self.assertEqual(value['trace_prefix_sha256_before'],hashlib.sha256(prefix).hexdigest());self.assertEqual(value['trace_start_offset'],len(prefix));prefix+=raw;self.assertEqual(value['trace_end_offset'],len(prefix))
 def test_budget_overflow_preserves_identical_first_rejected_records(self):
  evidence=[]
  with tempfile.TemporaryDirectory() as tmp:
   for module in [old,new]:
    root=Path(tmp)/module.__name__;root.mkdir()
    with patch.object(module.time,'time',return_value=123.5),patch.object(module,'pid_identity',return_value={'pid':42,'start_ticks':7}),patch.object(module.os,'fstat',return_value=SimpleNamespace(st_dev=1,st_ino=2)):
     sink=module.BufferedTrace(root/'trace',root/'combined',interval=1,max_total_bytes=16<<20)
     try:
      for _ in range(100):sink.emit({'kind':'native_receive','line':'x'*100000},combined_line='x'*100000)
     except ValueError:pass
     else:self.fail('Bounded sink did not reject overflow')
     with self.assertRaises(RuntimeError):sink.close()
     evidence.append(((root/'trace').read_bytes(),(root/'combined').read_bytes(),(root/'trace.overflow.raw').read_bytes(),sink.record_count,sink.line_count))
   self.assertEqual(evidence[0],evidence[1])
 def test_short_binary_write_fails_closed(self):
  class Short:
   def __init__(self,handle):self.handle=handle
   def write(self,raw):return self.handle.write(raw[:-1])
  with tempfile.TemporaryDirectory() as tmp:
   sink=new.BufferedTrace(Path(tmp)/'trace',Path(tmp)/'combined',interval=1)
   real=sink.trace_handle;sink.trace_handle=Short(real)
   with self.assertRaises(IOError):sink.emit({'kind':'native_receive','line':'original'},combined_line='original')
   sink.trace_handle=real
   with self.assertRaises(RuntimeError):sink.close()
   retained=(Path(tmp)/'trace').read_bytes();self.assertGreater(len(retained),0);self.assertFalse(retained.endswith(b'\n'));self.assertEqual(sink.record_count,0)
 def test_marker_short_write_fails_closed(self):
  class Short:
   def __init__(self,handle):self.handle=handle
   def write(self,raw):return self.handle.write(raw[:-1])
  with tempfile.TemporaryDirectory() as tmp:
   sink=new.BufferedTrace(Path(tmp)/'trace',Path(tmp)/'combined',interval=1);real=sink.combined_handle;sink.combined_handle=Short(real)
   with self.assertRaises(IOError):sink.marker('B70_PHASE_BEGIN index=0')
   sink.combined_handle=real
   with self.assertRaises(RuntimeError):sink.close()
   self.assertEqual((Path(tmp)/'combined').read_bytes(),b'B70_PHASE_BEGIN index=0');self.assertIsNone(sink.last_marker)
 def test_periodic_and_close_flush_failures_are_not_silently_successful(self):
  class BrokenFlush:
   def flush(self):raise IOError('CPU flush failure')
  with tempfile.TemporaryDirectory() as tmp:
   sink=new.BufferedTrace(Path(tmp)/'trace',Path(tmp)/'combined',interval=1);real=sink.trace_handle;sink.trace_handle=BrokenFlush()
   with self.assertRaises(IOError):sink.flush()
   sink.trace_handle=real
   with self.assertRaises(RuntimeError):sink.close()
   self.assertFalse(sink.status()['passed']);self.assertTrue(sink.status()['periodic_thread_retired'])
 def test_periodic_visibility_keeps_every_buffered_raw_byte(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);sink=new.BufferedTrace(root/'trace',root/'combined',interval=.02);visible=threading.Event();original_flush=sink._flush
   def published():
    original_flush()
    if sink.record_count:visible.set()
   sink._flush=published
   row=sink.emit({'kind':'native_receive','line':'all original bytes'},combined_line='all original bytes')
   self.assertTrue(visible.wait(1),'Periodic owned visibility did not publish within bounded CPU test')
   self.assertEqual(json.loads((root/'trace').read_bytes()),row);self.assertEqual((root/'combined').read_bytes(),b'all original bytes\n');self.assertTrue(sink.close()['periodic_thread_retired'])
 def test_text_tell_flushes_but_binary_offset_arithmetic_does_not(self):
  class Raw(io.RawIOBase):
   def __init__(self):self.count=0;self.bytes=0
   def writable(self):return True
   def seekable(self):return True
   def write(self,data):self.count+=1;self.bytes+=len(data);return len(data)
   def tell(self):return self.bytes
  raw=Raw();text=io.TextIOWrapper(io.BufferedWriter(raw),encoding='ascii');text.write('original\n');self.assertEqual(raw.count,0);self.assertEqual(text.tell(),9);self.assertEqual(raw.count,1)
  binary_raw=Raw();binary=io.BufferedWriter(binary_raw);count=binary.write(b'original\n');self.assertEqual(count,9);self.assertEqual(binary_raw.count,0);binary.flush();self.assertEqual(binary_raw.count,1)
if __name__=='__main__':unittest.main()
