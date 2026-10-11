"""Producer-shaped offsets/namespace/delayed metadata/atomic phase boundary."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from buffered_native_trace_epoch_v6 import BufferedTrace
from full_cache_semantic_phase_stream_v2 import PhaseStream,ack_binding,snapshot_binding
class Controls(unittest.TestCase):
 def fixture(self,root):
  sink=BufferedTrace(root/'api-native-trace.jsonl',root/'engine.combined.log')
  for i in range(100):sink.emit({'kind':'native_receive','engine_pid':7,'line':' ---> urStartup '+('z'*2048)},combined_line=' ---> urStartup '+('z'*2048))
  sink.marker('HARNESS FULLCACHE_PHASE index=1 name=warm');row=sink.emit({'kind':'fullcache_phase_begin','index':1,'name':'warm','engine_pid':7,'render_observation':None},semantic=True);ack={k:row[k]for k in('index','name','engine_pid','sequence','render_observation')};ack['trace_anchor']=sink.anchor(row);return sink,ack
 def test_original_marker_offsets_hash_and_global_line_range(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);sink,ack=self.fixture(root);stream=PhaseStream(root,ack,{'State':{'Pid':os.getpid()}})
   sink.emit({'kind':'native_receive','engine_pid':7,'line':'BDONE 0 2 stop 1 rid=1 slotgen=1'},combined_line='BDONE 0 2 stop 1 rid=1 slotgen=1',semantic=True);stream.poll()
   # Source really publishes metadata after DONE and an asynchronous proof.
   sink.emit({'kind':'eos_waiter_phase_proof','request':{'index':1,'name':'warm','calls':[1]},'proof':{}},semantic=True)
   sink.emit({'kind':'native_receive','engine_pid':7,'line':'FC39 {"kind":"native_lifetime"}'},combined_line='FC39 {"kind":"native_lifetime"}',semantic=True)
   rows,last,text,bounds,snapshot=stream.freeze();self.assertEqual(last,snapshot['last_sequence']);self.assertEqual(rows[-1]['sequence'],last);self.assertEqual(list(text.splitlines()),['BDONE 0 2 stop 1 rid=1 slotgen=1','FC39 {"kind":"native_lifetime"}']);self.assertEqual(bounds['original_first_line'],102);self.assertEqual(bounds['original_last_line'],103);self.assertTrue(ack_binding(ack,rows[0]));self.assertTrue(snapshot_binding(root,snapshot,ack));self.assertLess(snapshot['new_bytes_read'],5000);stream.close();sink.close()
 def test_foreign_namespace_and_changed_prefix_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);sink,ack=self.fixture(root);bad=json.loads(json.dumps(ack));bad['trace_anchor']['producer']['start_ticks']+=1
   self.assertRaises(ValueError,PhaseStream,root,bad,{'State':{'Pid':os.getpid()}});stream=PhaseStream(root,ack,{'State':{'Pid':os.getpid()}});sink.close();raw=(root/'api-native-trace.jsonl').read_bytes();(root/'api-native-trace.jsonl').write_bytes(raw.replace(b'urStartup',b'urChanged',1));self.assertRaises(ValueError,stream.action_boundary);stream.close()
if __name__=='__main__':unittest.main()
