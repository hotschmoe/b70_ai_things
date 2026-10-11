"""Producer-shaped UR backlog, actual owned reader and proof-sequence joins."""
import copy,json,os,tempfile,time,unittest
from pathlib import Path
from buffered_native_trace_epoch_v6 import BufferedTrace
from full_cache_semantic_phase_stream_v2 import PhaseStream
from full_cache_continuous_phase_drain_v1 import ContinuousDrain,binding
class Drain(unittest.TestCase):
 def write(self,p,row):p.write_text(json.dumps(row,ensure_ascii=True)+'\n')
 def fixture(self,root):
  sink=BufferedTrace(root/'api-native-trace.jsonl',root/'engine.combined.log');sink.marker('HARNESS FULLCACHE_PHASE index=1 name=warm');mark=sink.emit({'kind':'fullcache_phase_begin','index':1,'name':'warm','engine_pid':7,'render_observation':None},semantic=True);ack={k:mark[k]for k in('index','name','engine_pid','sequence','render_observation')};ack['trace_anchor']=sink.anchor(mark);return sink,ack
 def test_actual_large_backlog_during_http_reaches_waiter_before_freeze(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);sink,ack=self.fixture(root);stream=PhaseStream(root,ack,{'State':{'Pid':os.getpid()}})
   for index in range(2600):sink.emit({'kind':'native_receive','engine_pid':7,'line':' ---> urEvent '+('x'*4096)},combined_line=' ---> urEvent '+('x'*4096))
   proof=sink.emit({'kind':'eos_waiter_phase_proof','request':{'index':1,'name':'warm','calls':[1]},'proof':{'calls':[1]}},semantic=True)
   # A single post-return bounded read cannot reach the original proof.
   stream.poll();self.assertLess(stream.progress()['sequence'],proof['sequence'])
   with ContinuousDrain(stream,root,self.write)as drain:
    limit=time.monotonic()+10
    while stream.progress()['sequence']<proof['sequence']:
     drain.check();self.assertLess(time.monotonic(),limit);time.sleep(.005)
    drain.require_sequence(proof['sequence']);copy_rows=stream.poll();copy_rows[-1]['proof']['calls'].append(999);self.assertEqual(stream.poll()[-1]['proof']['calls'],[1])
    receipt=drain.finish();rows,last,text,bounds,snapshot=stream.freeze();self.assertTrue(binding(receipt,ack,snapshot,os.getpid()));self.assertEqual(last,proof['sequence']);self.assertEqual(rows[-1]['sequence'],last);self.assertFalse(drain.thread.is_alive())
   sink.close()
 def test_real_worker_failure_latches_and_retires_without_success(self):
  class Broken:
   def progress(self):return {'sequence':1,'bytes':1}
   def drain_poll(self):raise ValueError('actual poisoned source')
   def close(self):pass
  with tempfile.TemporaryDirectory()as temp:
   drain=ContinuousDrain(Broken(),temp,self.write);drain.thread.join(5);row=drain.finish();self.assertTrue(row['actual_thread_retired']);self.assertIn('poisoned',row['error']);self.assertRaises(ValueError,drain.check)
 def test_original_client_exception_joins_actual_reader_and_closes_stream(self):
  class Stream:
   closed=False
   def progress(self):return {'sequence':1,'bytes':1}
   def drain_poll(self):pass
   def close(self):self.closed=True
  with tempfile.TemporaryDirectory()as temp:
   stream=Stream();handle=None
   with self.assertRaises(KeyboardInterrupt):
    with ContinuousDrain(stream,temp,self.write)as handle:raise KeyboardInterrupt('owned client interruption')
   self.assertFalse(handle.thread.is_alive());self.assertTrue(stream.closed);self.assertTrue(json.loads((Path(temp)/'continuous-drain-receipt.json').read_bytes())['actual_thread_retired'])
 def test_foreign_missing_or_behind_original_proof_refused(self):
  with tempfile.TemporaryDirectory()as temp:
   root=Path(temp);sink,ack=self.fixture(root);stream=PhaseStream(root,ack,{'State':{'Pid':os.getpid()}})
   with ContinuousDrain(stream,root,self.write)as drain:
    with self.assertRaises(ValueError):drain.require_sequence(True)
    with self.assertRaises(ValueError):drain.require_sequence(ack['sequence']+100)
    receipt=drain.finish();rows,last,text,bounds,snapshot=stream.freeze();bad=copy.deepcopy(receipt);bad['owner']['pid']+=1;self.assertRaises(ValueError,binding,bad,ack,snapshot,os.getpid())
   sink.close()
if __name__=='__main__':unittest.main()
