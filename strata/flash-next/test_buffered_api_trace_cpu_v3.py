"""Exact frozenV2 record/line equivalence and owned buffered IO failure controls."""
import io,json,os,sys,tempfile,threading,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import batch_api_trace_v2 as frozen
import batch_api_trace_v3 as buffered
from buffered_api_trace_sink_v3 import BufferedTrace

class Controls(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='bufferedAPI-CPU-');self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def run_observer(self,module,folder):
  directory=self.root/folder;directory.mkdir();trace=directory/'trace.jsonl';combined=directory/'combined.log';armrequest=directory/'ARM.request';arm=directory/'ARM';cfg=directory/'cfg.json';cfg.write_text('{}');cancel=threading.Event();server=types.ModuleType('server');raw='UR CPU_SYNTHETIC\nT 5 rid=1 slotgen=0\nDONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2 0 rid=1 slotgen=0\n'
  class Proc:
   pid=77
   def __init__(self):self.stdin=io.StringIO();self.stdout=io.StringIO(raw)
   def poll(self):return 0
  class Service:
   def encode_prompt(self,messages,tools,kwargs):return [10,11]
  class Engine:
   gen=1
   def generate(self,ids,max_new,sampling,cancel,embeddings=None):
    self.proc.stdin.write('GEN 1 seed=1 rid=1 10,11\n');self.proc.stdin.flush()
    for _ in range(3):self.proc.stdout.readline()
    self.last={'request_id':1,'engine_generation':1,'finish':'length'};yield 5
   def close(self):self.proc.stdin.write('QUIT\n');self.proc.stdin.flush()
  server.Service=Service;server.StrataEngine=Engine;server.popen=lambda *a,**kw:Proc()
  def main():
   svc=server.Service();ids=svc.encode_prompt([{'role':'user','content':'CPU_ONLY'}],None,{});engine=server.StrataEngine();engine.proc=server.popen('native',['strata','--serve']);list(engine.generate(ids,1,{},cancel));engine.close();armrequest.write_text('CPU_ACTUAL_MARKER_REQUEST\n');deadline=time.monotonic()+1
   while not arm.exists():self.assertLess(time.monotonic(),deadline);time.sleep(.005)
   return 0
  server.main=main;serve=types.ModuleType('serve');serve.server=server;contract=types.ModuleType('c1_trace_contract');contract.pinned_eos=lambda cfg:[6];oldpath=list(sys.path)
  try:
   with patch.dict(sys.modules,{'serve':serve,'c1_trace_contract':contract}),patch.object(sys,'argv',['CPU_OBSERVER','--config',str(cfg)]),patch.dict(os.environ,{'B70_BATCH_TRACE':str(trace),'B70_BATCH_NATIVE_LOG':str(combined),'B70_BATCH_ARM_REQUEST':str(armrequest),'B70_BATCH_ARM':str(arm)}):self.assertEqual(module.main(),0)
  finally:sys.path[:]=oldpath
  records=[json.loads(line) for line in trace.read_text().splitlines()]
  for row in records:row.pop('epoch')
  return records,combined.read_bytes(),directory
 def test_frozenV2_equivalence_clock_only_normalized(self):
  old,oldlines,_=self.run_observer(frozen,'old');new,newlines,path=self.run_observer(buffered,'new');self.assertEqual(new,old);self.assertEqual(newlines,oldlines);status=json.loads((path/'trace.jsonl.buffered-status.json').read_text());self.assertTrue(status['passed']);self.assertTrue(status['ARM_thread_retired']);self.assertEqual(status['records'],len(new));self.assertEqual(status['owned_handles'],2)
 def test_concurrent_producers_exactsequence_and_lines(self):
  sink=BufferedTrace(self.root/'trace',self.root/'log',.02)
  def producer(index):
   for item in range(30):sink.emit({'kind':'CPU_THREAD','call':index,'line':str(item)},combined_line=str(index)+':'+str(item))
  threads=[threading.Thread(target=producer,args=(i,)) for i in range(4)]
  for thread in threads:thread.start()
  for thread in threads:thread.join()
  status=sink.close();rows=[json.loads(l) for l in (self.root/'trace').read_text().splitlines()];lines=(self.root/'log').read_text().splitlines();self.assertEqual([r['sequence'] for r in rows],list(range(1,121)));self.assertEqual(lines,[str(r['call'])+':'+r['line'] for r in rows]);self.assertEqual(status['records'],120)
 def test_marker_and_cancel_terminal_immediate_visibility(self):
  sink=BufferedTrace(self.root/'trace',self.root/'log',1);sink.emit({'kind':'native_send','line':'BSTOP 0 rid=1 slotgen=2'},semantic=True);self.assertIn('BSTOP',(self.root/'trace').read_text());sink.marker('HARNESS ARM CPU');self.assertEqual((self.root/'log').read_text(),'HARNESS ARM CPU\n');sink.emit({'kind':'engine_close','error':None},semantic=True);self.assertIn('engine_close',(self.root/'trace').read_text());sink.close()
 def test_bounded_periodic_visibility_and_owned_shutdown_drain(self):
  sink=BufferedTrace(self.root/'trace',self.root/'log',.02);sink.emit({'kind':'CPU_BUFFERED'},combined_line='CPU_LINE');deadline=time.monotonic()+1
  while not (self.root/'trace').read_text():self.assertLess(time.monotonic(),deadline);time.sleep(.005)
  status=sink.close();self.assertTrue(status['periodic_thread_retired']);self.assertTrue((self.root/'trace').read_text().endswith('\n'));self.assertTrue((self.root/'log').read_text().endswith('\n'))
 def test_write_or_flush_failure_never_reports_pass_or_silent_close(self):
  for method in ('write','flush'):
   directory=self.root/method;directory.mkdir();sink=BufferedTrace(directory/'trace',directory/'log',1)
   with patch.object(sink.trace_handle,method,side_effect=OSError('CPU_INJECTED_IO')):
    with self.assertRaises(OSError):sink.emit({'kind':'CPU_ERROR'},semantic=True)
   with self.assertRaises(RuntimeError):sink.close()
   self.assertFalse(sink.status()['passed']);self.assertIsNotNone(sink.status()['error'])
 def test_close_refuses_late_or_truncated_delivery(self):
  sink=BufferedTrace(self.root/'trace',self.root/'log',.02);sink.emit({'kind':'CPU_FINAL'});sink.close()
  with self.assertRaises(RuntimeError):sink.emit({'kind':'CPU_LATE'})
  self.assertTrue((self.root/'trace').read_text().endswith('\n'));self.assertEqual(len((self.root/'trace').read_text().splitlines()),1)
 def test_flush_worker_deadline_fails_without_waiting_for_lock(self):
  sink=BufferedTrace(self.root/'trace',self.root/'log',1);sink.stop.set();sink.worker.join();actual=sink.worker;sink.worker=types.SimpleNamespace(join=lambda **k:None,is_alive=lambda:True)
  with self.assertRaises(RuntimeError):sink.close()
  self.assertFalse(sink.status()['passed']);sink.worker=actual
  for handle in sink.handles:handle.close()
 def test_invalid_unbounded_flush_interval_refused(self):
  for interval in (0,2,True):
   with self.assertRaises(ValueError):BufferedTrace(self.root/'trace',self.root/'log',interval)
if __name__=='__main__':unittest.main()
