"""CPU protocol/clock controls; no model, process launch or GPU proof."""
import json,queue,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_source62_cached_window_profile_v1 as actor
import source62_profile_protocol_raw_v1 as raw
from inspect_source62_cached_RAW_clock_v1 import serial_uncertainty_ns
class Controls(unittest.TestCase):
 def test_actual_protected_request_and_RID_packet(self):
  cls=actor.NS['Protocol'];base=cls.__bases__[0]
  with tempfile.TemporaryDirectory() as d:
   obj=object.__new__(cls);obj.directory=Path(d);obj.roster=[dict(ids=[17]*2048,rid=1,fresh=1)];obj.stderr=[];obj.ready='READY';obj.raw_receives=[];obj.raw_bytes=0;obj.active_rid=0;obj.q=raw.ObservedQueue(obj,queue.Queue());sent=[]
   def send(_,line):
    sent.append(line)
    for text in ['REUSED 0','RESUME 0','T 7','DONE 1 2048 1 1 length']:
     obj.q.put((__import__('time').monotonic(),text))
   with patch.object(base,'send',send):record=obj.request('request-1',[17]*2048,max_new=512,fresh=0)
   self.assertEqual(sent,['GEN 8 rid=1 fresh=1 pin=1536 ckpt=1 temperature=0 '+' '.join(['17']*2048)])
   self.assertEqual(record['fresh'],1);self.assertEqual(record['pin'],1536);self.assertEqual(record['profile_request_rid'],1)
   self.assertEqual([x['line'] for x in record['profile_RAW_receive_rows']],record['stdout']);self.assertEqual(record,json.loads((Path(d)/'request-1.json').read_text()))
 def test_quota_refuses_before_original_queue(self):
  with tempfile.TemporaryDirectory() as d:
   owner=SimpleNamespace(active_rid=1,raw_receives=[],raw_bytes=(128<<10),directory=Path(d));q=queue.Queue();observer=raw.ObservedQueue(owner,q)
   with self.assertRaisesRegex(ValueError,'quota'):observer.put((1.,'T 7'))
   self.assertTrue(q.empty());self.assertFalse((Path(d)/'host-protocol-RAW.jsonl').exists())
 def test_source_serialization_bound_is_geometry(self):
  self.assertGreater(serial_uncertainty_ns('1790000000000000.000000'),100)
  self.assertLess(serial_uncertainty_ns('2.000000'),1)
if __name__=='__main__':unittest.main()
