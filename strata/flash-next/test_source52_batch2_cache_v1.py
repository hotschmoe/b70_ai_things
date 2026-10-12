"""Synthetic protocol controls only; no native process or model access."""
import json, queue, tempfile, unittest
from pathlib import Path
from batch_numerical_protocol_v2 import Roster
from run_source52_batch2_cache_v1 import Protocol
from source52_batch2_cache_contract_v1 import command_for

class CohortQueue:
 def __init__(self, protocol, lines): self.protocol=protocol; self.lines=iter(lines)
 def get(self, timeout):
  import time
  if len(self.protocol.sent_commands)!=2: raise AssertionError('Both actual BGENs must precede any drain')
  return time.monotonic(),next(self.lines)

class Controls(unittest.TestCase):
 def probe(self, directory, stale=False):
  p=Protocol.__new__(Protocol);p.directory=Path(directory);p.batch=Roster(2);p.completed={};p.sent_commands=[];p.send=p.sent_commands.append
  p.schedule=[dict(ordinal=i+1,rid=i+3,slot=i,ids=[12+i,19+i],fresh=1,pin=1536,max_new=8,cohort=2,role='AB'[i],expected_reused=0) for i in range(2)]
  lines=[]
  for slot,rid in ((0,3),(1,4)):
   lines.extend(['T 71 rid=%d slotgen=1'%rid,'DONE 1 2 1 0 length 0 0 0 0 0 0 0 0 2 0 rid=%d slotgen=1'%rid,'BADM %d 1 rid=%d slotgen=1'%(slot,rid)])
  lines.append('SBF batch_event pid=8 enginegen=1 event=1 rows=2 active_mask=3 completed=1')
  for slot,rid in ((0,3),(1,4)):
   lines.append('BT %d 72 rid=%d slotgen=%d'%(slot,rid,2 if stale and slot==1 else 1))
   lines.append('BDONE %d 2 stop rid=%d slotgen=1'%(slot,rid))
  p.q=CohortQueue(p,lines);return p
 def test_both_submitted_before_drain_and_second_owner_uses_saved_result(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.probe(d); first=p.request('request-1',p.schedule[0]['ids']);second=p.request('request-2',p.schedule[1]['ids'])
   self.assertEqual(p.sent_commands,[command_for(row) for row in p.schedule]);self.assertEqual([first['rid'],second['rid']],[3,4]);self.assertEqual(first['output_ids'],[71,72]);self.assertEqual(second['output_ids'],[71,72]);self.assertEqual([first['slot'],second['slot']],[0,1]);self.assertEqual(json.loads((Path(d)/'cohort-2.json').read_bytes())['commands'],p.sent_commands)
 def test_foreign_slot_generation_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.probe(d,stale=True)
   with self.assertRaisesRegex(ValueError,'Stale native slot generation'):p.request('request-1',p.schedule[0]['ids'])

if __name__=='__main__':unittest.main()
