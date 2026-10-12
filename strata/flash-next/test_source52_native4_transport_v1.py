"""Tiny current native4 adapter controls; no process/model/compile access."""
import tempfile,time,unittest
from pathlib import Path
from batch_numerical_protocol_v2 import Roster
from run_source52_native4_coherence_v1 import Protocol
class Queue:
 def __init__(self,p,lines):self.p=p;self.lines=iter(lines)
 def get(self,timeout):
  if len(self.p.sent)!=4:raise AssertionError('Four co-admissions must precede drain')
  return time.monotonic(),next(self.lines)
class Controls(unittest.TestCase):
 def probe(self,d,stale=False):
  p=Protocol.__new__(Protocol);p.directory=Path(d);p.batch=Roster(4);p.completed={};p.sent=[];p.send=p.sent.append
  p.schedule=[dict(ordinal=i+1,rid=i+1,slot=i,ids=list(range(2048)),fresh=1,pin=1536,max_new=8,cohort=0,role='ABCD'[i],expected_reused=0) for i in range(4)];lines=[]
  for i in range(4):lines.extend(['T 71 rid=%d slotgen=1'%(i+1),'DONE 1 2048 1 0 length 0 0 0 0 0 0 0 0 2048 0 rid=%d slotgen=1'%(i+1),'BADM %d 1 rid=%d slotgen=1'%(i,i+1)])
  lines.append('SBF batch_event pid=1 enginegen=1 event=1 rows=4 active_mask=15 completed=1')
  for i in range(4):lines.extend(['BT %d 248044 rid=%d slotgen=%d'%(i,i+1,2 if stale and i==3 else 1),'BDONE %d 2 stop rid=%d slotgen=1'%(i,i+1)])
  p.q=Queue(p,lines);return p
 def test_all_four_before_drain(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.probe(d);rows=[p.request('request-%d'%(i+1),row['ids']) for i,row in enumerate(p.schedule)];self.assertEqual(len(p.sent),4);self.assertEqual([r['slot'] for r in rows],list(range(4)));self.assertEqual([r['rid'] for r in rows],[1,2,3,4]);self.assertTrue(all(r['output_ids']==[71,248044] for r in rows))
 def test_stale_fourth_owner_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.probe(d,True)
   with self.assertRaisesRegex(ValueError,'Stale native slot generation'):p.request('request-1',p.schedule[0]['ids'])
if __name__=='__main__':unittest.main()
