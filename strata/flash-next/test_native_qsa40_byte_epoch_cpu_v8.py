"""Live process byte bracket and long semantic-before-READY ordering controls."""
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import native_qsa40_byte_epoch_v8 as e
import native_qsa40_runtime_v8 as d
import batch54_health_handshake_v1 as h
class Controls(unittest.TestCase):
 def test_live_complete_bytes_detect_same_size_and_roster_mutation(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);data=root/'data';data.mkdir();f=data/'a';f.write_bytes(b'original');models=[]
   for i in range(4):p=root/('model'+str(i));p.write_bytes(b'M');models.append(p)
   plan={'prepared':str(data),'engine_root':str(data),'pack':str(data),'own_producer_root':str(data)}
   with patch.object(e,'PREFIXES',(root,)),patch.object(e,'guard',return_value={'passed':True}):
    epoch=e.ByteEpoch(plan,[],models);epoch.ready();f.write_bytes(b'FOREIGN!');self.assertRaises(ValueError,epoch.predevice)
    f.write_bytes(b'original');epoch=e.ByteEpoch(plan,[],models);epoch.ready();(data/'extra').write_bytes(b'x');self.assertRaises(ValueError,epoch.predevice)
 def test_live_epoch_wrong_PID_and_duplicate_phase_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);data=root/'data';data.mkdir();(data/'a').write_bytes(b'x');models=[]
   for i in range(4):p=root/str(i);p.write_bytes(b'M');models.append(p)
   plan={k:str(data)for k in('prepared','engine_root','pack','own_producer_root')}
   with patch.object(e,'PREFIXES',(root,)),patch.object(e,'guard',return_value={'passed':True}):
    epoch=e.ByteEpoch(plan,[],models);epoch.ready();self.assertRaises(ValueError,epoch.ready)
    with patch.object(e.os,'getpid',return_value=os.getpid()+100):self.assertRaises(ValueError,epoch.predevice)
 def test_long_semantics_preREADY_postACK_only_current_byte_seal(self):
  clock=[100.];events=[]
  class Epoch:
   def __init__(self,*args):events.append('entry')
   def ready(self):events.append('ready');return {'label':'ready'}
   def predevice(self):events.append('byte_seal');clock[0]+=1;return {'label':'predevice'}
   def finish(self):events.append('post');return {'CPU_SYNTHETIC':True}
  def semantic(plan):events.append('semantic');clock[0]+=1022
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);out=root/'off';out.mkdir();plan={'source_plan_sha256':'a'*64,'driver_sha256':d.sha(d.__file__)};p=root/'plan.json';d.write(p,plan);actual_read=d.read;actual_write=h.write_new;d.write(root/'parent-start.json',{'started_epoch':100.});d.write(root/'off-pre-kernel.receipt.json',{'sha256':'CPU_FAKE','finished_epoch':1123.})
   def read(path):return {'files':{}}if Path(path)==d.PLAN else actual_read(path)
   def write(path,value):
    result=actual_write(path,value)
    if Path(path).name=='READY.json':
     clock[0]+=1;health={'passed':True,'finished_epoch':clock[0]};d.write(root/'off-pre-health.json',health);d.write(out/'ACK.json',{'health_sha256':d.sha(root/'off-pre-health.json'),'health_finished_epoch':clock[0],'journal_sha256':'CPU_FAKE','ack_epoch':clock[0],'ready_sha256':d.sha(path),'plan_sha256':d.sha(p)})
    return result
   with patch.object(e,'ByteEpoch',Epoch),patch.object(d,'read',side_effect=read),patch.object(d,'closure',return_value='a'*64),patch.object(d,'manifest',side_effect=semantic)as full,patch.object(d.os.path,'samefile',return_value=True),patch.object(d.time,'time',side_effect=lambda:clock[0]),patch.object(h,'write_new',side_effect=write),patch.object(d,'arm',side_effect=lambda *args:events.append('GPU')),patch('native_qsa40_evidence_v8.health'),patch('native_qsa40_evidence_v8.journal'):
    d.child(SimpleNamespace(plan=p,output=out,on=False));self.assertEqual(full.call_count,1)
   self.assertEqual(events,['entry','semantic','ready','byte_seal','GPU','post']);self.assertEqual(d.read(out/'GPU-seal.json')['finished_epoch'],1124.)
if __name__=='__main__':unittest.main()
