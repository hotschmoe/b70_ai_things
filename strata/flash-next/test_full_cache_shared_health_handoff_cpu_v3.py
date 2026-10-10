import copy,tempfile,time,unittest,ast
from unittest.mock import patch
from pathlib import Path
import full_cache_shared_health_handoff_v3 as h
import qualify_full_cache_shared_runtime_v3 as parent

class Handoff(unittest.TestCase):
 def fixture(self):
  now=time.time();plan={'driver_sha256':'a'*64,'engine_receipt_sha256':'b'*64,'prepared_sha256':'c'*64};manifest={'current_source':40};offered={'parent':{'pid':99,'start_ticks':456},'lease_identity':[{'card':0,'device':1,'inode':10},{'card':1,'device':1,'inode':11}],'parent_started_epoch':now-6};ready=h.ready_packet(plan,manifest,17,123,now-4,offered);health={'passed':True,'cards':[0,1],'started_epoch':now-3,'finished_epoch':now-2};ack={'parent':offered['parent'],'lease_identity':offered['lease_identity'],'offer_semantic_sha256':h.digest(offered),'schema':3,'kind':'source40_actual_parent_health_ACK','ready_sha256':h.digest(ready),'producer_pid':17,'producer_start_ticks':123,'health_sha256':'d'*64,'kernel_receipt_sha256':'e'*64,'observed_epoch':now-1};return plan,manifest,ready,health,ack,now
 def test_actual_postsemantic_owned_request_then_health_ACK(self):
  plan,manifest,ready,health,ack,now=self.fixture();h.admit_ready(ready,plan,manifest,17,123,now-5,{'parent':ready['parent'],'lease_identity':ready['lease_identity'],'parent_started_epoch':now-6});self.assertEqual(h.admit_ack(ready,ack,health),health)
 def test_changed_owner_source_stale_health_and_presemantic_health_rejected(self):
  for mutate in ('owner','source','ticks','stale','order'):
   plan,manifest,ready,health,ack,now=self.fixture()
   if mutate=='owner':ack['producer_pid']=18
   elif mutate=='source':ack['ready_sha256']='f'*64
   elif mutate=='ticks':ack['producer_start_ticks']=124
   elif mutate=='stale':health.update(started_epoch=now-400,finished_epoch=now-301);ready['observed_epoch']=now-500;ack['ready_sha256']=h.digest(ready)
   else:health['started_epoch']=now-5
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):h.admit_ack(ready,ack,health)
 def test_actual_parent_adapter_health_and_journal_before_ACK(self):
  source=parent.adapted_source();ast.parse(source);self.assertLess(source.index("health_path=health('leaf')"),source.index("atomic(out/'leaf-ack.json',ack)"));self.assertIn("start_ticks(child.pid)==ticks",source);self.assertIn("row['return_code']==0 and row['error'] is None",source)
 def test_postACK_seal_expiry_cannot_refresh_health_epoch(self):
  # Source control ensures the final check occurs after complete byte reads.
  import inspect
  body=inspect.getsource(h.post_ack_seal);self.assertLess(body.index('actual=sha(path)'),body.index('finished=time.time()'));self.assertIn('<=300',body);self.assertNotIn("handoff['actual_health']['finished_epoch']=",body)
 def test_actual_parent_reuse_and_fd_change_refused(self):
  offered={'parent':{'pid':99,'start_ticks':456},'lease_identity':[{'card':0,'device':1,'inode':10},{'card':1,'device':1,'inode':11}]}
  with patch.object(h.os,'getppid',return_value=99),patch.object(h,'process',return_value={'pid':99,'start_ticks':457}),patch.object(h,'leases',return_value=offered['lease_identity']),self.assertRaises(ValueError):h.parent_continuity(offered)
  with patch.object(h.os,'getppid',return_value=99),patch.object(h,'process',return_value=offered['parent']),patch.object(h,'leases',return_value=[{'card':0,'device':1,'inode':12},{'card':1,'device':1,'inode':11}]),self.assertRaises(ValueError):h.parent_continuity(offered)
 def test_typed_health_cards_and_ACK_parent_fields_rejected(self):
  _,_,ready,health,ack,_=self.fixture();health['cards']=[0.0,1]
  with self.assertRaises(ValueError):h.admit_ack(ready,ack,health)
  _,_,ready,health,ack,_=self.fixture();ack['parent']['pid']=99.0
  with self.assertRaises(ValueError):h.admit_ack(ready,ack,health)
if __name__=='__main__':unittest.main()
