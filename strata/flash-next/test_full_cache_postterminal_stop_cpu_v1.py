"""Exact observed producer EOS/send/BDONE/apply order, no old proof rewrite."""
import copy,json,unittest
from full_cache_postterminal_stop_contract_v1 import binding
class Controls(unittest.TestCase):
 def fixture(self):
  shape={'kind':'native_lifetime','pid':13,'rid':2,'slotgen':1,'slot':1,'position':183,'consumed_tokens':184,'actual_HTTP_client_terminal_qualified':False}
  phases=[('DONE_emitted',1,0,-1,'length',-1),('BADM_emitted',1,0,1,'length',-1),('BSTEP_before',1,20,-1,'',1076),('BT_emitted',2,20,-1,'',248046),('BDONE_emitted',2,20,0,'stop',248046),('BSTOP_applied',2,0,-1,'cancel',-1)]
  rows=[dict(shape,event=i+1,phase=p,generated=g,batch_window=w,continuation=c,finish=f,token=t)for i,(p,g,w,c,f,t)in enumerate(phases)]
  events=[{'kind':'native_receive','sequence':8,'engine_pid':13,'line':'BT 1 248046 rid=2 slotgen=1'},{'kind':'native_send','sequence':9,'engine_pid':13,'line':'BSTOP 1 rid=2 slotgen=1'},{'kind':'native_receive','sequence':10,'engine_pid':13,'line':'FC39 '+json.dumps(rows[-2])},{'kind':'native_receive','sequence':12,'engine_pid':13,'line':'FC39 '+json.dumps(rows[-1])},{'kind':'engine_end','sequence':13,'engine_pid':13,'rid':2,'pinned_eos_ids':[248046],'terminal_pinned_eos':True}];return rows,events
 def test_real_order_eos_send_then_terminal_then_inactive_apply(self):
  rows,events=self.fixture();value=binding(rows,2,1,1,events,13);self.assertEqual(value['generated'],2);self.assertFalse(value['postterminal_stop_housekeeping'][0]['live_cancel_or_migration_qualified']);self.assertEqual(rows[-2]['finish'],'stop');self.assertEqual(rows[-1]['finish'],'cancel')
 def test_foreign_owner_count_and_wrong_phase_refused(self):
  rows,events=self.fixture()
  for field,value in [('rid',9),('slotgen',2),('generated',3),('position',184),('batch_window',20),('phase','BSTEP_before')]:
   bad=copy.deepcopy(rows);bad[-1][field]=value;self.assertRaises(ValueError,binding,bad,2,1,1,events,13)
 def test_foreign_send_missing_eos_and_new_request_refused(self):
  rows,events=self.fixture()
  for changed in ('send','eos','new'):
   bad=copy.deepcopy(events)
   if changed=='send':bad[1]['line']='BSTOP 1 rid=9 slotgen=1'
   if changed=='eos':bad[-1]['pinned_eos_ids']=[]
   if changed=='new':bad.insert(2,{'kind':'native_send','sequence':11,'engine_pid':13,'line':'BGEN 1 32 rid=3'})
   self.assertRaises(ValueError,binding,rows,2,1,1,bad,13)
 def test_original_preterminal_stop_remains_strict(self):
  rows,events=self.fixture();rows[-1]['phase']='BADM_emitted';self.assertRaises(ValueError,binding,rows,2,1,1,events,13)
if __name__=='__main__':unittest.main()
