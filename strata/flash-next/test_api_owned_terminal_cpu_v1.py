"""Strict producer-owned EOS/cancel completion despite preserved stale observer snapshot."""
import copy,tempfile,unittest,json
from pathlib import Path
from api_owned_terminal_association_v1 import Terminals,finalized_associations
class Controls(unittest.TestCase):
 def rows(self):
  return [{'sequence':1,'kind':'engine_begin','call':1,'engine_pid':77,'engine_generation':1,'submitted_ids':[10,11],'rendered_matches_submitted':True},{'sequence':2,'kind':'native_send','call':1,'engine_pid':77,'line':'BGEN 0 32 seed=1 fresh=1 rid=1 10,11'},{'sequence':3,'kind':'native_receive','engine_pid':77,'line':'T 5 rid=1 slotgen=1'},{'sequence':4,'kind':'native_receive','engine_pid':77,'line':'DONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2 0 rid=1 slotgen=1'},{'sequence':5,'kind':'native_receive','engine_pid':77,'line':'BADM 0 1 rid=1 slotgen=1'},{'sequence':6,'kind':'native_receive','engine_pid':77,'line':'BT 0 6 rid=1 slotgen=1'},{'sequence':7,'kind':'native_receive','engine_pid':77,'line':'BDONE 0 2 stop 1 rid=1 slotgen=1'},{'sequence':8,'kind':'native_send','call':1,'engine_pid':77,'line':'BSTOP 0 rid=1 slotgen=1'},{'sequence':9,'kind':'engine_end','call':1,'engine_pid':77,'engine_generation':1,'rid':1,'generated_ids':[5,6],'pinned_eos_ids':[6],'cancelled':False,'consumer_closed':True,'engine_last':{'finish':'length'},'error':{'type':'GeneratorExit','message':'Noncancelled consumer close without pinned EOS native stop'}}]
 def qualify(self,rows):
  t=Terminals()
  for row in rows:t.consume(row)
  return t.association(1)
 def test_realterminal_over_stale_snapshot_raw_preserved(self):
  rows=self.rows();before=copy.deepcopy(rows);r=self.qualify(rows);self.assertEqual(rows,before);self.assertEqual(r['actual_native_terminal']['finish'],'stop');self.assertEqual(r['legacy_engine_last_preserved']['finish'],'length');self.assertFalse(r['frozen_raw_engine_end_rewritten']);self.assertFalse(r['cache_or_math_qualified'])
 def test_wrong_token_count_PID_slot_generation_or_finish_refused(self):
  for index,key,value in [(6,'line','BDONE 0 1 stop 1 rid=1 slotgen=1'),(6,'engine_pid',78),(6,'line','BDONE 0 2 stop 1 rid=1 slotgen=2'),(6,'line','BDONE 0 2 length 1 rid=1 slotgen=1'),(8,'engine_generation',2),(8,'generated_ids',[5,7])]:
   rows=self.rows();rows[index][key]=value
   with self.subTest(value=value),self.assertRaises(ValueError):self.qualify(rows)
 def test_missing_or_late_terminal_not_admitted(self):
  rows=self.rows();del rows[6]
  with self.assertRaises(ValueError):self.qualify(rows)
  rows=self.rows();rows[6]['sequence']=10;rows=rows[:6]+rows[7:]+[rows[6]]
  with self.assertRaises(ValueError):self.qualify(rows)
 def test_nonEOS_consumerclose_or_unknownerror_not_waived(self):
  for key,value in [('pinned_eos_ids',[7]),('error',{'type':'EngineGenerationChanged','message':'actual incarnation changed'}),('error',{'type':'OSError','message':'actual transport failure'})]:
   rows=self.rows();rows[-1][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):self.qualify(rows)
 def test_EOS_close_cancel_requires_owned_postEOS_stop_and_drain(self):
  rows=self.rows();rows[6]['line']='BSTOP 0 rid=1 slotgen=1';rows[6]['kind']='native_send';rows[6]['call']=1;rows[7]['line']='BDONE 0 2 cancel 1 rid=1 slotgen=1';rows[7]['kind']='native_receive';self.assertTrue(self.qualify(rows)['source_bound_EOS_close_stop_drain'])
  rows[6]['line']='BSTOP 0 rid=2 slotgen=1'
  with self.assertRaises(ValueError):self.qualify(rows)
 def test_realclientcancel_remains_canceled_not_naturalEOS(self):
  rows=self.rows();rows[6]['line']='BSTOP 0 rid=1 slotgen=1';rows[6]['kind']='native_send';rows[6]['call']=1;rows[7]['line']='BDONE 0 2 cancel 1 rid=1 slotgen=1';rows[7]['kind']='native_receive';rows[-1].update(cancelled=True,error=None);self.assertTrue(self.qualify(rows)['actual_client_cancelled'])
 def test_sparse_stream_requires_finalnewline_and_complete_roster(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'trace';path.write_text('\n'.join(json.dumps(row,sort_keys=True) for row in self.rows())+'\n');self.assertTrue(finalized_associations(path)[0]['terminal_association_qualified']);path.write_text(path.read_text()+'{"kind":"ignoredURtruncation"')
   with self.assertRaises(ValueError):finalized_associations(path)
if __name__=='__main__':unittest.main()
