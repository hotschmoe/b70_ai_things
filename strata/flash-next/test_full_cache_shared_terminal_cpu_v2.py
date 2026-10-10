import copy,hashlib,unittest
import full_cache_shared_terminal_v2 as s
import api_owned_terminal_association_v2 as old

class Partial(unittest.TestCase):
 def rows(self):
  return [{'sequence':1,'kind':'engine_begin','call':1,'engine_pid':7,'engine_generation':1,'max_new':64,'submitted_ids':list(range(10)),'rendered_matches_submitted':True},{'sequence':2,'kind':'native_send','call':1,'engine_pid':7,'line':'BGEN 0 64 seed=1 fresh=1 rid=1 '+','.join(map(str,range(10)))},{'sequence':3,'kind':'native_receive','engine_pid':7,'line':'PP 4 10 1.0'},{'sequence':4,'kind':'native_send','call':1,'engine_pid':7,'line':'STOP'},{'sequence':5,'kind':'native_receive','engine_pid':7,'line':'DONE 0 10 0 0 cancel 0 0 0 0 0 0 0 0 4 0 rid=1 slotgen=1'},{'sequence':6,'kind':'native_receive','engine_pid':7,'line':'BADM 0 0 rid=1 slotgen=1'},{'sequence':7,'kind':'engine_end','call':1,'engine_pid':7,'engine_generation':1,'rid':1,'generated_ids':[],'pinned_eos_ids':[9],'cancelled':True,'consumer_closed':True,'engine_last':{'finish':'cancel'},'error':None}]
 def test_actual_proper_prefill_cancel_and_frozen_old_refusal(self):
  rows=self.rows();before=copy.deepcopy(rows);self.assertTrue(s.associate_events(rows)[1]['actual_client_cancelled']);self.assertEqual(rows,before)
  with self.assertRaises(ValueError):old.associate_events(rows)
 def test_owned_PP_STOP_and_real_client_cancel_are_all_required(self):
  for index,key,value in [(2,'line','PP 5 10 1.0'),(2,'engine_pid',8),(3,'call',2),(6,'cancelled',False),(5,'line','BADM 0 1 rid=1 slotgen=1')]:
   rows=self.rows();rows[index][key]=value
   with self.subTest(index=index),self.assertRaises(ValueError):s.associate_events(rows)
  for index in [2,3]:
   rows=self.rows();del rows[index]
   with self.assertRaises(ValueError):s.associate_events(rows)
 def test_exact_three_boundary_port_restores_original_bytes(self):
  text=s.derived_source()
  for new,oldtext in [(s.NEW_STOP,s.OLD_STOP),(s.NEW_COUNT,s.OLD_COUNT),(s.NEW_RECEIVE,s.OLD_RECEIVE)]:self.assertEqual(text.count(new),1);text=text.replace(new,oldtext)
  self.assertEqual(hashlib.sha256(text.encode('ascii')).hexdigest(),s.TEMPLATE_SHA)

if __name__=='__main__':unittest.main()
