import copy,unittest
import full_cache_shared_stale_owner_v2 as s
import test_api_owned_terminal_cpu_v2 as old_fixture

class Stale(unittest.TestCase):
 def owners(self):
  old={'pid':7,'rid':1,'slotgen':1,'slot':0,'engine_generation':1,'send_sequence':1,'terminal_sequence':4,'mode':'BGEN','terminal':True};new={'pid':7,'rid':2,'slotgen':2,'slot':0,'engine_generation':1,'send_sequence':5,'mode':'BGEN','terminal':False,'admitted':True};return old,new
 def test_exact_prior_owner_not_current_owner(self):
  a,b=self.owners();p=s.command(a,b);self.assertEqual(p['command'],'BSTOP 0 rid=1 slotgen=1');self.assertFalse(p['full_cache_runtime_qualified'])
  for key,value in [('pid',8),('slotgen',1),('rid',1),('engine_generation',2),('terminal',True),('admitted',False),('slot',1),('rid',True)]:
   bad=copy.deepcopy(b);bad[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):s.command(a,bad)
 def test_refusal_requires_real_owner_continuation(self):
  a,b=self.owners();p=s.command(a,b);send={'kind':'fullcache_control_send','sequence':6,'engine_pid':7,'line':p['command']};events=[{'kind':'native_receive','sequence':7,'engine_pid':7,'line':p['expected_source_rejection']},{'kind':'native_receive','sequence':8,'engine_pid':7,'line':'BT 0 123 rid=2 slotgen=2'}];terminal={'call':2,'engine_pid':7,'rid':2,'actual_client_cancelled':False,'terminal_association_qualified':True,'actual_native_terminal':{'finish':'length','sequence':9}}
  self.assertTrue(s.recollect(p,send,events,terminal)['actual_old_owner_refused'])
  for changed in [events[:1],events[1:]]:
   with self.assertRaises(ValueError):s.recollect(p,send,changed,terminal)
  terminal['actual_client_cancelled']=True
  with self.assertRaises(ValueError):s.recollect(p,send,events,terminal)
 def test_actual_submitted_owner_rows_not_caller_snapshots(self):
  rows=old_fixture.Controls().rows();new=copy.deepcopy(rows[:5])
  for row in new:
   row['sequence']+=9
   if row.get('call')==1:row['call']=2
   if 'line' in row:row['line']=row['line'].replace('rid=1','rid=2').replace('slotgen=1','slotgen=2')
  seen=rows+new;owners=s.owner_rows(seen);a=next(r for r in owners if r['rid']==1);b=next(r for r in owners if r['rid']==2);proposal=s.command(a,b)
  self.assertEqual(s.admit_proposal(proposal,seen),proposal)
  wrong=copy.deepcopy(proposal);wrong['old_owner']['terminal_sequence']-=1;wrong=s.command(wrong['old_owner'],wrong['current_owner'])
  with self.assertRaises(ValueError):s.admit_proposal(wrong,seen)

if __name__=='__main__':unittest.main()
