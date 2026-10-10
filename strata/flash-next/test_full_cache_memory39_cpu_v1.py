import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
import full_cache_memory39_contract_v1 as c
import prepare_full_cache_memory39_v1 as s

class MemoryTests(unittest.TestCase):
 def sample(self,**updates):
  values=dict(budget_bytes=1000,parked_bytes=100,reusable_bytes=40,held_snapshot_bytes=60,incoming_estimate_bytes=400,local_reuse_bytes=80,allocated_snapshot_bytes=120)
  values.update(updates);return c.logical_sample(**values)
 def test_estimate_overlaps_pending_allocation(self):
  row=self.sample();self.assertEqual(row['observed_owned_snapshot_bytes'],400);self.assertEqual(row['logical_reservation_bytes'],600)
  row=self.sample(incoming_estimate_bytes=10);self.assertEqual(row['logical_reservation_bytes'],400)
 def test_reuse_transfer_not_double_charged(self):
  before=self.sample(reusable_bytes=80,local_reuse_bytes=0,allocated_snapshot_bytes=0,incoming_estimate_bytes=0)
  after=self.sample(reusable_bytes=0,local_reuse_bytes=80,allocated_snapshot_bytes=0,incoming_estimate_bytes=0)
  self.assertEqual(before['observed_owned_snapshot_bytes'],after['observed_owned_snapshot_bytes'])
 def test_capture_reuse_move_and_put_preserve_owned_bytes(self):
  local=self.sample(reusable_bytes=0,local_reuse_bytes=80,allocated_snapshot_bytes=120)
  image=self.sample(reusable_bytes=0,local_reuse_bytes=0,allocated_snapshot_bytes=200)
  parked=self.sample(parked_bytes=300,reusable_bytes=0,local_reuse_bytes=0,allocated_snapshot_bytes=0,incoming_estimate_bytes=0)
  self.assertEqual(local['observed_owned_snapshot_bytes'],image['observed_owned_snapshot_bytes']);self.assertEqual(image['observed_owned_snapshot_bytes'],parked['observed_owned_snapshot_bytes'])
 def test_refused_reservation_is_not_fake_allocation(self):
  row=self.sample(budget_bytes=500);self.assertFalse(row['reservation_within_budget']);self.assertLess(row['observed_owned_snapshot_bytes'],row['logical_reservation_bytes'])
 def test_types_extent_and_overflow(self):
  for update in ({'held_snapshot_bytes':True},{'reusable_bytes':-1},{'parked_bytes':2**64},{'parked_bytes':2**64-1}):
   with self.subTest(update=update),self.assertRaises(ValueError):self.sample(**update)
 def row(self,event=1,**updates):
  data=self.sample(**updates);return dict(data,kind='logical_memory',pid=20,cache_scope=400,event=event,observed_owned_snapshot_peak_bytes=data['observed_owned_snapshot_bytes'],logical_reservation_peak_bytes=data['logical_reservation_bytes'],incoming_estimate_overlaps_owned_pending=True,whole_process_peak_qualified=False,physical_reclamation_qualified=False,device_physical_memory_qualified=False)
 def test_sampled_peak_and_mutations(self):
  first=self.row();second=self.row(2,held_snapshot_bytes=0,incoming_estimate_bytes=0)
  second['observed_owned_snapshot_peak_bytes']=first['observed_owned_snapshot_bytes'];second['logical_reservation_peak_bytes']=first['logical_reservation_bytes']
  self.assertFalse(c.recollect([first,second])['actual_full_cache_qualified'])
  for key,value in [('event',1),('cache_total_bytes',0),('logical_reservation_peak_bytes',0),('whole_process_peak_qualified',True)]:
   bad=dict(second,**{key:value})
   with self.subTest(key=key),self.assertRaises(ValueError):c.recollect([first,bad])

class SourceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.old,cls.new=s.reconstruct()
 def test_pristine_patch_all66(self):
  with tempfile.TemporaryDirectory(prefix='source39-',dir='/tmp')as d:
   root=Path(d)
   for n,text in self.old.items():p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
   run=subprocess.run(['patch','--batch','--fuzz=0','-p1'],input=s.PATCH.read_bytes(),cwd=d,capture_output=True)
   self.assertEqual(run.returncode,0,run.stderr.decode('ascii'))
   for n,text in self.new.items():self.assertEqual((root/n).read_bytes(),text.encode(),n)
 def test_counts_off_and_immutable38(self):
  plan=s.build_plan();self.assertEqual((len(plan['expected_patched_source_sha256']),len(plan['added_header_payloads']),len(plan['patches'])),(66,30,39));self.assertEqual(set(plan['overlay_files']),set(self.new))
  self.assertEqual(self.new[s.source38.H],self.old[s.source38.H])
  h=self.new[s.H];self.assertIn('if(!full_cache38::enabled())return;',h);self.assertNotIn('sycl::',h);self.assertNotIn('queues_wait',h)
  resources=h[h.index('inline void chains('):]
  self.assertNotIn('\\"rid\\"',resources)
  self.assertIn('current_main_body_rid',resources)
  v=self.new['include/strata/core/conversation_cache.hpp'];self.assertIn('if(!full_cache38::enabled())return;\n        full_cache_memory39::record',v)
 def test_policy_conditions_unchanged(self):
  old=self.old['include/strata/core/conversation_cache.hpp'];new=self.new['include/strata/core/conversation_cache.hpp']
  for line in old.splitlines():
   if 'budget_ -'in line or 'std::find_if'in line:self.assertIn(line,new)
  g=self.new['sycl/src/program/generate.cpp']
  for phase in ['park_estimated','capture_before','capture_main_after','capture_stage_after','restore_before_held_release','restore_after_held_release','slot_capture_estimated','main_body_complete','BSTOP_applied','BSTEP_before','BT_emitted','BDONE_emitted','DONE_emitted','BADM_emitted']:self.assertIn('"'+phase+'"',g)
 def test_unique_direct_rid_constraint_stays_failclosed(self):
  observer=self.new['sycl/include/strata/core/batch_fidelity_observer.hpp']
  self.assertIn('!records().find(rid)',observer);self.assertIn('records().solo(rid,prompt,engine_generation())',observer)
  contract=self.new['sycl/include/strata/core/batch_fidelity_contract.hpp'];self.assertIn('!e->admission||!e->later||!e->closed_cancel',contract)

class LifetimeTests(unittest.TestCase):
 def rows(self):
  phases=[('DONE_emitted',1,0,-1,''),('BADM_emitted',1,0,1,''),('BSTOP_applied',1,0,-1,'cancel'),('BSTEP_before',1,1,-1,''),('BT_emitted',2,1,-1,''),('BDONE_emitted',2,1,0,'cancel')]
  return [dict(kind='native_lifetime',event=i+1,rid=20,slotgen=3,slot=0,phase=p,generated=n,batch_window=w,continuation=k,finish=f,actual_HTTP_client_terminal_qualified=False)for i,(p,n,w,k,f)in enumerate(phases)]
 def test_actual_later_cancel_terminal_scope(self):
  result=c.batch_terminal_binding(self.rows(),20,3,0);self.assertEqual(result['generated'],2);self.assertTrue(result['native_terminal_observed']);self.assertFalse(result['actual_HTTP_client_terminal_qualified'])
 def test_bad_owner_grammar_counts_and_chronology(self):
  rows=self.rows()
  for index,key,value in [(1,'continuation',0),(3,'slotgen',2),(4,'generated',4),(5,'event',1),(5,'finish','unknown'),(5,'actual_HTTP_client_terminal_qualified',True)]:
   bad=copy.deepcopy(rows);bad[index][key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):c.batch_terminal_binding(bad,20,3,0)
  with self.assertRaises(ValueError):c.batch_terminal_binding(rows[:-1],20,3,0)
  with self.assertRaises(ValueError):c.batch_terminal_binding(rows[:1]+rows[3:],20,3,0)
 def multiple_rows(self):
  original=self.rows();rows=original[:5]
  rows.append(dict(original[4],generated=3))
  rows.append(dict(original[3],generated=3,batch_window=2))
  rows.append(dict(original[4],generated=4,batch_window=2))
  rows.append(dict(original[4],generated=5,batch_window=2))
  rows.append(dict(original[5],generated=5,batch_window=2,finish='length'))
  return [dict(row,event=i+1)for i,row in enumerate(rows)]
 def test_multiple_bt_same_actual_window_and_next_bstep(self):
  rows=self.multiple_rows();result=c.batch_terminal_binding(rows,20,3,0)
  self.assertEqual(result['generated'],5)
  self.assertTrue(result['native_terminal_observed'])
  # The actual source accepts candidates in its existing j<keep[b] loop.
  _,new=s.reconstruct();source=new['sycl/src/program/generate.cpp']
  begin=source.index('for (int j = 0; j < keep[b]; ++j)')
  self.assertLess(begin,source.index('lifetime("BT_emitted"',begin))
 def test_multiple_bt_window_count_and_postterminal_negatives(self):
  rows=self.multiple_rows()
  for index,key,value in [(5,'generated',2),(5,'generated',4),(5,'batch_window',2),(6,'batch_window',1),(6,'generated',4),(9,'batch_window',1),(9,'generated',4),(5,'generated',True),(6,'batch_window',True)]:
   bad=copy.deepcopy(rows);bad[index][key]=value
   with self.subTest(index=index,key=key),self.assertRaises(ValueError):c.batch_terminal_binding(bad,20,3,0)
  bad=rows+[dict(rows[8],event=11,generated=6)]
  with self.assertRaises(ValueError):c.batch_terminal_binding(bad,20,3,0)
 def test_empty_window_cannot_finish_or_start_next_window(self):
  rows=self.rows();empty=rows[:4]+[dict(rows[-1],event=5,generated=1)]
  with self.assertRaises(ValueError):c.batch_terminal_binding(empty,20,3,0)
  nextstep=rows[:4]+[dict(rows[3],event=5,batch_window=2)]
  with self.assertRaises(ValueError):c.batch_terminal_binding(nextstep,20,3,0)

if __name__=='__main__':unittest.main()
