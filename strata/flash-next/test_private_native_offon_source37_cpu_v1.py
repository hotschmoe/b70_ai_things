"""Tiny source-indexed protocol/roster/raw negative controls; no model/GPU."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import private_native_protocol_recollection_v2 as p
import validate_private_native_offon_source37_v1 as r
from batch_numerical_protocol_v2 import Roster
class Protocol(unittest.TestCase):
 def fixture(self,diagnostic):
  plan={'slots':2,'diagnostic':diagnostic,'max_new':32,'max_new_by_request':[32,32],'cancel_index':0,'tokens':{'warm':[[10,11],[20,21]],'target':[[30,31],[40,41]]},'args':['--max-context','2048']};roster=Roster(2);lines=[]
  for i in range(2):roster.submit(1001+i,i,plan['tokens']['warm'][i],32)
  def emit(line):lines.append(line);roster.consume(line)
  emit('INFO batch_slots=2 batch_requested=2 batch_protocol=2');emit('READY CPU_SYNTHETIC')
  def admit(rid,slot,gen,token):
   emit(f'T {token} rid={rid} slotgen={gen}');emit(f'DONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2 0 rid={rid} slotgen={gen}');emit(f'BADM {slot} 1 rid={rid} slotgen={gen}')
  for i in range(2):admit(1001+i,i,1,100+i)
  if diagnostic:emit('SBF batch_event pid=1 enginegen=77 event=1 rows=2 active_mask=3 completed=1')
  for i in range(2):emit(f'BT {i} 248046 rid={1001+i} slotgen=1')
  for i in range(2):emit(f'BDONE {i} 2 stop 0 rid={1001+i} slotgen=1')
  if diagnostic:lines.append('HARNESS ARM after actual unarmed multi-row terminal')
  roster.bt_slots.clear()
  for i in range(2):roster.submit(2001+i,i,plan['tokens']['target'][i],32)
  for i in range(2):admit(2001+i,i,2,200+i)
  if diagnostic:emit('SBF batch_event pid=1 enginegen=77 event=1 rows=2 active_mask=3 completed=1');roster.cancel(2001)
  for i in range(2):emit(f'BT {i} {300+i} rid={2001+i} slotgen=2')
  if not diagnostic:roster.cancel(2001)
  emit('BDONE 0 2 cancel 0 rid=2001 slotgen=2');emit('BT 1 248046 rid=2002 slotgen=2');emit('BDONE 1 3 stop 0 rid=2002 slotgen=2')
  return plan,'\n'.join(lines),{str(k):v for k,v in roster.requests.items()}
 def test_exact_warm_target_ids_cancel_ack_and_offlimits(self):
  for diagnostic in (0,1):
   plan,trace,saved=self.fixture(diagnostic);result=p.recollect(plan,trace,saved);self.assertEqual(set(result['histories']),{'warm_0','warm_1','target_0','target_1'});self.assertTrue(result['histories']['target_0']['native_cancel_completed']);self.assertFalse(result['cancellation']['actual_stdin_bytes_logged']);self.assertFalse(result['actual_OFF_Nrow_tensor_event_observed']);self.assertFalse(result['actual_observer_tensor_capture_lines_present']);self.assertFalse(result['counter_rows']['2001']['read_from_reread_to_observer_fields_observed'])
 def test_off_cannotborrow_ON_marker_or_observer(self):
  plan,trace,saved=self.fixture(0)
  for line in ('HARNESS ARM after actual unarmed multi-row terminal','SBF batch_event pid=1 enginegen=77 event=1 rows=2 active_mask=3 completed=1','SBF vector CPU_SYNTHETIC'):
   with self.subTest(line=line),self.assertRaises(ValueError):p.recollect(plan,trace+'\n'+line,saved)
 def test_stale_generation_foreign_slot_unknown_error(self):
  plan,trace,saved=self.fixture(1)
  for bad in (trace.replace('slotgen=2','slotgen=1',1),trace.replace('BT 0 300','BT 1 300'),trace+'\nERR CPU_UNKNOWN_FAILURE'):
   with self.assertRaises((ValueError,KeyError)):p.recollect(plan,bad,saved)
 def test_missing_dup_terminal_or_changed_saved_records(self):
  plan,trace,saved=self.fixture(1)
  for bad in (trace.replace('BDONE 1 3 stop 0 rid=2002 slotgen=2',''),trace+'\nBDONE 1 3 stop 0 rid=2002 slotgen=2'):
   with self.assertRaises(ValueError):p.recollect(plan,bad,saved)
  saved['2002']['ids'][0]=9
  with self.assertRaises(ValueError):p.recollect(plan,trace,saved)
 def test_no32claim_from_short_length_or_nonnativeEOS(self):
  plan,trace,saved=self.fixture(1)
  for bad in (trace.replace('BDONE 0 2 stop','BDONE 0 2 length'),trace.replace('BT 0 248046 rid=1001','BT 0 123 rid=1001')):
   with self.assertRaises(ValueError):p.recollect(plan,bad,saved)
 def test_wrong_cold_resume_or_native64_budget(self):
  plan,trace,saved=self.fixture(0);plan['max_new_by_request']=[64,64]
  with self.assertRaises(ValueError):p.recollect(plan,trace,saved)
  plan,trace,saved=self.fixture(0);bad=trace.replace('length 0 0 0 0 0 0 0 0 2 0','length 0 0 1 0 0 0 0 0 1 0',1)
  with self.assertRaises(ValueError):p.recollect(plan,bad,saved)
 def test_compare_complete_warm_uncancelled_and_cancel_common_prefix(self):
  off=p.recollect(*self.fixture(0));on=p.recollect(*self.fixture(1));self.assertTrue(p.compare_histories(off,on,0)['passed']);on['histories']['warm_1']['native_generated_ids'][0]=99
  with self.assertRaises(ValueError):p.compare_histories(off,on,0)
class Plans(unittest.TestCase):
 def fixture(self):
  base={'harness_generation':40,'slots':2,'kind':'native','cards':[0,1],'lane':'source37','args':['--layer-split','32'],'env':{'ZE_AFFINITY_MASK':'0,1','STRATA_STAGE_MIRRORS':'1','STRATA_STAGE_MIRROR_SEGMENT_MIB':'1024'}};off=dict(base,diagnostic=0);on=dict(base,diagnostic=1)
  for plan in (off,on):
   alias=r.ctrl.api.experimental_alias({'args':plan['args'],'env':plan['env']},2,bool(plan['diagnostic']),'source37');plan.update(research_alias=alias,registry_binding=r.ctrl.api.registry_gate(alias))
  return off,on
 def test_exact_diagnostic_only_plan_match(self):self.assertTrue(r.matched_plans(*self.fixture()))
 def test_settings_token_or_math_env_drift_refused(self):
  off,on=self.fixture();on['args']=['--layer-split','16']
  with self.assertRaises(ValueError):r.matched_plans(off,on)
  off,on=self.fixture();on['env']=dict(on['env'],STRATA_CRITICAL_PATH_TRACE='1')
  with self.assertRaises(ValueError):r.matched_plans(off,on)
 def test_same_or_wrong_topology_and_old_harness_refused(self):
  for field,value in [('diagnostic',0),('cards',[0]),('harness_generation',39),('kind','api')]:
   off,on=self.fixture();on[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):r.matched_plans(off,on)
 def test_roster_external_exactpath_and_missing_groups(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);off=root/'OFF';on=root/'ON';off.mkdir();on.mkdir();self.assertEqual(r.serial_roster_path(on),root/'ON.serial49-roster-v1.json')
   with self.assertRaises(ValueError):r.serial_binding(off,on,{}, {},root/'WRONG.json')
class Serial(unittest.TestCase):
 def test_tiny_complete49_and_wrong_bytes_missing_duplicate_group(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);off=root/'OFF';on=root/'ON';serial=root/'SERIAL';off.mkdir();(on/'child').mkdir(parents=True);serial.mkdir();jobs=[{'rid':2001,'role':'admission','ids':[10,11]}];r.write(on/'child/serial-jobs.json',{'jobs':jobs})
   for path in [off/'parent-qualification.json',on/'parent-qualification.json',serial/'parent-qualification.json',serial/'input-plan.snapshot.json']:r.write(path,{'CPU_SYNTHETIC':True})
   batch={};actual={}
   for layer in range(-1,48):
    a=root/('a'+str(layer));b=root/('b'+str(layer));a.write_bytes(b'\x00\x00\x80\x3f');b.write_bytes(a.read_bytes());key=((2001,'admission'),layer);batch[key]=str(a);actual[key]=str(b)
   path=r.serial_roster_path(on);row={'root':str(serial),'group_index':0,'parent_sha256':r.sha(serial/'parent-qualification.json'),'plan_sha256':r.sha(serial/'input-plan.snapshot.json')};roster={'schema':1,'off_root':str(off),'on_root':str(on),'off_parent_sha256':r.sha(off/'parent-qualification.json'),'on_parent_sha256':r.sha(on/'parent-qualification.json'),'jobs_sha256':r.sha(on/'child/serial-jobs.json'),'groups':[row]};r.write(path,roster)
   plan={'engine_receipt_sha256':'CPU_NEW','prepared_sha256':'CPU_PREP','cards':[0,1],'args':['CPU_FIXED'],'env':{}};source=dict(plan,kind='serial',slots=2,group_index=0,batch_parent=str(on))
   with patch.object(r.audit,'parent_arm',return_value=({},source,{})),patch.object(r.audit,'serial_vectors',return_value=actual):
    self.assertTrue(r.serial_binding(off,on,plan,batch,path)['actual_complete_serial49_qualified'])
    Path(actual[((2001,'admission'),0)]).write_bytes(b'\x00\x00\x00\x40')
    with self.assertRaises(ValueError):r.serial_binding(off,on,plan,batch,path)
    roster['groups']=[];r.write(path,roster)
    with self.assertRaises(ValueError):r.serial_binding(off,on,plan,batch,path)
    roster['groups']=[row,row];r.write(path,roster)
    with self.assertRaises(ValueError):r.serial_binding(off,on,plan,batch,path)
if __name__=='__main__':unittest.main()
