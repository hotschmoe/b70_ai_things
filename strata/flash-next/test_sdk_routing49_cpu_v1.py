"""Complete SDK keyword routing and actual-shaped tiny witness joins."""
import ast,copy,hashlib,importlib,inspect,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import prepare_pack_gate_ports49_v1 as producer
import operation_sdk37_byte_witness_v1 as w
import sdk49_operation_scope_v1 as scope
import sdk_witness_binding49_v1 as saved
HERE=Path(__file__).resolve().parent
class Controls(unittest.TestCase):
 def test_every_reachable_sdk_callee_required_keyword_no_missing_or_foreign(self):
  names=list(producer.MAP.values())+['c137_prepared_pack49_v1','c137_sdk37_digest_ports_v1','batch_numerical_proofs_v49','batch_numerical_execution_v49','audit_batch_numerical_suite_v49','run_batch_numerical_pilot_v49','run_batch_serial_controls_v49','batch49_prelease_v1']
  for name in names:
   module=importlib.import_module(name);tree=ast.parse(Path(module.__file__).read_text());aliases={}
   for node in ast.walk(tree):
    if isinstance(node,ast.Import):
     for a in node.names:aliases[a.asname or a.name]=a.name
    if isinstance(node,ast.ImportFrom):
     for a in node.names:
      if a.name!='*':aliases[a.asname or a.name]=(node.module,a.name)
   for node in ast.walk(tree):
    if not isinstance(node,ast.Call):continue
    fn=node.func;obj=None
    if isinstance(fn,ast.Name):
     obj=getattr(module,fn.id,None)
     if fn.id in aliases and isinstance(aliases[fn.id],tuple):m,key=aliases[fn.id];obj=getattr(importlib.import_module(m),key)
    elif isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Name):
     if fn.value.id in aliases and isinstance(aliases[fn.value.id],str):obj=getattr(importlib.import_module(aliases[fn.value.id]),fn.attr,None)
     elif fn.value.id=='ctrl' and name=='batch49_prelease_v1':obj=getattr(importlib.import_module('batch_numerical_execution_v49'),fn.attr)
     elif hasattr(module,fn.value.id):obj=getattr(getattr(module,fn.value.id),fn.attr,None)
    elif isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Call) and isinstance(fn.value.func,ast.Name) and fn.value.func.id=='__import__':obj=getattr(importlib.import_module(fn.value.args[0].value),fn.attr)
    provided=any(k.arg=='sdk_epoch' for k in node.keywords)
    try:signature=inspect.signature(obj)
    except (TypeError,ValueError):
     self.assertFalse(provided,'Unresolved SDK callee '+name+' '+ast.unparse(fn));continue
    if 'sdk_epoch' in signature.parameters:self.assertTrue(provided,'Missing mandatory SDK propagation '+name+' '+ast.unparse(fn))
    else:self.assertFalse(provided,'Foreign SDK keyword '+name+' '+ast.unparse(fn))
 def test_parent_child_independent_scope_seal_leaf_and_no_health_age_relaxation(self):
  parent=(HERE/'qualify_batch_numerical_v49.py').read_text();self.assertLess(parent.index('sdk_epoch.seal_predevice()'),parent.index("pre=health('pre')"));self.assertIn("parent['sdk_operation_witness']=sdk_epoch.finalize()",parent)
  controller=(HERE/'batch_numerical_execution_v49.py').read_text();self.assertIn('sdk_for_plan(plan,10800)',controller.replace(' ',''))
  for name,launch in [('run_batch_numerical_pilot_v49.py','stream = NativeStream'),('run_batch_serial_controls_v49.py','protocol = CacheOffProtocol')]:
   source=(HERE/name).read_text();self.assertLess(source.index('sdk_epoch.seal_predevice()'),source.index(launch));self.assertIn('<= 300',source);self.assertLess(source.index('<= 300'),source.index('sdk_epoch.seal_predevice()'))
 def test_unchanged_health_age_sequence_fails_before_seal_or_leaf(self):
  import run_batch_numerical_pilot_v49 as native
  from types import SimpleNamespace
  events=[];plan={'lane':'source37','diagnostic':0,'env':{},'native_driver_sha256':'CPU','driver_sha256':'CPU','prepared':'/CPU','prepared_sha256':'CPU','engine_receipt_sha256':'SDK','engine_root':'/CPU'};c1=SimpleNamespace(leased=lambda cards:events.append('lease'),sha=lambda path:'CPU',read=lambda path:{'passed':True,'cards':[0,1],'finished_epoch':100})
  plan['cards']=[0,1];admitted=SimpleNamespace(plan=plan,verify=lambda p:None);args=SimpleNamespace(plan=Path('/CPU'),admitted=admitted,diagnostic=0,pre_health=Path('/CPUhealth'),output=Path('/CPUout'))
  def require(ok,msg):
   if not ok:raise ValueError(msg)
  c1.require=require
  with patch.object(native,'providers',return_value=(c1,None)),patch.object(native,'genuine_baseline',return_value=({'engine_receipt_sha256':'SDK'},{})),patch.object(native,'engine_binding',return_value={}),patch.object(native,'source_observers_off',return_value=None),patch.object(native.time,'time',return_value=401),patch.object(native,'NativeStream',side_effect=AssertionError('must not launch leaf')):
   epoch=SimpleNamespace(seal_predevice=lambda:events.append('seal'))
   self.assertRaises(ValueError,native.run,args,pack_epoch=epoch,sdk_epoch=epoch);self.assertNotIn('seal',events)
 def test_health_that_expires_during_byte_seals_fails_before_leaf(self):
  import run_batch_numerical_pilot_v49 as native
  from types import SimpleNamespace
  events=[];clock=[399];plan={'lane':'source37','diagnostic':0,'env':{},'native_driver_sha256':'CPU','driver_sha256':'CPU','prepared':'/CPU','prepared_sha256':'CPU','engine_receipt_sha256':'SDK','engine_root':'/CPU','cards':[0,1]};c1=SimpleNamespace(leased=lambda cards:None,sha=lambda path:'CPU',read=lambda path:{'passed':True,'cards':[0,1],'finished_epoch':100})
  args=SimpleNamespace(plan=Path('/CPU'),admitted=SimpleNamespace(plan=plan,verify=lambda p:None),diagnostic=0,pre_health=Path('/CPUhealth'),output=Path('/CPUout'))
  def seal():events.append('seal');clock[0]=401
  with patch.object(native,'providers',return_value=(c1,None)),patch.object(native,'genuine_baseline',return_value=({'engine_receipt_sha256':'SDK'},{})),patch.object(native,'engine_binding',return_value={}),patch.object(native,'source_observers_off',return_value=None),patch.object(native.time,'time',side_effect=lambda:clock[0]),patch.object(native,'NativeStream',side_effect=AssertionError('no expired leaf')):
   epoch=SimpleNamespace(seal_predevice=seal);self.assertRaises(ValueError,native.run,args,pack_epoch=epoch,sdk_epoch=epoch);self.assertEqual(events,['seal','seal'])
 def test_scope_union_selected_third_root_and_duplicate_selected_pair(self):
  self.assertEqual(scope.roots('/CPUone','/CPUpair','/CPUpair'),[Path('/CPUone/prepared.json'),Path('/CPUpair/prepared.json')]);self.assertEqual(len(scope.roots('/CPUone','/CPUpair','/CPUthird')),3)
 def tiny(self,r):
  rows=[];receipts=[]
  for i,role in enumerate(w.ROLES):
   p=r/str(i);p.write_bytes(b'\x7fELF test '+str(i).encode());rows.append((role,p,hashlib.sha256(p.read_bytes()).hexdigest()))
  for i in range(2):
   p=r/('receipt'+str(i));p.write_text('{}');receipts.append((p,hashlib.sha256(p.read_bytes()).hexdigest()))
  return rows,receipts
 def test_saved_full_three_boundaries_rejoin_current_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.tiny(Path(d));e=w.SDK37Epoch(rows,receipts);e.require_current_roster(rows,receipts);e.digest_for(*rows[0]);e.seal_predevice();proof=e.finalize();current=w.SDK37Epoch(rows,receipts);self.assertEqual(saved.binding(proof,current,proof['owner_pid'],proof['boundaries'][1]['finished_epoch'],proof['boundaries'][2]['started_epoch'])['unique_executables'],9)
 def test_saved_owner_counter_scope_shape_and_boundary_mutations_refused(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.tiny(Path(d));e=w.SDK37Epoch(rows,receipts);e.digest_for(*rows[0]);e.seal_predevice();proof=e.finalize();current=w.SDK37Epoch(rows,receipts)
   for key,value in [('owner_pid',True),('digest_calls',0),('saved_digest_imported',True),('unique_executables',8)]:
    bad=copy.deepcopy(proof);bad[key]=value;self.assertRaises(ValueError,saved.binding,bad,current,proof['owner_pid'],float('inf'),0)
   self.assertRaises(ValueError,saved.binding,proof,current,proof['owner_pid'],proof['boundaries'][1]['finished_epoch']-1,0);self.assertRaises(ValueError,saved.binding,proof,current,proof['owner_pid'],float('inf'),proof['boundaries'][2]['started_epoch']+1)
 def test_changed_current_byte_or_receipt_never_saved_proof_reuse(self):
  with tempfile.TemporaryDirectory() as d:
   rows,receipts=self.tiny(Path(d));e=w.SDK37Epoch(rows,receipts);e.digest_for(*rows[0]);e.seal_predevice();proof=e.finalize();rows[0][1].write_bytes(b'\x7fELF CHANGED');self.assertRaises(ValueError,w.SDK37Epoch,rows,receipts)
if __name__=='__main__':unittest.main()
