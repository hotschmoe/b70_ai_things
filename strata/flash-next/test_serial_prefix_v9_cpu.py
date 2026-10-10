"""CPU-only V9 protocol/early rejection tests; no engine or device execution."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import serial_prefix_qualification_v9 as v

class Tests(unittest.TestCase):
 def raw_meta(self):
  raw={'label':'other-evict_A','ids':[1,2,3],'fresh':0,'pin':0,'command':v.request_command([1,2,3],64,0,0)}
  meta={'ledger':{'actual_reused':0,'evaluated_prompt_rows':3},'selection':{'candidate_resume':0,'actual_reused':0}}
  return raw,meta
 def test_postterminal_metadata_association_rejects_vector_mutation(self):
  saved={'logits':{'sha256':'original'},'stage_ranges':[[0,48]]}
  self.assertTrue(v.cold_recollected_binding(saved,{'logits':{'sha256':'original'},'stage_ranges':[(0,48)]}))
  fresh=copy.deepcopy(saved);fresh['logits']['sha256']='mutated'
  with self.assertRaises(ValueError):v.cold_recollected_binding(saved,fresh)
 def test_exact_cold_protocol_passes(self):
  raw,meta=self.raw_meta();self.assertTrue(v.cold_establish_binding(raw,meta));self.assertIn('fresh=0 pin=0 ',raw['command'])
 def test_fresh_or_absent_pin_not_cold_establish(self):
  for key,value in [('fresh',1),('pin',None),('pin',1)]:
   raw,meta=self.raw_meta();raw[key]=value
   with self.assertRaises(ValueError):v.cold_establish_binding(raw,meta)
 def test_command_and_actual_work_mutations_rejected(self):
  raw,meta=self.raw_meta();raw['command']=v.request_command(raw['ids'],64,1,None)
  with self.assertRaises(ValueError):v.cold_establish_binding(raw,meta)
  for field,key in [('ledger','actual_reused'),('ledger','evaluated_prompt_rows'),('selection','candidate_resume'),('selection','actual_reused')]:
   raw,meta=self.raw_meta();meta[field][key]=8
   with self.assertRaises(ValueError):v.cold_establish_binding(raw,meta)
 def dispatch(self,group):
  # Execute the real source branch with CPU fake requests. This tests call
  # sequencing and argument propagation, not numerical/lifecycle GPU gates.
  tree=ast.parse(Path(v.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_process')
  branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=="group == 'basic'")
  calls=[];rows=[];cases={k:[1,2,3] for k in ['A','B','C','pin_A','pin_B','turn_A','turn_B','evict_A','evict_B','evict_C','evict_D']}
  def req(label,key,fresh=0,pin=None,cancel=None):
   calls.append((label,key,fresh,pin));raw={'ids':cases[key]};meta={'selection':{'route':'ram_snapshot'},'ledger':{'actual_reused':2 if label.startswith('restored') else 0,'evictions':len(calls)}};return raw,meta
  env={'group':group,'req':req,'plan':{'tokens':{'turn_boundaries':{'B':2},'pin':2,'root':1,'turn_shared':2}},'rows':rows,'requests':[],'require':v.require,'compare_requests':lambda *a:{'passed':True},'write':lambda *a:None,'directory':Path('/CPU-not-written'),'keyed_eviction':lambda *a:{'CPU_STUB':True}}
  exec(compile(ast.Module(body=[branch],type_ignores=[]),'<production V9 request dispatch>','exec'),env)
  return calls
 def test_parked_actual_dispatch_preserves_outgoing_without_reuse(self):
  calls=self.dispatch('parked');self.assertEqual(len(calls),6)
  self.assertEqual([x for x in calls if x[0].startswith('independent-context')],[('independent-context','A',0,0),('independent-context2','A',0,0)])
  self.assertEqual([x[2:] for x in calls if x[0].startswith('reference')],[(1,None),(1,None)])
 def test_eviction_actual_dispatch_cold_populates_three_victims(self):
  calls=self.dispatch('eviction');self.assertEqual(len(calls),6)
  self.assertEqual([x for x in calls if x[0].startswith('other-')],[(f'other-{k}',k,0,0) for k in ['evict_A','evict_C','evict_D']])
  self.assertEqual(calls[0][2:],(1,None));self.assertEqual(calls[-1][2:],(1,None))
 def test_remaining_roster_is_complete_ordered_seven(self):
  self.assertEqual(v.REMAINING_GROUPS,('turn','parked','eviction','cancel','cancel_decode','cancel_prefill_isolation','real_live'))
 def test_frozen_v8_dispatch_remains_unchanged(self):
  s=Path(v.v8.__file__).read_text();self.assertIn("req('independent-context'+suffix,'A',1)",s);self.assertIn("req('other-'+key,key,1)",s);self.assertEqual(v.sha(v.v8.__file__),v.V8_CONTROLLER_SHA)
 def test_missing_or_wrong_plan_rejected_before_prerequisite_io(self):
  with patch.object(v,'basic_receipt_binding') as bound:
   for plan in [{'schema':8},{'schema':9,'groups':[]},{'schema':9,'groups':list(v.REMAINING_GROUPS),'base_v8_controller_sha256':'wrong'}]:
    with self.assertRaises((ValueError,KeyError)):v.preflight(plan,None)
   bound.assert_not_called()
 def test_parent_preflight_is_outside_gpu_finally(self):
  tree=ast.parse(Path(v.__file__).with_name('qualify_serial_prefix_v9.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');tries=[n for n in main.body if isinstance(n,ast.Try)]
  early=next(n for n in tries if any(isinstance(c,ast.Call) and ast.unparse(c.func)=='ctrl.preflight' for c in ast.walk(n)))
  self.assertFalse(early.finalbody);self.assertTrue(any(isinstance(c,ast.Raise) for c in ast.walk(early.handlers[0])))
  gpu=next(n for n in tries if n.finalbody);self.assertLess(main.body.index(early),main.body.index(gpu))
if __name__=='__main__':unittest.main()
