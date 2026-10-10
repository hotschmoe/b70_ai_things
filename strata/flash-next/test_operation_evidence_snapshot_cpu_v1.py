"""Tiny immutable closed-evidence controls; no real logs/model/SDK reads."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import operation_evidence_snapshot_v1 as m
class Controls(unittest.TestCase):
 def fixture(self,r):
  code=r/'validator.py';code.write_text('CPU-pure-validator');log=r/'trace';log.write_text('allocate A\nfree A\n');roster=[(p,hashlib.sha256(p.read_bytes()).hexdigest()) for p in (code,log)];return code,log,roster
 def test_many_same_pure_calls_one_validator_and_three_full_byte_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));e=m.EvidenceEpoch(roster);calls=[]
   def validate(snapshot,parameters):calls.append(1);return {'all_original_predicates_CPU_fixture':snapshot.bytes_for(log)==b'allocate A\nfree A\n','shape':(2,1)}
   for i in range(80):e.require_roster(roster);self.assertEqual(e.pure_once('CPU-pure',code,roster[0][1],{'required_owners':True},[log],validate,current_roster=roster)['shape'],(2,1))
   e.seal_predevice();proof=e.finalize();self.assertEqual(len(calls),1);self.assertEqual(len(proof['boundaries']),3);self.assertEqual(proof['semantic_calls'],80);self.assertFalse(proof['live_guards_reuse_allowed'])
 def test_mutation_before_seal_and_afterseal_fails(self):
  for after in (False,True):
   with tempfile.TemporaryDirectory() as d:
    code,log,roster=self.fixture(Path(d));e=m.EvidenceEpoch(roster)
    if after:e.seal_predevice()
    log.write_text('CHANGED');self.assertRaises(ValueError,e.finalize if after else e.seal_predevice)
 def test_changed_parameters_cannot_borrow_same_semantic_result(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));e=m.EvidenceEpoch(roster);calls=[]
   def validate(snapshot,parameters):calls.append(1);return {'result':len(calls)}
   self.assertNotEqual(e.pure_once('CPU',code,roster[0][1],{'x':True},[log],validate,current_roster=roster),e.pure_once('CPU',code,roster[0][1],{'x':1},[log],validate,current_roster=roster));self.assertEqual(len(calls),2)
 def test_different_validator_code_cannot_borrow_first_result(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));e=m.EvidenceEpoch(roster);first=lambda snapshot,parameters:{'result':True};other=lambda snapshot,parameters:{'result':False};a=e.pure_once('CPU',code,roster[0][1],{},[log],first,current_roster=roster);b=e.pure_once('CPU',code,roster[0][1],{},[log],other,current_roster=roster);self.assertNotEqual(a,b);self.assertEqual(e.semantic_executions,2)
 def test_result_copy_preserves_types_and_prevents_caller_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));e=m.EvidenceEpoch(roster);fn=lambda snapshot,parameters:{'values':[1],'tuple':(1,2)};a=e.pure_once('CPU',code,roster[0][1],{},[log],fn,current_roster=roster);a['values'].append(2);self.assertEqual(e.pure_once('CPU',code,roster[0][1],{},[log],fn,current_roster=roster)['values'],[1])
 def test_failure_not_cached_and_missing_input_or_validator_source_refused(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));e=m.EvidenceEpoch(roster)
   def bad(snapshot,parameters):raise ValueError('original predicate failed')
   for i in range(2):self.assertRaises(ValueError,e.pure_once,'CPU',code,roster[0][1],{},[log],bad,current_roster=roster)
   self.assertEqual(e.memo,{});self.assertRaises(ValueError,e.bytes_for,Path(d)/'unknown');self.assertRaises(ValueError,e.pure_once,'CPU',code,'0'*64,{},[log],bad,current_roster=roster)
 def test_current_roster_alias_unpinned_scope_and_closed_phase(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));unpinned=[roster[0],(log,None)];e=m.EvidenceEpoch(unpinned);e.seal_predevice();proof=e.finalize();self.assertEqual(proof['externally_unpinned_current_inputs'],[str(log)]);self.assertRaises(ValueError,e.bytes_for,log);self.assertRaises(ValueError,m.EvidenceEpoch,roster+roster)
 def test_oversized_stat_refused_before_open(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d))
   with patch.object(m,'stat5',return_value=[1,2,129<<20,4,5]),patch.object(Path,'open',side_effect=AssertionError('no read')):self.assertRaises(ValueError,m.EvidenceEpoch,roster)
 def test_recursive_same_key_and_undeclared_gate_input_refused(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));aux=Path(d)/'extra';aux.write_text('CPU-extra');roster.append((aux,hashlib.sha256(aux.read_bytes()).hexdigest()));e=m.EvidenceEpoch(roster)
   def recursive(snapshot,parameters):return snapshot.pure_once('CPU',code,roster[0][1],{},[log],recursive,current_roster=roster)
   self.assertRaises(ValueError,e.pure_once,'CPU',code,roster[0][1],{},[log],recursive,current_roster=roster)
   self.assertRaises(ValueError,e.pure_once,'CPU',code,roster[0][1],{},[log],lambda snapshot,parameters:snapshot.bytes_for(aux),current_roster=roster);self.assertEqual(e.memo,{})
 def test_owner_and_expiry(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,roster=self.fixture(Path(d));e=m.EvidenceEpoch(roster)
   with patch.object(m.os,'getpid',return_value=e.owner+1):self.assertRaises(ValueError,e.seal_predevice)
   e.started-=901;self.assertRaises(ValueError,e.seal_predevice)
if __name__=='__main__':unittest.main()
