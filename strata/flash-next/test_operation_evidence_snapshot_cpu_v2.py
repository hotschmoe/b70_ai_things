"""Complete supported callback-state contract, tiny evidence fixtures only."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import operation_evidence_snapshot_v2 as m
GLOBAL_VALUE=1
MUTABLE_GLOBAL=[]
def basic(snapshot,params):return {'bytes_match':snapshot.bytes_for(params['log'])==b'allocate A\nfree A\n','tuple':(2,1)}
def global_value(snapshot,params):return {'value':GLOBAL_VALUE}
def mutable_global(snapshot,params):return {'value':len(MUTABLE_GLOBAL)}
def defaults(snapshot,params,value=1):return {'value':value}
def other(snapshot,params):return {'different':True}
class Controls(unittest.TestCase):
 def fixture(self,r):
  code=Path(__file__).resolve();log=r/'log';log.write_text('allocate A\nfree A\n');rows=[(p,hashlib.sha256(p.read_bytes()).hexdigest()) for p in (code,log)];return code,log,rows
 def call(self,e,code,log,rows,fn=basic,bindings=None):return e.pure_once('CPU-pure',code,rows[0][1],{'log':str(log)},[log],fn,current_roster=rows,global_bindings={} if bindings is None else bindings)
 def test_eighty_calls_one_execution_deepcopy_three_boundaries(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows)
   for index in range(80):a=self.call(e,code,log,rows);a['tuple']=(999,)
   self.assertEqual(e.semantic_executions,1);self.assertEqual(self.call(e,code,log,rows)['tuple'],(2,1));e.seal_predevice();proof=e.finalize();self.assertEqual(len(proof['boundaries']),3)
 def test_same_code_different_closure_rejected_before_any_memo(self):
  def factory(value):
   def validate(snapshot,params):return {'value':value}
   return validate
  with tempfile.TemporaryDirectory() as d:
   code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows)
   for value in (1,2):self.assertRaises(ValueError,self.call,e,code,log,rows,factory(value))
   self.assertEqual(e.memo,{})
 def test_changed_defaults_and_boundmethods_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows);self.assertRaises(ValueError,self.call,e,code,log,rows,defaults)
   class Method:
    def validate(self,snapshot,params):return {'value':1}
   self.assertRaises(ValueError,self.call,e,code,log,rows,Method().validate)
 def test_undeclared_mutable_or_module_globals_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows);self.assertRaises(ValueError,self.call,e,code,log,rows,global_value);self.assertRaises(ValueError,self.call,e,code,log,rows,mutable_global,{'MUTABLE_GLOBAL':[]})
 def test_changed_immutable_global_oldbinding_refused_newbinding_newexecution(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows)
   with patch.dict(global_value.__globals__,{'GLOBAL_VALUE':1}):self.assertEqual(self.call(e,code,log,rows,global_value,{'GLOBAL_VALUE':1})['value'],1)
   with patch.dict(global_value.__globals__,{'GLOBAL_VALUE':2}):self.assertRaises(ValueError,self.call,e,code,log,rows,global_value,{'GLOBAL_VALUE':1});self.assertEqual(self.call(e,code,log,rows,global_value,{'GLOBAL_VALUE':2})['value'],2)
   self.assertEqual(e.semantic_executions,2)
 def test_global_bool_int_signedzero_are_distinct(self):
  for a,b in [(True,1),(-0.0,0.0),(1,1.0)]:
   with patch.dict(global_value.__globals__,{'GLOBAL_VALUE':a}):self.assertRaises(ValueError,m.validator_state,global_value,{'GLOBAL_VALUE':b})
 def test_current_byte_mutation_caught_at_predevice_and_post(self):
  for sealed in (False,True):
   with tempfile.TemporaryDirectory() as d:
    code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows);self.call(e,code,log,rows)
    if sealed:e.seal_predevice()
    log.write_text('changed');self.assertRaises(ValueError,e.finalize if sealed else e.seal_predevice)
 def test_unknown_input_roster_owner_closed_phase(self):
  with tempfile.TemporaryDirectory() as d:
   code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows);self.assertRaises(ValueError,e.require_roster,rows[:-1])
   with patch.object(m.os,'getpid',return_value=e.owner+1):self.assertRaises(ValueError,e.seal_predevice)
   e.seal_predevice();e.finalize();self.assertRaises(ValueError,self.call,e,code,log,rows)
 def test_nested_global_and_live_snapshot_or_import_access_refused(self):
  def nested(snapshot,params):return {'value':[GLOBAL_VALUE for i in (0,)]}
  def unsafe(snapshot,params):return {'value':snapshot.phase}
  def imports(snapshot,params):return {'value':__import__('time').time()}
  self.assertRaises(ValueError,m.validator_state,nested,{})
  self.assertRaises(ValueError,m.validator_state,imports,{})
  with tempfile.TemporaryDirectory() as d:
   code,log,rows=self.fixture(Path(d));e=m.EvidenceEpoch(rows);self.assertRaises(AttributeError,self.call,e,code,log,rows,unsafe)
 def test_original_parser_wrapper_is_not_silently_pure_admitted(self):
  import collect_logical_free_snapshot_v1 as original
  self.assertRaises(ValueError,m.validator_state,original.validate,{})
if __name__=='__main__':unittest.main()
