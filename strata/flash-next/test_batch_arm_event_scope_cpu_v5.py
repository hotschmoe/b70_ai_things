#!/usr/bin/env python3
"""Synthetic event-scope controls and mocked frozen strictrawaudit boundary."""
import ast,queue,unittest
from pathlib import Path
from types import SimpleNamespace
import run_batch_numerical_pilot_v7 as driver
from unittest.mock import patch
import audit_batch_fidelity_coverage_v5 as q

def event(n,rows=2,mask=3,pid=1,gen=7):return f'SBF batch_event pid={pid} enginegen={gen} event={n} rows={rows} active_mask={mask} completed=1'
def trace():return '\n'.join([event(1),event(2),'HARNESS ARM after actual unarmed multi-row terminal','SBF request rid=2001',event(1),event(2)])
class ScopeTests(unittest.TestCase):
 def test_warm_armed_ordinal_reset_allowed_strict_separate_namespaces(self):
  target,proof=q.scoped_trace(trace(),2);self.assertEqual(proof['warm_to_armed_ordinal_overlap_count'],2);self.assertEqual(proof['warm_completed_events'],2);self.assertEqual(proof['armed_completed_events'],2);self.assertNotIn('HARNESS ARM',target);self.assertTrue(proof['frozen_strict_target_coverage_unchanged']);self.assertFalse(proof['full_model_math_qualified'])
 def test_target_raw_strict_function_receives_only_postARM_unchanged(self):
  called=[]
  def raw(*args):called.append(args);return {'status':'CPU_MOCK_RAW_ONLY','vectors':196}
  with patch.object(q,'strict_initial',raw):r=q.coverage(trace(),[2001,2002],[(0,0,48)],'/CPU_FAKE_CAPTURES')
  self.assertEqual(r['vectors'],196);self.assertEqual(called[0][0],q.scoped_trace(trace(),2)[0]);self.assertEqual(called[0][1:],[2001,2002] and ([2001,2002],[(0,0,48)],'/CPU_FAKE_CAPTURES'))
 def test_duplicate_within_warm_or_target_rejected(self):
  for changed in (trace().replace('HARNESS ARM',event(1)+'\nHARNESS ARM'),trace()+'\n'+event(1)):
   with self.assertRaisesRegex(ValueError,'Duplicate completed batch event'):q.scoped_trace(changed,2)
 def test_missing_duplicate_ARM_rejected(self):
  for changed in (trace().replace('HARNESS ARM','HARNESS UNARM'),trace()+'\nHARNESS ARM again'):
   with self.assertRaises(ValueError):q.scoped_trace(changed,2)
 def test_warm_observer_data_rejected(self):
  for kind in ('request','resume','replay','row','vector','allocation'):
   with self.assertRaises(ValueError):q.scoped_trace('SBF '+kind+' rid=1001\n'+trace(),2)
 def test_wrong_engine_between_phases_rejected(self):
  with self.assertRaises(ValueError):q.scoped_trace(trace().replace('SBF request rid=2001','SBF request rid=2001').rsplit('enginegen=7',1)[0]+'enginegen=8 event=2 rows=2 active_mask=3 completed=1',2)
 def test_invalid_geometry_missing_completed_and_Nrow_rejected(self):
  for changed in (trace().replace('active_mask=3','active_mask=1'),trace().replace('rows=2','rows=3'),trace().replace('completed=1','completed=0'),trace().replace('event=1','event=0'),trace().replace('rows=2 active_mask=3','rows=1 active_mask=1')):
   with self.assertRaises(ValueError):q.scoped_trace(changed,2)
 def test_immutable_strictcoverage_failure_type_and_cause_retained(self):
  def fail(*args):raise AssertionError()
  with patch.object(q,'strict_initial',fail):
   with self.assertRaisesRegex(ValueError,'AssertionError') as caught:q.coverage(trace(),[2001,2002],[(0,0,48)],'/CPU_FAKE')
  self.assertIsInstance(caught.exception.__cause__,AssertionError)
 def test_request_N_mismatch_and_frozen_parser_change_rejected(self):
  with self.assertRaises(ValueError):q.coverage(trace(),[2001,2002],[(0,0,48)],'/CPU_FAKE',4)
  with patch.object(q,'V2_SHA','0'*64):
   with self.assertRaises(ValueError):q.coverage(trace(),[2001,2002],[(0,0,48)],'/CPU_FAKE')
 def test_newdriver_emptyexception_type_stack_roster_preserved(self):
  stream=SimpleNamespace(roster=SimpleNamespace(requests={2001:{'generated':[16],'done':None}},events=[{}]))
  for exc in (AssertionError(),queue.Empty(),ValueError('precise gate')):
   try:raise exc
   except BaseException as caught:report=driver.diagnose_failure(caught,'armed strict raw coverage',stream)
   self.assertEqual(report['exception_type'],type(exc).__name__);self.assertIn(type(exc).__name__,report['traceback']);self.assertEqual(report['stage'],'armed strict raw coverage');self.assertEqual(report['completed_events_at_failure'],1);self.assertEqual(report['roster_at_failure'][2001]['generated'],[16]);stream.roster.requests[2001]['generated'].append(17);self.assertEqual(report['roster_at_failure'][2001]['generated'],[16]);stream.roster.requests[2001]['generated']=[16]
 def test_newdriver_preserves_oldcontrols_and_declares_maxnew_discrepancy(self):
  old=ast.parse(Path(driver.__file__).with_name('run_batch_numerical_pilot_v6.py').read_text());new=ast.parse(Path(driver.__file__).read_text())
  for name in ('require','write','set_arg'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name);self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
  text=Path(driver.__file__).read_text()
  for marker in ("from audit_batch_fidelity_coverage_v5 import coverage","stream.drain(warm)","stream.drain(targets", "genuine_baseline", "owner_proofs", "artifact_bindings", "constant32_vs_plan_maxnew_contract_unresolved", "actual_target_max_new", "diagnose_failure(exc", "stream.close()"):
   self.assertIn(marker,text)
if __name__=='__main__':unittest.main()
