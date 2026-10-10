"""Tiny synthetic real-parser bytes and own isolated CPU workers; no runtime logs."""
import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import logical_free_immutable_epoch_v1 as narrow
import parse_usm_logical_free_trace as original
from test_logical_free_snapshot_cpu_v1 import TRACE
class Controls(unittest.TestCase):
 def fixture(self,root,text=TRACE):
  log=root/'synthetic.log';log.write_text(text);logical=root/'synthetic-logical.json';checked=original.parse_trace(text);checked['negative_controls']=original.negative_controls(text,True)if checked['passed']else {};logical.write_text(json.dumps(checked));roster=[(p,hashlib.sha256(p.read_bytes()).hexdigest())for p in (log,logical,narrow.PARSER,narrow.WORKER,*(narrow.HERE/name for name in narrow.PROGRAM_SOURCE_NAMES))];return log,logical,roster
 def test_eighty_callers_have_one_positive_three_real_negative_parses(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60)
   for i in range(80):self.assertTrue(epoch.collect(log,logical,1,current_roster=roster)['original_upload_logical_subset_passed'])
   epoch.seal_predevice(current_roster=roster);proof=epoch.finalize(current_roster=roster);self.assertEqual(proof['semantic_calls'],80);self.assertEqual(proof['semantic_executions'],1);self.assertEqual(proof['positive_parse_count'],1);self.assertEqual(proof['negative_parse_count'],3)
   self.assertEqual(proof['evidence_byte_witness']['semantic_executions'],0);self.assertFalse(proof['generic_purity_contract_weakened']);self.assertEqual([r['label']for r in proof['evidence_byte_witness']['boundaries']],['entry','predevice','postoperation'])
 def test_complete_result_equals_frozen_original_positive_and_controls(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60);value=epoch.collect(log,logical,1,current_roster=roster)
   self.assertEqual(value['checked'],original.parse_trace(TRACE));self.assertEqual(value['negative_controls'],original.negative_controls(TRACE,True));self.assertFalse(value['outer_source_model_health_or_runtime_gates_proven'])
 def test_mutated_return_never_changes_memo(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60);value=epoch.collect(log,logical,1,current_roster=roster);value['checked']['passed']=False;self.assertTrue(epoch.collect(log,logical,1,current_roster=roster)['checked']['passed'])
 def test_parent_parser_globals_are_not_used_by_isolated_worker(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d))
   with patch.object(original,'parse_trace',side_effect=AssertionError('mutable parent state must not enter parser')):
    epoch=narrow.LogicalEpoch(roster,60);self.assertTrue(epoch.collect(log,logical,1,current_roster=roster)['checked']['passed'])
 def test_changed_owner_parameter_and_bool_fail_instead_of_borrowing_success(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60);epoch.collect(log,logical,1,current_roster=roster)
   self.assertRaises(ValueError,epoch.collect,log,logical,2,current_roster=roster);self.assertRaises(ValueError,epoch.collect,log,logical,True,current_roster=roster)
 def test_changed_input_bytes_fail_seal_and_post(self):
  for when in ('seal','post'):
   with tempfile.TemporaryDirectory()as d:
    log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60);epoch.collect(log,logical,1,current_roster=roster)
    if when=='post':epoch.seal_predevice(current_roster=roster)
    log.write_text(TRACE+'new immutable input bytes\n');self.assertRaises(ValueError,epoch.seal_predevice if when=='seal'else epoch.finalize,current_roster=roster)
 def test_changed_expected_roster_or_duplicate_refused(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60);changed=[(p,'0'*64 if p==log else v)for p,v in roster]
   self.assertRaises(ValueError,epoch.collect,log,logical,1,current_roster=changed);self.assertRaises(ValueError,epoch.collect,log,logical,1,current_roster=roster+[roster[0]])
 def test_process_ownership_lifetime_and_closed_phase_refused(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60)
   with patch.object(os,'getpid',return_value=epoch.evidence.owner+1):self.assertRaises(ValueError,epoch.collect,log,logical,1,current_roster=roster)
   epoch.evidence.started-=61;self.assertRaises(ValueError,epoch.collect,log,logical,1,current_roster=roster)
   epoch.evidence.started+=61;epoch.seal_predevice(current_roster=roster);epoch.finalize(current_roster=roster);self.assertRaises(ValueError,epoch.collect,log,logical,1,current_roster=roster)
 def test_wrong_saved_logical_or_failed_raw_trace_is_not_memoized(self):
  for change in ('ledger','raw'):
   with tempfile.TemporaryDirectory()as d:
    log,logical,roster=self.fixture(Path(d))
    if change=='ledger':v=json.loads(logical.read_text());v['counts']['owners']=2;logical.write_text(json.dumps(v))
    else:log.write_text(TRACE.replace('urUSMFree(.hContext','urUSMFree(.badContext'))
    roster=[(p,hashlib.sha256(p.read_bytes()).hexdigest())for p,v in roster];epoch=narrow.LogicalEpoch(roster,60);self.assertRaises(ValueError,epoch.collect,log,logical,1,current_roster=roster);self.assertEqual(epoch.executions,0);self.assertEqual(epoch.memo,{})
 def test_frozen_source_modified_or_missing_from_roster_is_refused(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));self.assertRaises(ValueError,narrow.LogicalEpoch,[r for r in roster if r[0]!=narrow.PARSER],60)
   worker=Path(d)/'foreign.py';worker.write_text('print("foreign worker")')
   with patch.object(narrow,'WORKER',worker):self.assertRaises(ValueError,narrow.LogicalEpoch,roster,60)
 def test_universal_newline_decoding_matches_original_read_text(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));log.write_bytes(TRACE.replace('\n','\r\n').encode());roster=[(p,hashlib.sha256(p.read_bytes()).hexdigest())for p,v in roster];epoch=narrow.LogicalEpoch(roster,60);self.assertEqual(epoch.collect(log,logical,1,current_roster=roster)['checked'],original.parse_trace(log.read_text()))
 def test_changed_worker_global_execution_context_cannot_reuse_success(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60);epoch.collect(log,logical,1,current_roster=roster)
   with patch.object(narrow,'MAX_OUTPUT',narrow.MAX_OUTPUT+1):self.assertRaisesRegex(ValueError,'context changed',epoch.collect,log,logical,1,current_roster=roster)
 def test_five_case_consumer_calls_execute_twenty_scans_total(self):
  import original_upload_logical_bytes_v1 as consumer
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);case_roster=[];cases=[]
   for i in range(5):
    log=root/('case'+str(i)+'.log');log.write_text(TRACE);logical=root/('case'+str(i)+'-logical-free.json');value=original.parse_trace(TRACE);value['negative_controls']=original.negative_controls(TRACE,True);logical.write_text(json.dumps(value));case_roster.extend([log,logical]);cases.append({'case':'case'+str(i),'report':{'stages':[{'unique_allocations':0}]}})
   roster=[(p,hashlib.sha256(p.read_bytes()).hexdigest())for p in (*case_roster,narrow.PARSER,narrow.WORKER,*(narrow.HERE/name for name in narrow.PROGRAM_SOURCE_NAMES))];epoch=narrow.LogicalEpoch(roster,60)
   for repeat in range(16):
    for case in cases:self.assertTrue(consumer.logical_case(epoch,root,case,current_roster=roster)['original_upload_logical_subset_passed'])
   self.assertEqual(epoch.executions,5);self.assertEqual(epoch.calls,80);self.assertEqual(4*epoch.executions,20)
 def test_every_host_program_dependency_is_mandatory(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d))
   for name in narrow.PROGRAM_SOURCE_NAMES:self.assertRaises(ValueError,narrow.LogicalEpoch,[r for r in roster if r[0]!=narrow.HERE/name],60)
 def test_failed_attempt_receipt_and_counts_are_retained(self):
  with tempfile.TemporaryDirectory()as d:
   log,logical,roster=self.fixture(Path(d));epoch=narrow.LogicalEpoch(roster,60);epoch.collect(log,logical,1,current_roster=roster);self.assertRaises(ValueError,epoch.collect,log,logical,2,current_roster=roster);epoch.seal_predevice(current_roster=roster);proof=epoch.finalize(current_roster=roster)
   self.assertFalse(proof['passed']);self.assertEqual(len(proof['worker_errors']),1);self.assertEqual(len(proof['worker_commands']),2);self.assertEqual(proof['positive_parse_count'],2);self.assertEqual(proof['negative_parse_count'],6);self.assertFalse(proof['worker_commands'][-1]['worker_passed'])
if __name__=='__main__':unittest.main()
