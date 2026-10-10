"""Producer-shaped original ledger has no invented saved log hash."""
import json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import upload_logical_epoch_factory_v2 as f
class Controls(unittest.TestCase):
 def fixture(self,root):
  receipt=root/'receipt.json';cases=[]
  for i in range(5):
   name='case'+str(i);cases.append({'case':name,'report':{'stages':[{'unique_allocations':i}]}});(root/(name+'.log')).write_text('synthetic original trace\n');(root/(name+'-logical-free.json')).write_text(json.dumps({'schema':1,'scope':'logical','passed':True,'counts':{},'owners':[],'events':[],'live':[],'errors':[],'negative_controls':{}}))
  receipt.write_text(json.dumps({'passed':True,'runner_generation':2,'cases':cases}));oracle=root/'oracle.json';oracle.write_text('{}');(root/'prepared.json').write_text(json.dumps({'upload_lifecycle':{'path':str(receipt),'sha256':f.sha(receipt),'oracle':str(oracle),'oracle_sha256':f.sha(oracle)}}));names=(*f.PROGRAM_SOURCE_NAMES,*f.COMPACT_SOURCES,'logical_free_immutable_worker_v2.py','parse_usm_logical_free_trace.py','original_upload_logical_requirement_v2.py','c137_sdk37_logical_ports_v2.py','c137_sdk37_digest_ports_v1.py','c1_serve_controller_combined_v137.py','sdk37_witness_scope_v1.py');return {'files':{str((f.HERE/name).relative_to(f.ROOT)):f.sha(f.HERE/name)for name in names}}
 def test_actual_no_log_sha_ledger_schema_and_typed_owner_corpus(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);plan=self.fixture(root)
   with patch.object(f,'sources',return_value=plan),patch.object(f,'RequireLogicalEpoch',side_effect=lambda items,seconds:SimpleNamespace(items=items)):
    epoch,roster,scope=f.for_prepared([root]);self.assertEqual([s['expected_owner_count']for s in scope['case_specs']],[1,2,3,4,5]);self.assertEqual(scope['case_specs'],f.case_specs_for_prepared([root]));self.assertEqual(len([p for p,h in roster if str(p).endswith('.log')]),5)
 def test_changed_receipt_and_boolean_stage_are_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);self.fixture(root);receipt=root/'receipt.json';value=json.loads(receipt.read_text());value['cases'][0]['report']['stages'][0]['unique_allocations']=True;receipt.write_text(json.dumps(value));prepared=json.loads((root/'prepared.json').read_text());prepared['upload_lifecycle']['sha256']=f.sha(receipt);(root/'prepared.json').write_text(json.dumps(prepared));self.assertRaises(ValueError,f.case_specs_for_prepared,[root]);receipt.write_text('{}');self.assertRaises(ValueError,f.case_specs_for_prepared,[root])
 def test_symlink_operand_is_not_a_current_byte_snapshot(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'real').write_text('bytes');(root/'alias').symlink_to(root/'real');self.assertRaises(ValueError,f.snapshot_sha,root/'alias')
if __name__=='__main__':unittest.main()
