"""Tiny actual CPU parser witness/current-byte join and mutation controls."""
import copy,tempfile,time,unittest,os
from pathlib import Path
import test_upload_logical_gate_ports_cpu_v2 as fixture
import logical_free_require_case_epoch_v2 as compact
from logical_witness_binding_v1 import binding
class Controls(unittest.TestCase):
 def test_independent_fresh_current_owner_bytes_and_bad_saved_scope(self):
  with tempfile.TemporaryDirectory()as d:
   parts=fixture.Controls().fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts;log=receipt.parent/'full390_card0.log';ledger=receipt.parent/'full390_card0-logical-free.json';epoch.require_case(log,ledger,1,current_roster=roster);epoch.seal_predevice(current_roster=roster);proof=epoch.finalize(current_roster=roster);fresh=compact.RequireLogicalEpoch(roster,60);device=proof['evidence_byte_witness']['boundaries'][1]['finished_epoch'];terminal=proof['evidence_byte_witness']['boundaries'][2]['started_epoch'];self.assertEqual(binding(proof,fresh,os.getpid(),device,terminal)['complete_byte_boundaries'],3)
   for key,value in [('passed',False),('owner_pid',True),('semantic_executions',0),('saved_result_imported',True)]:
    bad=copy.deepcopy(proof);bad[key]=value;self.assertRaises(ValueError,binding,bad,fresh,os.getpid(),device,terminal)
 def test_child_READY_ACK_and_current_byte_content_must_match(self):
  with tempfile.TemporaryDirectory()as d:
   parts=fixture.Controls().fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts;epoch.require_case(receipt.parent/'full390_card0.log',receipt.parent/'full390_card0-logical-free.json',1,current_roster=roster);epoch.evidence._fresh('ready');ready=epoch.evidence.boundaries[-1];ack=time.time();epoch.seal_predevice(current_roster=roster);proof=epoch.finalize(current_roster=roster);fresh=compact.RequireLogicalEpoch(roster,60);device=proof['evidence_byte_witness']['boundaries'][2]['finished_epoch'];terminal=proof['evidence_byte_witness']['boundaries'][3]['started_epoch'];self.assertEqual(binding(proof,fresh,os.getpid(),device,terminal,ready_boundary=ready,ack_epoch=ack)['complete_byte_boundaries'],4)
   self.assertRaises(ValueError,binding,proof,fresh,os.getpid(),device,terminal,ready_boundary=ready,ack_epoch=device+1);bad=copy.deepcopy(proof);bad['evidence_byte_witness']['boundaries'][0]['rows'][0]['sha256']='0'*64;self.assertRaises(ValueError,binding,bad,fresh,os.getpid(),device,terminal,ready_boundary=ready,ack_epoch=ack)
if __name__=='__main__':unittest.main()
