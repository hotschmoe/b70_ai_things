"""Tiny actual CPU parser witness/current-byte join and mutation controls."""
import copy,tempfile,time,unittest,os
from pathlib import Path
import test_upload_logical_gate_ports_cpu_v2 as fixture
import logical_free_require_case_epoch_v2 as compact
from logical_witness_binding_v2 import binding
class Controls(unittest.TestCase):
 def test_independent_fresh_current_owner_bytes_and_bad_saved_scope(self):
  with tempfile.TemporaryDirectory()as d:
   parts=fixture.Controls().fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts;log=receipt.parent/'full390_card0.log';ledger=receipt.parent/'full390_card0-logical-free.json';epoch.require_case(log,ledger,1,current_roster=roster);epoch.seal_predevice(current_roster=roster);proof=epoch.finalize(current_roster=roster);fresh=compact.RequireLogicalEpoch(roster,60);device=proof['evidence_byte_witness']['boundaries'][1]['finished_epoch'];terminal=proof['evidence_byte_witness']['boundaries'][2]['started_epoch'];self.assertEqual(binding(proof,fresh,os.getpid(),device,terminal,case_specs=[{'log_path':str(log),'logical_path':str(ledger),'expected_owner_count':1}])['complete_byte_boundaries'],3)
   for key,value in [('passed',False),('owner_pid',True),('semantic_executions',0),('saved_result_imported',True)]:
    bad=copy.deepcopy(proof);bad[key]=value;self.assertRaises(ValueError,binding,bad,fresh,os.getpid(),device,terminal,case_specs=[{'log_path':str(log),'logical_path':str(ledger),'expected_owner_count':1}])
 def test_child_READY_ACK_and_current_byte_content_must_match(self):
  with tempfile.TemporaryDirectory()as d:
   parts=fixture.Controls().fixture(Path(d));receipt,oracle,engine,pack,epoch,roster,sdk=parts;log=receipt.parent/'full390_card0.log';ledger=receipt.parent/'full390_card0-logical-free.json';epoch.require_case(log,ledger,1,current_roster=roster);epoch.evidence._fresh('ready');ready=epoch.evidence.boundaries[-1];ack=time.time();epoch.seal_predevice(current_roster=roster);proof=epoch.finalize(current_roster=roster);fresh=compact.RequireLogicalEpoch(roster,60);device=proof['evidence_byte_witness']['boundaries'][2]['finished_epoch'];terminal=proof['evidence_byte_witness']['boundaries'][3]['started_epoch'];self.assertEqual(binding(proof,fresh,os.getpid(),device,terminal,case_specs=[{'log_path':str(log),'logical_path':str(ledger),'expected_owner_count':1}],ready_boundary=ready,ack_epoch=ack)['complete_byte_boundaries'],4)
   self.assertRaises(ValueError,binding,proof,fresh,os.getpid(),device,terminal,case_specs=[{'log_path':str(log),'logical_path':str(ledger),'expected_owner_count':1}],ready_boundary=ready,ack_epoch=device+1);bad=copy.deepcopy(proof);bad['evidence_byte_witness']['boundaries'][0]['rows'][0]['sha256']='0'*64;self.assertRaises(ValueError,binding,bad,fresh,os.getpid(),device,terminal,case_specs=[{'log_path':str(log),'logical_path':str(ledger),'expected_owner_count':1}],ready_boundary=ready,ack_epoch=ack)
 def test_saved_original_packet_command_incarnation_and_aggregate_controls(self):
  with tempfile.TemporaryDirectory()as d:
   receipt,oracle,engine,pack,epoch,roster,sdk=fixture.Controls().fixture(Path(d));log=receipt.parent/'full390_card0.log';ledger=receipt.parent/'full390_card0-logical-free.json';specs=[{'log_path':str(log),'logical_path':str(ledger),'expected_owner_count':1}];epoch.require_case(log,ledger,1,current_roster=roster);epoch.seal_predevice(current_roster=roster);proof=epoch.finalize(current_roster=roster);fresh=compact.RequireLogicalEpoch(roster,60);device=proof['evidence_byte_witness']['boundaries'][1]['finished_epoch'];terminal=proof['evidence_byte_witness']['boundaries'][2]['started_epoch']
   binding(proof,fresh,os.getpid(),device,terminal,case_specs=specs)
   mutations=[(('worker_commands',0,'input_sha256'),'0'*64),(('worker_commands',0,'command'),['CPU_FOREIGN_WORKER']),(('worker_commands',0,'worker_retirement','worker_start_ticks'),0),(('worker_commands',0,'worker_identity','parent_pid'),True),(('worker_commands',0,'worker_retirement','owned_session_empty'),False),(('worker_commands',0,'worker_retirement','unexpected_descendant_rows'),[{}]),(('worker_commands',0,'parse_counts','positive'),True),(('worker_commands',0,'started_epoch'),device+1),(('worker_probe_command','input_sha256'),'0'*64),(('worker_probe_command','command'),['FOREIGN']),(('semantic_executions',),2),(('compact_success_tokens',),0),(('negative_parse_count',),4),(('first_full_result_isolation_copies',),True),(('worker_errors',),[{'error':'caught failure'}])]
   for path,value in mutations:
    with self.subTest(path=path):
     bad=copy.deepcopy(proof);target=bad
     for key in path[:-1]:target=target[key]
     target[path[-1]]=value;self.assertRaises(ValueError,binding,bad,fresh,os.getpid(),device,terminal,case_specs=specs)
   for count in (2,True):
    bad=copy.deepcopy(specs);bad[0]['expected_owner_count']=count;self.assertRaises(ValueError,binding,proof,fresh,os.getpid(),device,terminal,case_specs=bad)
   self.assertRaises(ValueError,binding,proof,fresh,os.getpid(),device,terminal,case_specs=specs+specs)
if __name__=='__main__':unittest.main()
