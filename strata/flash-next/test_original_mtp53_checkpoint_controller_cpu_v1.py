import unittest
import build_original_mtp53_checkpoint_v1 as b
import run_original_mtp53_checkpoint_v1 as r
class Recipe(unittest.TestCase):
 def test_native_cpu_command_required(self):
  commands=[{'name':'original-mtp53-checkpoint-session.o','argv':['icpx','-c','bridge.cpp']},{'name':'original-mtp53-shared-checkpoint','argv':['icpx','leaf.cpp']},{'name':'original-mtp53-checkpoint-cpu','argv':['icpx','cpu.cpp']}]
  shell=b.shell({'compile_commands':commands});self.assertIn('/out/original-mtp53-checkpoint-cpu > /out/checkpoint-cpu.stdout',shell);self.assertIn('test "$cpu_rc" -eq 0',shell)
  self.assertIn('test ! -e /dev/dri',shell)
 def test_reset_and_checkpoint_marker(self):
  for field in ['reset_slots=2','fresh_references=2','old_owner_after_reset_refused=1','private_reset_allocations_unchanged=1','checkpoint_branch_shared_roundtrip_exact=1','checkpoint_prefix_hash_negative=1']:self.assertIn(field,r.MARKER)
