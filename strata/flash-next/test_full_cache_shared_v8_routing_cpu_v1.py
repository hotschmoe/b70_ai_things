"""Producer-shaped lateBDONE routes; no synthetic GPU/model proof authority."""
import copy,unittest,ast
from pathlib import Path
import test_eos_waiter_observer40_cpu_v1 as fixture
from full_cache_shared_terminal_v2 import associate_events as old_gate
from eos_waiter_terminal_source40_v2 import adjudicate
from full_cache_shared_raw49_v2 import required_roles
class Controls(unittest.TestCase):
 def source_rows(self,finish='stop',cancelled=False):
  rows,proof=fixture.Controls().full_fixture()
  # Synthetic source-shaped terminal text is configured consistently across
  # original receive and queue-return observations; these are CPU fixtures.
  for row in rows:
   if type(row.get('line'))is str and row['line'].startswith('BDONE '):row['line']=row['line'].replace(' cancel ',' '+finish+' ')
  end=next(r for r in rows if r['kind']=='engine_end');end['cancelled']=cancelled
  if cancelled:
   end['error']=None;end['generated_ids']=[8,10]
   for row in rows:
    if row['kind']=='native_receive'and row['line'].startswith('BT '):row['line']=row['line'].replace(' 9 ',' 10 ')
  return rows,proof
 def test_old_gate_reproduces_late_failure_new_actual_terminals_drive_raw49(self):
  rows,proof=self.source_rows();before=copy.deepcopy(rows);self.assertRaises(ValueError,old_gate,rows,{3});result=adjudicate(rows,proof);terminals={r['call']:r for r in result['actual_API_terminal_associations']};phase={'native_multirow_required':True,'cancel_kind':None};work=[{'rid':9,'native_first_command':{'line':'BGEN 0 32 seed=1 fresh=1 rid=9 7'}}];self.assertEqual(required_roles(phase,work,terminals),{9:['admission','later']});self.assertEqual(rows,before);self.assertIsNotNone(terminals[3]['legacy_observer_error_preserved'])
 def test_non_cancelled_eos_cannot_borrow_native_cancel_or_length(self):
  for finish in('cancel','length'):
   rows,proof=self.source_rows(finish);self.assertRaises(ValueError,adjudicate,rows,proof)
 def test_actual_cancel_distinct_error_none_scope_allowed(self):
  rows,proof=self.source_rows('cancel',True);result=adjudicate(rows,proof);association=result['actual_API_terminal_associations'][0];self.assertTrue(association['actual_client_cancelled']);self.assertIsNone(association['legacy_observer_error_preserved'])
 def test_public_reader_raw49_path_has_no_old_terminal_gate(self):
  path=Path(__file__).with_name('validate_full_cache_shared_runtime_v8.py');source=path.read_text();tree=ast.parse(source);required=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='required_roles'];self.assertEqual(len(required),1);self.assertIsInstance(required[0].args[2],ast.Name);self.assertEqual(required[0].args[2].id,'actual_terminals')
if __name__=='__main__':unittest.main()
