"""Producer-shaped readline bytes, exact owner/order/content negative controls."""
import copy,unittest
import test_eos_waiter_observer40_cpu_v1 as fixture
import eos_waiter_terminal_source40_v3 as a
import eos_waiter_terminal_adjudication40_v1 as old
class Controls(unittest.TestCase):
 def fixture(self):
  events,proof=fixture.Controls().full_fixture()
  for row in events:
   if type(row.get('line'))is str and row['line'].startswith('BDONE '):row['line']=row['line'].replace(' cancel ',' stop ')
   if row['kind']=='eos_waiter_queue_return':row['line']+='\n'
  return events,proof
 def test_exact_actual_readline_terminator_and_raw_preservation(self):
  events,proof=self.fixture();before=copy.deepcopy(events);self.assertRaises(ValueError,old.adjudicate,events,proof);result=a.adjudicate(events,proof);self.assertTrue(result['actual_readline_single_LF_verified']);self.assertEqual(events,before)
 def test_no_whitespace_or_missing_duplicate_CR_newline_waiver(self):
  for suffix in('', '\n\n','\r\n',' \n','\t\n'):
   events,proof=self.fixture();row=next(x for x in events if x['kind']=='eos_waiter_queue_return');row['line']=row['line'][:-1]+suffix;self.assertRaises(ValueError,a.adjudicate,events,proof)
 def test_terminal_owner_finish_content_sequence_and_thread_gates_remain(self):
  for mode in('rid','slotgen','token_count','finish','sequence','thread'):
   events,proof=self.fixture();row=next(x for x in events if x['kind']=='eos_waiter_queue_return')
   if mode=='rid':row['line']=row['line'].replace('rid=9','rid=8')
   elif mode=='slotgen':row['line']=row['line'].replace('slotgen=4','slotgen=5')
   elif mode=='token_count':row['line']=row['line'].replace('BDONE 0 2','BDONE 0 3')
   elif mode=='finish':row['line']=row['line'].replace(' stop ',' length ')
   elif mode=='sequence':row['sequence']=1
   else:row['thread_ident']+=1
   self.assertRaises(ValueError,a.adjudicate,events,proof)
if __name__=='__main__':unittest.main()
