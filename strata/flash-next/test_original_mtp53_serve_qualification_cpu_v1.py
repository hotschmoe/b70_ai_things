import unittest
from run_original_mtp53_serve_qualification_v1 import output_view,request_command
class Contracts(unittest.TestCase):
 def test_real_protocol_bounds(self):
  command=request_command(9,3,list(range(54)),0);self.assertTrue(command.startswith('BGEN 3 64 fresh=0 pin=53 ckpt=1 temperature=0 rid=9 '))
  row={'ids':[2],'generated':[198]*64,'done':'BDONE 3 64 length 1.0 rid=9 slotgen=1','slotgen':1,'admission_done':'DONE'};self.assertEqual(len(output_view(row)['output_ids']),64)
  row['generated']=[198]*63
  with self.assertRaises(ValueError):output_view(row)
 def test_natural_eos_and_cancel(self):
  row={'ids':[2],'generated':[198,248046],'done':'BDONE 3 2 stop 1.0 rid=9 slotgen=1','slotgen':1,'admission_done':'DONE'};self.assertEqual(output_view(row)['finish'],'stop')
  row['generated']=[248046,198]
  with self.assertRaises(ValueError):output_view(row)
  row['generated']=[198,32];row['done']='BDONE 3 2 cancel 1.0 rid=9 slotgen=1';self.assertEqual(output_view(row)['finish'],'cancel')
