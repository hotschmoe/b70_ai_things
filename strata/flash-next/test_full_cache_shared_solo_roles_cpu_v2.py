import unittest
from full_cache_shared_raw49_v2 import required_roles

class Roles(unittest.TestCase):
 def test_decode_cancel_requires_real_survivor_solo_49(self):
  phase={'cancel_kind':'decode','native_multirow_required':True};work=[{'rid':3,'native_first_command':{'line':'BGEN 0'}},{'rid':4,'native_first_command':{'line':'BGEN 1'}}];ends={1:{'rid':3,'actual_client_cancelled':True},2:{'rid':4,'actual_client_cancelled':False}}
  actual=required_roles(phase,work,ends);self.assertEqual(actual[3],['admission','later']);self.assertEqual(actual[4],['admission','later','solo_migration'])
 def test_missing_or_two_cancelled_cannot_substitute_solo(self):
  phase={'cancel_kind':'decode','native_multirow_required':True};work=[{'rid':3,'native_first_command':{'line':'BGEN 0'}},{'rid':4,'native_first_command':{'line':'BGEN 1'}}]
  for ends in ({1:{'rid':3,'actual_client_cancelled':False},2:{'rid':4,'actual_client_cancelled':False}},{1:{'rid':3,'actual_client_cancelled':True},2:{'rid':4,'actual_client_cancelled':True}}):
   with self.assertRaises(ValueError):required_roles(phase,work,ends)
if __name__=='__main__':unittest.main()
