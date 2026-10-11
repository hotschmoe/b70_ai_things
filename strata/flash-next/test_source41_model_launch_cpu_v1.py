import copy,unittest
from source41_model_cli_launch_binding_v1 import binding
class Controls(unittest.TestCase):
 def fixture(self):
  owner={'pid':10,'start_ticks':11};client={'pid':12,'start_ticks':13,'session':12,'state':'S'};cmd=['owned','exact']
  launch={'actual_client_started':True,'client_identity_observed':True,'owner':owner,'client':client,'command':cmd,'started_epoch':3}
  retired={'client':client,'command':cmd,'return_code':0,'session_members':[],'retired_epoch':6}
  capture={'return_code':0,'process_started_epoch':2,'started_epoch':4,'completed_epoch':5,'finished_epoch':7}
  result={'return_code':0,'created_epoch':1,'finished_epoch':8,'terminal':{'State':{'FinishedAt':'1970-01-01T00:00:04Z'}}};phase={'started_epoch':1,'finished_epoch':8};parent={'producer_pid':10,'producer_start_ticks':11,'actual_command':cmd,'actual_model_terminal_epoch':9}
  return [launch,retired,capture,result,phase,parent]
 def test_actual_shaped_complete_launch_join(self):self.assertTrue(binding(*self.fixture())['original_launch_pipe_retirement_crossjoined'])
 def test_foreign_command_parent_pid_start_and_session(self):
  for kind in('command','owner','owner_start','client','client_start','session','exit'):
   args=self.fixture()
   if kind=='command':args[0]['command']=['foreign']
   elif kind=='owner':args[0]['owner']['pid']=99
   elif kind=='owner_start':args[0]['owner']['start_ticks']=True
   elif kind=='client':args[0]['client']['pid']=True
   elif kind=='client_start':args[0]['client']['start_ticks']=0
   elif kind=='session':args[0]['client']['session']=99
   else:args[2]['return_code']=False
   with self.subTest(kind=kind),self.assertRaises(ValueError):binding(*args)
 def test_capture_and_phase_ordering_mutations(self):
  for kind in('late_launch','late_eof','early_capture_end','phase','nan','bool_epoch'):
   args=self.fixture()
   if kind=='late_launch':args[0]['started_epoch']=5
   elif kind=='late_eof':args[2]['completed_epoch']=10
   elif kind=='early_capture_end':args[2]['finished_epoch']=3
   elif kind=='phase':args[4]['finished_epoch']=9
   elif kind=='nan':args[3]['created_epoch']=float('nan')
   else:args[4]['started_epoch']=True
   with self.subTest(kind=kind),self.assertRaises(ValueError):binding(*args)
 def test_container_terminal_after_eof_refused(self):
  args=self.fixture();args[3]['terminal']['State']['FinishedAt']='1970-01-01T00:00:06Z'
  with self.assertRaises(ValueError):binding(*args)
if __name__=='__main__':unittest.main()
